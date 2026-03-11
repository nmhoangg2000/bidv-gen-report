"""
Pipeline API
============
POST /api/pipeline/start              — create run, kick off LangGraph steps 1-3
GET  /api/pipeline/{run_id}           — get current run state
GET  /api/pipeline/{run_id}/results   — get field results (step 3 output)
POST /api/pipeline/{run_id}/approve   — submit human review edits → resume graph
POST /api/pipeline/{run_id}/export    — generate and download the filled .docx
GET  /api/pipeline                    — list recent runs
"""

import base64
import json
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from agent.pipeline import PipelineState, compiled_graph
from utils.database import get_db
from utils.docx_parser import fill_and_export

router = APIRouter()


# ─── Request / Response Schemas ──────────────────────────────────────────────

class StartPipelineRequest(BaseModel):
    template_id: str
    source_doc_ids: List[str] = []


class ApproveRequest(BaseModel):
    edits: Dict[str, str] = {}   # {field_key: edited_value}
    export_mode: str = "clean"   # clean | tracked


class FieldResultOut(BaseModel):
    field_key:   str
    para_idx:    int
    placeholder: str
    context:     str
    ai_value:    str
    final_value: str
    confidence:  str
    reason:      str
    approved:    bool
    human_edited: bool = False


class RunOut(BaseModel):
    id:           str
    template_id:  str
    status:       str
    current_step: int
    created_at:   str
    updated_at:   str


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/start")
async def start_pipeline(
    req:              StartPipelineRequest,
    background_tasks: BackgroundTasks,
    db:               AsyncSession = Depends(get_db),
):
    """Create a pipeline run and execute steps 1-3 asynchronously."""
    # Load template from DB
    tmpl_row = await db.execute(text(
        "SELECT id::text, name, description FROM templates WHERE id = :id AND is_active"
    ), {"id": req.template_id})
    tmpl = tmpl_row.mappings().first()
    if not tmpl:
        raise HTTPException(404, "Template not found")

    # Load template fields
    fields_row = await db.execute(text("""
        SELECT field_key as key, para_idx, placeholder, context, field_order
        FROM template_fields WHERE template_id = :tid ORDER BY field_order
    """), {"tid": req.template_id})
    fields = [dict(r) for r in fields_row.mappings()]

    # Load source documents
    source_docs = []
    for doc_id in req.source_doc_ids:
        src_row = await db.execute(text(
            "SELECT filename, extracted_text as text FROM source_documents WHERE id = :id"
        ), {"id": doc_id})
        src = src_row.mappings().first()
        if src:
            source_docs.append(dict(src))

    # Create run in DB
    run_id = str(uuid.uuid4())
    await db.execute(text("""
        INSERT INTO pipeline_runs (id, template_id, status, current_step)
        VALUES (:id, :tid, 'created', 0)
    """), {"id": run_id, "tid": req.template_id})

    # Link source docs to run
    for doc_id in req.source_doc_ids:
        await db.execute(text("""
            INSERT INTO run_source_documents (run_id, doc_id) VALUES (:rid, :did)
            ON CONFLICT DO NOTHING
        """), {"rid": run_id, "did": doc_id})

    await db.commit()

    # Build initial LangGraph state
    initial_state = PipelineState(
        run_id         = run_id,
        template_id    = req.template_id,
        template_name  = tmpl["name"],
        template_fields= fields,
        source_docs    = source_docs,
        source_context = "",
        field_results  = [],
        write_progress = 0,
        review_submitted = False,
        human_edits    = {},
        export_mode    = "clean",
        exported_bytes = None,
        current_step   = 0,
        status         = "created",
        error          = None,
        messages       = [],
    )

    # Run the graph in background (steps 1–3, then interrupt before step 4)
    background_tasks.add_task(_run_pipeline_steps, run_id, initial_state)

    return {"run_id": run_id, "status": "started"}


@router.get("", response_model=List[RunOut])
async def list_runs(db: AsyncSession = Depends(get_db)):
    rows = await db.execute(text("""
        SELECT id::text, template_id::text, status, current_step,
               created_at::text, updated_at::text
        FROM pipeline_runs ORDER BY created_at DESC LIMIT 50
    """))
    return [RunOut(**dict(r)) for r in rows.mappings()]


@router.delete("/{run_id}")
async def delete_run(run_id: str, db: AsyncSession = Depends(get_db)):
    await db.execute(text("DELETE FROM field_results WHERE run_id = :id"), {"id": run_id})
    await db.execute(text("DELETE FROM run_source_documents WHERE run_id = :id"), {"id": run_id})
    await db.execute(text("DELETE FROM exported_documents WHERE run_id = :id"), {"id": run_id})
    result = await db.execute(text("DELETE FROM pipeline_runs WHERE id = :id RETURNING id"), {"id": run_id})
    await db.commit()
    if not result.rowcount:
        raise HTTPException(404, "Run not found")
    return {"deleted": True}


@router.delete("")
async def clear_all_runs(db: AsyncSession = Depends(get_db)):
    await db.execute(text("DELETE FROM field_results"))
    await db.execute(text("DELETE FROM run_source_documents"))
    await db.execute(text("DELETE FROM exported_documents"))
    await db.execute(text("DELETE FROM pipeline_runs"))
    await db.commit()
    return {"cleared": True}


@router.get("/{run_id}")
async def get_run(run_id: str, db: AsyncSession = Depends(get_db)):
    row = await db.execute(text("""
        SELECT id::text, template_id::text, status, current_step,
               langgraph_state, error_message,
               created_at::text, updated_at::text
        FROM pipeline_runs WHERE id = :id
    """), {"id": run_id})
    r = row.mappings().first()
    if not r:
        raise HTTPException(404)
    return dict(r)


@router.get("/{run_id}/results", response_model=List[FieldResultOut])
async def get_results(run_id: str, db: AsyncSession = Depends(get_db)):
    rows = await db.execute(text("""
        SELECT fr.field_key, tf.para_idx, tf.placeholder, tf.context,
               fr.ai_value, fr.final_value, fr.confidence, fr.reason,
               fr.approved, fr.human_edited
        FROM field_results fr
        JOIN template_fields tf ON tf.id = fr.field_id
        WHERE fr.run_id = :rid
        ORDER BY tf.field_order
    """), {"rid": run_id})
    results = [dict(r) for r in rows.mappings()]
    if not results:
        # Try to get from LangGraph checkpointer
        results = _get_results_from_graph(run_id)
    return [FieldResultOut(**r) for r in results]


@router.post("/{run_id}/attach_sources")
async def attach_sources(
    run_id: str,
    req:    dict,
    db:     AsyncSession = Depends(get_db),
):
    """Gắn source docs vào run sau khi upload — cập nhật source_docs trong LangGraph state."""
    from agent.pipeline import compiled_graph, PipelineState
    from utils.database import AsyncSessionLocal

    source_doc_ids = req.get("source_doc_ids", [])

    # Lưu mapping run ↔ source docs vào DB
    for doc_id in source_doc_ids:
        await db.execute(text("""
            INSERT INTO run_source_documents (run_id, doc_id)
            VALUES (:rid, :did) ON CONFLICT DO NOTHING
        """), {"rid": run_id, "did": doc_id})
    await db.commit()

    # Load source docs text
    source_docs = []
    for doc_id in source_doc_ids:
        row = await db.execute(text(
            "SELECT filename, extracted_text as text FROM source_documents WHERE id = :id"
        ), {"id": doc_id})
        src = row.mappings().first()
        if src:
            source_docs.append(dict(src))

    # Update LangGraph checkpoint state với source_docs
    config = {"configurable": {"thread_id": run_id}}
    try:
        state = await compiled_graph.aget_state(config)
        if state and state.values:
            await compiled_graph.aupdate_state(
                config,
                {"source_docs": source_docs}
            )
    except Exception:
        pass  # Nếu chưa có checkpoint thì bỏ qua

    return {"attached": len(source_doc_ids)}


@router.post("/{run_id}/confirm_upload")
async def confirm_upload(
    run_id:           str,
    background_tasks: BackgroundTasks,
    db:               AsyncSession = Depends(get_db),
):
    """
    Người dùng bấm 'Xác nhận Upload' → resume graph từ interrupt trước upload_sources.
    source_docs đã có trong initial_state từ lúc start.
    """
    row = await db.execute(text(
        "SELECT status FROM pipeline_runs WHERE id = :id"
    ), {"id": run_id})
    r = row.mappings().first()
    if not r:
        raise HTTPException(404)
    # Cho phép confirm_upload khi pipeline mới start hoặc đang chờ
    if r["status"] not in ("awaiting_upload", "analyzing", "created", "uploading"):
        raise HTTPException(400, f"Run status '{r['status']}' không thể confirm upload")

    await db.execute(text(
        "UPDATE pipeline_runs SET status='uploading', updated_at=NOW() WHERE id=:id"
    ), {"id": run_id})
    await db.commit()

    background_tasks.add_task(_resume_upload, run_id)
    return {"status": "resuming_upload"}


@router.post("/{run_id}/confirm_extract")
async def confirm_extract(
    run_id:           str,
    background_tasks: BackgroundTasks,
    db:               AsyncSession = Depends(get_db),
):
    """
    Người dùng bấm 'Xác nhận Trích xuất' → resume graph từ interrupt trước extract_sources.
    """
    row = await db.execute(text(
        "SELECT status FROM pipeline_runs WHERE id = :id"
    ), {"id": run_id})
    r = row.mappings().first()
    if not r:
        raise HTTPException(404)
    if r["status"] not in ("awaiting_extract", "uploading", "uploaded"):
        raise HTTPException(400, f"Run status '{r['status']}' không thể confirm extract")

    await db.execute(text(
        "UPDATE pipeline_runs SET status='extracting', updated_at=NOW() WHERE id=:id"
    ), {"id": run_id})
    await db.commit()

    background_tasks.add_task(_resume_extract, run_id)
    return {"status": "resuming_extract"}


@router.post("/{run_id}/approve")
async def approve_and_resume(
    run_id:           str,
    req:              ApproveRequest,
    background_tasks: BackgroundTasks,
    db:               AsyncSession = Depends(get_db),
):
    """
    Human submits their review edits.
    Resumes the LangGraph from the interrupt point (before human_review node).
    """
    row = await db.execute(text(
        "SELECT status, langgraph_state FROM pipeline_runs WHERE id = :id"
    ), {"id": run_id})
    r = row.mappings().first()
    if not r:
        raise HTTPException(404)
    if r["status"] not in ("reviewing", "error"):
        raise HTTPException(400, f"Run is in status '{r['status']}', cannot approve")

    # Update run status
    await db.execute(text("""
        UPDATE pipeline_runs
        SET status = 'approved', export_mode = :mode, updated_at = NOW()
        WHERE id = :id
    """), {"id": run_id, "mode": req.export_mode})
    await db.commit()

    # Resume graph in background
    background_tasks.add_task(_resume_pipeline, run_id, req.edits, req.export_mode)
    return {"status": "resuming"}


@router.post("/{run_id}/export")
async def export_docx(
    run_id:      str,
    export_mode: str = "clean",
    db:          AsyncSession = Depends(get_db),
):
    """Generate and return the filled .docx file."""
    # Get template bytes
    run_row = await db.execute(text(
        "SELECT template_id FROM pipeline_runs WHERE id = :id"
    ), {"id": run_id})
    run = run_row.mappings().first()
    if not run:
        raise HTTPException(404)

    tmpl_row = await db.execute(text(
        "SELECT filename, file_data FROM templates WHERE id = :id"
    ), {"id": str(run["template_id"])})
    tmpl = tmpl_row.mappings().first()
    if not tmpl:
        raise HTTPException(404, "Template not found")

    # Get approved field results
    results_row = await db.execute(text("""
        SELECT fr.field_key, tf.para_idx, fr.final_value, fr.confidence
        FROM field_results fr
        JOIN template_fields tf ON tf.id = fr.field_id
        WHERE fr.run_id = :rid
    """), {"rid": run_id})
    field_values = [dict(r) for r in results_row.mappings()]

    if not field_values:
        raise HTTPException(400, "No field results found — run the pipeline first")

    # Fill the template
    try:
        filled_bytes = fill_and_export(
            bytes(tmpl["file_data"]),
            field_values,
            apply_colors=(export_mode == "tracked"),
        )
    except Exception as e:
        raise HTTPException(500, f"Export failed: {e}")

    # Save to DB
    await db.execute(text("""
        INSERT INTO exported_documents (id, run_id, filename, file_data, export_mode)
        VALUES (:id, :rid, :fname, :data, :mode)
    """), {
        "id":    str(uuid.uuid4()),
        "rid":   run_id,
        "fname": tmpl["filename"].replace(".docx", f"_{export_mode}.docx"),
        "data":  filled_bytes,
        "mode":  export_mode,
    })
    await db.execute(text(
        "UPDATE pipeline_runs SET status = 'done', updated_at = NOW() WHERE id = :id"
    ), {"id": run_id})
    await db.commit()

    from urllib.parse import quote
    out_name = tmpl["filename"].replace(".docx", f"_{export_mode}.docx")
    out_name_encoded = quote(out_name)
    cd = "attachment; filename*=UTF-8''" + out_name_encoded
    return Response(
        content=filled_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": cd},
    )



@router.get("/{run_id}/preview-html")
async def preview_html(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Convert filled .docx → HTML for in-browser preview."""
    import subprocess, tempfile, os

    run_row = await db.execute(text(
        "SELECT template_id FROM pipeline_runs WHERE id = :id"
    ), {"id": run_id})
    run = run_row.mappings().first()
    if not run:
        raise HTTPException(404)

    tmpl_row = await db.execute(text(
        "SELECT filename, file_data FROM templates WHERE id = :id"
    ), {"id": str(run["template_id"])})
    tmpl = tmpl_row.mappings().first()
    if not tmpl:
        raise HTTPException(404, "Template not found")

    results_row = await db.execute(text("""
        SELECT fr.field_key, tf.para_idx, fr.final_value, fr.confidence
        FROM field_results fr
        JOIN template_fields tf ON tf.id = fr.field_id
        WHERE fr.run_id = :rid
    """), {"rid": run_id})
    field_values = [dict(r) for r in results_row.mappings()]

    if not field_values:
        raise HTTPException(400, "No field results — run pipeline first")

    try:
        filled_bytes = fill_and_export(
            bytes(tmpl["file_data"]),
            field_values,
            apply_colors=True,  # giữ màu highlight để phân biệt AI-filled
        )
    except Exception as e:
        raise HTTPException(500, f"Fill failed: {e}")

    # Convert docx → HTML bằng pandoc
    try:
        with tempfile.TemporaryDirectory() as tmp:
            docx_path = os.path.join(tmp, "preview.docx")
            html_path = os.path.join(tmp, "preview.html")
            open(docx_path, "wb").write(filled_bytes)

            result = subprocess.run(
                ["pandoc", docx_path, "-o", html_path,
                 "--standalone", "--embed-resources",
                 "--metadata", "title=Preview",
                 "--css=/dev/null"],
                capture_output=True, timeout=30
            )
            if result.returncode != 0:
                raise Exception(result.stderr.decode())

            html = open(html_path, "r", encoding="utf-8").read()

            # Inject A4 styling
            inject_css = """<style>
body{margin:0;padding:32px 48px;font-family:'Times New Roman',serif;
  font-size:13pt;line-height:1.8;color:#111;background:#fff}
p{margin:.4em 0}
table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccc;padding:6px 10px;font-size:12pt}
h1,h2,h3{font-size:13pt;font-weight:bold}
</style>"""
            html = html.replace("</head>", inject_css + "</head>")

    except FileNotFoundError:
        # pandoc không có → fallback render đơn giản
        from utils.docx_parser import extract_fields
        fields_raw = extract_fields(bytes(tmpl["file_data"]))
        val_map = {fv["field_key"]: fv.get("final_value","") for fv in field_values}
        conf_map = {fv["field_key"]: fv.get("confidence","mid") for fv in field_values}

        rows = []
        for f in fields_raw:
            key  = f["key"]
            ctx  = f["context"]
            val  = val_map.get(key, f["placeholder"])
            conf = conf_map.get(key, "mid")
            color = "#d1fae5" if conf=="high" else "#fee2e2" if conf=="low" else "#fef3c7"
            border= "#059669" if conf=="high" else "#dc2626" if conf=="low" else "#d97706"
            rows.append(f"""<p style="margin:10px 0">{ctx.replace(f["placeholder"],
                f'<mark style="background:{color};border-bottom:2px solid {border};padding:1px 3px">{val}</mark>')}</p>""")

        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{{margin:0;padding:32px 48px;font-family:'Times New Roman',serif;
font-size:13pt;line-height:1.8;color:#111;background:#fff}}</style></head>
<body>{"".join(rows)}</body></html>"""

    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)


# ─── Background task helpers ─────────────────────────────────────────────────

async def _run_pipeline_steps(run_id: str, initial_state: PipelineState):
    """
    Execute the LangGraph until the interrupt (before human_review).
    Persist field_results to DB after step 3 completes.
    """
    from utils.database import AsyncSessionLocal

    config = {"configurable": {"thread_id": run_id}}
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text(
                "UPDATE pipeline_runs SET status='analyzing', current_step=1, updated_at=NOW() WHERE id=:id"
            ), {"id": run_id})
            await db.commit()

        # Chạy graph → dừng tại interrupt trước extract_sources
        # (analyze_template + upload_sources chạy tự động)
        await compiled_graph.ainvoke(initial_state, config=config)

        async with AsyncSessionLocal() as db:
            await db.execute(text(
                "UPDATE pipeline_runs SET status='awaiting_extract', current_step=2, updated_at=NOW() WHERE id=:id"
            ), {"id": run_id})
            await db.commit()

    except Exception as e:
        async with AsyncSessionLocal() as db:
            await db.execute(text("""
                UPDATE pipeline_runs
                SET status='error', error_message=:err, updated_at=NOW()
                WHERE id=:id
            """), {"id": run_id, "err": str(e)})
            await db.commit()


async def _resume_upload(run_id: str):
    """Không cần resume graph — interrupt upload_sources đã bị bỏ.
    Chỉ đảm bảo status = awaiting_extract để frontend hiện Phase B."""
    from utils.database import AsyncSessionLocal
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text(
                "UPDATE pipeline_runs SET status='awaiting_extract', current_step=2, updated_at=NOW() WHERE id=:id"
            ), {"id": run_id})
            await db.commit()
    except Exception as e:
        async with AsyncSessionLocal() as db:
            await db.execute(text(
                "UPDATE pipeline_runs SET status='error', error_message=:err, updated_at=NOW() WHERE id=:id"
            ), {"id": run_id, "err": str(e)})
            await db.commit()


async def _resume_extract(run_id: str):
    """Resume graph từ interrupt trước extract_sources → write_fields → dừng trước human_review."""
    from utils.database import AsyncSessionLocal
    import uuid as _uuid
    config = {"configurable": {"thread_id": run_id}}
    try:
        final_state = await compiled_graph.ainvoke(None, config=config)
        async with AsyncSessionLocal() as db:
            run_row = await db.execute(text(
                "SELECT template_id FROM pipeline_runs WHERE id=:id"
            ), {"id": run_id})
            run = run_row.mappings().first()
            tid = str(run["template_id"])

            for r in final_state.get("field_results", []):
                fid_row = await db.execute(text(
                    "SELECT id FROM template_fields WHERE template_id=:tid AND field_key=:key"
                ), {"tid": tid, "key": r["field_key"]})
                fid = fid_row.scalar_one_or_none()
                if not fid:
                    continue
                await db.execute(text("""
                    INSERT INTO field_results
                        (id, run_id, field_id, field_key, ai_value, final_value, confidence, reason, approved)
                    VALUES (:id, :rid, :fid, :key, :ai, :final, :conf, :reason, FALSE)
                    ON CONFLICT (run_id, field_key) DO UPDATE SET
                        ai_value=EXCLUDED.ai_value, final_value=EXCLUDED.final_value,
                        confidence=EXCLUDED.confidence, reason=EXCLUDED.reason, updated_at=NOW()
                """), {
                    "id":     str(_uuid.uuid4()),
                    "rid":    run_id,
                    "fid":    str(fid),
                    "key":    r["field_key"],
                    "ai":     r["ai_value"],
                    "final":  r["final_value"],
                    "conf":   r["confidence"],
                    "reason": r["reason"],
                })
            await db.execute(text("""
                UPDATE pipeline_runs
                SET status='reviewing', current_step=4,
                    langgraph_state=:state, updated_at=NOW()
                WHERE id=:id
            """), {
                "id":    run_id,
                "state": json.dumps({"status": "reviewing", "step": 4}),
            })
            await db.commit()
    except Exception as e:
        async with AsyncSessionLocal() as db:
            await db.execute(text(
                "UPDATE pipeline_runs SET status='error', error_message=:err, updated_at=NOW() WHERE id=:id"
            ), {"id": run_id, "err": str(e)})
            await db.commit()


async def _resume_pipeline(run_id: str, edits: Dict, export_mode: str):
    """Resume the LangGraph after human review interrupt."""
    from utils.database import AsyncSessionLocal

    config = {"configurable": {"thread_id": run_id}}
    try:
        # Inject human edits vào checkpoint state trước khi resume
        await compiled_graph.aupdate_state(
            config,
            {"human_edits": edits, "export_mode": export_mode, "review_submitted": True},
        )
        final_state = await compiled_graph.ainvoke(None, config=config)

        # Persist approved values
        async with AsyncSessionLocal() as db:
            for r in final_state.get("field_results", []):
                await db.execute(text("""
                    UPDATE field_results
                    SET final_value=:val, human_edited=:he, approved=TRUE, updated_at=NOW()
                    WHERE run_id=:rid AND field_key=:key
                """), {
                    "val":  r["final_value"],
                    "he":   r.get("human_edited", False),
                    "rid":  run_id,
                    "key":  r["field_key"],
                })
            await db.execute(text(
                "UPDATE pipeline_runs SET status='exporting', current_step=4, updated_at=NOW() WHERE id=:id"
            ), {"id": run_id})
            await db.commit()

    except Exception as e:
        async with AsyncSessionLocal() as db:
            await db.execute(text("""
                UPDATE pipeline_runs
                SET status='error', error_message=:err, updated_at=NOW()
                WHERE id=:id
            """), {"id": run_id, "err": str(e)})
            await db.commit()
