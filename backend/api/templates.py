"""
Templates API
=============
POST   /api/templates/upload     — upload a .docx template
GET    /api/templates             — list all templates
GET    /api/templates/{id}        — get template + its fields
DELETE /api/templates/{id}        — delete
"""

import uuid
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from utils.database import get_db
from utils.docx_parser import extract_fields

router = APIRouter()


# ─── Schemas ─────────────────────────────────────────────────────────────────

class FieldOut(BaseModel):
    id:          str
    field_key:   str
    para_idx:    int
    placeholder: str
    context:     str
    field_order: int
    field_mode:  str = "replace"


class TemplateOut(BaseModel):
    id:          str
    name:        str
    description: str
    filename:    str
    file_size:   int
    version:     int
    is_active:   bool
    created_at:  str
    fields:      List[FieldOut] = []


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=TemplateOut)
async def upload_template(
    file:        UploadFile = File(...),
    name:        str        = Form(...),
    description: str        = Form(""),
    db:          AsyncSession = Depends(get_db),
):
    """Upload a .docx template file, extract its fields, persist to DB."""
    if not file.filename.endswith(".docx"):
        raise HTTPException(400, "Only .docx files supported")

    # Kiểm tra trùng tên
    dup = await db.execute(text(
        "SELECT id, filename FROM templates WHERE LOWER(name) = LOWER(:name) AND is_active"
    ), {"name": name})
    existing = dup.mappings().first()
    if existing:
        raise HTTPException(409, f"Template tên '{name}' đã tồn tại (file: {existing['filename']}). Dùng tên khác hoặc xóa template cũ trước.")

    content   = await file.read()
    file_size = len(content)

    try:
        raw_fields = extract_fields(content)
    except Exception as e:
        raise HTTPException(422, f"Failed to parse DOCX: {e}")

    # Insert template
    tmpl_id = str(uuid.uuid4())
    await db.execute(text("""
        INSERT INTO templates (id, name, description, filename, file_data, file_size)
        VALUES (:id, :name, :desc, :filename, :data, :size)
    """), {
        "id":       tmpl_id,
        "name":     name,
        "desc":     description,
        "filename": file.filename,
        "data":     content,
        "size":     file_size,
    })

    # Insert fields
    for order, f in enumerate(raw_fields):
        await db.execute(text("""
            INSERT INTO template_fields
                (id, template_id, field_key, para_idx, placeholder, context,
                 field_type, field_mode, field_order)
            VALUES (:id, :tid, :key, :pidx, :ph, :ctx, :ftype, :fmode, :order)
        """), {
            "id":    str(uuid.uuid4()),
            "tid":   tmpl_id,
            "key":   f["key"],
            "pidx":  f["para_idx"],
            "ph":    f["placeholder"],
            "ctx":   f["context"],
            "ftype": f.get("field_type", "sentence"),
            "fmode": f.get("field_mode", "replace"),
            "order": order,
        })

    await db.commit()

    return await _get_template_out(tmpl_id, db)


@router.get("", response_model=List[TemplateOut])
async def list_templates(db: AsyncSession = Depends(get_db)):
    rows = await db.execute(text("""
        SELECT id, name, description, filename, file_size, version, is_active,
               created_at::text
        FROM templates WHERE is_active = TRUE ORDER BY created_at DESC
    """))
    templates = []
    for row in rows.mappings():
        t = dict(row)
        t["fields"] = await _get_fields(t["id"], db)
        t["id"] = str(t["id"])
        templates.append(TemplateOut(**t))
    return templates


@router.get("/{template_id}", response_model=TemplateOut)
async def get_template(template_id: str, db: AsyncSession = Depends(get_db)):
    return await _get_template_out(template_id, db)


@router.get("/{template_id}/preview-html")
async def preview_template_html(template_id: str, db: AsyncSession = Depends(get_db)):
    """Convert template .docx → HTML để xem trước trong browser."""
    import subprocess, tempfile, os as _os
    row = await db.execute(text(
        "SELECT name, filename, file_data FROM templates WHERE id = :id"
    ), {"id": template_id})
    r = row.mappings().first()
    if not r:
        raise HTTPException(404)

    docx_bytes = bytes(r["file_data"])

    try:
        with tempfile.TemporaryDirectory() as tmp:
            docx_path = _os.path.join(tmp, "template.docx")
            html_path = _os.path.join(tmp, "template.html")
            open(docx_path, "wb").write(docx_bytes)

            result = subprocess.run(
                ["pandoc", docx_path, "-o", html_path,
                 "--standalone", "--embed-resources",
                 "--metadata", f"title={r['name']}",
                 "--css=/dev/null"],
                capture_output=True, timeout=30
            )

            if result.returncode == 0:
                html = open(html_path, "r", encoding="utf-8").read()
            else:
                raise FileNotFoundError("pandoc failed")

            # Inject A4 styling + highlight vàng dễ thấy
            inject_css = """<style>
body{margin:0;padding:32px 48px;font-family:'Times New Roman',serif;
  font-size:13pt;line-height:1.8;color:#111;background:#fff}
p{margin:.5em 0}
table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccc;padding:6px 10px;font-size:12pt}
mark{background:#fff59d;padding:0 2px;border-radius:2px}
</style>"""
            html = html.replace("</head>", inject_css + "</head>")

    except FileNotFoundError:
        # Fallback: render fields list nếu không có pandoc
        fields_row = await db.execute(text(
            "SELECT placeholder, context, field_mode FROM template_fields WHERE template_id=:id ORDER BY field_order"
        ), {"id": template_id})
        fields = [dict(f) for f in fields_row.mappings()]
        items = "".join(f"""
            <div style="margin:10px 0;padding:8px 12px;border-left:3px solid #f59e0b;background:#fffbeb;border-radius:4px">
              <div style="font-size:9px;color:#92400e;font-weight:700;margin-bottom:4px">
                {'🟢 Chèn bên dưới' if f.get('field_mode')=='insert_below' else '🟡 Thay thế'}
              </div>
              <mark style="background:#fef08a;padding:1px 4px;border-radius:2px;font-size:12pt">{f['placeholder']}</mark>
              <div style="font-size:10px;color:#6b7280;margin-top:4px">{f['context'][:120]}...</div>
            </div>""" for f in fields)
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{{margin:0;padding:32px 48px;font-family:'Times New Roman',serif;font-size:13pt;line-height:1.8;color:#111}}</style>
</head><body>
<h2 style="font-size:16pt;margin-bottom:4px">{r['name']}</h2>
<p style="font-size:10pt;color:#6b7280">Danh sách {len(fields)} trường cần điền:</p>
{items}
</body></html>"""

    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)


@router.get("/{template_id}/download")
async def download_template(template_id: str, db: AsyncSession = Depends(get_db)):
    row = await db.execute(text(
        "SELECT filename, file_data FROM templates WHERE id = :id"
    ), {"id": template_id})
    r = row.mappings().first()
    if not r:
        raise HTTPException(404)
    return Response(
        content=bytes(r["file_data"]),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{r["filename"]}"'},
    )


@router.delete("")
async def delete_all_templates(db: AsyncSession = Depends(get_db)):
    """Xóa toàn bộ templates và dữ liệu liên quan."""
    # Lấy tất cả run_ids
    runs = await db.execute(text("SELECT id FROM pipeline_runs"))
    run_ids = [str(r["id"]) for r in runs.mappings()]
    for rid in run_ids:
        await db.execute(text("DELETE FROM field_results WHERE run_id = :rid"), {"rid": rid})
        await db.execute(text("DELETE FROM run_source_documents WHERE run_id = :rid"), {"rid": rid})
        await db.execute(text("DELETE FROM exported_documents WHERE run_id = :rid"), {"rid": rid})
    await db.execute(text("DELETE FROM pipeline_runs"))
    await db.execute(text("DELETE FROM template_fields"))
    await db.execute(text("DELETE FROM templates"))
    await db.commit()
    return {"cleared": True}


@router.delete("/{template_id}")
async def delete_template(template_id: str, db: AsyncSession = Depends(get_db)):
    # Lấy danh sách run_ids liên quan
    runs = await db.execute(text(
        "SELECT id FROM pipeline_runs WHERE template_id = :id"
    ), {"id": template_id})
    run_ids = [str(r["id"]) for r in runs.mappings()]

    # Xóa theo thứ tự: child tables trước
    for rid in run_ids:
        await db.execute(text(
            "DELETE FROM field_results WHERE run_id = :rid"
        ), {"rid": rid})
        await db.execute(text(
            "DELETE FROM run_source_documents WHERE run_id = :rid"
        ), {"rid": rid})
        await db.execute(text(
            "DELETE FROM exported_documents WHERE run_id = :rid"
        ), {"rid": rid})

    await db.execute(text(
        "DELETE FROM pipeline_runs WHERE template_id = :id"
    ), {"id": template_id})
    await db.execute(text(
        "DELETE FROM template_fields WHERE template_id = :id"
    ), {"id": template_id})
    result = await db.execute(text(
        "DELETE FROM templates WHERE id = :id RETURNING id"
    ), {"id": template_id})
    await db.commit()
    if not result.rowcount:
        raise HTTPException(404, "Template not found")
    return {"deleted": True}


# ─── Helpers ─────────────────────────────────────────────────────────────────

async def _get_template_out(template_id: str, db: AsyncSession) -> TemplateOut:
    row = await db.execute(text("""
        SELECT id, name, description, filename, file_size, version, is_active,
               created_at::text
        FROM templates WHERE id = :id
    """), {"id": template_id})
    r = row.mappings().first()
    if not r:
        raise HTTPException(404, f"Template {template_id} not found")
    t            = dict(r)
    t["id"]      = str(t["id"])
    t["fields"]  = await _get_fields(template_id, db)
    return TemplateOut(**t)


async def _get_fields(template_id: str, db: AsyncSession) -> List[FieldOut]:
    rows = await db.execute(text("""
        SELECT id::text, field_key, para_idx, placeholder, context, field_order,
               COALESCE(field_mode,'replace') as field_mode
        FROM template_fields WHERE template_id = :tid ORDER BY field_order
    """), {"tid": template_id})
    return [FieldOut(**dict(r)) for r in rows.mappings()]