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
                (id, template_id, field_key, para_idx, placeholder, context, field_order)
            VALUES (:id, :tid, :key, :pidx, :ph, :ctx, :order)
        """), {
            "id":    str(uuid.uuid4()),
            "tid":   tmpl_id,
            "key":   f["key"],
            "pidx":  f["para_idx"],
            "ph":    f["placeholder"],
            "ctx":   f["context"],
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
        SELECT id::text, field_key, para_idx, placeholder, context, field_order
        FROM template_fields WHERE template_id = :tid ORDER BY field_order
    """), {"tid": template_id})
    return [FieldOut(**dict(r)) for r in rows.mappings()]