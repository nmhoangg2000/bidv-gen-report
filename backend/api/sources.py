"""
Source Documents API
====================
POST /api/sources/upload   — upload one or more Word/PDF source files
GET  /api/sources/{id}     — get extracted text
"""

import uuid
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from utils.database import get_db
from utils.docx_parser import extract_text_from_file

router = APIRouter()

ALLOWED_TYPES = {".pdf", ".docx", ".doc", ".txt"}


class SourceDocOut(BaseModel):
    id:        str
    filename:  str
    file_type: str
    file_size: int
    preview:   str
    text_len:  int


@router.post("/upload", response_model=List[SourceDocOut])
async def upload_sources(
    files: List[UploadFile] = File(...),
    db:    AsyncSession      = Depends(get_db),
):
    results = []
    for f in files:
        ext = "." + f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED_TYPES:
            continue

        content      = await f.read()
        extracted    = extract_text_from_file(content, f.filename)
        doc_id       = str(uuid.uuid4())

        await db.execute(sa_text("""
            INSERT INTO source_documents
                (id, filename, file_type, file_size, extracted_text)
            VALUES (:id, :fname, :ftype, :fsize, :txt)
        """), {
            "id":    doc_id,
            "fname": f.filename,
            "ftype": ext.lstrip("."),
            "fsize": len(content),
            "txt":   extracted,
        })

        results.append(SourceDocOut(
            id        = doc_id,
            filename  = f.filename,
            file_type = ext.lstrip("."),
            file_size = len(content),
            preview   = extracted[:400],
            text_len  = len(extracted),
        ))

    await db.commit()
    return results


@router.get("")
async def list_sources(db: AsyncSession = Depends(get_db)):
    """Lấy danh sách tất cả tài liệu nguồn, kèm tên template đã dùng."""
    rows = await db.execute(sa_text("""
        SELECT
            sd.id::text, sd.filename, sd.file_type, sd.file_size,
            LEFT(sd.extracted_text, 300) AS preview,
            LENGTH(sd.extracted_text) AS text_len,
            sd.created_at::text,
            ARRAY_AGG(DISTINCT t.name) FILTER (WHERE t.name IS NOT NULL) AS used_in_templates
        FROM source_documents sd
        LEFT JOIN run_source_documents rsd ON rsd.doc_id = sd.id
        LEFT JOIN pipeline_runs pr ON pr.id = rsd.run_id
        LEFT JOIN templates t ON t.id = pr.template_id
        GROUP BY sd.id, sd.filename, sd.file_type, sd.file_size, sd.extracted_text, sd.created_at
        ORDER BY sd.created_at DESC
    """))
    result = []
    for r in rows.mappings():
        d = dict(r)
        d['used_in_templates'] = d.get('used_in_templates') or []
        result.append(d)
    return result


@router.get("/{doc_id}")
async def get_source(doc_id: str, db: AsyncSession = Depends(get_db)):
    row = await db.execute(
        sa_text("SELECT * FROM source_documents WHERE id = :id"),
        {"id": doc_id}
    )
    r = row.mappings().first()
    if not r:
        raise HTTPException(404)
    return dict(r)


@router.delete("/{doc_id}")
async def delete_source(doc_id: str, db: AsyncSession = Depends(get_db)):
    await db.execute(sa_text("DELETE FROM run_source_documents WHERE doc_id = :id"), {"id": doc_id})
    result = await db.execute(sa_text("DELETE FROM source_documents WHERE id = :id RETURNING id"), {"id": doc_id})
    await db.commit()
    if not result.rowcount:
        raise HTTPException(404, "Tài liệu không tồn tại")
    return {"deleted": True}