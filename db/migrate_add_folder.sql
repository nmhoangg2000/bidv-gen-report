-- Migration: thêm cột folder vào source_documents
-- Chạy 1 lần: docker exec -i bidv_db psql -U bidv -d bidv_agent < db/migrate_add_folder.sql

ALTER TABLE source_documents
ADD COLUMN IF NOT EXISTS folder VARCHAR(200) DEFAULT 'Chưa phân loại';
