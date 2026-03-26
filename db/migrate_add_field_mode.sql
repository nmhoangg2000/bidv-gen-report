-- Migration: thêm cột field_mode vào template_fields
-- Chạy: Get-Content db/migrate_add_field_mode.sql | docker exec -i bidv_db psql -U bidv -d bidv_agent

ALTER TABLE template_fields
  ADD COLUMN IF NOT EXISTS field_mode VARCHAR(20) DEFAULT 'replace';

-- Kiểm tra kết quả
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'template_fields' AND column_name = 'field_mode';