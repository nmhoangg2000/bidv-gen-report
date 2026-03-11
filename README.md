# BIDV Report AI Agent

LangGraph agentic pipeline + PostgreSQL + Docker cho việc tự động điền các mẫu báo cáo BIDV.

## Kiến trúc

```
┌─────────────┐     ┌──────────────────────────────────────────┐     ┌──────────────┐
│   Nginx     │────▶│  FastAPI Backend                         │────▶│  PostgreSQL  │
│  :3000      │     │  :8000                                   │     │  + pgvector  │
│  (frontend) │     │  ┌────────────────────────────────────┐  │     │  :5432       │
└─────────────┘     │  │  LangGraph Pipeline                │  │     └──────────────┘
                    │  │                                    │  │
                    │  │  analyze → extract → write         │  │
                    │  │           ↓                        │  │
                    │  │     [INTERRUPT]                    │  │
                    │  │           ↓                        │  │
                    │  │  human_review → export             │  │
                    │  └────────────────────────────────────┘  │
                    └──────────────────────────────────────────┘
```

## Database Schema

| Table | Mô tả |
|-------|-------|
| `templates` | File .docx + metadata, lưu binary |
| `template_fields` | Các trường tô vàng của mỗi template |
| `source_documents` | File Word/PDF nguồn được upload |
| `pipeline_runs` | Mỗi lần chạy pipeline |
| `field_results` | Kết quả AI + human edit cho từng field |
| `exported_documents` | File .docx output đã xuất |

## Cài đặt & Chạy

### 1. Yêu cầu
- Docker Desktop
- OpenAI API key

### 2. Clone và cấu hình
```bash
git clone <repo>
cd bidv-agent
cp .env.example .env
# Mở .env, điền OPENAI_API_KEY của bạn
```

### 3. Khởi động
```bash
docker compose up --build
```

Lần đầu build mất ~3-5 phút (cài Python packages, pandoc).

### 4. Truy cập
- **Web UI**: http://localhost:3000
- **API docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432 (user: `bidv`, pass: `bidv_secret`, db: `bidv_agent`)

## Sử dụng

### Bước 0: Upload template
1. Mở http://localhost:3000
2. Click **"+ Upload Template"**
3. Upload file `.docx` có các trường **tô màu vàng**
4. Đặt tên và mô tả

### Bước 1-5: Chạy pipeline
1. **Chọn template** trong sidebar
2. **Upload tài liệu nguồn** (báo cáo cũ, file số liệu Word/PDF) — hoặc bỏ qua
3. Click **"Chạy Pipeline"** — LangGraph sẽ tự động:
   - Node 1: Phân tích template fields
   - Node 2: Chuẩn bị source context
   - Node 3: Claude điền từng trường + đánh giá confidence
   - *(INTERRUPT — chờ người dùng)*
4. **Review** kết quả: 🟢 cao · 🟡 trung bình · 🔴 cần kiểm tra
5. Chỉnh sửa nếu cần, click **"Phê duyệt"**
6. Click **"Tải xuống"** — file .docx được generate từ server

## API Endpoints

```
POST /api/templates/upload          Upload template .docx
GET  /api/templates                 Danh sách templates
GET  /api/templates/{id}            Chi tiết template + fields
GET  /api/templates/{id}/download   Download file gốc

POST /api/sources/upload            Upload source files (Word/PDF)

POST /api/pipeline/start            Bắt đầu pipeline run
GET  /api/pipeline/{run_id}         Trạng thái run
GET  /api/pipeline/{run_id}/results Field results (step 3 output)
POST /api/pipeline/{run_id}/approve Submit human edits, resume graph
POST /api/pipeline/{run_id}/export  Generate & download filled .docx
GET  /api/pipeline                  Lịch sử runs
```

## LangGraph Flow

```python
# agent/pipeline.py

# Graph nodes
analyze_template  → Validate fields loaded from DB
extract_sources   → Prepare concatenated source context  
write_fields      → Claude fills each field with confidence score
human_review      → Apply human edits (RESUMED after interrupt)
export_doc        → Mark done, signal ready for export

# Key: interrupt_before=["human_review"]
# Pipeline PAUSES here, waits for POST /approve
```

## Dừng & Reset

```bash
# Dừng
docker compose down

# Dừng và xóa database
docker compose down -v
```
