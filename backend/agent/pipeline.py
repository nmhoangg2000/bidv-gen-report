"""
BIDV Report AI Agent — LangGraph Pipeline (OpenAI)
Graph: analyze_template → upload_sources ⏸ → extract_sources ⏸ → write_fields ⏸ → human_review → export_doc
"""

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Annotated, Dict, List, Optional, TypedDict

from openai import AsyncOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ── LangSmith tracing ──────────────────────────────────────────────────────
try:
    from langchain_core.tracers.langchain import LangChainTracer
    _ls_project = os.getenv("LANGCHAIN_PROJECT", "bidv-agent")
    _ls_tracer  = LangChainTracer(project_name=_ls_project)
    _ls_enabled = bool(os.getenv("LANGCHAIN_API_KEY"))
    if _ls_enabled:
        print(f"[LangSmith] Tracing enabled → project: {_ls_project}")
except Exception as e:
    _ls_tracer  = None
    _ls_enabled = False
    print(f"[LangSmith] Tracing disabled: {e}")

def get_ls_callbacks():
    return [_ls_tracer] if (_ls_enabled and _ls_tracer) else []


# ── OpenAI client singleton ────────────────────────────────────────────────
_client: Optional[AsyncOpenAI] = None

def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    return _client


# ─── State Schema ─────────────────────────────────────────────────────────────

class FieldResult(TypedDict):
    field_key:   str
    para_idx:    int
    placeholder: str
    context:     str
    ai_value:    str
    final_value: str
    confidence:  str   # high | mid | low
    reason:      str
    approved:    bool


class PipelineState(TypedDict):
    run_id:           str
    template_id:      str
    template_name:    str
    template_fields:  List[Dict]
    source_docs:      List[Dict]
    source_context:   str
    field_results:    List[FieldResult]
    write_progress:   int
    review_submitted: bool
    human_edits:      Dict[str, str]
    export_mode:      str
    exported_bytes:   Optional[str]
    current_step:     int
    status:           str
    error:            Optional[str]
    messages:         Annotated[List, add_messages]


# ─── Node 1: analyze_template ─────────────────────────────────────────────────

async def node_analyze_template(state: PipelineState) -> dict:
    n = len(state["template_fields"])
    return {
        "current_step": 1,
        "status": "analyzing",
        "messages": [HumanMessage(content=
            f"[Step 1] Template '{state['template_name']}' loaded. {n} fields to fill."
        )],
    }


# ─── Node 2a: upload_sources ──────────────────────────────────────────────────

async def node_upload_sources(state: PipelineState) -> dict:
    docs = state.get("source_docs", [])
    return {
        "current_step": 2,
        "status": "uploaded",
        "messages": [HumanMessage(content=
            f"[Step 2a] {len(docs)} source doc(s) uploaded."
        )],
    }


# ─── Node 2b: extract_sources (interrupt trước node này) ──────────────────────

async def node_extract_sources(state: PipelineState) -> dict:
    docs = state.get("source_docs", [])
    context = "\n\n".join(
        f"=== {d['filename']} ===\n{d['text'][:10000]}"
        for d in docs
    ) if docs else "(Không có tài liệu nguồn — AI sẽ generate dựa trên ngữ cảnh template)"

    return {
        "current_step":   3,
        "status":         "extracting",
        "source_context": context,
        "messages": [HumanMessage(content=
            f"[Step 2b] Context built: {len(context):,} chars."
        )],
    }


# ─── Node 3: write_fields ─────────────────────────────────────────────────────

async def node_write_fields(state: PipelineState) -> dict:
    client    = get_client()
    fields    = state["template_fields"]
    context   = state.get("source_context", "")
    model     = os.getenv("OPENAI_MODEL", "gpt-4o")
    semaphore = asyncio.Semaphore(int(os.getenv("OPENAI_CONCURRENCY", "3")))

    today      = datetime.now()
    today_str  = today.strftime("%d/%m/%Y")
    month_year = today.strftime("tháng %m/%Y")
    quarter    = f"Quý {(today.month - 1) // 3 + 1}/{today.year}"

    system_msg = f"""Bạn là chuyên gia soạn thảo văn bản hành chính ngân hàng BIDV.
Nhiệm vụ: điền nội dung CHI TIẾT, ĐẦY ĐỦ vào các trường còn trống dựa trên tài liệu nguồn.

NGÀY HIỆN TẠI: {today_str} ({month_year}, {quarter})

NGUYÊN TẮC BẮT BUỘC — QUAN TRỌNG NHẤT:
1. LUÔN trích xuất số liệu cụ thể từ tài liệu nguồn: con số, %, tỷ đồng, số khách hàng, số dự án, ngày tháng
2. KHÔNG ĐƯỢC viết câu chung chung: "đã triển khai tốt", "đạt kết quả khả quan", "tiếp tục phát triển"
3. Mỗi câu PHẢI có ít nhất 1 thông tin định lượng hoặc định danh cụ thể
4. Nếu tài liệu nguồn có nhiều số liệu liên quan → ĐƯA TẤT CẢ vào, không bỏ sót
5. Output PHẢI dài bằng hoặc HƠN placeholder gốc — TUYỆT ĐỐI không rút gọn

ĐỘ DÀI BẮT BUỘC theo độ dài placeholder:
- < 50 ký tự: tối thiểu 2 câu có số liệu cụ thể
- 50-200 ký tự: tối thiểu 4-6 câu, đủ ý, đủ số liệu
- 200-500 ký tự: tối thiểu 6-10 câu, liệt kê nhiều kết quả
- > 500 ký tự: viết đầy đủ nhiều đoạn, tương đương bản gốc

VĂN PHONG:
- Ngôn ngữ trang trọng, hành chính
- Chủ ngữ: "BIDV", "Ngân hàng" — không dùng "chúng tôi"
- Đơn vị đầy đủ: tỷ đồng, %, triệu người, dự án
- Thời gian cụ thể: "Quý I/2026", "tháng 3/2026", "ngày 12/05/2025"
- Viết hoa danh từ riêng: BIDV, NHNN, Bộ Công an, NQ57, CSDLQG

SO SÁNH ĐÚNG/SAI:
✗ SAI: "BIDV đã triển khai nhiều giải pháp, đạt kết quả tốt."
✓ ĐÚNG: "BIDV đã xác thực sinh trắc học cho 9,8 triệu khách hàng cá nhân qua CCCD gắn chip và VNeID, trong đó 9,4 triệu qua kênh quầy và Smartbanking, 442.000 qua ứng dụng VNeID tính đến ngày 10/11/2025."

Trả lời CHỈ bằng JSON hợp lệ, không markdown."""

    async def call_one(field) -> FieldResult:
        field_type = field.get("field_type", "sentence")
        type_hint  = field.get("type_hint", "")
        ph_len     = len(field.get("placeholder", ""))

        user_msg = f"""Điền nội dung vào trường còn trống trong báo cáo BIDV.

LOẠI TRƯỜNG: {field_type.upper()}
{type_hint}

NGỮ CẢNH ĐOẠN VĂN (phần văn bản xung quanh trường cần điền):
{field['context']}

NỘI DUNG CẦN THAY THẾ (placeholder hiện tại):
"{field['placeholder']}"

TÀI LIỆU NGUỒN (dùng để tìm số liệu và thông tin chính xác):
{context[:14000]}

ĐỘ DÀI YÊU CẦU: Placeholder gốc dài {ph_len} ký tự → output phải dài TỐI THIỂU {max(ph_len, 50)} ký tự
BẮT BUỘC: Trích xuất và liệt kê TẤT CẢ số liệu, kết quả, dự án có trong tài liệu nguồn liên quan đến nội dung này

HƯỚNG DẪN:
1. Đọc kỹ tài liệu nguồn, tìm TẤT CẢ số liệu liên quan đến placeholder này
2. Liệt kê đầy đủ: số lượng, %, tỷ đồng, tên dự án, ngày tháng, đơn vị thực hiện
3. Nếu placeholder là đoạn văn dài → viết nhiều câu, nhiều ý, không bỏ sót
4. KHÔNG viết câu chung chung — mỗi câu phải có ít nhất 1 con số hoặc tên cụ thể
5. Nội dung phải PHÙ HỢP ngữ cảnh xung quanh

Trả về JSON (chỉ JSON, không markdown):
{{
  "value": "nội dung điền vào theo đúng văn phong hành chính ngân hàng",
  "confidence": "high|mid|low",
  "reason": "lý do ngắn gọn: tìm thấy ở đâu trong tài liệu nguồn, hoặc lý do tự generate"
}}

Quy tắc confidence:
- high: tìm thấy số liệu/thông tin cụ thể từ tài liệu nguồn, điền trực tiếp
- mid : suy luận có căn cứ từ ngữ cảnh, không có số liệu cụ thể
- low : không có tài liệu nguồn hoặc không tìm thấy thông tin liên quan, tự generate"""

        if ph_len < 50:
            max_tok = 500
        elif ph_len < 200:
            max_tok = 1500
        elif ph_len < 500:
            max_tok = 3000
        else:
            max_tok = 4000

        parsed = None
        async with semaphore:
            for attempt in range(5):
                try:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_msg},
                                {"role": "user",   "content": user_msg},
                            ],
                            max_tokens=max_tok,
                            temperature=0.3,
                            response_format={"type": "json_object"},
                        ),
                        timeout=90,
                    )
                    parsed = json.loads(response.choices[0].message.content.strip())
                    break
                except asyncio.TimeoutError:
                    parsed = {"value": "", "confidence": "low", "reason": "Timeout sau 90s"}
                    break
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
                    is_transient  = any(x in err_str for x in ("502", "503", "504", "connection"))
                    if (is_rate_limit or is_transient) and attempt < 4:
                        wait_match = re.search(r"try again in (\d+\.?\d*)s", err_str)
                        wait_secs  = float(wait_match.group(1)) if wait_match else 2 ** (attempt + 1)
                        wait_secs  = min(wait_secs + 1, 60)
                        print(f"[write_fields] retry {attempt + 1}/5 — wait {wait_secs:.1f}s ({err_str[:60]})")
                        await asyncio.sleep(wait_secs)
                        continue
                    parsed = {"value": "", "confidence": "low", "reason": f"Lỗi: {err_str[:200]}"}
                    break

        if parsed is None:
            parsed = {"value": "", "confidence": "low", "reason": "Vượt quá số lần thử lại"}

        key = field.get("key") or field.get("field_key", "")
        return FieldResult(
            field_key   = key,
            para_idx    = field.get("para_idx", 0),
            placeholder = field.get("placeholder", ""),
            context     = field.get("context", ""),
            ai_value    = parsed.get("value", ""),
            final_value = parsed.get("value", ""),
            confidence  = parsed.get("confidence", "low"),
            reason      = parsed.get("reason", ""),
            approved    = False,
        )

    raw = await asyncio.gather(*[call_one(f) for f in fields], return_exceptions=True)

    results: List[FieldResult] = []
    for i, item in enumerate(raw):
        if isinstance(item, Exception):
            f   = fields[i]
            key = f.get("key") or f.get("field_key", "")
            results.append(FieldResult(
                field_key   = key,
                para_idx    = f.get("para_idx", 0),
                placeholder = f.get("placeholder", ""),
                context     = f.get("context", ""),
                ai_value    = "",
                final_value = "",
                confidence  = "low",
                reason      = f"Exception: {item}",
                approved    = False,
            ))
        else:
            results.append(item)

    return {
        "current_step":   4,
        "status":         "reviewing",
        "field_results":  results,
        "write_progress": len(results),
        "messages": [HumanMessage(content=
            f"[Step 3] {len(results)} fields filled."
        )],
    }


# ─── Node 4: human_review ─────────────────────────────────────────────────────

async def node_human_review(state: PipelineState) -> dict:
    edits   = state.get("human_edits", {})
    results = state.get("field_results", [])

    updated = []
    for r in results:
        edited_value = edits.get(r["field_key"])
        updated.append(FieldResult(**{
            **r,
            "final_value": edited_value if edited_value is not None else r["final_value"],
            "approved":    True,
        }))

    return {
        "current_step":  5,
        "status":        "exporting",
        "field_results": updated,
        "messages": [HumanMessage(content=
            f"[Step 4] Review complete. {len(edits)} fields edited."
        )],
    }


# ─── Node 5: export_doc ───────────────────────────────────────────────────────

async def node_export_doc(state: PipelineState) -> dict:
    return {
        "current_step": 6,
        "status":       "done",
        "messages": [HumanMessage(content="[Step 5] Pipeline complete.")],
    }


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(PipelineState)
    builder.add_node("analyze_template", node_analyze_template)
    builder.add_node("upload_sources",   node_upload_sources)
    builder.add_node("extract_sources",  node_extract_sources)
    builder.add_node("write_fields",     node_write_fields)
    builder.add_node("human_review",     node_human_review)
    builder.add_node("export_doc",       node_export_doc)

    builder.add_edge(START,              "analyze_template")
    builder.add_edge("analyze_template", "upload_sources")
    builder.add_edge("upload_sources",   "extract_sources")
    builder.add_edge("extract_sources",  "write_fields")
    builder.add_edge("write_fields",     "human_review")
    builder.add_edge("human_review",     "export_doc")
    builder.add_edge("export_doc",       END)
    return builder


# compiled_graph — for LangGraph Studio (no checkpointer, no interrupt/resume support)
compiled_graph = build_graph().compile()

# compiled_graph_with_memory — for FastAPI/Docker (MemorySaver checkpointer, supports interrupt/resume)
_checkpointer = MemorySaver()
compiled_graph_with_memory = build_graph().compile(
    checkpointer=_checkpointer,
    interrupt_before=["extract_sources", "human_review"],
)
