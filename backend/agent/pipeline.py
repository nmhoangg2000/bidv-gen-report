"""
BIDV Report AI Agent — LangGraph Pipeline (OpenAI)
Graph: analyze_template → upload_sources ⏸ → extract_sources ⏸ → write_fields ⏸ → human_review → export_doc
"""

import json
import os
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


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


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

async def node_analyze_template(state: PipelineState) -> PipelineState:
    n = len(state["template_fields"])
    return {
        **state,
        "current_step": 1,
        "status": "analyzing",
        "messages": [HumanMessage(content=
            f"[Step 1] Template '{state['template_name']}' loaded. {n} fields to fill."
        )],
    }


# ─── Node 2a: upload_sources ──────────────────────────────────────────────────

async def node_upload_sources(state: PipelineState) -> PipelineState:
    docs = state.get("source_docs", [])
    return {
        **state,
        "current_step": 2,
        "status": "uploaded",
        "messages": [HumanMessage(content=
            f"[Step 2a] {len(docs)} source doc(s) uploaded."
        )],
    }


# ─── Node 2b: extract_sources (interrupt trước node này) ──────────────────────

async def node_extract_sources(state: PipelineState) -> PipelineState:
    docs = state.get("source_docs", [])
    context = "\n\n".join(
        f"=== {d['filename']} ===\n{d['text'][:10000]}"
        for d in docs
    ) if docs else "(Không có tài liệu nguồn — AI sẽ generate dựa trên ngữ cảnh template)"

    return {
        **state,
        "current_step":   3,
        "status":         "extracting",
        "source_context": context,
        "messages": [HumanMessage(content=
            f"[Step 2b] Context built: {len(context):,} chars."
        )],
    }


# ─── Node 3: write_fields ─────────────────────────────────────────────────────

async def node_write_fields(state: PipelineState) -> PipelineState:
    import asyncio, re as _re
    client    = get_client()
    fields    = state["template_fields"]
    full_ctx  = state.get("source_context", "")
    model     = os.getenv("OPENAI_MODEL", "gpt-4o")  # dùng gpt-4o cho chất lượng tốt hơn
    semaphore = asyncio.Semaphore(int(os.getenv("OPENAI_CONCURRENCY", "2")))

    from datetime import datetime
    today      = datetime.now()
    today_str  = today.strftime("%d/%m/%Y")
    month_year = today.strftime("tháng %m/%Y")
    quarter    = f"Quý {(today.month-1)//3+1}/{today.year}"

    # ── Smart context: tìm đoạn liên quan nhất cho từng field ────────────────
    def extract_relevant_context(placeholder: str, context: str, max_chars: int = 8000) -> str:
        """Tìm các đoạn trong tài liệu nguồn liên quan nhất đến placeholder."""
        if not context or len(context) <= max_chars:
            return context

        # Tách thành các đoạn nhỏ (~500 ký tự mỗi đoạn)
        paragraphs = [p.strip() for p in _re.split(r'\n{2,}', context) if p.strip()]

        # Keywords từ placeholder
        ph_lower = placeholder.lower()
        keywords = set(_re.findall(r'[a-zA-ZÀ-ỹ]{4,}', ph_lower))

        # Score từng đoạn theo keyword overlap
        scored = []
        for para in paragraphs:
            para_lower = para.lower()
            score = sum(1 for kw in keywords if kw in para_lower)
            # Ưu tiên đoạn có số liệu
            score += len(_re.findall(r'\d+[.,]?\d*\s*(%|tỷ|triệu|nghìn|người|dự án)', para_lower))
            scored.append((score, para))

        # Lấy các đoạn điểm cao nhất, giữ nguyên thứ tự
        top = sorted(scored, key=lambda x: -x[0])[:15]
        top_set = {id(p) for _, p in top}
        ordered = [p for _, p in scored if id(p) in top_set]

        result = "\n\n".join(ordered)
        return result[:max_chars] if len(result) > max_chars else result

    system_msg = f"""Bạn là chuyên gia cao cấp soạn thảo văn bản hành chính ngân hàng BIDV.

NGÀY HIỆN TẠI: {today_str} ({month_year}, {quarter})

NGUYÊN TẮC VÀNG — KHÔNG ĐƯỢC VI PHẠM:
1. CHỈ viết những gì có bằng chứng trong tài liệu nguồn — không suy đoán, không bịa số liệu
2. Trích dẫn CHÍNH XÁC: số, %, ngày tháng, tên văn bản phải đúng với tài liệu nguồn
3. Mỗi câu phải có ít nhất 1 thông tin định lượng (số, %, tỷ đồng) hoặc định danh cụ thể
4. KHÔNG dùng: "đã triển khai tốt", "kết quả khả quan", "tiếp tục phát huy" — quá chung chung
5. Output PHẢI dài ít nhất bằng placeholder gốc

VĂN PHONG CHUẨN BIDV:
- Chủ ngữ: "BIDV", "Ngân hàng" (không dùng "chúng tôi", "đơn vị")
- Đơn vị: tỷ đồng, %, triệu người, lệnh/ngày
- Thời gian: "Quý I/2026", "tháng 3/2026", "ngày 12/05/2025"
- Tên viết hoa đúng: BIDV, NHNN, CSDLQG, VNeID, NQ57, Bộ Công an

MẪU CHUẨN:
✗ "BIDV đã hoàn thành nhiều nhiệm vụ quan trọng trong năm 2025."
✓ "Trong năm 2025, BIDV ban hành Nghị quyết 488/NQ-BIDV ngày 12/05/2025, xác thực sinh trắc học cho 9,8 triệu KHCN qua CCCD gắn chip và VNeID, đạt kỷ lục 20 triệu giao dịch/ngày vào 20/10/2025, hoàn thành Core Banking Myanmar với thời gian kỷ lục."

Trả lời CHỈ bằng JSON hợp lệ."""

    async def call_one(field) -> FieldResult:
        placeholder = field.get("placeholder", "")
        ph_len      = len(placeholder)
        field_type  = field.get("field_type", "sentence")
        type_hint   = field.get("type_hint", "")

        # Trích xuất context liên quan nhất cho field này
        relevant_ctx = extract_relevant_context(placeholder, full_ctx, max_chars=10000)

        user_msg = f"""Điền nội dung vào ô còn trống trong báo cáo BIDV.

=== LOẠI TRƯỜNG (BẮT BUỘC ĐỌC KỸ) ===
{type_hint}

=== THÔNG TIN TRƯỜNG ===
Placeholder ({ph_len} ký tự): "{placeholder}"
Ngữ cảnh xung quanh: {field['context']}

=== TÀI LIỆU NGUỒN ===
{relevant_ctx}

=== YÊU CẦU ===
- Độ dài TỐI THIỂU: {max(ph_len, 80)} ký tự
- Liệt kê TẤT CẢ số liệu liên quan từ tài liệu nguồn
- KHÔNG viết chung chung — phải có số hoặc tên cụ thể

Trả về JSON:
{{
  "value": "nội dung điền theo đúng LOẠI TRƯỜNG và văn phong hành chính BIDV",
  "confidence": "high|mid|low",
  "reason": "lấy từ đâu hoặc tự generate"
}}"""

        if ph_len < 50:
            max_tok = 600
        elif ph_len < 200:
            max_tok = 2000
        elif ph_len < 500:
            max_tok = 3500
        else:
            max_tok = 4096

        async with semaphore:
            parsed   = None
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user",   "content": user_msg},
                        ],
                        max_tokens=max_tok,
                        temperature=0.2,
                        response_format={"type": "json_object"},
                    )
                    parsed = json.loads(response.choices[0].message.content.strip())
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "rate_limit" in err_str.lower():
                        import re as _re2
                        wait_match = _re2.search(r'try again in (\d+\.?\d*)s', err_str)
                        wait_secs  = float(wait_match.group(1)) if wait_match else (2 ** attempt * 3)
                        wait_secs  = min(wait_secs + 1, 60)
                        print(f"[Rate limit] attempt {attempt+1}/{max_retries} — wait {wait_secs:.1f}s")
                        import asyncio as _asyncio
                        await _asyncio.sleep(wait_secs)
                        continue
                    parsed = {"value": "", "confidence": "low", "reason": f"Lỗi: {e}"}
                    break
            if parsed is None:
                parsed = {"value": "", "confidence": "low", "reason": "Vượt quá rate limit"}

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

    results: List[FieldResult] = list(await asyncio.gather(*[call_one(f) for f in fields]))

    return {
        **state,
        "current_step":  4,
        "status":        "reviewing",
        "field_results": results,
        "messages": [HumanMessage(content=
            f"[Step 3] {len(results)} fields filled."
        )],
    }


# ─── Node 4: human_review ─────────────────────────────────────────────────────

async def node_human_review(state: PipelineState) -> PipelineState:
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
        **state,
        "current_step":  5,
        "status":        "exporting",
        "field_results": updated,
        "messages": [HumanMessage(content=
            f"[Step 4] Review complete. {len(edits)} fields edited."
        )],
    }


# ─── Node 5: export_doc ───────────────────────────────────────────────────────

async def node_export_doc(state: PipelineState) -> PipelineState:
    return {
        **state,
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


# compiled_graph — dùng cho LangGraph Studio/API (không có checkpointer)
compiled_graph = build_graph().compile(
    interrupt_before=["extract_sources", "human_review"],
)

# compiled_graph_with_memory — dùng cho FastAPI/Docker (có MemorySaver)
_checkpointer = MemorySaver()
compiled_graph_with_memory = build_graph().compile(
    checkpointer=_checkpointer,
    interrupt_before=["extract_sources", "human_review"],
)