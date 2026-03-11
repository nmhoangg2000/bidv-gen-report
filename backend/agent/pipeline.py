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
    source_docs:      List[Dict]   # inject sau interrupt upload_sources
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


# ─── Node 2a: upload_sources (interrupt trước node này) ───────────────────────
# Node này chỉ đánh dấu status — dữ liệu source_docs được inject từ bên ngoài
# khi người dùng bấm "Xác nhận Upload"

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
# Node này thực sự ghép text — chạy sau khi người dùng bấm "Xác nhận Trích xuất"

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
    client  = get_client()
    fields  = state["template_fields"]
    context = state.get("source_context", "")
    model   = os.getenv("OPENAI_MODEL", "gpt-4o")
    results: List[FieldResult] = []

    system_msg = """Bạn là chuyên gia soạn thảo văn bản hành chính ngân hàng BIDV.
Nhiệm vụ: điền nội dung vào các trường còn trống trong báo cáo dựa trên tài liệu nguồn được cung cấp.

VĂN PHONG BẮT BUỘC:
- Ngôn ngữ trang trọng, hành chính — tuyệt đối không dùng ngôn ngữ thông thường
- Câu văn súc tích, đầy đủ thông tin, không dài dòng
- Dùng chủ ngữ "BIDV", "Ngân hàng", "Đơn vị" — không dùng "chúng tôi", "tôi"
- Số liệu phải có đơn vị rõ ràng: tỷ đồng, %, người, dự án...
- Thời gian viết đầy đủ: "Quý I/2025", "tháng 10/2025", "năm 2025"
- Viết hoa đúng danh từ riêng: BIDV, NHNN, Bộ Công an, NQ57...
- Không dùng dấu chấm lửng (...), không viết tắt tùy tiện
- Kết quả phải cụ thể, có số liệu — tránh chung chung kiểu "đã triển khai tốt"

VÍ DỤ VĂN PHONG ĐÚNG:
✓ "BIDV đã hoàn thành xác thực sinh trắc học cho 8,2 triệu khách hàng cá nhân, đạt 94% kế hoạch năm."
✓ "Trong Quý III/2025, BIDV triển khai 12 dự án CNTT trọng điểm, giải ngân 320 tỷ đồng."
✗ SAI: "Ngân hàng đã làm tốt việc này và đạt được nhiều kết quả khả quan..."

Luôn trả lời bằng JSON hợp lệ, không thêm markdown hay giải thích ngoài JSON."""

    for field in fields:
        user_msg = f"""Điền nội dung vào trường còn trống trong báo cáo BIDV.

NGỮ CẢNH ĐOẠN VĂN (phần văn bản xung quanh trường cần điền):
{field['context']}

NỘI DUNG CẦN THAY THẾ (placeholder hiện tại):
"{field['placeholder']}"

TÀI LIỆU NGUỒN (dùng để tìm số liệu và thông tin chính xác):
{context[:14000]}

HƯỚNG DẪN:
1. Ưu tiên tìm số liệu/thông tin CỤ THỂ từ tài liệu nguồn
2. Nội dung phải PHÙ HỢP với ngữ cảnh đoạn văn xung quanh
3. Độ dài phù hợp với placeholder — nếu placeholder ngắn thì điền ngắn, dài thì điền dài
4. Giữ nguyên định dạng nếu placeholder có cấu trúc đặc biệt (danh sách, bảng...)

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

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            raw    = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
        except Exception as e:
            parsed = {"value": "", "confidence": "low", "reason": f"Lỗi: {e}"}

        key = field.get("key") or field.get("field_key", "")
        results.append(FieldResult(
            field_key   = key,
            para_idx    = field.get("para_idx", 0),
            placeholder = field.get("placeholder", ""),
            context     = field.get("context", ""),
            ai_value    = parsed.get("value", ""),
            final_value = parsed.get("value", ""),
            confidence  = parsed.get("confidence", "low"),
            reason      = parsed.get("reason", ""),
            approved    = False,
        ))

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
    builder.add_node("upload_sources",   node_upload_sources)   # 2a
    builder.add_node("extract_sources",  node_extract_sources)  # 2b
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


_checkpointer = MemorySaver()

compiled_graph = build_graph().compile(
    checkpointer=_checkpointer,
    interrupt_before=["extract_sources", "human_review"],
)