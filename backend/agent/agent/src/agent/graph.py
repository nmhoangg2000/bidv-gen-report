"""
BIDV Report AI Agent — LangGraph Pipeline
==========================================

Graph nodes:
  analyze_template → extract_sources → write_fields → human_review → export_doc
                                                            ↑
                                                     INTERRUPT here
                                                   (wait for user approve)

State is persisted to PostgreSQL between steps so the run can be
resumed after the human_review interrupt.
"""

"""
BIDV Report AI Agent — LangGraph Pipeline (OpenAI)
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


# ─── Node 2: extract_sources ──────────────────────────────────────────────────

async def node_extract_sources(state: PipelineState) -> PipelineState:
    docs = state.get("source_docs", [])
    context = "\n\n".join(
        f"=== {d['filename']} ===\n{d['text'][:10000]}"
        for d in docs
    ) if docs else "(Không có tài liệu nguồn — AI sẽ generate dựa trên ngữ cảnh template)"

    return {
        **state,
        "current_step":   2,
        "status":         "extracting",
        "source_context": context,
        "messages": [HumanMessage(content=
            f"[Step 2] {len(docs)} source doc(s). Context: {len(context):,} chars."
        )],
    }


# ─── Node 3: write_fields ─────────────────────────────────────────────────────

async def node_write_fields(state: PipelineState) -> PipelineState:
    client  = get_client()
    fields  = state["template_fields"]
    context = state.get("source_context", "")
    model   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    results: List[FieldResult] = []

    system_msg = (
        "Bạn là AI agent chuyên điền báo cáo ngân hàng BIDV. "
        "Nhiệm vụ: với mỗi trường tô vàng, điền nội dung phù hợp dựa trên tài liệu nguồn, chỗ nào tô vàng mà có (...*) thì điền vào chỗ đó thông tin phù hợp như ngày tháng hay số liệu. "
        """VĂN PHONG BẮT BUỘC:
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
            ✗ SAI: "Ngân hàng đã làm tốt việc này và đạt được nhiều kết quả khả quan..." """
        "Luôn trả lời bằng JSON hợp lệ theo schema được yêu cầu, không thêm markdown hay giải thích."
    )

    for field in fields:
        user_msg = f"""Điền nội dung vào trường tô vàng trong báo cáo.

NGỮ CẢNH ĐOẠN VĂN:
{field['context']}

PLACEHOLDER CẦN THAY THẾ:
"{field['placeholder']}"

TÀI LIỆU NGUỒN:
{context[:14000]}

Trả về JSON (chỉ JSON, không markdown):
{{
  "value": "nội dung điền vào — tiếng Việt, văn phong báo cáo ngân hàng chuyên nghiệp",
  "confidence": "high|mid|low",
  "reason": "lý do ngắn gọn về mức độ tin cậy"
}}

Quy tắc confidence:
- high: tìm thấy số liệu/thông tin cụ thể từ tài liệu nguồn
- mid : suy luận hợp lý từ ngữ cảnh
- low : không có tài liệu nguồn, tự generate"""

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

        results.append(FieldResult(
            field_key   = field["key"],
            para_idx    = field["para_idx"],
            placeholder = field["placeholder"],
            context     = field["context"],
            ai_value    = parsed.get("value", ""),
            final_value = parsed.get("value", ""),
            confidence  = parsed.get("confidence", "low"),
            reason      = parsed.get("reason", ""),
            approved    = False,
        ))

    high = sum(1 for r in results if r["confidence"] == "high")
    low  = sum(1 for r in results if r["confidence"] == "low")

    return {
        **state,
        "current_step":   3,
        "status":         "reviewing",
        "field_results":  results,
        "write_progress": len(results),
        "messages": [HumanMessage(content=
            f"[Step 3] Wrote {len(results)} fields. High: {high}, Low: {low}."
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
        "current_step":  4,
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
        "current_step": 5,
        "status":       "done",
        "messages": [HumanMessage(content="[Step 5] Pipeline complete.")],
    }


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(PipelineState)
    builder.add_node("analyze_template", node_analyze_template)
    builder.add_node("extract_sources",  node_extract_sources)
    builder.add_node("write_fields",     node_write_fields)
    builder.add_node("human_review",     node_human_review)
    builder.add_node("export_doc",       node_export_doc)

    builder.add_edge(START,              "analyze_template")
    builder.add_edge("analyze_template", "extract_sources")
    builder.add_edge("extract_sources",  "write_fields")
    builder.add_edge("write_fields",     "human_review")   # ← INTERRUPT trước node này
    builder.add_edge("human_review",     "export_doc")
    builder.add_edge("export_doc",       END)
    return builder


# _checkpointer = MemorySaver()

graph = build_graph().compile(
    # checkpointer=_checkpointer,
    interrupt_before=["human_review"],
)