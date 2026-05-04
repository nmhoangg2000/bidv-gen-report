"""
BIDV Report AI Agent - Multi-Agent LangGraph Pipeline
=====================================================

Graph:
  analyze_template -> upload_sources -> extract_sources (interrupt)
    -> write_fields [Researcher + Writer per field]
    -> verify_fields [Verifier per field]
    -> fix_fields [Editor for failed fields]
    -> human_review (interrupt)
    -> export_doc

4 Agents:
  1. Researcher  - search_source, extract_facts, verify_facts_locally
  2. Writer      - compose_content / compose_simple (adapts to field_type)
  3. Verifier    - verify_content (cross-check with source)
  4. Editor      - rewrite_with_feedback (self-correction for fabricated content)
"""

import json
import os
import re as _re
import asyncio
from typing import Annotated, Dict, List, Optional, TypedDict
from datetime import datetime

from openai import AsyncOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.tools import ResearcherAgent, WriterAgent, VerifierAgent, EditorAgent

# -- LangSmith tracing --
try:
    from langchain_core.tracers.langchain import LangChainTracer
    _ls_project = os.getenv("LANGCHAIN_PROJECT", "bidv-agent")
    _ls_tracer  = LangChainTracer(project_name=_ls_project)
    _ls_enabled = bool(os.getenv("LANGCHAIN_API_KEY"))
    if _ls_enabled:
        print(f"[LangSmith] Tracing enabled -> project: {_ls_project}")
except Exception as e:
    _ls_tracer  = None
    _ls_enabled = False
    print(f"[LangSmith] Tracing disabled: {e}")

def get_ls_callbacks():
    return [_ls_tracer] if (_ls_enabled and _ls_tracer) else []

def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


# === State Schema ============================================================

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


# === Node 1: analyze_template ================================================

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


# === Node 2a: upload_sources =================================================

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


# === Node 2b: extract_sources ================================================

async def node_extract_sources(state: PipelineState) -> PipelineState:
    docs = state.get("source_docs", [])
    print(f"[extract_sources] source_docs count: {len(docs)}")
    for d in docs:
        text_len = len(d.get('text', '')) if d.get('text') else 0
        print(f"  -> {d.get('filename', '?')}: {text_len:,} chars")

    context = "\n\n".join(
        f"=== {d['filename']} ===\n{d['text'][:10000]}"
        for d in docs if d.get('text')
    ) if docs else ""

    print(f"[extract_sources] Built source_context: {len(context):,} chars")

    return {
        **state,
        "current_step":   3,
        "status":         "extracting",
        "source_context": context,
        "messages": [HumanMessage(content=
            f"[Step 2b] Context built: {len(context):,} chars."
        )],
    }


# === Node 3: write_fields (Researcher + Writer) ==============================

async def node_write_fields(state: PipelineState) -> PipelineState:
    """
    Multi-agent: Researcher trich xuat facts -> Writer viet noi dung.
    Chay song song cho tat ca fields.
    """
    client    = get_client()
    fields    = state["template_fields"]
    full_ctx  = state.get("source_context", "")
    semaphore = asyncio.Semaphore(int(os.getenv("OPENAI_CONCURRENCY", "2")))

    # -- DEFENSIVE: rebuild source_context neu rong --
    docs = state.get("source_docs", [])
    if (not full_ctx) and docs:
        full_ctx = "\n\n".join(
            f"=== {d['filename']} ===\n{d['text'][:10000]}"
            for d in docs if d.get('text')
        )
        print(f"[write_fields] Rebuilt source_context from {len(docs)} docs -> {len(full_ctx):,} chars")

    has_source = bool(full_ctx and len(full_ctx) > 50)
    print(f"[write_fields] source_context: {len(full_ctx):,} chars, "
          f"source_docs: {len(docs)}, fields: {len(fields)}, has_source: {has_source}")

    from utils.docx_parser import _FIELD_TYPE_HINTS

    async def process_one_field(field: dict) -> FieldResult:
        placeholder  = field.get("placeholder", "")
        ph_len       = len(placeholder)
        field_type   = field.get("field_type", "sentence")
        type_hint    = field.get("type_hint") or _FIELD_TYPE_HINTS.get(field_type, "")
        context_text = field.get("context", "")
        key          = field.get("key") or field.get("field_key", "")

        # == AGENT 1: Researcher ==
        relevant_ctx = ResearcherAgent.search_source(
            placeholder, context_text, full_ctx, max_chars=10000
        )

        # == AGENT 2: Writer ==
        needs_deep = (
            field_type in ("paragraph", "sentence", "bullet_list")
            and has_source
            and len(relevant_ctx) > 300
        )

        if needs_deep:
            # Researcher: extract + verify facts
            facts_result = await ResearcherAgent.extract_facts(
                client, placeholder, context_text, relevant_ctx, semaphore
            )
            raw_facts = facts_result.get("facts", [])
            verified_facts = ResearcherAgent.verify_facts_locally(raw_facts, relevant_ctx)

            # Writer: compose with verified facts
            parsed = await WriterAgent.compose_content(
                client, field, verified_facts, field_type, type_hint,
                ph_len, semaphore, relevant_ctx
            )
        else:
            # Short fields: compose directly
            parsed = await WriterAgent.compose_simple(
                client, field, relevant_ctx, field_type, type_hint,
                ph_len, semaphore
            )

        ai_value = parsed.get("value", "")

        # Build rich citation reason from Writer output
        citations = parsed.get("citations", [])
        base_reason = parsed.get("reason", "")
        if citations:
            cite_parts = []
            seen_files = set()
            for c in citations:
                f = c.get("file", "")
                q = c.get("quote", "")
                if f and f not in seen_files:
                    seen_files.add(f)
                if f and q:
                    cite_parts.append(f'[{f}]: "{q[:80]}"')
                elif f:
                    cite_parts.append(f'[{f}]')
            file_list = ", ".join(seen_files) if seen_files else ""
            reason = f"Nguon: {file_list}"
            if cite_parts:
                reason += " | " + "; ".join(cite_parts[:4])
            if base_reason:
                reason += f" | {base_reason}"
        elif needs_deep and verified_facts:
            # Fallback: build from verified_facts source info
            src_files = set()
            cite_parts = []
            for vf in verified_facts:
                if vf.get("verified"):
                    sf = vf.get("source_file", "")
                    ss = vf.get("source_sentence", "")
                    if sf:
                        src_files.add(sf)
                    if sf and ss:
                        cite_parts.append(f'[{sf}]: "{ss[:60]}"')
            file_list = ", ".join(src_files) if src_files else "tai lieu nguon"
            reason = f"Nguon: {file_list}"
            if cite_parts:
                reason += " | " + "; ".join(cite_parts[:3])
        else:
            reason = base_reason or "Trich xuat truc tiep tu tai lieu nguon"

        return FieldResult(
            field_key   = key,
            para_idx    = field.get("para_idx", 0),
            placeholder = placeholder,
            context     = context_text,
            ai_value    = ai_value,
            final_value = ai_value,
            confidence  = parsed.get("confidence", "low"),
            reason      = reason,
            approved    = False,
        )

    results: List[FieldResult] = list(
        await asyncio.gather(*[process_one_field(f) for f in fields])
    )

    non_empty = sum(1 for r in results if r.get("ai_value", "").strip())
    print(f"[write_fields] Done: {len(results)} fields, {non_empty} non-empty")

    return {
        **state,
        "current_step":  4,
        "status":        "verifying",
        "field_results": results,
        "messages": [HumanMessage(content=
            f"[Step 3] Researcher + Writer: {len(results)} fields ({non_empty} non-empty)."
        )],
    }


# === Node 4: verify_fields (Verifier Agent) ==================================

async def node_verify_fields(state: PipelineState) -> PipelineState:
    """Verifier Agent: doi chieu AI output voi tai lieu nguon."""
    client    = get_client()
    results   = state.get("field_results", [])
    full_ctx  = state.get("source_context", "")
    semaphore = asyncio.Semaphore(int(os.getenv("OPENAI_CONCURRENCY", "2")))

    if not full_ctx or len(full_ctx) < 50:
        updated = [{**r, "qc_status": "warning",
                    "qc_note": "Khong co tai lieu nguon"} for r in results]
        return {**state, "field_results": updated,
                "messages": [HumanMessage(content="[Verifier] No source - all warning.")]}

    async def verify_one(r: dict) -> dict:
        ai_value = r.get("ai_value", "")
        if not ai_value.strip():
            return {**r, "qc_status": "fail",
                    "qc_note": "AI khong tao duoc noi dung", "qc_fabricated": []}

        relevant_ctx = ResearcherAgent.search_source(
            r.get("placeholder", ""), r.get("context", ""),
            full_ctx, max_chars=6000
        )

        qc = await VerifierAgent.verify_content(
            client, ai_value, relevant_ctx, semaphore
        )

        fabricated = qc.get("fabricated", [])
        note = qc.get("note", "")
        if fabricated:
            note += f" | Nghi bia: {', '.join(str(f) for f in fabricated)}"

        return {**r,
                "qc_status": qc.get("status", "warning"),
                "qc_note": note,
                "qc_fabricated": fabricated}

    qc_results = list(await asyncio.gather(*[verify_one(r) for r in results]))

    n_ok   = sum(1 for r in qc_results if r.get("qc_status") == "ok")
    n_warn = sum(1 for r in qc_results if r.get("qc_status") == "warning")
    n_fail = sum(1 for r in qc_results if r.get("qc_status") == "fail")
    print(f"[Verifier] ok={n_ok}, warning={n_warn}, fail={n_fail}")

    return {
        **state,
        "field_results": qc_results,
        "messages": [HumanMessage(content=
            f"[Verifier] ok={n_ok} warn={n_warn} fail={n_fail}"
        )],
    }


# === Node 5: fix_fields (Editor Agent) =======================================

async def node_fix_fields(state: PipelineState) -> PipelineState:
    """Editor Agent: viet lai cac field bi Verifier danh fail."""
    client    = get_client()
    results   = state.get("field_results", [])
    fields    = state.get("template_fields", [])
    full_ctx  = state.get("source_context", "")
    semaphore = asyncio.Semaphore(int(os.getenv("OPENAI_CONCURRENCY", "2")))

    from utils.docx_parser import _FIELD_TYPE_HINTS

    field_map = {}
    for f in fields:
        key = f.get("key") or f.get("field_key", "")
        field_map[key] = f

    async def fix_one(r: dict) -> dict:
        fabricated = r.get("qc_fabricated", [])
        if r.get("qc_status") != "fail" or not fabricated:
            return r

        field = field_map.get(r["field_key"], {})
        field_type = field.get("field_type", "sentence")
        type_hint  = field.get("type_hint") or _FIELD_TYPE_HINTS.get(field_type, "")
        ph_len     = len(r.get("placeholder", ""))

        relevant_ctx = ResearcherAgent.search_source(
            r.get("placeholder", ""), r.get("context", ""),
            full_ctx, max_chars=8000
        )

        facts_result = await ResearcherAgent.extract_facts(
            client, r.get("placeholder", ""), r.get("context", ""),
            relevant_ctx, semaphore
        )
        verified_facts = ResearcherAgent.verify_facts_locally(
            facts_result.get("facts", []), relevant_ctx
        )

        new_parsed = await EditorAgent.rewrite_with_feedback(
            client, field, r.get("ai_value", ""),
            fabricated, r.get("qc_note", ""),
            verified_facts, field_type, type_hint, ph_len, semaphore
        )

        new_value = new_parsed.get("value", "").strip()
        if new_value:
            print(f"[Editor] Fixed {r['field_key']}: removed {len(fabricated)} fabricated items")
            return {
                **r,
                "ai_value": new_value,
                "final_value": new_value,
                "confidence": new_parsed.get("confidence", "mid"),
                "reason": new_parsed.get("reason", ""),
                "qc_status": "warning",
                "qc_note": f"Da sua lai - ban goc co: {', '.join(str(f) for f in fabricated)}",
            }
        return r

    fixed_results = list(await asyncio.gather(*[fix_one(r) for r in results]))
    n_fixed = sum(1 for r in fixed_results if "Da sua lai" in r.get("qc_note", ""))
    print(f"[Editor] Fixed {n_fixed} fields")

    return {
        **state,
        "current_step": 4,
        "status": "reviewing",
        "field_results": fixed_results,
        "messages": [HumanMessage(content=
            f"[Editor] Auto-fixed {n_fixed} fields."
        )],
    }


# === Node 6: human_review ====================================================

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
            f"[Review] Complete. {len(edits)} fields edited."
        )],
    }


# === Node 7: export_doc ======================================================

async def node_export_doc(state: PipelineState) -> PipelineState:
    return {
        **state,
        "current_step": 6,
        "status":       "done",
        "messages": [HumanMessage(content="[Export] Pipeline complete.")],
    }


# === Build Graph ==============================================================

def build_graph() -> StateGraph:
    builder = StateGraph(PipelineState)

    builder.add_node("analyze_template", node_analyze_template)
    builder.add_node("upload_sources",   node_upload_sources)
    builder.add_node("extract_sources",  node_extract_sources)
    builder.add_node("write_fields",     node_write_fields)
    builder.add_node("verify_fields",    node_verify_fields)
    builder.add_node("fix_fields",       node_fix_fields)
    builder.add_node("human_review",     node_human_review)
    builder.add_node("export_doc",       node_export_doc)

    builder.add_edge(START,              "analyze_template")
    builder.add_edge("analyze_template", "upload_sources")
    builder.add_edge("upload_sources",   "extract_sources")
    builder.add_edge("extract_sources",  "write_fields")
    builder.add_edge("write_fields",     "verify_fields")
    builder.add_edge("verify_fields",    "fix_fields")
    builder.add_edge("fix_fields",       "human_review")
    builder.add_edge("human_review",     "export_doc")
    builder.add_edge("export_doc",       END)

    return builder


# Compiled graphs
compiled_graph = build_graph().compile(
    interrupt_before=["extract_sources", "human_review"],
)

_checkpointer = MemorySaver()
compiled_graph_with_memory = build_graph().compile(
    checkpointer=_checkpointer,
    interrupt_before=["extract_sources", "human_review"],
)