"""
BIDV Report AI Agent - LangGraph Pipeline (OpenAI)
Graph: analyze_template -> upload_sources | -> extract_sources | -> write_fields -> qc_fields -> human_review -> export_doc

=== IMPROVEMENTS v2 ===
1. Semantic context matching (TF-IDF style scoring thay vì keyword overlap đơn giản)
2. 3-step chain: Extract Facts -> Verify -> Compose (thay vì 2-step)
3. Self-correction loop: QC fail -> re-generate với feedback (tối đa 1 lần)
4. Anti-fabrication guardrails: cross-check số liệu trước khi output
5. Source citation tracking: mỗi câu phải trace được về đoạn nguồn
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

# ── LangSmith tracing ──────────────────────────────────────────────────────
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


# ===============================================================================
# IMPROVED: Semantic Context Matching
# ===============================================================================

def _tokenize_vi(text: str) -> List[str]:
    """Tokenize tiếng Việt: tách từ >= 2 ký tự, lowercase."""
    return _re.findall(r'[a-zA-ZÀ-ỹ]{2,}', text.lower())


def _compute_idf(paragraphs: List[str]) -> Dict[str, float]:
    """Tính IDF cho tập đoạn văn."""
    import math
    n = len(paragraphs)
    if n == 0:
        return {}
    df = {}
    for para in paragraphs:
        tokens = set(_tokenize_vi(para))
        for t in tokens:
            df[t] = df.get(t, 0) + 1
    return {t: math.log(n / (1 + freq)) + 1 for t, freq in df.items()}


def extract_relevant_context(
    placeholder: str,
    context_field: str,
    full_context: str,
    max_chars: int = 10000,
) -> str:
    """
    Tìm đoạn liên quan nhất bằng TF-IDF scoring + bonus cho số liệu.
    Cải thiện so với keyword overlap đơn giản.
    """
    if not full_context or len(full_context) <= max_chars:
        return full_context

    # Tách thành đoạn
    paragraphs = [p.strip() for p in _re.split(r'\n{2,}', full_context) if p.strip() and len(p.strip()) > 20]
    if not paragraphs:
        return full_context[:max_chars]

    # Tính IDF
    idf = _compute_idf(paragraphs)

    # Query tokens từ placeholder + context xung quanh
    query_text = f"{placeholder} {context_field}"
    query_tokens = _tokenize_vi(query_text)
    query_tf = {}
    for t in query_tokens:
        query_tf[t] = query_tf.get(t, 0) + 1

    # Score từng đoạn
    scored = []
    for para in paragraphs:
        para_tokens = _tokenize_vi(para)
        para_tf = {}
        for t in para_tokens:
            para_tf[t] = para_tf.get(t, 0) + 1

        # TF-IDF cosine-like score
        score = 0.0
        for t, qtf in query_tf.items():
            if t in para_tf:
                score += qtf * para_tf[t] * idf.get(t, 1.0)

        # Bonus cho đoạn có số liệu cụ thể
        num_data_points = len(_re.findall(
            r'\d+[.,]?\d*\s*(%|tỷ|triệu|nghìn|người|dự án|lệnh|đồng|KHCN|KH)',
            para, _re.I
        ))
        score += num_data_points * 2.0

        # Bonus cho đoạn có tên riêng liên quan
        proper_nouns = len(_re.findall(
            r'(BIDV|NHNN|CSDLQG|VNeID|NQ\d+|Bộ [A-ZÀ-Ỹ]|Ban [A-ZÀ-Ỹ])',
            para
        ))
        score += proper_nouns * 1.5

        scored.append((score, para))

    # Sắp xếp theo score, lấy top, giữ thứ tự gốc
    scored_with_idx = [(s, p, i) for i, (s, p) in enumerate(scored)]
    scored_with_idx.sort(key=lambda x: -x[0])
    top_n = min(20, len(scored_with_idx))
    top = scored_with_idx[:top_n]

    # Giữ thứ tự gốc
    top.sort(key=lambda x: x[2])
    result = "\n\n".join(p for _, p, _ in top)

    return result[:max_chars] if len(result) > max_chars else result


# ===============================================================================
# IMPROVED: Fact Extraction (Step 1 of 3-step chain)
# ===============================================================================

async def _extract_facts(
    client: AsyncOpenAI,
    model: str,
    placeholder: str,
    context_field: str,
    relevant_ctx: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Bước 1: Trích xuất FACTS có cấu trúc từ tài liệu nguồn.
    Output: JSON với facts + source_sentences (để trace).
    """
    prompt = f"""Từ tài liệu nguồn, trích xuất TẤT CẢ dữ kiện liên quan đến chủ đề:
Chủ đề: "{placeholder}"
Ngữ cảnh: {context_field[:200]}

QUY TẮC:
1. CHỈ trích xuất thông tin CÓ TRONG tài liệu - KHÔNG suy luận, KHÔNG bịa
2. Mỗi fact phải kèm câu gốc (source_sentence) để đối chiếu
3. Phân loại fact: number (số liệu), name (tên/mã), date (ngày), description (mô tả)

TÀI LIỆU NGUỒN:
{relevant_ctx[:8000]}

Trả về JSON:
{{
  "facts": [
    {{
      "type": "number|name|date|description",
      "value": "giá trị cụ thể",
      "source_sentence": "câu gốc từ tài liệu chứa thông tin này",
      "relevance": "high|medium|low"
    }}
  ],
  "has_sufficient_data": true/false,
  "missing_info": "mô tả ngắn thông tin còn thiếu (nếu có)"
}}"""

    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content.strip())
        except Exception as e:
            return {"facts": [], "has_sufficient_data": False, "missing_info": f"Lỗi extract: {e}"}


# ===============================================================================
# IMPROVED: Fact Verification (Step 2 of 3-step chain)
# ===============================================================================

def _verify_facts_locally(facts: List[dict], source_text: str) -> List[dict]:
    """
    Kiểm tra nhanh bằng string matching: mỗi fact.value có xuất hiện
    trong tài liệu nguồn không. Gắn cờ verified/unverified.
    Chạy local, không tốn API call.
    """
    src_lower = source_text.lower()
    verified = []
    for fact in facts:
        val = str(fact.get("value", "")).strip()
        if not val:
            continue

        # Kiểm tra số có trong source
        numbers_in_val = _re.findall(r'[\d.,]+', val)
        is_verified = False

        if numbers_in_val:
            # Với số liệu: check từng số có trong source
            is_verified = all(
                num in source_text for num in numbers_in_val
            )
        else:
            # Với text: check substring (case-insensitive, >= 4 chars)
            val_lower = val.lower()
            if len(val_lower) >= 4:
                is_verified = val_lower in src_lower
            else:
                is_verified = True  # short values get benefit of doubt

        verified.append({
            **fact,
            "verified": is_verified,
        })

    return verified


# ===============================================================================
# IMPROVED: Content Composition (Step 3 of 3-step chain)
# ===============================================================================

def _build_system_prompt() -> str:
    today     = datetime.now()
    today_str = today.strftime("%d/%m/%Y")
    month_year= today.strftime("tháng %m/%Y")
    quarter   = f"Quý {(today.month-1)//3+1}/{today.year}"

    return f"""Bạn là chuyên gia soạn thảo văn bản hành chính ngân hàng BIDV.

NGÀY HIỆN TẠI: {today_str} ({month_year}, {quarter})

=== NGUYÊN TẮC BẮT BUỘC ===

1. CHỈ DÙNG DỮ KIỆN ĐÃ XÁC MINH
   - Bạn được cung cấp danh sách facts đã trích xuất và xác minh từ tài liệu nguồn
   - CHỈ sử dụng facts có "verified": true
   - Với facts "verified": false -> BỎ QUA, không đưa vào nội dung
   - KHÔNG thêm bất kỳ số liệu, tên, ngày tháng nào NGOÀI danh sách facts

2. VIẾT LẠI HOÀN TOÀN - KHÔNG COPY
   - Đọc facts -> hiểu -> viết lại bằng câu chữ MỚI HOÀN TOÀN
   - Cấu trúc câu PHẢI KHÁC tài liệu nguồn
   - GIỐNG: số liệu chính xác (9,8 triệu -> 9,8 triệu)
   - KHÁC: cách diễn đạt, cấu trúc câu, từ nối

3. MỖI CÂU / MỖI DÒNG PHẢI CÓ CĂN CỨ
   - Mỗi câu (hoặc mỗi gạch đầu dòng) phải chứa ít nhất 1 fact cụ thể
   - KHÔNG viết câu chung chung: "đã triển khai tốt", "kết quả khả quan", "tiếp tục phát huy"
   - Nếu không đủ facts -> viết ngắn hơn, KHÔNG bịa thêm

=== QUAN TRỌNG: NHẬN BIẾT LOẠI TRƯỜNG ===

Bạn sẽ nhận mô tả "LOẠI TRƯỜNG" trong mỗi request. TUÂN THỦ NGHIÊM NGẶT:

• date         -> CHỈ trả về ngày cụ thể, đúng format (15/03/2026). KHÔNG viết câu.
• time_range   -> CHỈ trả về khoảng thời gian (Quý I/2026, giai đoạn 2021-2025). KHÔNG viết câu.
• number       -> CHỈ trả về con số + đơn vị (1.847,3 tỷ đồng). KHÔNG viết câu.
• percentage   -> CHỈ trả về % (94,5%). Có thể kèm so sánh ngắn nếu source có.
• name         -> CHỈ trả về tên/mã (NQ57-NQ/TW, Ban CNTT). KHÔNG viết câu.
• short        -> Cụm từ ngắn, không cần câu hoàn chỉnh.
• sentence     -> 1-3 câu VĂN XUÔI liền mạch. KHÔNG gạch đầu dòng.
• paragraph    -> Nhiều câu VĂN XUÔI liền mạch. Dùng từ nối tự nhiên. KHÔNG gạch đầu dòng trừ khi placeholder gốc có.
• bullet_list  -> PHẢI dùng gạch đầu dòng "- ". Mỗi dòng 1 ý, phân cách bằng \\n.

=== VĂN PHONG ===
- Chủ ngữ: "BIDV", "Ngân hàng", "Ban/Phòng..." (không dùng "chúng tôi")
- Đơn vị: tỷ đồng, %, triệu người, lệnh/ngày
- Thời gian: "Quý I/2026", "tháng 3/2026", "ngày 15/03/2026"
- Tên viết hoa đúng: BIDV, NHNN, CSDLQG, VNeID, NQ57, Bộ Công an
- Câu văn xuôi: đa dạng cấu trúc, dùng từ nối ("Trong đó,", "Cụ thể,", "Bên cạnh đó,")

=== VÍ DỤ ===

Loại "sentence":
  ✓ "BIDV hoàn thành triển khai Core Banking mới với 156 chi nhánh kết nối, đạt uptime 99,7% trong Quý I/2026."
  ✗ "Đã triển khai tốt các nhiệm vụ được giao."

Loại "bullet_list":
  ✓ "- Dự án Core Banking: kết nối 156 chi nhánh, uptime 99,7%\\n- QR Pay: 2,3 triệu lệnh/ngày, tăng 45%\\n- eKYC VNeID: xác thực 9,8 triệu KHCN"
  ✗ "- Đã triển khai nhiều dự án\\n- Kết quả tốt\\n- Tiếp tục phát huy"

Loại "date":
  ✓ "15/03/2026"
  ✗ "trong năm 2026"

Loại "number":
  ✓ "1.847,3 tỷ đồng"
  ✗ "khoảng 2.000 tỷ"

Trả lời CHỈ bằng JSON hợp lệ."""


def _format_instruction(field_type: str, ph_len: int) -> str:
    """Tạo hướng dẫn format cụ thể cho từng loại trường."""
    instructions = {
        "date": (
            "- OUTPUT: CHỈ ngày tháng, KHÔNG viết câu\n"
            "- Format: DD/MM/YYYY hoặc 'ngày DD tháng MM năm YYYY'\n"
            "- Nếu không tìm được ngày cụ thể -> dùng ngày hiện tại"
        ),
        "time_range": (
            "- OUTPUT: CHỈ khoảng thời gian, KHÔNG viết câu\n"
            "- Format: 'Quý I/2026', '6 tháng đầu năm 2026', 'giai đoạn 2021-2025'\n"
            "- Tìm kỳ báo cáo từ tài liệu nguồn hoặc ngữ cảnh"
        ),
        "number": (
            "- OUTPUT: CHỈ con số + đơn vị, KHÔNG viết câu\n"
            "- Format: '1.847,3 tỷ đồng', '9,8 triệu KHCN'\n"
            "- KHÔNG làm tròn, KHÔNG ước tính"
        ),
        "percentage": (
            "- OUTPUT: CHỈ tỷ lệ %, có thể kèm so sánh ngắn\n"
            "- Format: '94,5%' hoặc 'đạt 94,5%, tăng 5,3 điểm %'\n"
            "- KHÔNG làm tròn, giữ đúng số thập phân từ source"
        ),
        "name": (
            "- OUTPUT: CHỈ tên/mã, KHÔNG viết câu\n"
            "- Viết hoa đúng: BIDV, NHNN, NQ57\n"
            "- Số văn bản đúng format: số 123/QĐ-BIDV"
        ),
        "short": (
            "- OUTPUT: 1 cụm từ ngắn (3-25 ký tự)\n"
            "- KHÔNG viết câu hoàn chỉnh"
        ),
        "sentence": (
            f"- OUTPUT: 1-3 câu VĂN XUÔI liền mạch (tham khảo {max(ph_len, 80)} ký tự)\n"
            "- KHÔNG dùng gạch đầu dòng, KHÔNG dùng bullet\n"
            "- Mỗi câu phải chứa ít nhất 1 dữ kiện cụ thể\n"
            "- Dùng từ nối tự nhiên giữa các câu"
        ),
        "paragraph": (
            f"- OUTPUT: đoạn văn LIỀN MẠCH (tối thiểu {max(ph_len, 150)} ký tự)\n"
            "- CẤU TRÚC: câu tổng quát -> chi tiết số liệu -> kết luận\n"
            "- KHÔNG dùng gạch đầu dòng - viết văn xuôi liền mạch\n"
            "- Dùng từ nối: 'Trong đó,', 'Cụ thể,', 'Bên cạnh đó,', 'Đáng chú ý,'\n"
            "- Mỗi câu phải chứa ít nhất 1 dữ kiện cụ thể"
        ),
        "bullet_list": (
            f"- OUTPUT: danh sách gạch đầu dòng, phân cách bằng \\n\n"
            "- Mỗi dòng bắt đầu bằng '- ' (dấu gạch ngang + khoảng trắng)\n"
            "- Mỗi gạch đầu dòng PHẢI có dữ kiện cụ thể (số, tên, ngày)\n"
            "- Format: '- [Tên mục]: [số liệu/chi tiết cụ thể]'\n"
            "- KHÔNG viết gạch đầu dòng chung chung không có số liệu"
        ),
    }
    return instructions.get(field_type, instructions["sentence"])


def _build_compose_prompt(
    field: dict,
    verified_facts: List[dict],
    field_type: str,
    type_hint: str,
    ph_len: int,
    correction_feedback: str = "",
) -> str:
    """Tạo prompt cho bước Compose, có thể kèm feedback từ QC nếu là lần retry."""

    # Lọc chỉ lấy verified facts
    good_facts = [f for f in verified_facts if f.get("verified", False)]
    weak_facts = [f for f in verified_facts if not f.get("verified", False)]

    facts_section = ""
    if good_facts:
        facts_section += "=== DỮ KIỆN ĐÃ XÁC MINH (ĐƯỢC PHÉP DÙNG) ===\n"
        for i, f in enumerate(good_facts, 1):
            facts_section += f"{i}. [{f.get('type','?')}] {f['value']}\n"
            if f.get('source_sentence'):
                facts_section += f"   (Nguồn: \"{f['source_sentence'][:100]}\")\n"

    if weak_facts:
        facts_section += "\n=== DỮ KIỆN CHƯA XÁC MINH (KHÔNG ĐƯỢC DÙNG) ===\n"
        for f in weak_facts:
            facts_section += f"⚠️ {f['value']} - KHÔNG dùng trong output\n"

    correction_section = ""
    if correction_feedback:
        correction_section = f"""
=== ⚠️ PHẢN HỒI TỪ KIỂM ĐỊNH VIÊN - BẮT BUỘC SỬA ===
{correction_feedback}
Bạn PHẢI sửa các lỗi trên. Xóa bỏ mọi thông tin bị gắn cờ "bịa"."""

    return f"""Viết nội dung cho trường trong báo cáo BIDV.

=== LOẠI TRƯỜNG (ĐỌC KỸ - QUYẾT ĐỊNH FORMAT OUTPUT) ===
{type_hint}

=== THÔNG TIN TRƯỜNG ===
Placeholder ({ph_len} ký tự): "{field.get('placeholder', '')}"
Ngữ cảnh: {field.get('context', '')}

{facts_section}
{correction_section}

=== YÊU CẦU NGHIÊM NGẶT ===
- CHỈ dùng dữ kiện từ mục "ĐÃ XÁC MINH" ở trên
- KHÔNG thêm số liệu, tên, ngày tháng nào khác ngoài facts
- Viết lại bằng CÂU CHỮ MỚI - cấu trúc câu phải khác source
- Nếu ít facts -> viết ngắn, KHÔNG bịa thêm cho đủ dài
- CHÍNH XÁC quan trọng hơn ĐẦY ĐỦ, đầy đủ quan trọng hơn DÀI
{_format_instruction(field_type, ph_len)}

Trả về JSON:
{{
  "value": "nội dung (format đúng loại trường ở trên)",
  "facts_used": [1, 3, 5],
  "confidence": "high|mid|low",
  "reason": "dữ kiện lấy từ đâu, đủ hay thiếu"
}}"""


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


# ─── Node 2b: extract_sources ─────────────────────────────────────────────────

async def node_extract_sources(state: PipelineState) -> PipelineState:
    docs = state.get("source_docs", [])
    print(f"[extract_sources] source_docs count: {len(docs)}")
    for d in docs:
        text_len = len(d.get('text', '')) if d.get('text') else 0
        print(f"  -> {d.get('filename', '?')}: {text_len:,} chars")

    context = "\n\n".join(
        f"=== {d['filename']} ===\n{d['text'][:10000]}"
        for d in docs if d.get('text')
    ) if docs else "(Không có tài liệu nguồn - AI sẽ generate dựa trên ngữ cảnh template)"

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


# ===============================================================================
# IMPROVED: Node 3 - write_fields (3-step chain)
# ===============================================================================

async def node_write_fields(state: PipelineState) -> PipelineState:
    client    = get_client()
    fields    = state["template_fields"]
    full_ctx  = state.get("source_context", "")
    model     = os.getenv("OPENAI_MODEL", "gpt-4o")
    semaphore = asyncio.Semaphore(int(os.getenv("OPENAI_CONCURRENCY", "2")))

    # ── DEFENSIVE: rebuild source_context nếu rỗng nhưng source_docs có data ──
    docs = state.get("source_docs", [])
    if (not full_ctx or full_ctx.startswith("(Không có")) and docs:
        full_ctx = "\n\n".join(
            f"=== {d['filename']} ===\n{d['text'][:10000]}"
            for d in docs if d.get('text')
        )
        print(f"[write_fields] Rebuilt source_context from {len(docs)} docs -> {len(full_ctx):,} chars")
    
    print(f"[write_fields] source_context length: {len(full_ctx):,} chars, "
          f"source_docs count: {len(docs)}, fields: {len(fields)}")

    system_msg = _build_system_prompt()

    from utils.docx_parser import _FIELD_TYPE_HINTS

    async def process_one_field(field: dict) -> FieldResult:
        placeholder  = field.get("placeholder", "")
        ph_len       = len(placeholder)
        field_type   = field.get("field_type", "sentence")
        type_hint    = field.get("type_hint") or _FIELD_TYPE_HINTS.get(field_type, "")
        context_text = field.get("context", "")

        # ── Step 1: Semantic context matching ─────────────────────────────
        relevant_ctx = extract_relevant_context(
            placeholder, context_text, full_ctx, max_chars=10000
        )

        # ── Step 2: Extract structured facts ──────────────────────────────
        # Full extraction for: paragraph, sentence, bullet_list (content-heavy types)
        needs_extraction = field_type in ("paragraph", "sentence", "bullet_list")
        if needs_extraction and len(relevant_ctx) > 300:
            facts_result = await _extract_facts(
                client, model, placeholder, context_text, relevant_ctx, semaphore
            )
            raw_facts = facts_result.get("facts", [])
            has_data  = facts_result.get("has_sufficient_data", False)

            # ── Step 2b: Local verification ───────────────────────────────
            verified_facts = _verify_facts_locally(raw_facts, relevant_ctx)
        else:
            # Short fields (date, number, percentage, name, short, time_range)
            verified_facts = []
            has_data = bool(relevant_ctx and not relevant_ctx.startswith("(Không có"))

        # ── Step 3: Compose with verified facts ───────────────────────────
        if verified_facts:
            user_msg = _build_compose_prompt(
                field, verified_facts, field_type, type_hint, ph_len
            )
        else:
            # Fallback cho field ngắn hoặc không có facts
            user_msg = _build_compose_prompt_simple(
                field, relevant_ctx, field_type, type_hint, ph_len
            )

        # ── Max tokens theo loại trường ───────────────────────────────────
        _MAX_TOKENS_BY_TYPE = {
            "date": 200, "time_range": 200, "number": 200,
            "percentage": 200, "name": 300, "short": 300,
        }
        if field_type in _MAX_TOKENS_BY_TYPE:
            max_tok = _MAX_TOKENS_BY_TYPE[field_type]
        elif ph_len < 50:
            max_tok = 600
        elif ph_len < 200:
            max_tok = 2000
        elif ph_len < 500:
            max_tok = 3500
        else:
            max_tok = 4096

        # ── Temperature: thấp cho field cần chính xác, cao hơn cho văn xuôi ──
        _TEMP_BY_TYPE = {
            "date": 0.0, "time_range": 0.0, "number": 0.0,
            "percentage": 0.0, "name": 0.05, "short": 0.1,
            "sentence": 0.2, "paragraph": 0.35, "bullet_list": 0.25,
        }
        temperature = _TEMP_BY_TYPE.get(field_type, 0.2)

        parsed = await _call_openai_with_retry(
            client, model, system_msg, user_msg,
            max_tok, temperature, semaphore
        )

        key = field.get("key") or field.get("field_key", "")
        ai_value = parsed.get("value", "")

        # Gắn metadata cho QC
        return FieldResult(
            field_key   = key,
            para_idx    = field.get("para_idx", 0),
            placeholder = placeholder,
            context     = context_text,
            ai_value    = ai_value,
            final_value = ai_value,
            confidence  = parsed.get("confidence", "low"),
            reason      = parsed.get("reason", ""),
            approved    = False,
        )

    results: List[FieldResult] = list(
        await asyncio.gather(*[process_one_field(f) for f in fields])
    )

    return {
        **state,
        "current_step":  4,
        "status":        "reviewing",
        "field_results": results,
        "messages": [HumanMessage(content=
            f"[Step 3] {len(results)} fields filled via 3-step chain."
        )],
    }


def _build_compose_prompt_simple(
    field: dict,
    relevant_ctx: str,
    field_type: str,
    type_hint: str,
    ph_len: int,
) -> str:
    """Prompt cho field ngan (date, number, name, percentage, time_range, short) - dung context truc tiep."""
    return f"""Điền nội dung cho trường trong báo cáo BIDV.

=== LOẠI TRƯỜNG (ĐỌC KỸ - QUYẾT ĐỊNH FORMAT OUTPUT) ===
{type_hint}

=== THÔNG TIN TRƯỜNG ===
Placeholder ({ph_len} ký tự): "{field.get('placeholder', '')}"
Ngữ cảnh: {field.get('context', '')}

=== TÀI LIỆU NGUỒN ===
{relevant_ctx[:6000]}

=== YÊU CẦU ===
- Tim thong tin CU THE tu tai lieu nguon phu hop voi truong nay
- Neu khong tim duoc, ghi: 'chua co du lieu tu tai lieu nguon'
- KHONG bia so lieu, ten, ngay thang
- Giu dung FORMAT yeu cau cua loai truong (xem muc LOAI TRUONG o tren)
{_format_instruction(field_type, ph_len)}

Trả về JSON:
{{
  "value": "nội dung điền vào",
  "confidence": "high|mid|low",
  "reason": "nguồn dữ liệu"
}}"""


async def _call_openai_with_retry(
    client: AsyncOpenAI,
    model: str,
    system_msg: str,
    user_msg: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> dict:
    """Call OpenAI với retry logic cho rate limit."""
    async with semaphore:
        parsed = None
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": user_msg},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                parsed = json.loads(response.choices[0].message.content.strip())
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    wait_match = _re.search(r'try again in (\d+\.?\d*)s', err_str)
                    wait_secs  = float(wait_match.group(1)) if wait_match else (2 ** attempt * 3)
                    wait_secs  = min(wait_secs + 1, 60)
                    print(f"[Rate limit] attempt {attempt+1}/{max_retries} - wait {wait_secs:.1f}s")
                    await asyncio.sleep(wait_secs)
                    continue
                parsed = {"value": "", "confidence": "low", "reason": f"Lỗi: {e}"}
                break
        if parsed is None:
            parsed = {"value": "", "confidence": "low", "reason": "Vượt quá rate limit"}
    return parsed


# ===============================================================================
# IMPROVED: Node 3b - QC with self-correction
# ===============================================================================

async def node_qc_fields(state: PipelineState) -> PipelineState:
    """
    QC Agent: đối chiếu AI output với tài liệu nguồn.
    Nếu phát hiện bịa -> tự sửa 1 lần (self-correction loop).
    """
    client    = get_client()
    results   = state.get("field_results", [])
    context   = state.get("source_context", "")
    fields    = state.get("template_fields", [])
    model     = os.getenv("OPENAI_MODEL", "gpt-4.1")
    semaphore = asyncio.Semaphore(int(os.getenv("OPENAI_CONCURRENCY", "2")))

    if not context or context.startswith("(Không có tài liệu"):
        updated = []
        for r in results:
            updated.append({**r,
                "qc_status": "warning",
                "qc_note":   "Không có tài liệu nguồn để đối chiếu - cần kiểm tra thủ công"
            })
        return {**state, "field_results": updated,
                "messages": [HumanMessage(content="[QC] No source docs - all flagged warning.")]}

    system_qc = """Bạn là chuyên gia kiểm định nội dung báo cáo ngân hàng BIDV.

NHIỆM VỤ: Đối chiếu từng thông tin trong nội dung AI viết với tài liệu nguồn.

QUY TẮC KIỂM TRA:
1. Mỗi CON SỐ (%, tỷ, triệu, nghìn) -> phải tìm được trong tài liệu nguồn
2. Mỗi TÊN (dự án, văn bản, tổ chức) -> phải có trong tài liệu nguồn
3. Mỗi NGÀY THÁNG -> phải khớp hoặc suy ra được từ tài liệu nguồn
4. Câu chung chung không có dữ kiện -> gắn cờ warning

QUAN TRỌNG: Nếu AI viết "tương đương X%" mà X% KHÔNG có trong nguồn -> đó là BỊA.

Trả lời CHỈ bằng JSON hợp lệ."""

    from utils.docx_parser import _FIELD_TYPE_HINTS

    async def qc_and_fix(r: dict, field: dict) -> dict:
        ai_value = r.get("ai_value", "")
        if not ai_value.strip():
            return {**r, "qc_status": "fail", "qc_note": "AI không tạo được nội dung"}

        # Tìm context liên quan
        relevant_ctx = extract_relevant_context(
            r.get("placeholder", ""),
            r.get("context", ""),
            context,
            max_chars=6000,
        )

        # ── QC Check ─────────────────────────────────────────────────────
        user_qc = f"""Kiểm tra nội dung AI viết có căn cứ từ tài liệu nguồn không.

NỘI DUNG AI VIẾT:
"{ai_value}"

TÀI LIỆU NGUỒN:
{relevant_ctx}

Kiểm tra từng thông tin:
1. Liệt kê MỌI con số, tên, ngày trong nội dung AI
2. Với mỗi thông tin: tìm câu gốc trong tài liệu nguồn
3. Nếu KHÔNG tìm được -> gắn cờ "fabricated"

Trả về JSON:
{{
  "status": "ok|warning|fail",
  "checks": [
    {{"info": "9,8 triệu KHCN", "found_in_source": true, "source_quote": "câu gốc..."}},
    {{"info": "94% tổng KHCN", "found_in_source": false, "source_quote": null}}
  ],
  "fabricated": ["liệt kê cụm từ/số liệu bịa"],
  "note": "giải thích ngắn"
}}

Quy tắc status:
- ok:      TẤT CẢ checks đều found_in_source = true
- warning: có 1-2 checks không rõ ràng
- fail:    có checks rõ ràng found_in_source = false (bịa)"""

        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_qc},
                        {"role": "user",   "content": user_qc},
                    ],
                    max_tokens=600,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                qc = json.loads(resp.choices[0].message.content.strip())
            except Exception as e:
                qc = {"status": "warning", "note": f"QC lỗi: {e}", "fabricated": []}

        fabricated = qc.get("fabricated", [])
        qc_status  = qc.get("status", "warning")
        note       = qc.get("note", "")

        # ── Self-correction: nếu QC fail -> re-generate 1 lần ─────────────
        if qc_status == "fail" and fabricated:
            print(f"[QC] Field {r['field_key']} FAIL - fabricated: {fabricated}. Re-generating...")

            correction_feedback = f"""CÁC LỖI CẦN SỬA:
- Thông tin bịa (KHÔNG có trong tài liệu nguồn): {', '.join(fabricated)}
- {note}

YÊU CẦU: Viết lại, XÓA BỎ hoàn toàn các thông tin bịa trên.
Chỉ giữ lại thông tin CÓ TRONG tài liệu nguồn."""

            # Re-extract facts
            facts_result = await _extract_facts(
                client, model, r.get("placeholder", ""),
                r.get("context", ""), relevant_ctx, semaphore
            )
            verified_facts = _verify_facts_locally(
                facts_result.get("facts", []), relevant_ctx
            )

            field_type = field.get("field_type", "sentence") if field else "sentence"
            type_hint  = field.get("type_hint") or _FIELD_TYPE_HINTS.get(field_type, "")

            retry_prompt = _build_compose_prompt(
                field or r, verified_facts, field_type, type_hint,
                len(r.get("placeholder", "")),
                correction_feedback=correction_feedback,
            )

            system_msg = _build_system_prompt()
            new_parsed = await _call_openai_with_retry(
                client, model, system_msg, retry_prompt,
                2000, 0.15, semaphore
            )

            new_value = new_parsed.get("value", "")
            if new_value.strip():
                return {
                    **r,
                    "ai_value":    new_value,
                    "final_value": new_value,
                    "confidence":  new_parsed.get("confidence", "mid"),
                    "reason":      new_parsed.get("reason", "") + " [QC: đã sửa lại sau kiểm tra]",
                    "qc_status":   "warning",
                    "qc_note":     f"Đã sửa lại - bản gốc có thông tin bịa: {', '.join(fabricated)}",
                }

        # Không sửa được hoặc không cần sửa
        if fabricated:
            note += f" | Nghi bịa: {', '.join(fabricated)}"

        return {**r, "qc_status": qc_status, "qc_note": note}

    # Map field_key -> field dict để truyền vào qc_and_fix
    field_map = {}
    for f in fields:
        key = f.get("key") or f.get("field_key", "")
        field_map[key] = f

    qc_results = list(await asyncio.gather(*[
        qc_and_fix(r, field_map.get(r["field_key"], {}))
        for r in results
    ]))

    n_ok   = sum(1 for r in qc_results if r.get("qc_status") == "ok")
    n_warn = sum(1 for r in qc_results if r.get("qc_status") == "warning")
    n_fail = sum(1 for r in qc_results if r.get("qc_status") == "fail")
    n_fixed= sum(1 for r in qc_results if "đã sửa lại" in r.get("qc_note", ""))

    return {
        **state,
        "field_results": qc_results,
        "messages": [HumanMessage(content=
            f"[QC] ✅ {n_ok} ok * ⚠️ {n_warn} warning * ❌ {n_fail} fail * 🔄 {n_fixed} auto-fixed"
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
    builder.add_node("qc_fields",        node_qc_fields)
    builder.add_node("human_review",     node_human_review)
    builder.add_node("export_doc",       node_export_doc)

    builder.add_edge(START,              "analyze_template")
    builder.add_edge("analyze_template", "upload_sources")
    builder.add_edge("upload_sources",   "extract_sources")
    builder.add_edge("extract_sources",  "write_fields")
    builder.add_edge("write_fields",     "qc_fields")
    builder.add_edge("qc_fields",        "human_review")
    builder.add_edge("human_review",     "export_doc")
    builder.add_edge("export_doc",       END)
    return builder


# compiled_graph - dùng cho LangGraph Studio/API (không có checkpointer)
compiled_graph = build_graph().compile(
    interrupt_before=["extract_sources", "human_review"],
)

# compiled_graph_with_memory - dùng cho FastAPI/Docker (có MemorySaver)
_checkpointer = MemorySaver()
compiled_graph_with_memory = build_graph().compile(
    checkpointer=_checkpointer,
    interrupt_before=["extract_sources", "human_review"],
)