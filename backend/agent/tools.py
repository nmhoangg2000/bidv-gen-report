"""
Multi-Agent Tools for BIDV Report AI Agent
==========================================
4 specialized agents, each with dedicated tools:

1. Researcher  - search_source, extract_facts, rank_relevance
2. Writer      - compose_content (adapts to field_type automatically)
3. Verifier    - cross_check_facts, detect_fabrication
4. Editor      - rewrite_with_feedback (self-correction loop)
"""

import json
import os
import re as _re
import math
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from openai import AsyncOpenAI


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


# =============================================================================
# MODEL ROUTING — mix model per agent for cost/quality optimization
# =============================================================================
# Writer + Editor: need best quality -> flagship model (gpt-5.4)
# Researcher + Verifier: structured extraction/check -> mini model (gpt-5.4-mini)
#
# Config via env vars:
#   OPENAI_MODEL_WRITER   = gpt-5.4        (default)
#   OPENAI_MODEL_RESEARCH = gpt-5.4-mini   (default)
#   OPENAI_MODEL          = fallback for all if specific not set

def _get_model_for(agent: str) -> str:
    """Get model string for specific agent role."""
    # Per-agent env vars (highest priority)
    env_map = {
        "writer":     "OPENAI_MODEL_WRITER",
        "editor":     "OPENAI_MODEL_EDITOR",
        "researcher": "OPENAI_MODEL_RESEARCH",
        "verifier":   "OPENAI_MODEL_VERIFIER",
    }
    env_key = env_map.get(agent)
    if env_key:
        val = os.getenv(env_key)
        if val:
            return val

    # Default routing: writer/editor -> flagship, researcher/verifier -> mini
    fallback = os.getenv("OPENAI_MODEL", "gpt-5.4")
    if agent in ("researcher", "verifier"):
        return os.getenv("OPENAI_MODEL_MINI", "gpt-5.4-mini")
    return fallback


# Log model routing on import
print("[Model Routing]")
for _agent in ("researcher", "writer", "verifier", "editor"):
    print(f"  {_agent:12s} -> {_get_model_for(_agent)}")


# =============================================================================
# SHARED UTILS
# =============================================================================

def _tokenize_vi(text: str) -> List[str]:
    return _re.findall(r'[a-zA-Z\u00C0-\u024F\u1EA0-\u1EFF]{2,}', text.lower())


def _today_context() -> dict:
    today = datetime.now()
    q = (today.month - 1) // 3 + 1
    return {
        "date_str": today.strftime("%d/%m/%Y"),
        "date_formal": f"ngay {today.day:02d} thang {today.month:02d} nam {today.year}",
        "month_year": f"thang {today.month:02d}/{today.year}",
        "quarter": f"Quy {q}/{today.year}",
        "quarter_roman": f"Quy {'I' if q==1 else 'II' if q==2 else 'III' if q==3 else 'IV'}/{today.year}",
        "half_year": f"{'6 thang dau' if today.month <= 6 else '6 thang cuoi'} nam {today.year}",
        "year": str(today.year),
        "prev_quarter": f"Quy {q-1 if q>1 else 4}/{today.year if q>1 else today.year-1}",
    }


# =============================================================================
# AGENT 1: RESEARCHER
# Tools: search_source, extract_facts
# =============================================================================

class ResearcherAgent:
    """
    Tim va trich xuat du kien tu tai lieu nguon.
    Khong viet van - chi tra ve structured data.
    """

    @staticmethod
    def search_source(
        placeholder: str,
        context_field: str,
        full_context: str,
        max_chars: int = 10000,
    ) -> str:
        """
        Tool 1: Tim doan lien quan nhat trong tai lieu nguon bang TF-IDF scoring.
        Returns: string chua cac doan lien quan nhat (giu nguyen header === filename ===).
        """
        if not full_context or len(full_context) <= max_chars:
            return full_context or ""

        paragraphs = [p.strip() for p in _re.split(r'\n{2,}', full_context)
                      if p.strip() and len(p.strip()) > 20]
        if not paragraphs:
            return full_context[:max_chars]

        # Compute IDF
        n = len(paragraphs)
        df = {}
        for para in paragraphs:
            for t in set(_tokenize_vi(para)):
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log(n / (1 + freq)) + 1 for t, freq in df.items()}

        # Query tokens
        query_text = f"{placeholder} {context_field}"
        query_tokens = _tokenize_vi(query_text)
        query_tf = {}
        for t in query_tokens:
            query_tf[t] = query_tf.get(t, 0) + 1

        # Score
        scored = []
        for idx, para in enumerate(paragraphs):
            para_tokens = _tokenize_vi(para)
            para_tf = {}
            for t in para_tokens:
                para_tf[t] = para_tf.get(t, 0) + 1

            score = 0.0
            for t, qtf in query_tf.items():
                if t in para_tf:
                    score += qtf * para_tf[t] * idf.get(t, 1.0)

            # Bonus: so lieu
            num_data = len(_re.findall(
                r'\d+[.,]?\d*\s*(%|ty|trieu|nghin|nguoi|du an|lenh|dong|KHCN|KH)',
                para, _re.I
            ))
            score += num_data * 2.0

            # Bonus: ten rieng
            proper = len(_re.findall(
                r'(BIDV|NHNN|CSDLQG|VNeID|NQ\d+|Bo [A-Z]|Ban [A-Z])',
                para
            ))
            score += proper * 1.5

            scored.append((score, para, idx))

        scored.sort(key=lambda x: -x[0])
        top_n = min(20, len(scored))
        top = scored[:top_n]
        top.sort(key=lambda x: x[2])  # giu thu tu goc

        result = "\n\n".join(p for _, p, _ in top)
        return result[:max_chars]

    @staticmethod
    def detect_source_files(full_context: str) -> List[str]:
        """Lay danh sach ten file nguon tu context (header === filename ===)."""
        return _re.findall(r'===\s*(.+?)\s*===', full_context)

    @staticmethod
    async def extract_facts(
        client: AsyncOpenAI,
        placeholder: str,
        context_field: str,
        relevant_ctx: str,
        semaphore: asyncio.Semaphore,
    ) -> dict:
        """
        Tool 2: Trich xuat structured facts tu tai lieu nguon.
        Moi fact co: type, value, source_sentence, relevance.
        """
        prompt = f"""Tu tai lieu nguon, trich xuat NHIEU NHAT CO THE cac du kien lien quan den chu de:
Chu de: "{placeholder}"
Ngu canh: {context_field[:200]}

QUY TAC:
1. CHI trich xuat thong tin CO TRONG tai lieu - KHONG suy luan, KHONG bia
2. Moi fact PHAI kem:
   - source_file: ten file nguon (lay tu dong "=== ten_file ===" trong tai lieu)
   - source_sentence: cau goc CHINH XAC tu tai lieu chua thong tin nay
3. Phan loai fact: number (so lieu), name (ten/ma), date (ngay), description (mo ta)
4. Trich xuat CA thong tin truc tiep VA thong tin bo sung/lien quan
5. Voi moi so lieu: lay ca ngu canh xung quanh (tang/giam bao nhieu, so voi ky nao)
6. Trich xuat TOI THIEU 5-10 facts neu tai lieu co du thong tin

LUU Y VE TEN FILE:
- Tai lieu nguon co dang: "=== ten_file.docx ===" hoac "=== ten_file.pdf ==="
- Neu khong thay ten file, ghi "khong ro file"

TAI LIEU NGUON:
{relevant_ctx[:10000]}

Tra ve JSON:
{{
  "facts": [
    {{
      "type": "number|name|date|description",
      "value": "gia tri cu the",
      "source_file": "ten_file_nguon.docx",
      "source_sentence": "trich dan CHINH XAC cau goc tu tai lieu",
      "relevance": "high|medium|low"
    }}
  ],
  "has_sufficient_data": true/false,
  "source_files_found": ["file1.docx", "file2.pdf"],
  "topic_summary": "tom tat 1 dong ve chu de trong tai lieu"
}}"""

        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=_get_model_for("researcher"),
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=2000,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content.strip())
            except Exception as e:
                return {"facts": [], "has_sufficient_data": False,
                        "topic_summary": f"Loi extract: {e}"}

    @staticmethod
    def verify_facts_locally(facts: List[dict], source_text: str) -> List[dict]:
        """
        Tool 3: Cross-check nhanh bang string matching (khong ton API call).
        Gan co verified=true/false cho tung fact.
        """
        src_lower = source_text.lower()
        verified = []
        for fact in facts:
            val = str(fact.get("value", "")).strip()
            if not val:
                continue

            numbers_in_val = _re.findall(r'[\d.,]+', val)
            is_verified = False

            if numbers_in_val:
                is_verified = all(num in source_text for num in numbers_in_val)
            else:
                val_lower = val.lower()
                if len(val_lower) >= 4:
                    is_verified = val_lower in src_lower
                else:
                    is_verified = True

            verified.append({**fact, "verified": is_verified})

        return verified


# =============================================================================
# AGENT 2: WRITER
# Tools: compose_content (adapts format by field_type)
# =============================================================================

class WriterAgent:
    """
    Viet noi dung dua tren verified facts.
    Tu dong chon van phong theo field_type.
    """

    @staticmethod
    def _get_format_rules(field_type: str, ph_len: int) -> str:
        today = _today_context()
        rules = {
            "date": (
                "DAY LA TRUONG NGAY THANG — PHAI DUNG THOI GIAN THUC.\n"
                "TUYET DOI KHONG lay ngay cu tu tai lieu nguon.\n"
                f"NGAY HIEN TAI: {today['date_str']}\n"
                f"NGAY DANG HANH CHINH: {today['date_formal']}\n"
                "OUTPUT: CHI ngay thang, KHONG viet cau.\n"
                "Format: DD/MM/YYYY hoac 'ngay DD thang MM nam YYYY'.\n"
                f"Mac dinh dung: {today['date_str']}\n"
                "Chi dung ngay khac neu ngu canh yeu cau ro (VD: 'ngay ky', 'ngay hop' cu the)."
            ),
            "time_range": (
                "DAY LA TRUONG KHOANG THOI GIAN / KY BAO CAO — PHAI DUNG THOI GIAN THUC.\n"
                "TUYET DOI KHONG lay ky bao cao cu tu tai lieu nguon.\n"
                f"THOI DIEM HIEN TAI: {today['quarter_roman']}, {today['month_year']}, nam {today['year']}\n"
                f"Quy hien tai: {today['quarter_roman']}\n"
                f"Quy truoc: {today['prev_quarter']}\n"
                f"Nua nam: {today['half_year']}\n"
                "OUTPUT: CHI khoang thoi gian, KHONG viet cau.\n"
                "Chon ky phu hop voi ngu canh:\n"
                f"  - Bao cao quy -> '{today['quarter_roman']}'\n"
                f"  - Bao cao 6 thang -> '{today['half_year']}'\n"
                f"  - Bao cao nam -> 'nam {today['year']}'\n"
                f"  - Giai doan -> 'giai doan 2021-{today['year']}'\n"
                "KHONG copy thoi gian cu tu tai lieu nguon — luon dung thoi gian thuc."
            ),
            "number": (
                "OUTPUT: CHI con so + don vi, KHONG viet cau.\n"
                "Format: '1.847,3 ty dong', '9,8 trieu KHCN'.\n"
                "KHONG lam tron, KHONG uoc tinh."
            ),
            "percentage": (
                "OUTPUT: CHI ty le %, co the kem so sanh ngan.\n"
                "Format: '94,5%' hoac 'dat 94,5%, tang 5,3 diem %'."
            ),
            "name": (
                "OUTPUT: CHI ten/ma, KHONG viet cau.\n"
                "Viet hoa dung: BIDV, NHNN, NQ57."
            ),
            "short": (
                "OUTPUT: 1 cum tu ngan (3-30 ky tu).\n"
                "KHONG viet cau hoan chinh."
            ),
            "sentence": (
                f"OUTPUT: 2-5 cau VAN XUOI lien mach, DAY DU va CHI TIET.\n"
                f"DO DAI TOI THIEU: {max(ph_len * 2, 200)} ky tu — PHAI viet DU DAI.\n"
                "KHONG dung gach dau dong.\n"
                "Moi cau phai chua it nhat 1 du kien cu the (so lieu, ten, ngay).\n"
                "Dung tu noi tu nhien: 'Trong do,', 'Cu the,', 'Dong thoi,', 'Ngoai ra,'.\n"
                "Neu co nhieu facts -> PHAI dung HET, khong bo sot.\n"
                "Viet PHONG PHU, chi tiet — khong tom tat qua ngan."
            ),
            "paragraph": (
                f"OUTPUT: doan van LIEN MACH, DAY DU va CHI TIET.\n"
                f"DO DAI TOI THIEU: {max(ph_len * 2, 400)} ky tu — cang dai cang tot neu co du lieu.\n"
                "CAU TRUC: cau mo dau tong quat -> nhieu cau chi tiet so lieu -> cau ket.\n"
                "KHONG dung gach dau dong — viet van xuoi lien mach.\n"
                "Dung tu noi da dang: 'Trong do,', 'Cu the,', 'Ben canh do,', 'Dang chu y,',\n"
                "  'Ngoai ra,', 'Ve mat...', 'Lien quan den...', 'Dong thoi,'.\n"
                "Moi cau phai chua it nhat 1 du kien cu the.\n"
                "Neu co nhieu facts -> PHAI dung TAT CA, trien khai moi fact thanh 1-2 cau.\n"
                "KHONG viet tom tat — viet PHONG PHU, day du nhu bao cao chinh thuc."
            ),
            "bullet_list": (
                f"OUTPUT: danh sach gach dau dong CHI TIET, phan cach bang \\n.\n"
                f"DO DAI TOI THIEU: {max(ph_len * 2, 300)} ky tu.\n"
                "Moi dong bat dau bang '- '.\n"
                "Moi gach dau dong PHAI co du kien cu the (so, ten, ngay).\n"
                "Moi gach dau dong dai 1-2 cau, KHONG chi viet cum tu ngan.\n"
                "Format: '- [Ten muc/Du an]: [mo ta chi tiet + so lieu cu the]'.\n"
                "Neu co nhieu facts -> tao nhieu gach dau dong, KHONG gop chung."
            ),
        }
        return rules.get(field_type, rules["sentence"])

    @staticmethod
    async def compose_content(
        client: AsyncOpenAI,
        field: dict,
        verified_facts: List[dict],
        field_type: str,
        type_hint: str,
        ph_len: int,
        semaphore: asyncio.Semaphore,
        relevant_ctx: str = "",
    ) -> dict:
        """
        Tool: Viet noi dung tu verified facts theo dung field_type.
        """
        today = _today_context()

        # Build facts section with source citations
        good_facts = [f for f in verified_facts if f.get("verified", False)]
        weak_facts = [f for f in verified_facts if not f.get("verified", False)]

        facts_section = ""
        if good_facts:
            facts_section += "=== DU KIEN DA XAC MINH (DUOC PHEP DUNG) ===\n"
            for i, f in enumerate(good_facts, 1):
                src_file = f.get('source_file', 'khong ro file')
                facts_section += f"{i}. [{f.get('type','?')}] {f['value']}\n"
                facts_section += f"   [File: {src_file}]\n"
                if f.get('source_sentence'):
                    facts_section += f"   [Trich dan: \"{f['source_sentence'][:120]}\"]\n"

        if weak_facts:
            facts_section += "\n=== DU KIEN CHUA XAC MINH (KHONG DUOC DUNG) ===\n"
            for f in weak_facts:
                facts_section += f"!! {f['value']} - KHONG dung trong output\n"

        format_rules = WriterAgent._get_format_rules(field_type, ph_len)

        system_msg = f"""Ban la chuyen gia soan thao van ban hanh chinh ngan hang BIDV.

NGAY HIEN TAI: {today['date_str']} ({today['month_year']}, {today['quarter']})

NGUYEN TAC BAT BUOC:
1. CHI dung du kien da xac minh trong danh sach facts
2. VIET LAI HOAN TOAN - cau truc cau PHAI khac tai lieu nguon
3. Moi cau/dong phai co can cu tu facts — KHONG viet cau chung chung
4. KHONG dung: "da trien khai tot", "ket qua kha quan", "tiep tuc phat huy"
5. Dung TAT CA facts duoc cung cap — KHONG bo sot du kien nao
6. Viet DAY DU, CHI TIET, PHONG PHU — nhu bao cao chinh thuc cua ngan hang

CACH VIET CHI TIET:
- Moi fact nen duoc trien khai thanh 1-2 cau day du
- Dung tu noi da dang de lien ket cac y: "Trong do,", "Cu the,", "Ben canh do,",
  "Dang chu y,", "Ngoai ra,", "Ve mat...", "Dong thoi,", "Ket qua cho thay,"
- Ket hop so lieu voi ngu canh/y nghia: khong chi neu con so ma giai thich y nghia cua no
- Vi du: thay vi "dat 9,8 trieu KHCN" -> viet "BIDV hoan thanh xac thuc sinh trac hoc voi quy mo
  9,8 trieu khach hang ca nhan, the hien buoc tien vuot bac trong cong tac chuyen doi so"

VAN PHONG CHUAN BIDV:
- Chu ngu: "BIDV", "Ngan hang", "Ban/Phong..." (khong dung "chung toi")
- Don vi: ty dong, %, trieu nguoi, lenh/ngay
- Ten viet hoa: BIDV, NHNN, CSDLQG, VNeID, NQ57

Tra loi CHI bang JSON hop le."""

        user_msg = f"""Viet noi dung cho truong trong bao cao BIDV.

=== LOAI TRUONG (QUYET DINH FORMAT OUTPUT) ===
{type_hint}

=== QUY TAC FORMAT ===
{format_rules}

=== THONG TIN TRUONG ===
Placeholder ({ph_len} ky tu): "{field.get('placeholder', '')}"
Ngu canh: {field.get('context', '')}

{facts_section}

=== YEU CAU ===
- Dung TAT CA facts da xac minh — KHONG bo sot bat ky du kien nao
- Moi fact trien khai thanh 1-2 cau day du (khong chi liet ke so lieu)
- Viet lai bang CAU CHU MOI — cau truc cau phai khac tai lieu nguon
- Giu nguyen chinh xac so lieu, ten, ngay thang tu facts
- Viet DAY DU, CHI TIET — output phai dai hon placeholder goc

Tra ve JSON:
{{
  "value": "noi dung viet theo dung format o tren",
  "facts_used": [1, 3, 5],
  "confidence": "high|mid|low",
  "citations": [
    {{"file": "ten_file.docx", "quote": "cau goc trich dan tu tai lieu"}},
    {{"file": "ten_file.pdf", "quote": "cau goc khac"}}
  ],
  "reason": "tom tat ngan: du lieu lay tu file nao, du kien gi"
}}"""

        # Adjust params by field type
        temp_map = {
            "date": 0.0, "time_range": 0.0, "number": 0.0,
            "percentage": 0.0, "name": 0.05, "short": 0.1,
            "sentence": 0.2, "paragraph": 0.35, "bullet_list": 0.25,
        }
        tok_map = {
            "date": 200, "time_range": 200, "number": 200,
            "percentage": 200, "name": 300, "short": 300,
        }

        temperature = temp_map.get(field_type, 0.2)
        if field_type in tok_map:
            max_tok = tok_map[field_type]
        elif field_type == "paragraph":
            max_tok = 4096
        elif field_type == "bullet_list":
            max_tok = 3500
        elif field_type == "sentence":
            max_tok = max(2500, min(ph_len * 8, 4096))
        elif ph_len < 200:
            max_tok = 2500
        else:
            max_tok = 4096

        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=_get_model_for("writer"),
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    max_completion_tokens=max_tok,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content.strip())
            except Exception as e:
                return {"value": "", "confidence": "low",
                        "reason": f"Writer error: {e}"}

    @staticmethod
    async def compose_simple(
        client: AsyncOpenAI,
        field: dict,
        relevant_ctx: str,
        field_type: str,
        type_hint: str,
        ph_len: int,
        semaphore: asyncio.Semaphore,
    ) -> dict:
        """
        Tool: Compose cho field ngan (date/number/name) - dung context truc tiep.
        """
        today = _today_context()
        format_rules = WriterAgent._get_format_rules(field_type, ph_len)

        user_msg = f"""Dien noi dung cho truong trong bao cao BIDV.
NGAY HIEN TAI: {today['date_str']}
QUY HIEN TAI: {today['quarter_roman']}
THANG/NAM: {today['month_year']}
NAM: {today['year']}

=== LOAI TRUONG ===
{type_hint}

=== QUY TAC FORMAT ===
{format_rules}

=== THONG TIN TRUONG ===
Placeholder ({ph_len} ky tu): "{field.get('placeholder', '')}"
Ngu canh: {field.get('context', '')}

=== TAI LIEU NGUON (CHI DUNG CHO SO LIEU, TEN — KHONG DUNG CHO NGAY THANG) ===
{relevant_ctx[:6000]}

=== YEU CAU ===
- Voi truong NGAY THANG / THOI GIAN: LUON dung thoi gian thuc (hien tai), KHONG lay ngay cu tu tai lieu nguon
- Voi truong SO LIEU / TEN: Tim thong tin CU THE tu tai lieu nguon
- Neu khong tim duoc so lieu -> ghi 'chua co du lieu'
- KHONG bia so lieu, ten
- Giu dung FORMAT yeu cau

LUU Y: Tai lieu nguon co dang "=== ten_file ===" de nhan biet file.
Ghi ro ten file nguon trong reason (tru truong ngay/thoi gian thi ghi "thoi gian thuc").

Tra ve JSON:
{{
  "value": "noi dung dien vao",
  "confidence": "high|mid|low",
  "source_file": "ten_file_nguon.docx (hoac 'thoi gian thuc' neu la truong ngay)",
  "source_quote": "trich dan chinh xac cau goc (hoac 'realtime' neu la ngay/thoi gian)",
  "reason": "Nguon: giai thich cu the"
}}"""

        temp_map = {
            "date": 0.0, "time_range": 0.0, "number": 0.0,
            "percentage": 0.0, "name": 0.05, "short": 0.1,
        }
        temperature = temp_map.get(field_type, 0.15)
        max_tok = 300

        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=_get_model_for("writer"),
                    messages=[{"role": "user", "content": user_msg}],
                    max_completion_tokens=max_tok,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content.strip())
            except Exception as e:
                return {"value": "", "confidence": "low",
                        "reason": f"Writer error: {e}"}


# =============================================================================
# AGENT 3: VERIFIER
# Tools: cross_check_facts, detect_fabrication
# =============================================================================

class VerifierAgent:
    """
    Kiem dinh noi dung AI viet, doi chieu voi tai lieu nguon.
    Phat hien thong tin bia, danh gia confidence.
    """

    @staticmethod
    async def verify_content(
        client: AsyncOpenAI,
        ai_value: str,
        relevant_ctx: str,
        semaphore: asyncio.Semaphore,
    ) -> dict:
        """
        Tool: Cross-check toan bo noi dung AI viet voi tai lieu nguon.
        Returns: {status, checks, fabricated, note}
        """
        if not ai_value.strip():
            return {"status": "fail", "note": "AI khong tao duoc noi dung",
                    "fabricated": [], "checks": []}

        system_qc = """Ban la chuyen gia kiem dinh noi dung bao cao ngan hang BIDV.
Nhiem vu: doi chieu tung thong tin trong noi dung AI viet voi tai lieu nguon.

QUY TAC:
1. Moi CON SO (%, ty, trieu) -> phai tim duoc trong tai lieu nguon
2. Moi TEN (du an, van ban, to chuc) -> phai co trong tai lieu nguon
3. Moi NGAY THANG -> phai khop voi tai lieu nguon
4. Cau chung chung khong co du kien -> gan co warning

QUAN TRONG: Neu AI viet "tuong duong X%" ma X% KHONG co trong nguon -> do la BIA.

Tra loi CHI bang JSON hop le."""

        user_qc = f"""Kiem tra noi dung AI viet co can cu tu tai lieu nguon khong.

NOI DUNG AI VIET:
"{ai_value}"

TAI LIEU NGUON:
{relevant_ctx[:6000]}

Kiem tra tung thong tin:
1. Liet ke MOI con so, ten, ngay trong noi dung AI
2. Voi moi thong tin: tim cau goc trong tai lieu nguon
3. Neu KHONG tim duoc -> gan co "fabricated"

Tra ve JSON:
{{
  "status": "ok|warning|fail",
  "checks": [
    {{"info": "9,8 trieu KHCN", "found_in_source": true, "source_quote": "cau goc..."}},
    {{"info": "94% tong KHCN", "found_in_source": false, "source_quote": null}}
  ],
  "fabricated": ["liet ke cum tu/so lieu bia"],
  "note": "giai thich ngan"
}}

Quy tac status:
- ok:      TAT CA checks deu found_in_source = true
- warning: co 1-2 checks khong ro rang
- fail:    co checks ro rang found_in_source = false (bia)"""

        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=_get_model_for("verifier"),
                    messages=[
                        {"role": "system", "content": system_qc},
                        {"role": "user", "content": user_qc},
                    ],
                    max_completion_tokens=800,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content.strip())
            except Exception as e:
                return {"status": "warning", "note": f"Verifier error: {e}",
                        "fabricated": [], "checks": []}


# =============================================================================
# AGENT 4: EDITOR
# Tools: rewrite_with_feedback (self-correction)
# =============================================================================

class EditorAgent:
    """
    Nhan feedback tu Verifier, viet lai noi dung da sua loi.
    Chi chay khi Verifier phat hien fabrication.
    """

    @staticmethod
    async def rewrite_with_feedback(
        client: AsyncOpenAI,
        field: dict,
        original_value: str,
        fabricated_items: List[str],
        verifier_note: str,
        verified_facts: List[dict],
        field_type: str,
        type_hint: str,
        ph_len: int,
        semaphore: asyncio.Semaphore,
    ) -> dict:
        """
        Tool: Viet lai noi dung, xoa bo thong tin bia, chi giu facts verified.
        """
        today = _today_context()
        format_rules = WriterAgent._get_format_rules(field_type, ph_len)

        good_facts = [f for f in verified_facts if f.get("verified", False)]
        facts_list = ""
        if good_facts:
            for i, f in enumerate(good_facts, 1):
                facts_list += f"{i}. [{f.get('type','?')}] {f['value']}\n"

        system_msg = f"""Ban la bien tap vien cao cap bao cao ngan hang BIDV.
NGAY HIEN TAI: {today['date_str']} ({today['quarter']})

NHIEM VU: Sua lai noi dung da bi kiem dinh vien phat hien loi.
- XOA BO hoan toan cac thong tin bia
- CHI giu lai thong tin co trong danh sach facts
- Viet lai bang cau chu moi, van phong hanh chinh

Tra loi CHI bang JSON hop le."""

        user_msg = f"""Noi dung GOC (co loi):
"{original_value}"

=== LOI CAN SUA ===
- Thong tin bia: {', '.join(fabricated_items)}
- Ghi chu kiem dinh: {verifier_note}

=== DU KIEN DUOC PHEP DUNG ===
{facts_list or '(Khong co facts - viet ngan gon nhat co the)'}

=== LOAI TRUONG ===
{type_hint}

=== QUY TAC FORMAT ===
{format_rules}

=== YEU CAU ===
- XOA BO hoan toan: {', '.join(fabricated_items)}
- CHI dung du kien trong danh sach tren
- Viet lai bang cau chu MOI
- CHINH XAC quan trong hon DAI

Tra ve JSON:
{{
  "value": "noi dung da sua",
  "confidence": "mid",
  "reason": "da sua lai sau kiem tra - xoa {len(fabricated_items)} thong tin bia"
}}"""

        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=_get_model_for("editor"),
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    max_completion_tokens=2000,
                    temperature=0.15,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content.strip())
            except Exception as e:
                return {"value": original_value, "confidence": "low",
                        "reason": f"Editor error: {e}"}