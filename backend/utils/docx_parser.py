"""
DOCX field extractor and filler.
Finds yellow-highlighted runs → extracts placeholder text + context.
On export: replaces highlighted runs with filled values.
"""
import os
import re
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace('wpc', 'http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas')
ET.register_namespace('cx',  'http://schemas.microsoft.com/office/drawing/2014/chartex')
ET.register_namespace('mc',  'http://schemas.openxmlformats.org/markup-compatibility/2006')
ET.register_namespace('o',   'urn:schemas-microsoft-com:office:office')
ET.register_namespace('r',   'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
ET.register_namespace('m',   'http://schemas.openxmlformats.org/officeDocument/2006/math')
ET.register_namespace('v',   'urn:schemas-microsoft-com:vml')
ET.register_namespace('wp14','http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing')
ET.register_namespace('wp',  'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing')
ET.register_namespace('w10', 'urn:schemas-microsoft-com:office:word')
ET.register_namespace('w',   'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
ET.register_namespace('w14', 'http://schemas.microsoft.com/office/word/2010/wordml')
ET.register_namespace('w15', 'http://schemas.microsoft.com/office/word/2012/wordml')
ET.register_namespace('w16cex','http://schemas.microsoft.com/office/word/2018/wordml/cex')
ET.register_namespace('w16se','http://schemas.microsoft.com/office/word/2015/wordml/symex')
ET.register_namespace('wpg', 'http://schemas.microsoft.com/office/word/2010/wordprocessingGroup')
ET.register_namespace('wpi', 'http://schemas.microsoft.com/office/word/2010/wordprocessingInk')
ET.register_namespace('wne', 'http://schemas.microsoft.com/office/word/2006/wordml')
ET.register_namespace('wps', 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape')
ET.register_namespace('xml', 'http://www.w3.org/XML/1998/namespace')


# ─── Extract yellow fields ────────────────────────────────────────────────────

def extract_fields(docx_bytes: bytes) -> List[Dict]:
    with tempfile.TemporaryDirectory() as tmp:
        docx_path  = os.path.join(tmp, "template.docx")
        unpack_dir = os.path.join(tmp, "unpacked")
        with open(docx_path, "wb") as f:
            f.write(docx_bytes)
        with zipfile.ZipFile(docx_path, "r") as z:
            z.extractall(unpack_dir)
        return _parse_highlights(os.path.join(unpack_dir, "word", "document.xml"))


import re as _re

# ─── Patterns detect loại field ──────────────────────────────────────────────

_DATE_PATTERNS = [
    _re.compile(r'\bDD/MM/YYYY\b', _re.I),
    _re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),
    _re.compile(r'\bngày\b.{0,20}\btháng\b', _re.I),
    _re.compile(r'\btháng\s*\d+\b', _re.I),
    _re.compile(r'\bQuý\s+[IVX\d]+\b', _re.I),
    _re.compile(r'\bnăm\s+20\d{2}\b', _re.I),
    _re.compile(r'\b20\d{2}\b'),
    _re.compile(r'ngày\s*\.\.\.|tháng\s*\.\.\.|năm\s*\.\.\.', _re.I),
]

_NUMBER_PATTERNS = [
    _re.compile(r'\bXXX\b|\b\.\.\.\b|\bN\b'),
    _re.compile(r'\d+[.,]\d+\s*(tỷ|triệu|nghìn|%|người|dự án|lệnh)', _re.I),
    _re.compile(r'\bsố\s*(lượng|liệu|tiền|vốn|dự án)\b', _re.I),
    _re.compile(r'\[\s*số\s*\]|\[\s*tỷ\s*\]|\[\s*%\s*\]', _re.I),
]

_FIELD_TYPE_HINTS = {
    "date": (
        "ĐÂY LÀ TRƯỜNG NGÀY THÁNG.\n"
        "→ Điền đúng ngày/tháng/năm thực tế từ tài liệu nguồn hoặc ngày hiện tại.\n"
        "→ Giữ đúng định dạng: DD/MM/YYYY, tháng X/YYYY, Quý X/YYYY.\n"
        "→ KHÔNG điền text khác, KHÔNG để trống."
    ),
    "number": (
        "ĐÂY LÀ TRƯỜNG SỐ LIỆU / CON SỐ.\n"
        "→ Tìm con số cụ thể từ tài liệu nguồn.\n"
        "→ Giữ đơn vị: tỷ đồng, %, triệu người, dự án...\n"
        "→ Nếu không tìm được → ghi rõ 'chưa có số liệu cụ thể'."
    ),
    "short": (
        "ĐÂY LÀ TRƯỜNG NGẮN (tên, mã, cụm từ).\n"
        "→ Điền 1 cụm từ hoặc tên cụ thể từ tài liệu nguồn.\n"
        "→ Không cần câu hoàn chỉnh."
    ),
    "paragraph": (
        "ĐÂY LÀ TRƯỜNG ĐOẠN VĂN DÀI.\n"
        "→ Viết nhiều câu, đầy đủ ý, có số liệu cụ thể.\n"
        "→ Độ dài output PHẢI tương đương hoặc DÀI HƠN placeholder gốc.\n"
        "→ Liệt kê tất cả kết quả, dự án, số liệu liên quan từ tài liệu nguồn."
    ),
    "sentence": (
        "ĐÂY LÀ TRƯỜNG 1-3 CÂU.\n"
        "→ Viết 1-3 câu hoàn chỉnh, có số liệu cụ thể.\n"
        "→ Văn phong hành chính, súc tích."
    ),
}


def _classify_field(placeholder: str, context: str) -> str:
    """Phân loại field dựa trên placeholder text và ngữ cảnh."""
    ph     = placeholder.strip()
    ph_len = len(ph)

    # Với placeholder DÀI (> 60 ký tự) → ưu tiên phân loại theo độ dài
    if ph_len > 60:
        return "paragraph" if ph_len > 200 else "sentence"

    # Ngày tháng — chỉ check khi placeholder ngắn
    for pat in _DATE_PATTERNS:
        if pat.search(ph):
            return "date"

    # Số liệu
    for pat in _NUMBER_PATTERNS:
        if pat.search(ph):
            return "number"

    # Độ dài
    if ph_len <= 25:
        return "short"
    elif ph_len <= 200:
        return "sentence"
    else:
        return "paragraph"


def _parse_highlights(doc_xml_path: str) -> List[Dict]:
    raw = open(doc_xml_path, "rb").read()
    tree = ET.fromstring(raw)
    fields = []

    all_paras = list(tree.iter(f"{{{NS_W}}}p"))

    for para_idx, para in enumerate(all_paras):
        runs = list(para.iter(f"{{{NS_W}}}r"))
        full_text = "".join(
            (r.find(f"{{{NS_W}}}t").text or "")
            for r in runs if r.find(f"{{{NS_W}}}t") is not None
        )
        highlights = []
        for run in runs:
            rpr = run.find(f"{{{NS_W}}}rPr")
            t   = run.find(f"{{{NS_W}}}t")
            if rpr is not None and t is not None and t.text:
                hl = rpr.find(f"{{{NS_W}}}highlight")
                if hl is not None and hl.get(f"{{{NS_W}}}val") == "yellow":
                    highlights.append(t.text)
        if not highlights:
            continue

        placeholder = "".join(highlights)

        # Context: lấy đoạn trước + hiện tại + sau
        prev_text = ""
        if para_idx > 0:
            prev_runs = list(all_paras[para_idx-1].iter(f"{{{NS_W}}}r"))
            prev_text = "".join(
                (r.find(f"{{{NS_W}}}t").text or "")
                for r in prev_runs if r.find(f"{{{NS_W}}}t") is not None
            )
        next_text = ""
        if para_idx < len(all_paras) - 1:
            next_runs = list(all_paras[para_idx+1].iter(f"{{{NS_W}}}r"))
            next_text = "".join(
                (r.find(f"{{{NS_W}}}t").text or "")
                for r in next_runs if r.find(f"{{{NS_W}}}t") is not None
            )
        rich_context = f"{prev_text[:100]} | {full_text[:200]} | {next_text[:100]}".strip()

        # Phân loại field
        field_type = _classify_field(placeholder, rich_context)

        fields.append({
            "key":          f"para_{para_idx}",
            "para_idx":     para_idx,
            "placeholder":  placeholder,
            "context":      rich_context[:400],
            "field_type":   field_type,
            "type_hint":    _FIELD_TYPE_HINTS[field_type],
        })
    return fields



# ─── Fill and export ──────────────────────────────────────────────────────────

def fill_and_export(
    docx_bytes:   bytes,
    field_values: List[Dict],
    apply_colors: bool = False,  # True = giữ highlight vàng, False = xóa highlight
) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        docx_path  = os.path.join(tmp, "template.docx")
        unpack_dir = os.path.join(tmp, "unpacked")
        out_path   = os.path.join(tmp, "output.docx")

        with open(docx_path, "wb") as f:
            f.write(docx_bytes)
        with zipfile.ZipFile(docx_path, "r") as z:
            z.extractall(unpack_dir)

        doc_xml_path = os.path.join(unpack_dir, "word", "document.xml")
        raw = open(doc_xml_path, "rb").read()
        raw = _process_fields(raw, field_values, keep_highlight=apply_colors)
        open(doc_xml_path, "wb").write(raw)

        with zipfile.ZipFile(docx_path, "r") as orig:
            with zipfile.ZipFile(out_path, "w") as out_zip:
                for item in orig.infolist():
                    modified = os.path.join(unpack_dir, item.filename)
                    if os.path.isfile(modified):
                        out_zip.write(modified, item.filename,
                                      compress_type=item.compress_type)
                    else:
                        out_zip.writestr(item, orig.read(item.filename))

        return open(out_path, "rb").read()


def _process_fields(raw: bytes, field_values: List[Dict], keep_highlight: bool = False) -> bytes:
    """
    Thay thế yellow highlight bằng giá trị AI.
    keep_highlight=True  → giữ highlight vàng trên nội dung AI (mode "tracked")
    keep_highlight=False → xóa highlight, file sạch (mode "clean")
    """
    val_map = {fv["field_key"]: fv.get("final_value", "") for fv in field_values}

    xml_str      = raw.decode("utf-8")
    root_open_end = xml_str.find('>') + 1
    root_tag_end  = xml_str.find('>', root_open_end) + 1
    orig_hdr      = xml_str[:root_tag_end]

    tree  = ET.fromstring(raw)
    paras = list(tree.iter(f"{{{NS_W}}}p"))

    for para_idx, para in enumerate(paras):
        key   = f"para_{para_idx}"
        value = val_map.get(key)
        if value is None:
            continue

        # Thu thập tất cả highlighted runs
        h_runs = []
        for run in para.iter(f"{{{NS_W}}}r"):
            rpr = run.find(f"{{{NS_W}}}rPr")
            t   = run.find(f"{{{NS_W}}}t")
            if rpr is not None and t is not None and t.text:
                hl = rpr.find(f"{{{NS_W}}}highlight")
                if hl is not None and hl.get(f"{{{NS_W}}}val") == "yellow":
                    h_runs.append((t, rpr, hl))

        if not h_runs:
            continue

        # Run đầu tiên: đặt text = giá trị AI
        first_t, first_rpr, first_hl = h_runs[0]
        first_t.text = value
        if value.startswith(' ') or value.endswith(' '):
            first_t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')

        # Xử lý highlight: giữ hay xóa
        if not keep_highlight:
            # Clean mode: xóa highlight khỏi tất cả runs
            first_rpr.remove(first_hl)
            for t, rpr, hl in h_runs[1:]:
                t.text = ""
                rpr.remove(hl)
        else:
            # Tracked mode: giữ highlight vàng trên run đầu, xóa text các run còn lại
            for t, rpr, hl in h_runs[1:]:
                t.text = ""

    # Serialize — restore header gốc để giữ XML declaration + namespace declarations
    new_xml = ET.tostring(tree, encoding="unicode", xml_declaration=False)
    new_end = new_xml.find('>') + 1
    new_xml = orig_hdr + new_xml[new_end:]

    return new_xml.encode("utf-8")


def _replace_highlights_raw(raw: bytes, field_values: List[Dict], apply_colors: bool) -> bytes:
    return _process_fields(raw, field_values)




# ─── Extract text from source files ──────────────────────────────────────────

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        if ext == ".pdf":
            r = subprocess.run(["pdftotext", tmp_path, "-"], capture_output=True)
            return r.stdout.decode("utf-8", errors="replace")[:60000]
        elif ext in (".docx", ".doc"):
            r = subprocess.run(
                ["pandoc", "--track-changes=all", tmp_path, "-t", "plain"],
                capture_output=True
            )
            return r.stdout.decode("utf-8", errors="replace")[:60000]
        else:
            with open(tmp_path, errors="replace") as f:
                return f.read()[:60000]
    finally:
        os.unlink(tmp_path)