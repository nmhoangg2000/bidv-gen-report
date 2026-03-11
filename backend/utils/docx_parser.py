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


def _parse_highlights(doc_xml_path: str) -> List[Dict]:
    raw = open(doc_xml_path, "rb").read()
    tree = ET.fromstring(raw)
    fields = []

    for para_idx, para in enumerate(tree.iter(f"{{{NS_W}}}p")):
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
        if highlights:
            fields.append({
                "key":         f"para_{para_idx}",
                "para_idx":    para_idx,
                "placeholder": "".join(highlights),
                "context":     full_text[:200].strip(),
            })
    return fields


# ─── Fill and export ──────────────────────────────────────────────────────────

def fill_and_export(
    docx_bytes:   bytes,
    field_values: List[Dict],
    apply_colors: bool = False,
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

        # Đọc raw XML bytes — không dùng ET.write để tránh mất namespace
        raw = open(doc_xml_path, "rb").read()
        raw = _replace_highlights_raw(raw, field_values, apply_colors)
        open(doc_xml_path, "wb").write(raw)

        # Zip lại — giữ nguyên compression từ file gốc
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


def _replace_highlights_raw(raw: bytes, field_values: List[Dict], apply_colors: bool) -> bytes:
    """
    Thay thế yellow highlight bằng cách parse XML trong memory,
    sửa trực tiếp trên string XML rồi trả về bytes — giữ nguyên namespace declarations.
    """
    xml_str = raw.decode("utf-8")

    val_map  = {fv["field_key"]: fv.get("final_value", "") for fv in field_values}
    conf_map = {fv["field_key"]: fv.get("confidence", "mid") for fv in field_values}

    # Parse để tìm vị trí các paragraph + highlight
    tree = ET.fromstring(raw)
    paras = list(tree.iter(f"{{{NS_W}}}p"))

    for para_idx, para in enumerate(paras):
        key   = f"para_{para_idx}"
        value = val_map.get(key)
        if value is None:
            continue

        h_runs = []
        for run in para.iter(f"{{{NS_W}}}r"):
            rpr = run.find(f"{{{NS_W}}}rPr")
            t   = run.find(f"{{{NS_W}}}t")
            if rpr is not None and t is not None and t.text:
                hl = rpr.find(f"{{{NS_W}}}highlight")
                if hl is not None and hl.get(f"{{{NS_W}}}val") == "yellow":
                    h_runs.append((t.text, hl.get(f"{{{NS_W}}}val")))

        if not h_runs:
            continue

        # Thay placeholder đầu tiên bằng value, xóa highlight tag
        placeholder = h_runs[0][0]
        escaped_val = (value
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))

        if apply_colors:
            conf  = conf_map.get(key, "mid")
            color = "green" if conf == "high" else "red"
            # Đổi màu highlight
            xml_str = xml_str.replace(
                f'w:val="yellow"',
                f'w:val="{color}"',
                1
            )
        else:
            # Xóa toàn bộ thẻ highlight yellow trong đoạn này
            xml_str = xml_str.replace(
                '<w:highlight w:val="yellow"/>',
                '',
                len(h_runs)
            )

        # Thay text placeholder
        if placeholder in xml_str:
            xml_str = xml_str.replace(
                f'>{placeholder}<',
                f'>{escaped_val}<',
                1
            )

    return xml_str.encode("utf-8")


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