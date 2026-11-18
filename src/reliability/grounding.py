"""Locate an extracted value in OCR text (shared by Signal 1 and Signal 2)."""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from ..utils.normalize import parse_amount, norm_amount, date_interps

DATE_PATTERNS = [r"\b\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}\b", r"\b\d{1,2}\s*[A-Za-z]{3,9}\.?,?\s*\d{2,4}\b", r"\b[A-Za-z]{3,9}\.?\s*\d{1,2},?\s*\d{2,4}\b", r"\b[A-Za-z]{3,9}\.?\s*\d{1,2}[/\-.]\d{2,4}\b"]

def ground_score(kind: str, value, text: str) -> float:
    """1.0 if the (normalised) value is found in `text`; otherwise longest-common-substring coverage in [0,1)."""
    if not text or value is None: return 0.0
    if kind == "amount":
        want = norm_amount(value)
        return 1.0 if want is not None and any(norm_amount(c) == want for c in re.findall(r"\d[\d.,]*\d|\d", text)) else 0.0
    if kind == "date":
        want = date_interps(value)
        return 1.0 if want and any(date_interps(m) & want for pat in DATE_PATTERNS for m in re.findall(pat, text)) else 0.0
    v = re.sub(r"[^0-9a-z]", "", str(value).lower()); t = re.sub(r"[^0-9a-z]", "", text.lower())
    if not v: return 0.0
    if v in t: return 1.0
    return SequenceMatcher(None, v, t, autojunk=False).find_longest_match(0, len(v), 0, len(t)).size / len(v)

def tok(s) -> str: return re.sub(r"[^0-9a-z]", "", str(s).lower())

def locate_lines(kind: str, value, ocr, thr: float = 0.99) -> list[str]:
    """Line IDs whose own text contains the value (used when the model gave no usable citation)."""
    return [l.id for l in ocr.lines if ground_score(kind, value, l.text) >= thr]
