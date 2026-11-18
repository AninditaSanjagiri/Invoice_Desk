"""Convert DocTR-format OCR (live DocTR export OR the DocILE-provided ocr/*.json) into an OCRDocument."""
from __future__ import annotations
import numpy as np
from .models import Word, Line, OCRDocument

def flatten_export(export: dict, max_pages: int | None = None) -> tuple[list[Word], int, str]:
    words, plain, pages = [], [], export["pages"][:max_pages] if max_pages else export["pages"]
    for pi, page in enumerate(pages):
        for b in page.get("blocks", []):
            for l in b.get("lines", []):
                plain.append(" ".join(w["value"] for w in l.get("words", [])))
                for w in l.get("words", []):
                    (x0, y0), (x1, y1) = w["geometry"]
                    words.append(Word(pi, str(w["value"]), float(w.get("confidence", 1.0)), x0, y0, x1, y1))
            plain.append("")
    return words, len(pages), "\n".join(plain).strip()

def build_rows(words: list[Word], gap_sep: float = 0.03) -> list[Line]:
    """Group words into visual rows per page so labels and values on one row share a line (reading order)."""
    lines, n = [], 0
    for page in sorted({w.page for w in words}):
        ws = sorted([(i, w) for i, w in enumerate(words) if w.page == page], key=lambda t: (t[1].yc, t[1].x0)); rows = []
        for i, w in ws:
            if rows and abs(w.yc - np.mean([words[j].yc for j in rows[-1]])) <= 0.6 * np.median([words[j].h for j in rows[-1]]): rows[-1].append(i)
            else: rows.append([i])
        for r in rows:
            r = sorted(r, key=lambda j: words[j].x0); parts = []
            for k, j in enumerate(r):
                if k and words[j].x0 - words[r[k - 1]].x1 > gap_sep: parts.append("|")
                parts.append(words[j].text)
            n += 1
            lines.append(Line(f"L{n:03d}", page, " ".join(parts), r, float(np.mean([words[j].conf for j in r]))))
    return lines

def ocr_from_export(export: dict, source: str = "doctr", max_pages: int | None = None) -> OCRDocument:
    words, n_pages, plain = flatten_export(export, max_pages)
    return OCRDocument(words, build_rows(words), n_pages, source, plain)
