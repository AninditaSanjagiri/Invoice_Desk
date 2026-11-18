"""OCR data model: words with confidence + geometry, visual-row lines with stable IDs."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class Word:
    page: int; text: str; conf: float; x0: float; y0: float; x1: float; y1: float
    @property
    def yc(self): return (self.y0 + self.y1) / 2
    @property
    def h(self): return max(self.y1 - self.y0, 1e-4)

@dataclass
class Line:
    id: str; page: int; text: str; word_idx: list; conf: float

class OCRDocument:
    def __init__(self, words: list[Word], lines: list[Line], n_pages: int, source: str = "doctr", plain_text: str = ""):
        self.words, self.lines, self.n_pages, self.source, self.plain_text = words, lines, n_pages, source, plain_text
        self.line_by_id = {l.id: l for l in lines}
    @property
    def mean_conf(self) -> float: return float(np.mean([w.conf for w in self.words])) if self.words else 0.0
    @property
    def full_text(self) -> str: return "\n".join(l.text for l in self.lines)

    def indexed_text(self, max_chars: int = 30000) -> tuple[str, int, bool]:
        """Readable text, one row per line: '[L003] Invoice No: | 12345'. Returns (text, lines_used, truncated)."""
        out, used, total, cur = [], 0, 0, None
        for l in self.lines:
            if l.page != cur:
                cur = l.page; hdr = f"=== PAGE {cur + 1} ==="; out.append(hdr); total += len(hdr) + 1
            s = f"[{l.id}] {l.text}"
            if total + len(s) + 1 > max_chars: out.append("[TRUNCATED: remaining lines omitted]"); return "\n".join(out), used, True
            out.append(s); total += len(s) + 1; used += 1
        return "\n".join(out), used, False

    def to_dict(self) -> dict:
        return dict(n_pages=self.n_pages, source=self.source, plain_text=self.plain_text, words=[asdict(w) for w in self.words], lines=[asdict(l) for l in self.lines])
    @classmethod
    def from_dict(cls, d: dict) -> "OCRDocument":
        return cls([Word(**w) for w in d["words"]], [Line(**l) for l in d["lines"]], d["n_pages"], d.get("source", "doctr"), d.get("plain_text", ""))
