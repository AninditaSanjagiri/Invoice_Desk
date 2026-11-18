"""Signals 1 (OCR/layout quality) and 2 (LLM grounding + self-consistency); Signal 3 comes from src.validation."""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict, field
import numpy as np
from ..config import Settings
from ..extraction.schema import KIND, FieldOut, FIELD_NAMES
from ..ocr.models import OCRDocument
from ..utils.normalize import canon
from .grounding import ground_score, locate_lines, tok

# ---------------- Signal 1: OCR and layout quality (per field, from the words that support THIS value) ----------------
@dataclass
class OCRSignal:
    status: str                      # HIGH | LOW | UNAVAILABLE (UNAVAILABLE = could not be linked to OCR evidence - not a low score)
    conf_mean: float | None = None; conf_min: float | None = None; conf_p10: float | None = None
    n_words: int = 0; located: bool = False; evidence_lines: list = field(default_factory=list)
    fragmented: bool = False; page: int | None = None; bbox: list | None = None; reason: str = ""
    def to_dict(self): return asdict(self)

def ocr_signal(name: str, fo: FieldOut, ocr: OCRDocument, s: Settings) -> OCRSignal | None:
    if fo.value is None: return None
    kind = KIND[name]; cited = [i for i in fo.source_lines if i in ocr.line_by_id]; basis = cited
    if not cited or ground_score(kind, fo.value, " ".join(ocr.line_by_id[i].text for i in cited)) < 0.8:
        found = locate_lines(kind, fo.value, ocr); basis = found       # no usable citation -> try to locate the value ourselves
    if not basis: return OCRSignal("UNAVAILABLE", located=False, reason="Value could not be located in the OCR text; no OCR evidence to score.")
    idx = [j for i in basis for j in ocr.line_by_id[i].word_idx]
    toks = {tok(t) for t in re.split(r"\s+", str(fo.value)) if tok(t)}
    ws = [ocr.words[j] for j in idx if tok(ocr.words[j].text) in toks or (kind in ("amount", "date") and any(tok(ocr.words[j].text) and tok(ocr.words[j].text) in t or t in tok(ocr.words[j].text) for t in toks if len(t) > 1))]
    if not ws: return OCRSignal("UNAVAILABLE", located=True, evidence_lines=basis, reason="Located the lines but no individual OCR words match the value.")
    cs = np.array([w.conf for w in ws]); fragmented = kind in ("id", "date", "amount", "email") and len({l for l in basis}) > 1 and len(basis) > 2
    st = "HIGH" if (cs.mean() >= s.min_ocr_conf_mean and cs.min() >= s.min_ocr_conf_min and not fragmented) else "LOW"
    reason = f"mean conf {cs.mean():.2f}, min {cs.min():.2f} over {len(ws)} supporting word(s)" + ("; value is split across many lines" if fragmented else "")
    return OCRSignal(st, float(cs.mean()), float(cs.min()), float(np.percentile(cs, 10)), len(ws), True, basis, fragmented, ws[0].page,
                     [min(w.x0 for w in ws), min(w.y0 for w in ws), max(w.x1 for w in ws), max(w.y1 for w in ws)], reason)

# ---------------- Signal 2: LLM extraction reliability (evidence grounding + self-consistency) ----------------
@dataclass
class LLMSignal:
    status: str                      # SUPPORTED | UNCITED_FOUND | MISCITED | INVALID_CITATION | UNSUPPORTED
    has_citation: bool = False; cited_exist: bool = True
    g_cited: float | None = None; g_any: float = 0.0
    sc_agree: float | None = None; n_samples: int = 0; reason: str = ""
    def to_dict(self): return asdict(self)

def llm_signal(name: str, fo: FieldOut, ocr: OCRDocument, samples: list[dict[str, FieldOut]], s: Settings) -> LLMSignal | None:
    if fo.value is None: return None
    kind = KIND[name]; has = bool(fo.source_lines); exist = all(i in ocr.line_by_id for i in fo.source_lines)
    g_any = ground_score(kind, fo.value, ocr.full_text)
    g_cit = ground_score(kind, fo.value, " ".join(ocr.line_by_id[i].text for i in fo.source_lines if i in ocr.line_by_id)) if has and any(i in ocr.line_by_id for i in fo.source_lines) else None
    thr = s.min_grounding
    if not has: st, why = ("UNCITED_FOUND", "Value is in the OCR text but the model cited no lines.") if g_any >= thr else ("UNSUPPORTED", "No citation and the value is not found in the OCR text.")
    elif not exist: st, why = "INVALID_CITATION", "The model cited OCR line IDs that do not exist."
    elif g_cit is not None and g_cit >= thr: st, why = "SUPPORTED", "Value found in the cited OCR line(s)."
    elif g_any >= thr: st, why = "MISCITED", "Value exists in the document but not in the cited lines (ambiguous evidence)."
    else: st, why = "UNSUPPORTED", f"Value is not supported by the OCR text (best match {max(g_cit or 0, g_any):.2f})."
    sc = float(np.mean([canon(kind, smp[name].value) == canon(kind, fo.value) for smp in samples])) if samples else None
    return LLMSignal(st, has, exist, g_cit, g_any, sc, len(samples), why + (f" Self-consistency {sc:.2f} over {len(samples)} sample(s)." if sc is not None else ""))

@dataclass
class FieldSignals:
    name: str; value: str | None; source_lines: list
    ocr: OCRSignal | None; llm: LLMSignal | None; validation: dict      # validation: n_applicable, n_fail, fails, messages
    def to_dict(self): return dict(name=self.name, value=self.value, source_lines=self.source_lines, ocr=self.ocr.to_dict() if self.ocr else None,
                                   llm=self.llm.to_dict() if self.llm else None, validation=self.validation)

def assess_fields(ocr: OCRDocument, fields: dict[str, FieldOut], samples: list[dict[str, FieldOut]], rule_by_field: dict, s: Settings) -> dict[str, FieldSignals]:
    out = {}
    for n in FIELD_NAMES:
        fo = fields[n]
        out[n] = FieldSignals(n, fo.value, fo.source_lines, ocr_signal(n, fo, ocr, s), llm_signal(n, fo, ocr, samples, s),
                              rule_by_field.get(n, dict(n_applicable=0, n_fail=0, fails=[], messages=[])))
    return out

def evidence_for_rules(ocr: OCRDocument, fields: dict[str, FieldOut]) -> dict:
    ev = {}
    for n, fo in fields.items():
        if fo.value is None: continue
        has = bool(fo.source_lines); exist = all(i in ocr.line_by_id for i in fo.source_lines)
        g = ground_score(KIND[n], fo.value, " ".join(ocr.line_by_id[i].text for i in fo.source_lines if i in ocr.line_by_id)) if has and exist else None
        ev[n] = dict(has_citation=has, cited_exist=exist, g_cited=g)
    return ev
