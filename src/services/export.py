"""JSON / CSV export of the PERSISTED records (final values = human edits where present)."""
from __future__ import annotations
import csv, io, json
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..database import repository as repo
from ..database.models import Document
from ..extraction.schema import FIELD_BY_NAME, FIELD_NAMES
from ..utils.normalize import norm_amount

def export_records(s: Session, doc_ids: list[int] | None = None) -> list[dict]:
    q = select(Document).order_by(Document.id)
    if doc_ids is not None: q = q.where(Document.id.in_(doc_ids))
    out = []
    for d in s.scalars(q):
        fields = {}
        for r in repo.current_fields(s, d.id):
            o, l = r.ocr_signal or {}, r.llm_signal or {}
            fields[r.name] = dict(value=r.final_value, extracted_value=r.value, edited=r.was_edited, decision=r.decision, source_lines=r.source_lines, issues=r.issues,
                                  ocr_status=o.get("status"), ocr_conf_mean=o.get("conf_mean"), grounding_status=l.get("status"), self_consistency=l.get("sc_agree"),
                                  validation_failures=(r.validation or {}).get("fails", []))
        out.append(dict(document_id=d.id, filename=d.original_name, folder=repo.folder_path(s, d.folder_id), sha256=d.file_hash, processing_status=d.processing_status,
                        system_decision=d.decision, decision_reasons=d.decision_reasons, review_status=d.review_status, retry_count=d.retry_count,
                        processing_seconds=round(d.processing_seconds or 0, 2), error=d.error, created_at=d.created_at.isoformat(timespec="seconds"), fields=fields))
    return out

def to_json(records: list[dict]) -> str: return json.dumps(records, indent=2, ensure_ascii=False)

def to_csv(records: list[dict]) -> str:
    """One row per document. Nulls -> empty cell. Amount fields additionally get a normalised numeric column (e.g. 21000.00)."""
    base = ["document_id", "filename", "folder", "processing_status", "system_decision", "review_status", "retry_count", "processing_seconds", "created_at"]
    cols = base[:]
    for n in FIELD_NAMES:
        cols += [n, f"{n}__decision"] + ([f"{n}__numeric"] if FIELD_BY_NAME[n].kind == "amount" else [])
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in records:
        row = {k: ("" if r.get(k) is None else r[k]) for k in base}
        for n in FIELD_NAMES:
            f = r["fields"].get(n)
            row[n] = "" if not f or f["value"] is None else f["value"]; row[f"{n}__decision"] = "" if not f else f["decision"]
            if FIELD_BY_NAME[n].kind == "amount": row[f"{n}__numeric"] = "" if not f or f["value"] is None else (norm_amount(f["value"]) or "")
        w.writerow(row)
    return buf.getvalue()
