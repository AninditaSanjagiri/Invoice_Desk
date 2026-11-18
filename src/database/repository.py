"""All database operations (folders, documents, fields, review, search, statistics)."""
from __future__ import annotations
from sqlalchemy import case, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from ..extraction.schema import FIELD_BY_NAME
from ..utils.normalize import norm_amount, norm_text
from .models import Document, ExtractionRun, FieldRecord, Folder, ReviewEvent, utcnow

class RepoError(ValueError): pass

# ---------------- folders ----------------
def _clean_name(name: str) -> str:
    n = (name or "").strip()
    if not n or len(n) > 120 or "/" in n or "\\" in n: raise RepoError("Folder names must be 1-120 characters and must not contain slashes.")
    return n

def create_folder(s: Session, name: str, parent_id: int | None = None) -> Folder:
    name = _clean_name(name)
    if parent_id is not None and s.get(Folder, parent_id) is None: raise RepoError("Parent folder does not exist.")
    if _dup(s, name, parent_id): raise RepoError("A folder with this name already exists here.")
    f = Folder(name=name, parent_id=parent_id)
    try:
        with s.begin_nested(): s.add(f)        # savepoint: a failure must not discard the caller's other pending work
    except IntegrityError as e: raise RepoError("A folder with this name already exists here.") from e
    return f

def _dup(s, name, parent_id, exclude_id=None):
    q = select(Folder).where(Folder.name == name, Folder.parent_id.is_(None) if parent_id is None else Folder.parent_id == parent_id)
    return [f for f in s.scalars(q) if f.id != exclude_id]

def rename_folder(s: Session, folder_id: int, new_name: str) -> Folder:
    f = s.get(Folder, folder_id)
    if f is None: raise RepoError("Folder not found.")
    new_name = _clean_name(new_name)
    if _dup(s, new_name, f.parent_id, f.id): raise RepoError("A folder with this name already exists here.")
    f.name = new_name; s.flush(); return f

def descendants(s: Session, folder_id: int) -> set[int]:
    out, frontier = set(), [folder_id]
    while frontier:
        kids = [k for k in s.scalars(select(Folder.id).where(Folder.parent_id.in_(frontier)))]
        frontier = [k for k in kids if k not in out]; out.update(frontier)
    return out

def move_folder(s: Session, folder_id: int, new_parent_id: int | None) -> Folder:
    f = s.get(Folder, folder_id)
    if f is None: raise RepoError("Folder not found.")
    if new_parent_id is not None:
        if s.get(Folder, new_parent_id) is None: raise RepoError("Target folder does not exist.")
        if new_parent_id == folder_id or new_parent_id in descendants(s, folder_id): raise RepoError("A folder cannot be moved inside itself or one of its subfolders.")
    if _dup(s, f.name, new_parent_id, f.id): raise RepoError("A folder with this name already exists in the target.")
    f.parent_id = new_parent_id; s.flush(); return f

def delete_folder(s: Session, folder_id: int) -> None:
    """Only empty folders can be deleted (no documents, no subfolders)."""
    f = s.get(Folder, folder_id)
    if f is None: raise RepoError("Folder not found.")
    if s.scalar(select(func.count(Folder.id)).where(Folder.parent_id == folder_id)): raise RepoError("Folder contains subfolders; delete or move them first.")
    if s.scalar(select(func.count(Document.id)).where(Document.folder_id == folder_id)): raise RepoError("Folder contains documents; move them first.")
    s.delete(f); s.flush()

def folder_path(s: Session, folder_id: int | None) -> str:
    parts, seen = [], set()
    while folder_id is not None and folder_id not in seen:
        seen.add(folder_id); f = s.get(Folder, folder_id)
        if f is None: break
        parts.append(f.name); folder_id = f.parent_id
    return "/" + "/".join(reversed(parts)) if parts else "/"

def list_folders(s: Session) -> list[dict]:
    """Flat list ordered as a tree: [{id, name, parent_id, path, depth}]."""
    fs = list(s.scalars(select(Folder).order_by(Folder.name))); kids: dict = {}
    for f in fs: kids.setdefault(f.parent_id, []).append(f)
    out = []
    def walk(pid, depth, prefix):
        for f in kids.get(pid, []):
            path = f"{prefix}/{f.name}"; out.append(dict(id=f.id, name=f.name, parent_id=f.parent_id, path=path, depth=depth)); walk(f.id, depth + 1, path)
    walk(None, 0, ""); return out

def move_document(s: Session, doc_id: int, folder_id: int | None) -> Document:
    d = s.get(Document, doc_id)
    if d is None: raise RepoError("Document not found.")
    if folder_id is not None and s.get(Folder, folder_id) is None: raise RepoError("Target folder does not exist.")
    old = d.folder_id; d.folder_id = folder_id
    s.add(ReviewEvent(document_id=d.id, action="moved", changes=dict(from_folder=old, to_folder=folder_id))); s.flush(); return d

# ---------------- documents / fields ----------------
def get_document_by_hash(s: Session, h: str) -> Document | None: return s.scalar(select(Document).where(Document.file_hash == h))

def current_fields(s: Session, doc_id: int) -> list[FieldRecord]:
    order = {n: i for i, n in enumerate(FIELD_BY_NAME)}
    rows = list(s.scalars(select(FieldRecord).where(FieldRecord.document_id == doc_id, FieldRecord.is_current.is_(True))))
    return sorted(rows, key=lambda r: order.get(r.name, 99))

def norm_for_search(name: str, value: str | None) -> str | None:
    if value is None: return None
    spec = FIELD_BY_NAME.get(name)
    return (norm_amount(value) if spec and spec.kind == "amount" else None) or norm_text(value)

def update_field_values(s: Session, doc_id: int, edits: dict[str, str | None], note: str | None = None) -> dict:
    """Human edits (reviewed_value); returns {field: (old, new)} for fields that actually changed."""
    changes = {}
    for r in current_fields(s, doc_id):
        if r.name not in edits: continue
        new = (edits[r.name] or "").strip() or None; old = r.final_value
        if new == old: continue
        r.reviewed_value, r.was_edited = new, True; r.norm_value = norm_for_search(r.name, new); changes[r.name] = (old, new)
    if changes:
        d = s.get(Document, doc_id); d.review_status = "corrected"
        s.add(ReviewEvent(document_id=doc_id, action="edited", note=note, changes={k: list(v) for k, v in changes.items()}))
    s.flush(); return changes

def set_review_status(s: Session, doc_id: int, status: str, note: str | None = None) -> Document:
    if status not in ("approved", "rejected", "unreviewed"): raise RepoError("Invalid review status.")
    d = s.get(Document, doc_id)
    if d is None: raise RepoError("Document not found.")
    if status == "approved" and d.processing_status != "completed": raise RepoError("A document whose processing failed cannot be approved; reprocess it or enter values manually first.")
    d.review_status = status; s.add(ReviewEvent(document_id=doc_id, action=status, note=note)); s.flush(); return d

def delete_document(s: Session, doc_id: int) -> None:
    d = s.get(Document, doc_id)
    if d is not None: s.delete(d); s.flush()

# ---------------- search ----------------
def search_documents(s: Session, query: str = "", folder_id: int | None = None, decision: str | None = None, review_status: str | None = None,
                     include_subfolders: bool = True, limit: int = 500) -> list[Document]:
    q = select(Document).order_by(Document.created_at.desc())
    if folder_id is not None:
        ids = {folder_id} | (descendants(s, folder_id) if include_subfolders else set()); q = q.where(Document.folder_id.in_(ids))
    if decision: q = q.where(Document.decision == decision)
    if review_status: q = q.where(Document.review_status == review_status)
    query = (query or "").strip()
    if query:
        like = f"%{query.lower()}%"; nq = norm_text(query); amt = norm_amount(query)
        final = case((FieldRecord.was_edited.is_(True), FieldRecord.reviewed_value), else_=FieldRecord.value)      # search the FINAL (human-corrected) value
        conds = [func.lower(final).like(like)]
        if nq: conds.append(FieldRecord.norm_value.like(f"%{nq}%"))
        if amt: conds.append(FieldRecord.norm_value.like(f"%{amt}%"))
        sub = select(FieldRecord.document_id).where(FieldRecord.is_current.is_(True), or_(*conds))
        q = q.where(or_(Document.id.in_(sub), func.lower(Document.original_name).like(like)))
    return list(s.scalars(q.limit(limit)))

# ---------------- statistics (computed from stored runs only) ----------------
def dashboard_stats(s: Session) -> dict:
    count = lambda *w: s.scalar(select(func.count(Document.id)).where(*w)) or 0
    total = count(); completed = count(Document.processing_status == "completed")
    cur = select(FieldRecord).where(FieldRecord.is_current.is_(True))
    n_fields = s.scalar(select(func.count(FieldRecord.id)).where(FieldRecord.is_current.is_(True), FieldRecord.decision != "ABSENT")) or 0
    by_dec = dict(s.execute(select(FieldRecord.decision, func.count(FieldRecord.id)).where(FieldRecord.is_current.is_(True)).group_by(FieldRecord.decision)).all())
    runs = s.scalar(select(func.count(ExtractionRun.id))) or 0; ok_runs = s.scalar(select(func.count(ExtractionRun.id)).where(ExtractionRun.status == "ok")) or 0
    outcomes = dict(s.execute(select(ExtractionRun.outcome, func.count(ExtractionRun.id)).where(ExtractionRun.kind == "retry").group_by(ExtractionRun.outcome)).all())
    rule_fail = {}
    for (v,) in s.execute(select(ExtractionRun.validation).where(ExtractionRun.selected.is_(True))).all():
        for r in v or []:
            if r["status"] != "NOT_APPLICABLE":
                d = rule_fail.setdefault(r["rule"], dict(applicable=0, failed=0)); d["applicable"] += 1; d["failed"] += r["status"] == "FAIL"
    return dict(total=total, completed=completed, failed=count(Document.processing_status == "failed"), accepted=count(Document.decision == "ACCEPT"),
                review=count(Document.decision == "REVIEW"), retried=count(Document.retry_count > 0), approved=count(Document.review_status == "approved"),
                corrected=count(Document.review_status == "corrected"), extraction_success_rate=(completed / total if total else None),
                avg_seconds=s.scalar(select(func.avg(Document.processing_seconds)).where(Document.processing_status == "completed")),
                field_decisions=by_dec, fields_extracted=n_fields, llm_runs=runs, llm_runs_ok=ok_runs, retry_outcomes=outcomes, rule_stats=rule_fail)

def recent_documents(s: Session, n: int = 10) -> list[Document]: return list(s.scalars(select(Document).order_by(Document.created_at.desc()).limit(n)))
