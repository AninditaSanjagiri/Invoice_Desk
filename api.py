"""FastAPI backend for the React invoice application."""
from __future__ import annotations
import csv, io, os
from pathlib import Path
from typing import Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from src.config import get_settings
from src.database import repository as repo
from src.database.models import Document, FieldRecord, Folder
from src.database.session import init_db, session_scope
from src.services.pipeline import InvoicePipeline
from src.services import export as export_svc

settings = get_settings()
# Keep the baseline pipeline inexpensive and predictable; validation remains enabled.
settings.max_retries = 0
settings.sc_samples = 0
SF = init_db(settings.db_path)
PIPELINE = InvoicePipeline(settings, SF)
app = FastAPI(title="Invoice Extraction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class FolderIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    parent_id: int | None = None
class MoveIn(BaseModel): folder_id: int | None = None
class ReviewIn(BaseModel):
    action: str
    edits: dict[str, str | None] = Field(default_factory=dict)


def doc_json(d: Document, s) -> dict[str, Any]:
    fields = {}
    for r in repo.current_fields(s, d.id):
        fields[r.name] = {"value": r.final_value, "extracted_value": r.value, "edited": r.was_edited,
                          "source_lines": r.source_lines, "decision": r.decision, "issues": r.issues,
                          "reasons": r.reasons, "ocr": r.ocr_signal, "llm": r.llm_signal,
                          "validation": r.validation}
    return {"id": d.id, "filename": d.original_name, "folder_id": d.folder_id, "folder": repo.folder_path(s, d.folder_id),
            "processing_status": d.processing_status, "decision": d.decision, "decision_reasons": d.decision_reasons,
            "review_status": d.review_status, "retry_count": d.retry_count, "processing_seconds": d.processing_seconds,
            "pages": d.n_pages, "ocr_confidence": d.ocr_mean_conf, "error": d.error,
            "created_at": d.created_at.isoformat() if d.created_at else None, "fields": fields}

@app.get("/api/health")
def health(): return {"status": "ok", "api_key_configured": settings.has_api_key, "provider": settings.llm_provider, "model": settings.model}

@app.get("/api/dashboard")
def dashboard():
    with session_scope(SF) as s:
        stats = repo.dashboard_stats(s); recent = repo.recent_documents(s, 8)
        return {"stats": {"total": stats["total"], "processed": stats["completed"], "pending_review": sum(1 for d in s.scalars(select(Document)) if d.review_status == "unreviewed" and d.processing_status == "completed"), "approved": stats["approved"], "rejected": sum(1 for d in s.scalars(select(Document)) if d.review_status == "rejected"), "failed": stats["failed"]}, "recent": [doc_json(d, s) for d in recent]}

@app.get("/api/folders")
def folders():
    with session_scope(SF) as s:
        return [{**f, "document_count": s.scalar(select(func.count(Document.id)).where(Document.folder_id == f["id"])) or 0} for f in repo.list_folders(s)]

@app.post("/api/folders")
def create_folder(body: FolderIn):
    try:
        with session_scope(SF) as s:
            f = repo.create_folder(s, body.name, body.parent_id)
            return {"id": f.id, "name": f.name, "parent_id": f.parent_id, "path": repo.folder_path(s, f.id)}
    except repo.RepoError as e: raise HTTPException(400, str(e))

@app.patch("/api/folders/{folder_id}")
def rename_folder(folder_id: int, body: FolderIn):
    try:
        with session_scope(SF) as s:
            f = repo.rename_folder(s, folder_id, body.name)
            return {"id": f.id, "name": f.name, "parent_id": f.parent_id, "path": repo.folder_path(s, f.id)}
    except repo.RepoError as e: raise HTTPException(400, str(e))

@app.delete("/api/folders/{folder_id}")
def delete_folder(folder_id: int):
    try:
        with session_scope(SF) as s: repo.delete_folder(s, folder_id)
        return {"ok": True}
    except repo.RepoError as e: raise HTTPException(400, str(e))

@app.get("/api/documents")
def documents(q: str = "", folder_id: int | None = None, review_status: str | None = None):
    with session_scope(SF) as s:
        ds = repo.search_documents(s, query=q, folder_id=folder_id, review_status=review_status)
        return [doc_json(d, s) for d in ds]

@app.get("/api/documents/{doc_id}")
def document_detail(doc_id: int):
    with session_scope(SF) as s:
        d = s.get(Document, doc_id)
        if d is None: raise HTTPException(404, "Document not found")
        return doc_json(d, s)

@app.get("/api/documents/{doc_id}/file")
def document_file(doc_id: int):
    with session_scope(SF) as s:
        d = s.get(Document, doc_id)
        if d is None or not Path(d.stored_path).is_file(): raise HTTPException(404, "Document file not found")
        path, mime, name = d.stored_path, d.mime or "application/octet-stream", d.original_name
    return FileResponse(path, media_type=mime, filename=name, content_disposition_type="inline")

@app.post("/api/documents")
def upload_document(file: UploadFile = File(...), folder_id: int | None = Query(default=None)):
    data = file.file.read()
    result = PIPELINE.process_upload(file.filename or "invoice.pdf", data, folder_id=folder_id)
    if result.doc_id is None: raise HTTPException(400, result.message or "Upload failed")
    with session_scope(SF) as s:
        d = s.get(Document, result.doc_id)
        if d is None: raise HTTPException(500, "Uploaded document could not be loaded")
        return {**doc_json(d, s), "duplicate": result.duplicate, "message": result.message}

@app.post("/api/documents/{doc_id}/retry")
def retry_document(doc_id: int):
    result = PIPELINE.reprocess(doc_id)
    if result.doc_id is None: raise HTTPException(404, result.message)
    with session_scope(SF) as s:
        d = s.get(Document, result.doc_id)
        return {**doc_json(d, s), "message": result.message}

@app.post("/api/documents/{doc_id}/review")
def review_document(doc_id: int, body: ReviewIn):
    action = body.action.lower().strip()
    if action not in {"approve", "reject", "retry"}: raise HTTPException(400, "Action must be approve, reject, or retry")
    if action == "retry": return retry_document(doc_id)
    try:
        with session_scope(SF) as s:
            d = s.get(Document, doc_id)
            if d is None: raise HTTPException(404, "Document not found")
            if body.edits: repo.update_field_values(s, doc_id, body.edits, note="Edited in React review screen")
            repo.set_review_status(s, doc_id, "approved" if action == "approve" else "rejected")
            return doc_json(d, s)
    except repo.RepoError as e: raise HTTPException(400, str(e))

@app.patch("/api/documents/{doc_id}/folder")
def move_document(doc_id: int, body: MoveIn):
    try:
        with session_scope(SF) as s:
            repo.move_document(s, doc_id, body.folder_id)
            return doc_json(s.get(Document, doc_id), s)
    except repo.RepoError as e: raise HTTPException(400, str(e))

@app.get("/api/folders/{folder_id}/export.csv")
def export_folder(folder_id: int):
    with session_scope(SF) as s:
        ids = [d.id for d in repo.search_documents(s, folder_id=folder_id, include_subfolders=False) if d.review_status == "approved"]
        records = export_svc.export_records(s, ids)
        if not records: raise HTTPException(404, "No approved invoices in this folder to export")
        content = export_svc.to_csv(records)
    return StreamingResponse(iter([content]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="folder_{folder_id}_approved_invoices.csv"'})

@app.get("/api/folders/{folder_id}/export-sheets-status")
def sheets_status(folder_id: int):
    # Exporting to Sheets requires user-provided Google service-account credentials; never pretend it is configured otherwise.
    return {"configured": bool(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")) and bool(os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")), "csv_url": f"/api/folders/{folder_id}/export.csv"}

@app.post("/api/folders/{folder_id}/export-sheets")
def export_folder_sheets(folder_id: int):
    """Append approved folder rows to a configured Google Sheet. Credentials stay server-side."""
    import json
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "").strip()
    if not raw or not spreadsheet_id:
        raise HTTPException(400, "Google Sheets is not configured. Set GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_SHEETS_SPREADSHEET_ID in the backend .env file, or use CSV export.")
    try:
        service_info = json.loads(raw) if raw.startswith("{") else json.loads(Path(raw).read_text(encoding="utf-8"))
        creds = Credentials.from_service_account_info(service_info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
        with session_scope(SF) as s:
            folder = s.get(Folder, folder_id)
            if folder is None: raise HTTPException(404, "Folder not found")
            ids = [d.id for d in repo.search_documents(s, folder_id=folder_id, include_subfolders=False) if d.review_status == "approved"]
            records = export_svc.export_records(s, ids)
        if not records: raise HTTPException(404, "No approved invoices in this folder to export")
        from src.extraction.schema import FIELD_NAMES
        headers = ["document_id", "filename", "folder", "review_status", "created_at"] + FIELD_NAMES
        values_by_id = {}
        for rec in records:
            row = [rec.get("document_id"), rec.get("filename"), rec.get("folder"), rec.get("review_status"), rec.get("created_at")]
            row.extend((rec.get("fields", {}).get(k) or {}).get("value") or "" for k in FIELD_NAMES)
            values_by_id[str(rec.get("document_id"))] = row
        existing = sheets.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range="A:ZZ").execute().get("values", [])
        if not existing:
            sheets.spreadsheets().values().update(spreadsheetId=spreadsheet_id, range="A1", valueInputOption="RAW", body={"values": [headers]}).execute()
            existing = [headers]
        existing_headers = existing[0]
        if existing_headers != headers:
            raise ValueError("The destination sheet headers do not match this application's export schema. Use a new blank sheet or restore the exported header row.")
        id_col = existing_headers.index("document_id")
        id_to_row = {str(row[id_col]): idx + 1 for idx, row in enumerate(existing[1:], start=1) if len(row) > id_col and str(row[id_col]).strip()}
        updates, appends = [], []
        for doc_key, row in values_by_id.items():
            if doc_key in id_to_row:
                updates.append({"range": f"A{id_to_row[doc_key]}", "values": [row]})
            else:
                appends.append(row)
        if updates:
            sheets.spreadsheets().values().batchUpdate(spreadsheetId=spreadsheet_id, body={"valueInputOption": "RAW", "data": updates}).execute()
        if appends:
            sheets.spreadsheets().values().append(spreadsheetId=spreadsheet_id, range="A1", valueInputOption="RAW", insertDataOption="INSERT_ROWS", body={"values": appends}).execute()
        return {"ok": True, "rows": len(records), "inserted": len(appends), "updated": len(updates), "spreadsheet_id": spreadsheet_id}
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(502, f"Google Sheets export failed. Check credentials, spreadsheet access, and API setup. Details: {type(e).__name__}")
