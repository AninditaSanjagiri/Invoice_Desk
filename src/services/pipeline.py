"""Application service: upload -> validate -> hash/dedupe -> OCR (cached) -> LLM extraction -> signals -> routing -> persistence."""
from __future__ import annotations
import logging, time
from dataclasses import dataclass
from pathlib import Path
from sqlalchemy import delete
from ..config import Settings
from ..database import repository as repo
from ..database.models import Document, ExtractionRun, FieldRecord, ReviewEvent
from ..database.session import session_scope
from ..extraction.extractor import Extractor
from ..extraction.llm_client import create_llm
from ..extraction.schema import FIELD_NAMES
from ..ocr.engine import DoctrEngine, OCRError, cached_ocr
from ..utils.files import ALLOWED_EXT, UploadError, safe_filename, store_upload, validate_upload
from ..utils.hashing import sha256_bytes
from .core import CoreResult, run_core, run_simple_core

log = logging.getLogger(__name__)

@dataclass
class ProcessResult:
    doc_id: int | None; status: str; duplicate: bool = False; message: str = ""

class InvoicePipeline:
    def __init__(self, settings: Settings, session_factory, ocr_engine=None, llm=None):
        self.s, self.sf = settings, session_factory
        self.ocr_engine = ocr_engine or DoctrEngine(); self.extractor = Extractor(settings, llm or create_llm(settings))   # no LLM cache in the app

    # ---- public API ----
    def process_upload(self, filename: str, data: bytes, folder_id: int | None = None, progress=None) -> ProcessResult:
        say = progress or (lambda m: None)
        try: ext = validate_upload(filename, data, self.s.max_file_mb)
        except UploadError as e: return ProcessResult(None, "rejected", message=str(e))
        h = sha256_bytes(data)
        with session_scope(self.sf) as ses:
            existing = repo.get_document_by_hash(ses, h)
            if existing is not None: return ProcessResult(existing.id, existing.processing_status, True, f"Duplicate of '{existing.original_name}' (already processed).")
            path = store_upload(data, safe_filename(filename), self.s.upload_dir, h)
            doc = Document(file_hash=h, original_name=safe_filename(filename), stored_path=str(path), mime=ALLOWED_EXT[ext], size_bytes=len(data), folder_id=folder_id, processing_status="processing")
            ses.add(doc); ses.flush(); doc_id = doc.id
        return self._process(doc_id, say)

    def reprocess(self, doc_id: int, progress=None) -> ProcessResult:
        with session_scope(self.sf) as ses:
            d = ses.get(Document, doc_id)
            if d is None: return ProcessResult(None, "rejected", message="Document not found.")
            ses.execute(delete(FieldRecord).where(FieldRecord.document_id == doc_id)); ses.execute(delete(ExtractionRun).where(ExtractionRun.document_id == doc_id))
            d.processing_status, d.error, d.decision, d.review_status, d.retry_count = "processing", None, None, "unreviewed", 0
            ses.add(ReviewEvent(document_id=doc_id, action="reprocessed", note="Re-run of OCR/LLM pipeline (previous results and human edits were replaced)."))
        return self._process(doc_id, progress or (lambda m: None))

    # ---- internals ----
    def _process(self, doc_id: int, say) -> ProcessResult:
        t0 = time.time()
        with session_scope(self.sf) as ses: d = ses.get(Document, doc_id); path, h = Path(d.stored_path), d.file_hash
        try:
            say("Running OCR (DocTR)...")
            ocr, cached = cached_ocr(self.ocr_engine, path, h, self.s.ocr_cache_dir, self.s.max_pages)
        except OCRError as e: return self._fail(doc_id, f"OCR error: {e}", t0)
        except Exception as e: log.exception("unexpected OCR failure"); return self._fail(doc_id, f"Unexpected OCR failure: {type(e).__name__}", t0)
        if cached: say("OCR loaded from cache.")
        try: res = run_simple_core(ocr, self.extractor, self.s, None, say)
        except Exception as e: log.exception("unexpected pipeline failure"); return self._fail(doc_id, f"Unexpected processing failure: {type(e).__name__}: {str(e)[:150]}", t0, ocr=ocr)
        say("Saving results...")
        with session_scope(self.sf) as ses: self._persist(ses, doc_id, ocr, res, time.time() - t0)
        return ProcessResult(doc_id, "completed" if res.status == "completed" else "failed", False, res.error or "")

    def _fail(self, doc_id: int, msg: str, t0: float, ocr=None) -> ProcessResult:
        with session_scope(self.sf) as ses:
            d = ses.get(Document, doc_id); d.processing_status, d.error, d.decision, d.processing_seconds = "failed", msg, "REVIEW", time.time() - t0
            d.decision_reasons = [f"Processing failed: {msg}"]
            if ocr is not None: d.n_pages, d.ocr_source, d.ocr_mean_conf = ocr.n_pages, ocr.source, ocr.mean_conf
        return ProcessResult(doc_id, "failed", False, msg)

    def _persist(self, ses, doc_id: int, ocr, res: CoreResult, seconds: float):
        d = ses.get(Document, doc_id); cfg = self.s.llm_config_dict()
        d.n_pages, d.ocr_source, d.ocr_mean_conf, d.processing_seconds, d.error = ocr.n_pages, ocr.source, ocr.mean_conf, seconds, None
        if res.status == "failed":
            a = res.failed_attempt
            ses.add(ExtractionRun(document_id=doc_id, attempt_no=0, kind="initial", status=a.status, model=a.model, prompt_hash=a.prompt_hash, config=cfg, raw_output=a.raw, usage=a.usage, latency_s=a.latency_s, error=a.error, selected=False))
            d.processing_status, d.error, d.decision, d.decision_reasons = "failed", res.error, "REVIEW", ["Processing failed (no usable extraction): " + (res.error or "")]; return
        final = res.final; run_by_attempt = {}
        for i, ev in enumerate(res.history):
            a = ev.attempt; outcome = res.retry_outcomes[i - 1] if i > 0 and i - 1 < len(res.retry_outcomes) else None
            run = ExtractionRun(document_id=doc_id, attempt_no=i, kind=a.kind, status=a.status, model=a.model, prompt_hash=a.prompt_hash, config=cfg, raw_output=a.raw, usage=a.usage,
                                latency_s=a.latency_s, error=a.error, outcome=outcome, validation=[r.to_dict() for r in ev.rules], doc_decision=ev.routing.doc_decision if ev.routing else None,
                                selected=(ev.attempt is final.attempt))
            ses.add(run); ses.flush(); run_by_attempt[id(ev.attempt)] = run.id
        for sm in res.samples:
            ses.add(ExtractionRun(document_id=doc_id, attempt_no=100 + sm.sample_idx, kind="sample", status=sm.status, model=sm.model, prompt_hash=sm.prompt_hash, config=cfg, raw_output=sm.raw, usage=sm.usage, latency_s=sm.latency_s, error=sm.error))
        run_id = run_by_attempt[id(final.attempt)]
        for n in FIELD_NAMES:
            sg, dec = final.signals[n], final.routing.fields[n]
            ses.add(FieldRecord(document_id=doc_id, run_id=run_id, name=n, value=sg.value, norm_value=repo.norm_for_search(n, sg.value), source_lines=sg.source_lines,
                                ocr_signal=sg.ocr.to_dict() if sg.ocr else None, llm_signal=sg.llm.to_dict() if sg.llm else None, validation=sg.validation,
                                decision=dec.decision, issues=dec.issues, reasons=dec.reasons, is_current=True))
        d.processing_status = "completed"; d.retry_count = sum(1 for e in res.history if e.role == "retry")
        d.decision = "ACCEPT" if final.routing.doc_decision == "ACCEPT" else "REVIEW"; d.decision_reasons = final.routing.doc_reasons
