import csv, io, json
import pytest
from conftest import FakeOCREngine, good_ocr, good_llm_fields, png_bytes, FakeClient
from src.database import repository as repo
from src.database.models import Document, ExtractionRun, FieldRecord
from src.database.session import init_db, session_scope
from src.extraction.llm_client import GroqLLM
from src.ocr.engine import OCRError, cached_ocr
from src.services import export as ex
from src.services.pipeline import InvoicePipeline
from src.utils.files import safe_filename, validate_upload, UploadError

@pytest.fixture
def sf(): return init_db(":memory:")

@pytest.fixture
def make_pipeline(settings, sf):
    def make(script, ocr=None, ocr_error=None):
        c = FakeClient(script); p = InvoicePipeline(settings, sf, FakeOCREngine(ocr, ocr_error), GroqLLM(settings, client=c, sleep=lambda x: None)); return p, c
    return make

# ---------------- folders ----------------
def test_folder_rules(sf):
    with session_scope(sf) as s:
        a = repo.create_folder(s, "Acme"); b = repo.create_folder(s, "2024", a.id); c = repo.create_folder(s, "Q1", b.id)
        assert repo.folder_path(s, c.id) == "/Acme/2024/Q1" and [f["depth"] for f in repo.list_folders(s)] == [0, 1, 2]
        with pytest.raises(repo.RepoError): repo.create_folder(s, "2024", a.id)                  # duplicate sibling
        with pytest.raises(repo.RepoError): repo.create_folder(s, "bad/name")
        with pytest.raises(repo.RepoError): repo.move_folder(s, a.id, a.id)                      # inside itself
        with pytest.raises(repo.RepoError): repo.move_folder(s, a.id, c.id)                      # inside its own descendant
        with pytest.raises(repo.RepoError): repo.delete_folder(s, a.id)                          # non-empty (has subfolders)
        assert repo.rename_folder(s, b.id, "2025").name == "2025"
        with pytest.raises(repo.RepoError): repo.rename_folder(s, b.id, "  ")
        repo.move_folder(s, c.id, None); assert repo.folder_path(s, c.id) == "/Q1"
        repo.delete_folder(s, c.id); assert all(f["id"] != c.id for f in repo.list_folders(s))

# ---------------- upload validation ----------------
def test_upload_validation_and_safe_names():
    with pytest.raises(UploadError): validate_upload("x.exe", b"MZ", 5)
    with pytest.raises(UploadError): validate_upload("x.pdf", b"", 5)
    with pytest.raises(UploadError): validate_upload("x.pdf", b"not a pdf", 5)
    with pytest.raises(UploadError): validate_upload("x.png", b"%PDF-1.4" + b"0" * 10, 5)
    with pytest.raises(UploadError): validate_upload("x.pdf", b"%PDF" + b"0" * (6 * 1024 * 1024), 5)
    assert validate_upload("a.PNG", png_bytes(), 5) == ".png" and safe_filename("../../etc/pass wd?.PDF") == "pass_wd.pdf"

# ---------------- end-to-end (fake OCR + fake LLM; LOCAL INTEGRATION ONLY) ----------------
def test_full_workflow_persists_everything(make_pipeline, sf, settings):
    p, c = make_pipeline([good_llm_fields()]); msgs = []
    r = p.process_upload("inv1.png", png_bytes(), progress=msgs.append)
    assert r.status == "completed" and not r.duplicate and any("OCR" in m for m in msgs) and any("validation" in m.lower() for m in msgs)
    with session_scope(sf) as s:
        d = s.get(Document, r.doc_id); assert d.decision == "ACCEPT" and d.processing_status == "completed" and d.ocr_source == "test" and (settings.upload_dir / f"{d.file_hash[:32]}.png").exists()
        f = {x.name: x for x in repo.current_fields(s, d.id)}; assert len(f) == 18 and f["document_id"].value == "INV-2041" and f["document_id"].decision == "ACCEPT" and f["vendor_email"].decision == "ABSENT"
        assert f["document_id"].ocr_signal["status"] == "HIGH" and f["document_id"].llm_signal["status"] == "SUPPORTED" and f["amount_total_gross"].norm_value == "1100.00"
        run = s.query(ExtractionRun).filter_by(document_id=d.id).one(); assert run.selected and run.model == settings.model and run.prompt_hash and run.config["model"] == settings.model and "test-key" not in json.dumps(run.config)
        assert any(v["rule"] == "net_plus_tax_eq_gross" and v["status"] == "PASS" for v in run.validation)

def test_duplicate_upload_is_detected_and_not_reprocessed(make_pipeline, sf):
    p, c = make_pipeline([good_llm_fields()]); a = p.process_upload("a.png", png_bytes(), None); b = p.process_upload("copy_of_a.png", png_bytes())
    assert b.duplicate and b.doc_id == a.doc_id and len(c.calls) == 1 and p.ocr_engine.calls == 1
    with session_scope(sf) as s: assert s.query(Document).count() == 1

def test_ocr_cache_is_by_hash_and_llm_is_not_reused_across_documents(make_pipeline, sf):
    p, c = make_pipeline([good_llm_fields()]); p.process_upload("a.png", png_bytes((1, 2, 3))); p.process_upload("b.png", png_bytes((9, 9, 9)))
    assert p.ocr_engine.calls == 2 and len(c.calls) == 2                                      # different documents -> own OCR and own LLM call
    r = p.reprocess(1); assert r.status == "completed" and p.ocr_engine.calls == 2 and len(c.calls) == 3       # same document: OCR cached, LLM called again

def test_invalid_upload_is_rejected_without_side_effects(make_pipeline, sf):
    p, c = make_pipeline([good_llm_fields()]); r = p.process_upload("evil.exe", b"MZ....")
    assert r.status == "rejected" and "Unsupported" in r.message and len(c.calls) == 0
    with session_scope(sf) as s: assert s.query(Document).count() == 0

def test_ocr_failure_is_a_processing_failure_not_fake_data(make_pipeline, sf):
    p, c = make_pipeline([good_llm_fields()], ocr_error=OCRError("no text")); r = p.process_upload("a.png", png_bytes())
    assert r.status == "failed" and len(c.calls) == 0
    with session_scope(sf) as s:
        d = s.get(Document, r.doc_id); assert d.processing_status == "failed" and d.decision == "REVIEW" and "no text" in d.error and repo.current_fields(s, d.id) == []
        with pytest.raises(repo.RepoError): repo.set_review_status(s, d.id, "approved")      # cannot approve a failed extraction

def test_llm_api_failure_is_recorded_distinct_from_uncertain_extraction(make_pipeline, sf):
    p, c = make_pipeline([TimeoutError("down")]); r = p.process_upload("a.png", png_bytes())
    assert r.status == "failed"
    with session_scope(sf) as s:
        d = s.get(Document, r.doc_id); run = s.query(ExtractionRun).filter_by(document_id=d.id).one()
        assert d.processing_status == "failed" and run.status == "failed" and "TimeoutError" in run.error and repo.current_fields(s, d.id) == []

def test_simple_pipeline_flags_bad_total_for_review_without_automatic_retry(make_pipeline, sf):
    ocr = good_ocr(); bad = good_llm_fields(ocr); bad["amount_total_gross"]["value"] = "$1,500.00"
    p, c = make_pipeline([bad, good_llm_fields(ocr)], ocr=ocr); r = p.process_upload("a.png", png_bytes())
    with session_scope(sf) as s:
        d = s.get(Document, r.doc_id); runs = s.query(ExtractionRun).filter_by(document_id=d.id).order_by(ExtractionRun.attempt_no).all()
        assert d.retry_count == 0 and d.decision == "REVIEW" and [x.kind for x in runs] == ["initial"]
        assert runs[0].selected and runs[0].doc_decision == "REVIEW"
        assert {x.run_id for x in repo.current_fields(s, d.id)} == {runs[0].id}

# ---------------- edits, review, search, stats ----------------
def _seed(make_pipeline, sf, n=2):
    p, c = make_pipeline([good_llm_fields()]); ids = []
    for i in range(n): ids.append(p.process_upload(f"i{i}.png", png_bytes((i, i, i))).doc_id)
    return p, ids

def test_edit_review_search_and_stats(make_pipeline, sf):
    p, ids = _seed(make_pipeline, sf)
    with session_scope(sf) as s:
        ch = repo.update_field_values(s, ids[0], {"vendor_name": "Acme Tools Limited", "document_id": "INV-2041"}, "typo"); assert list(ch) == ["vendor_name"]
        assert s.get(Document, ids[0]).review_status == "corrected"
        repo.set_review_status(s, ids[1], "approved", "ok")
        assert [d.id for d in repo.search_documents(s, "limited")] == [ids[0]]                       # finds the edited value
        assert len(repo.search_documents(s, "acme tools")) == 2 and len(repo.search_documents(s, "acme tools ltd")) == 1 and len(repo.search_documents(s, "inv-2041")) == 2
        assert len(repo.search_documents(s, "1100")) == 2 and len(repo.search_documents(s, "$1,100.00")) == 2 and repo.search_documents(s, "nonexistent") == []
        assert len(repo.search_documents(s, "i1.png")) == 1
        st = repo.dashboard_stats(s)
        assert st["total"] == 2 and st["accepted"] == 2 and st["review"] == 0 and st["approved"] == 1 and st["corrected"] == 1 and st["extraction_success_rate"] == 1.0
        assert st["rule_stats"]["net_plus_tax_eq_gross"] == dict(applicable=2, failed=0) and st["llm_runs"] == 2
        assert [e.action for e in s.get(Document, ids[0]).reviews] == ["edited"]

def test_move_documents_and_filter_by_folder(make_pipeline, sf):
    p, ids = _seed(make_pipeline, sf)
    with session_scope(sf) as s:
        a = repo.create_folder(s, "Acme"); sub = repo.create_folder(s, "2024", a.id); repo.move_document(s, ids[0], sub.id)
        assert [d.id for d in repo.search_documents(s, folder_id=a.id)] == [ids[0]] and repo.search_documents(s, folder_id=a.id, include_subfolders=False) == []
        with pytest.raises(repo.RepoError): repo.delete_folder(s, sub.id)                            # contains a document
        with pytest.raises(repo.RepoError): repo.move_document(s, ids[0], 999)
        repo.move_document(s, ids[0], None); repo.delete_folder(s, sub.id)

def test_export_json_and_csv_use_persisted_records(make_pipeline, sf):
    p, ids = _seed(make_pipeline, sf, 1)
    with session_scope(sf) as s:
        repo.update_field_values(s, ids[0], {"vendor_name": "Edited Vendor", "amount_total_net": None}); recs = ex.export_records(s)
    js = json.loads(ex.to_json(recs)); f = js[0]["fields"]
    assert f["vendor_name"]["value"] == "Edited Vendor" and f["vendor_name"]["extracted_value"] == "Acme Tools Ltd" and f["vendor_name"]["edited"] and f["amount_total_net"]["value"] is None and f["vendor_email"]["value"] is None
    assert js[0]["system_decision"] == "ACCEPT" and js[0]["review_status"] == "corrected" and f["document_id"]["ocr_status"] == "HIGH"
    rows = list(csv.DictReader(io.StringIO(ex.to_csv(recs)))); r = rows[0]
    assert r["vendor_name"] == "Edited Vendor" and r["vendor_email"] == "" and r["amount_total_net"] == "" and r["amount_total_net__numeric"] == "" and r["amount_total_gross"] == "$1,100.00" and r["amount_total_gross__numeric"] == "1100.00"
    assert r["system_decision"] == "ACCEPT" and r["document_id__decision"] == "ACCEPT" and r["processing_status"] == "completed"

def test_database_initialisation_is_idempotent(tmp_path):
    f1 = init_db(tmp_path / "x.db")
    with session_scope(f1) as s: repo.create_folder(s, "Keep")
    f2 = init_db(tmp_path / "x.db")
    with session_scope(f2) as s: assert [f["name"] for f in repo.list_folders(s)] == ["Keep"]

def test_cached_ocr_roundtrip(tmp_path):
    eng = FakeOCREngine(); f = tmp_path / "a.png"; f.write_bytes(png_bytes())
    d1, hit1 = cached_ocr(eng, f, "h" * 64, tmp_path / "cache", 5); d2, hit2 = cached_ocr(eng, f, "h" * 64, tmp_path / "cache", 5)
    assert (hit1, hit2) == (False, True) and eng.calls == 1 and [l.text for l in d1.lines] == [l.text for l in d2.lines] and d2.words[0].conf == d1.words[0].conf
