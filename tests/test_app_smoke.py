"""Launches the real Streamlit script headlessly (streamlit.testing AppTest) against a temporary database seeded via the pipeline
with fake OCR/LLM objects. This verifies the entry point, imports and that every page renders; it does NOT exercise real OCR or Groq."""
from pathlib import Path
import pytest
from conftest import FakeClient, FakeOCREngine, good_llm_fields, good_ocr, png_bytes
from src.extraction.llm_client import GroqLLM

APP = str(Path(__file__).resolve().parent.parent / "app.py")

@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "ad")); monkeypatch.setenv("GROQ_API_KEY", "")
    import streamlit as st; st.cache_resource.clear()
    from src.config import get_settings
    from src.database import repository as repo
    from src.database.session import init_db, session_scope
    from src.services.pipeline import InvoicePipeline
    s = get_settings(); sf = init_db(s.db_path); ocr = good_ocr(); bad = good_llm_fields(ocr); bad["amount_total_gross"]["value"] = "$9,999.00"
    c = FakeClient([good_llm_fields(ocr), bad, bad]); p = InvoicePipeline(s, sf, FakeOCREngine(ocr), GroqLLM(s, client=c, sleep=lambda x: None))
    a = p.process_upload("good.png", png_bytes((1, 1, 1))); b = p.process_upload("bad.png", png_bytes((2, 2, 2)))
    with session_scope(sf) as ses: fid = repo.create_folder(ses, "Acme").id
    return dict(a=a.doc_id, b=b.doc_id, sf=sf, folder=fid)

def test_all_pages_render_with_real_persisted_data(seeded):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60).run(); assert not at.exception, at.exception
    m = {x.label: x.value for x in at.metric}; assert m["Documents"] == "2" and m["Accepted"] == "1" and m["Need review"] == "1"     # metrics come from stored runs
    for page in ["Upload", "Documents", "Document detail", "Folders", "Export", "About", "Dashboard"]:
        at.sidebar.radio[0].set_value(page).run(); assert not at.exception, (page, at.exception)

def test_detail_page_edit_and_approve_flow(seeded):
    from streamlit.testing.v1 import AppTest
    from src.database import repository as repo
    from src.database.session import session_scope
    at = AppTest.from_file(APP, default_timeout=60).run(); at.session_state["doc_id"] = seeded["b"]; at.sidebar.radio[0].set_value("Document detail").run(); assert not at.exception, at.exception
    assert len(at.tabs) == 6 and any("REVIEW" in str(md.value) for md in at.markdown)
    approve = next(b for b in at.button if "Approve" in b.label); approve.click().run(); assert not at.exception
    with session_scope(seeded["sf"]) as s:
        assert repo.search_documents(s, review_status="approved")[0].id == seeded["b"]

def test_export_page_offers_downloads_and_folders_page_actions(seeded):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60).run(); at.sidebar.radio[0].set_value("Folders").run(); assert not at.exception
    at.text_input(key="new_name").set_value("2024"); next(b for b in at.button if b.label == "Create folder").click().run(); assert not at.exception
    from src.database import repository as repo
    from src.database.session import session_scope
    with session_scope(seeded["sf"]) as s: assert {f["path"] for f in repo.list_folders(s)} == {"/Acme", "/2024"}
    at.sidebar.radio[0].set_value("Export").run(); assert not at.exception and len(at.dataframe) >= 1
