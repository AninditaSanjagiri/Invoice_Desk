"""AI-Powered Invoice Extraction with Multi-Signal Reliability Routing - Streamlit application.  Run:  streamlit run app.py"""
from __future__ import annotations
import json
import pandas as pd
import streamlit as st
from src.config import get_settings
from src.database import repository as repo
from src.database.models import Document
from src.database.session import init_db, session_scope
from src.extraction.schema import FIELD_BY_NAME, FIELD_NAMES
from src.ocr.models import OCRDocument
from src.services import export as export_svc
from src.services.pipeline import InvoicePipeline
from src.services.preview import highlight, render_pages

st.set_page_config(page_title="Invoice Extraction & Reliability Routing", page_icon="🧾", layout="wide")
BADGE = {"ACCEPT": "🟢 ACCEPT", "RETRY": "🟠 RETRY", "REVIEW": "🔴 REVIEW", "ABSENT": "⚪ absent"}
PAGES = ["Dashboard", "Upload", "Documents", "Document detail", "Folders", "Export", "About"]

@st.cache_resource
def services():
    s = get_settings(); sf = init_db(s.db_path); return s, sf, InvoicePipeline(s, sf)
settings, SF, pipeline = services()

def go(page: str, doc_id: int | None = None):
    st.session_state["page"] = page
    if doc_id is not None: st.session_state["doc_id"] = doc_id

def folder_options(s, with_root=True):
    opts = {"(no folder)": None} if with_root else {}
    for f in repo.list_folders(s): opts["\u00a0\u00a0" * f["depth"] + f["path"]] = f["id"]
    return opts

def doc_rows(docs, s):
    return pd.DataFrame([dict(ID=d.id, File=d.original_name, Folder=repo.folder_path(s, d.folder_id), Processing=d.processing_status, Decision=BADGE.get(d.decision, d.decision or "-"),
                              Review=d.review_status, Retries=d.retry_count, Seconds=round(d.processing_seconds or 0, 1), Uploaded=d.created_at.strftime("%Y-%m-%d %H:%M")) for d in docs])

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🧾 Invoice AI")
    st.session_state.setdefault("page", "Dashboard")
    st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.divider(); st.caption("Pipeline settings (this session)")
    settings.sc_samples = st.slider("Self-consistency samples", 0, 5, settings.sc_samples, help="Extra independent LLM calls per document. 0 = off (cheapest).")
    settings.max_retries = st.slider("Max retries", 0, 2, settings.max_retries, help="Controlled re-extractions of flagged fields.")
    st.caption(f"Provider: `{settings.llm_provider}` · Model: `{settings.model}`"); st.caption("LLM API key: " + ("✅ configured" if settings.has_api_key else f"❌ missing (check the {settings.llm_provider.upper()} API key in .env)"))
page = st.session_state["page"]

# ------------------------------------------------------------------ dashboard
def page_dashboard():
    st.header("Dashboard")
    with session_scope(SF) as s: st_ = repo.dashboard_stats(s); recent = repo.recent_documents(s, 10); rows = doc_rows(recent, s)
    c = st.columns(6)
    c[0].metric("Documents", st_["total"]); c[1].metric("Accepted", st_["accepted"]); c[2].metric("Need review", st_["review"]); c[3].metric("Retried", st_["retried"]); c[4].metric("Processing failures", st_["failed"])
    c[5].metric("Extraction success", "-" if st_["extraction_success_rate"] is None else f"{st_['extraction_success_rate']:.0%}")
    if st_["total"] == 0: st.info("No documents yet. Go to **Upload** to process your first invoice."); return
    a, b = st.columns(2)
    with a:
        st.subheader("Field routing decisions"); fd = {BADGE.get(k, k): v for k, v in st_["field_decisions"].items()}
        st.bar_chart(pd.Series(fd, name="fields")); st.caption(f"{st_['fields_extracted']} non-null fields across current results. Average processing time: " + ("-" if st_["avg_seconds"] is None else f"{st_['avg_seconds']:.1f}s"))
    with b:
        st.subheader("Validation rules (current results)")
        rs = pd.DataFrame([dict(rule=k, applicable=v["applicable"], failed=v["failed"], fail_rate=(v["failed"] / v["applicable"] if v["applicable"] else None)) for k, v in st_["rule_stats"].items()])
        st.dataframe(rs, hide_index=True, use_container_width=True) if len(rs) else st.write("No rule results yet.")
        if st_["retry_outcomes"]: st.write("Retry outcomes:", {k or "n/a": v for k, v in st_["retry_outcomes"].items()})
        st.caption(f"LLM runs: {st_['llm_runs']} ({st_['llm_runs_ok']} successful) · human-approved: {st_['approved']} · corrected: {st_['corrected']}")
    st.subheader("Recent processing history"); st.dataframe(rows, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ upload
def page_upload():
    st.header("Upload & process")
    if not settings.has_api_key: st.warning("The configured LLM API key is missing - OCR will run but extraction will fail. Copy `.env.example` to `.env` and add your key.")
    with session_scope(SF) as s: opts = folder_options(s)
    folder = st.selectbox("Store in folder", list(opts), key="up_folder")
    files = st.file_uploader("PDF, PNG, JPG or JPEG (max %d MB, %d pages)" % (settings.max_file_mb, settings.max_pages), type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
    if st.button("Process documents", type="primary", disabled=not files):
        for f in files:
            with st.status(f"Processing {f.name}", expanded=True) as status:
                r = pipeline.process_upload(f.name, f.getvalue(), opts[folder], progress=lambda m: st.write(m))
                if r.status == "rejected": status.update(label=f"{f.name}: rejected", state="error"); st.error(r.message)
                elif r.duplicate: status.update(label=f"{f.name}: duplicate", state="complete"); st.info(r.message)
                elif r.status == "failed": status.update(label=f"{f.name}: processing failed", state="error"); st.error(r.message or "Processing failed.")
                else: status.update(label=f"{f.name}: done", state="complete")
            if r.doc_id: st.button(f"Open {f.name}", key=f"open_{f.name}_{r.doc_id}", on_click=go, args=("Document detail", r.doc_id))

# ------------------------------------------------------------------ documents list
def page_documents():
    st.header("Documents")
    with session_scope(SF) as s: fopts = folder_options(s)
    c = st.columns([3, 2, 1.3, 1.5]); q = c[0].text_input("Search vendor, invoice no., date, amount, any extracted value, file name"); fl = c[1].selectbox("Folder", ["(all)"] + [k for k in fopts if k != "(no folder)"])
    dec = c[2].selectbox("Decision", ["(all)", "ACCEPT", "REVIEW"]); rev = c[3].selectbox("Review", ["(all)", "unreviewed", "approved", "rejected", "corrected"])
    with session_scope(SF) as s:
        docs = repo.search_documents(s, q, fopts.get(fl) if fl != "(all)" else None, None if dec == "(all)" else dec, None if rev == "(all)" else rev); rows = doc_rows(docs, s)
    st.caption(f"{len(rows)} document(s)")
    if rows.empty: st.info("No documents match."); return
    st.dataframe(rows, hide_index=True, use_container_width=True)
    pick = st.selectbox("Open document", rows["ID"].tolist(), format_func=lambda i: f"#{i} - {rows.loc[rows.ID == i, 'File'].iloc[0]}")
    st.button("Open details", on_click=go, args=("Document detail", int(pick)))

# ------------------------------------------------------------------ document detail
def load_ocr(doc):
    p = settings.ocr_cache_dir / f"{doc.file_hash[:32]}.json"
    try: return OCRDocument.from_dict(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None
    except Exception: return None

def page_detail():
    st.header("Document detail")
    with session_scope(SF) as s:
        ids = [d.id for d in repo.search_documents(s)]
        if not ids: st.info("No documents yet."); return
        did = st.session_state.get("doc_id", ids[0]); did = did if did in ids else ids[0]
        did = st.selectbox("Document", ids, index=ids.index(did), format_func=lambda i: f"#{i}"); st.session_state["doc_id"] = did
        d = s.get(Document, did); fields = repo.current_fields(s, did); fopts = folder_options(s); path = d.stored_path
        runs = [dict(attempt=r.attempt_no, kind=r.kind, status=r.status, routing=r.doc_decision, outcome=r.outcome, selected=r.selected, tokens=(r.usage or {}).get("prompt", 0) + (r.usage or {}).get("completion", 0),
                     latency_s=round(r.latency_s or 0, 1), error=r.error) for r in d.runs]
        validation = next((r.validation for r in d.runs if r.selected), []); events = [dict(when=e.created_at.strftime("%Y-%m-%d %H:%M"), action=e.action, note=e.note, changes=json.dumps(e.changes) if e.changes else "") for e in d.reviews]
        info = dict(name=d.original_name, proc=d.processing_status, dec=d.decision, reasons=d.decision_reasons, review=d.review_status, err=d.error, folder=d.folder_id, retries=d.retry_count, secs=d.processing_seconds, conf=d.ocr_mean_conf, pages=d.n_pages)
        recs = [dict(name=f.name, value=f.value, final=f.final_value, edited=f.was_edited, decision=f.decision, reasons=f.reasons, issues=f.issues, lines=f.source_lines, ocr=f.ocr_signal, llm=f.llm_signal, val=f.validation) for f in fields]
        ocr = load_ocr(d)
    top = st.columns([3, 1.5, 1.5, 1.5]); top[0].subheader(info["name"]); top[1].markdown(f"**Processing:** {info['proc']}"); top[2].markdown(f"**System decision:** {BADGE.get(info['dec'], info['dec'] or '-')}"); top[3].markdown(f"**Review:** {info['review']}")
    if info["proc"] == "failed": st.error(f"Processing failed (this is a failure, not an uncertain extraction): {info['err']}")
    for r in info["reasons"] or []: st.write("• " + r)
    st.caption(f"{info['pages']} page(s) · OCR mean confidence {('%.2f' % info['conf']) if info['conf'] is not None else '-'} · {info['retries']} retry(ies) · {info['secs']:.1f}s")
    a = st.columns(5)
    if a[0].button("✅ Approve", disabled=info["proc"] != "completed"):
        with session_scope(SF) as s: repo.set_review_status(s, did, "approved", "Approved in UI")
        st.rerun()
    if a[1].button("⛔ Reject"):
        with session_scope(SF) as s: repo.set_review_status(s, did, "rejected", "Rejected in UI")
        st.rerun()
    if a[2].button("🔁 Reprocess"):
        with st.spinner("Re-running OCR/LLM pipeline (human edits are replaced)..."): pipeline.reprocess(did)
        st.rerun()
    mv = a[3].selectbox("Move to folder", list(fopts), index=list(fopts.values()).index(info["folder"]) if info["folder"] in fopts.values() else 0, label_visibility="collapsed")
    if a[4].button("Move") and fopts[mv] != info["folder"]:
        with session_scope(SF) as s: repo.move_document(s, did, fopts[mv])
        st.rerun()
    t1, t2, t3, t4, t5, t6 = st.tabs(["Extracted fields", "Evidence & signals", "Validation", "Attempts & history", "Document", "OCR text"])
    with t1:
        if not recs: st.info("No extraction is available for this document.")
        else:
            df = pd.DataFrame([dict(Field=FIELD_BY_NAME[r["name"]].label, key=r["name"], Value=r["final"] or "", Status=BADGE.get(r["decision"], r["decision"]), Why="; ".join(r["reasons"])[:300], Edited="✎" if r["edited"] else "") for r in recs])
            edited = st.data_editor(df.drop(columns=["key"]), hide_index=True, use_container_width=True, disabled=["Field", "Status", "Why", "Edited"], key=f"editor_{did}_{info['review']}")
            st.caption("🟢 evidence, OCR quality and rules all pass · 🟠 repairable by re-extraction · 🔴 needs human verification · ⚪ not present. Edit the Value column, then save.")
            if st.button("💾 Save edits", type="primary"):
                edits = {df.loc[i, "key"]: (edited.loc[i, "Value"] or None) for i in df.index}
                with session_scope(SF) as s: ch = repo.update_field_values(s, did, edits, "Edited in UI")
                st.success(f"Saved {len(ch)} change(s)." if ch else "No changes."); st.rerun()
    with t2:
        sel = st.selectbox("Field", [r["name"] for r in recs if r["value"] is not None] or [""], format_func=lambda n: FIELD_BY_NAME[n].label if n else "-") if recs else ""
        r = next((x for x in recs if x["name"] == sel), None)
        if r:
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown("**Signal 1 - OCR / layout quality**"); st.json(r["ocr"] or {"status": "UNAVAILABLE"}, expanded=False)
            with c2: st.markdown("**Signal 2 - LLM grounding / consistency**"); st.json(r["llm"] or {}, expanded=False)
            with c3: st.markdown("**Signal 3 - Validation**"); st.json(r["val"], expanded=False)
            st.markdown(f"**Decision:** {BADGE.get(r['decision'])} — " + " ".join(r["reasons"]))
            if ocr is not None:
                cited = set(r["lines"]) | set((r["ocr"] or {}).get("evidence_lines", []))
                st.markdown("**Supporting OCR evidence**"); st.dataframe(pd.DataFrame([dict(line=l.id, text=l.text, conf=round(l.conf, 3)) for l in ocr.lines if l.id in cited]), hide_index=True, use_container_width=True)
                bbox, pg = (r["ocr"] or {}).get("bbox"), (r["ocr"] or {}).get("page") or 0; imgs = render_pages(path, max(pg + 1, 1))
                if bbox and len(imgs) > pg: st.image(highlight(imgs[pg], bbox), caption="Region the value was read from", use_container_width=True)
    with t3:
        vd = pd.DataFrame([dict(rule=v["rule"], status=v["status"], fields=", ".join(v["fields"]), explanation=v["message"]) for v in validation])
        st.dataframe(vd, hide_index=True, use_container_width=True) if len(vd) else st.write("No validation results.")
    with t4:
        st.markdown("**Extraction attempts (initial and controlled retries)**"); st.dataframe(pd.DataFrame(runs), hide_index=True, use_container_width=True)
        st.markdown("**Review history**"); st.dataframe(pd.DataFrame(events), hide_index=True, use_container_width=True) if events else st.write("No review events yet.")
    with t5:
        imgs = render_pages(path, 3)
        for i, im in enumerate(imgs): st.image(im, caption=f"Page {i + 1}", use_container_width=True)
        if not imgs: st.warning("Preview unavailable for this file.")
    with t6:
        if ocr is None: st.write("OCR text not available.")
        else: st.code(ocr.indexed_text()[0], language=None)

# ------------------------------------------------------------------ folders
def page_folders():
    st.header("Folders")
    with session_scope(SF) as s: tree = repo.list_folders(s); fopts = folder_options(s)
    st.write("\n".join("\u00a0\u00a0\u00a0\u00a0" * f["depth"] + f"📁 {f['name']}" for f in tree) or "No folders yet.")
    def act(fn, ok):
        try:
            with session_scope(SF) as s: fn(s)
            st.success(ok); st.rerun()
        except repo.RepoError as e: st.error(str(e))
    c = st.columns(2)
    with c[0]:
        st.subheader("Create"); nm = st.text_input("Folder name", key="new_name"); par = st.selectbox("Parent", list(fopts), key="new_parent")
        if st.button("Create folder"): act(lambda s: repo.create_folder(s, nm, fopts[par]), "Folder created.")
    real = {k: v for k, v in fopts.items() if v is not None}
    with c[1]:
        st.subheader("Rename / move / delete")
        if not real: st.write("Create a folder first.")
        else:
            tgt = st.selectbox("Folder", list(real), key="tgt"); tid = real[tgt]
            new = st.text_input("New name", key="ren"); 
            if st.button("Rename"): act(lambda s: repo.rename_folder(s, tid, new), "Renamed.")
            dest = st.selectbox("Move into", list(fopts), key="mvdest")
            if st.button("Move folder"): act(lambda s: repo.move_folder(s, tid, fopts[dest]), "Moved.")
            if st.button("Delete (only if empty)"): act(lambda s: repo.delete_folder(s, tid), "Deleted.")

# ------------------------------------------------------------------ export
def page_export():
    st.header("Export")
    c = st.columns(3); dec = c[0].selectbox("System decision", ["(all)", "ACCEPT", "REVIEW"]); rev = c[1].selectbox("Review status", ["(all)", "unreviewed", "approved", "rejected", "corrected"])
    with session_scope(SF) as s:
        fopts = folder_options(s, False); fl = c[2].selectbox("Folder", ["(all)"] + list(fopts))
        docs = repo.search_documents(s, "", fopts.get(fl) if fl != "(all)" else None, None if dec == "(all)" else dec, None if rev == "(all)" else rev, limit=100000); recs = export_svc.export_records(s, [d.id for d in docs])
    st.caption(f"{len(recs)} persisted record(s). Values are the final values (human edits override extracted values).")
    if not recs: return
    csv_text, json_text = export_svc.to_csv(recs), export_svc.to_json(recs)
    d = st.columns(2); d[0].download_button("⬇ Download CSV", csv_text, "invoices.csv", "text/csv"); d[1].download_button("⬇ Download JSON", json_text, "invoices.json", "application/json")
    st.dataframe(pd.read_csv(__import__("io").StringIO(csv_text), dtype=str).fillna(""), hide_index=True, use_container_width=True)

def page_about():
    st.header("About")
    st.markdown("""**Pipeline:** upload → DocTR OCR (words, confidence, geometry) → readable text with line IDs → Groq LLM (strict JSON + cited line IDs) → three reliability signals → field-level ACCEPT / RETRY / REVIEW → SQLite.

* **Signal 1 – OCR/layout quality:** confidence of the OCR words that support *this* value (mean/min/p10), whether the value can be located, fragmentation. *Unavailable* is reported separately from *low*.
* **Signal 2 – LLM grounding:** do the cited lines exist and contain the value (after normalisation)? Optional self-consistency uses real independent API calls. The model's own confidence is never used.
* **Signal 3 – Deterministic validation:** transparent rules returning PASS / FAIL / NOT_APPLICABLE with explanations, attributed to affected fields.

No claim is made here that this routing beats simpler alternatives - run `python scripts/evaluate.py` against DocILE ground truth to measure it.""")
    st.json({k: v for k, v in settings.llm_config_dict().items()}, expanded=False)

{"Dashboard": page_dashboard, "Upload": page_upload, "Documents": page_documents, "Document detail": page_detail, "Folders": page_folders, "Export": page_export, "About": page_about}[page]()
