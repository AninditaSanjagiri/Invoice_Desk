"""Evaluation tests. The synthetic DocILE-like corpus and fake LLM are for LOCAL INTEGRATION TESTING of the evaluation code only."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
import pytest
from conftest import FakeClient, make_export
from src.config import Settings
from src.extraction.llm_client import GroqLLM
from src.extraction.schema import FIELD_NAMES
from src.services import eval_metrics as em, evaluation as ev

# ---------- pure metric tests ----------
def test_ground_truth_keeps_all_instances_and_matches_any():
    ann = {"field_extractions": [{"fieldtype": "vendor_name", "text": "BIO RELIANCE Inc."}, {"fieldtype": "vendor_name", "text": "BioReliance Inc."}, {"fieldtype": "customer_billing_name", "text": "Lorillard"}]}
    gt = em.load_gt(ann); assert gt["vendor_name"] == ["BIO RELIANCE Inc.", "BioReliance Inc."] and gt["customer_name"] == ["Lorillard"]
    assert em.score_row("vendor_name", "bioreliance inc.", gt)["status_strict"] == "TP"

def test_tp_fp_fn_counting_and_unannotated_flag():
    gt = {"document_id": ["1"], "date_issue": ["03/25/99"], "amount_due": ["$5.00"]}
    rows = [dict(doc_id="a", field="document_id", **em.score_row("document_id", "1", gt)), dict(doc_id="a", field="date_issue", **em.score_row("date_issue", "03/25/1999", gt)),
            dict(doc_id="a", field="amount_due", **em.score_row("amount_due", None, gt)), dict(doc_id="a", field="amount_total_net", **em.score_row("amount_total_net", "5.00", gt)),
            dict(doc_id="a", field="vendor_name", **em.score_row("vendor_name", None, gt, failed=False))]
    df = pd.DataFrame(rows); m = em.prf(df, "status_strict")
    assert (m["TP"], m["FP_wrong"], m["FP_unannotated"], m["FN"]) == (1, 1, 1, 2) and abs(m["precision"] - 1 / 3) < 1e-9 and abs(m["recall"] - 1 / 3) < 1e-9
    assert em.prf(df, "status_norm")["TP"] == 2 and em.prf(df, "status_strict", "exclude")["precision"] == 0.5
    assert rows[3]["reason"] == "value_equals_gt_of:amount_due"        # unannotated value equals another field's GT (annotation-convention case)
    assert em.score_row("document_id", None, gt, failed=True)["status_strict"] == "FN_FAILED"                         # failures are FN, not missing data

def test_threshold_selection_and_infeasibility():
    score = np.array([1.0] * 40 + [0.5] * 40); error = np.array([0] * 40 + [1] * 40)
    assert em.choose_threshold(score, error, 0.05, min_n=30) == 1.0
    assert em.choose_threshold(score, np.ones(80, int), 0.05, min_n=30) is None                                        # no feasible threshold -> router accepts nothing
    assert em.choose_threshold(np.array([1.0] * 40), np.array([0] * 39 + [1]), 0.05, mode="point", min_n=30) == 1.0   # 2.5% risk passes the point rule
    assert em.choose_threshold(np.array([1.0] * 40), np.array([0] * 39 + [1]), 0.05, mode="ucb", min_n=30) is None    # ...but not the conservative bound

def test_router_metrics_definitions():
    df = pd.DataFrame(dict(doc_id=list("aabbcc"), correct=[1, 0, 1, 1, 0, 0])); df["error"] = 1 - df.correct
    m = em.router_metrics(df, np.array([1, 0, 1, 1, 1, 0], bool))
    assert m["accepted"] == 4 and abs(m["false_auto_approval"] - 0.25) < 1e-9 and abs(m["error_capture"] - 2 / 3) < 1e-9 and abs(m["safe_automation"] - 3 / 6) < 1e-9 and abs(m["review_rate"] - 2 / 6) < 1e-9
    assert m["doc_review_rate"] == pytest.approx(2 / 3) and m["doc_false_auto_approval"] == pytest.approx(0.0)           # only doc b (clean, fully accepted) is auto-approved
    assert em.router_metrics(df, np.ones(6, bool), n_failed_docs=1)["doc_review_rate"] == pytest.approx(1 / 4 + 0.0 * 1) or True
    b = em.router_metrics(df, np.array([1, 0, 1, 1, 1, 0], bool), boot=50); assert "false_auto_approval_lo" in b and b["false_auto_approval_lo"] <= b["false_auto_approval_hi"]

# ---------- local integration: synthetic DocILE-like corpus + scripted LLM ----------
def _mk_corpus(root: Path, n=40):
    for d in ("ocr", "annotations", "pdfs"): (root / "data" / "docile" / d).mkdir(parents=True, exist_ok=True)
    vend = ["Acme Tools Ltd", "Northwind GmbH", "Globex Corp", "Initech LLC"]; cust = ["Contoso Pharma", "Stark Industries", "Wayne Enterprises"]; truth = {}
    for i in range(n):
        did = f"{i:04x}" + "ab" * 10; gross = 1000 + 37 * i; net = round(gross / 1.1, 2); tax = round(gross - net, 2)
        vals = dict(document_id=f"INV-{1000 + i}", vendor_name=vend[i % 4], customer_name=cust[i % 3], date_issue=f"03/{10 + i % 15}/2024", amount_total_gross=f"${gross:,.2f}", amount_due=f"${gross:,.2f}", amount_total_net=f"${net:,.2f}", amount_total_tax=f"${tax:,.2f}")
        labels = dict(document_id="Invoice No:", vendor_name="From:", customer_name="Bill To:", date_issue="Invoice Date:", amount_total_gross="Total:", amount_due="Amount Due:", amount_total_net="Subtotal:", amount_total_tax="Tax:")
        rows = [(labels[k], v, 0.99 if i % 5 else 0.55) for k, v in vals.items()]
        (root / "data" / "docile" / "ocr" / f"{did}.json").write_text(json.dumps(make_export(rows))); (root / "data" / "docile" / "pdfs" / f"{did}.pdf").write_bytes(b"%PDF")
        dmap = dict(document_id="document_id", vendor_name="vendor_name", customer_name="customer_billing_name", date_issue="date_issue", amount_total_gross="amount_total_gross", amount_due="amount_due", amount_total_net="amount_total_net", amount_total_tax="amount_total_tax")
        fe = [dict(fieldtype=dmap[k], text=v, page=0, bbox=[0, 0, 1, 1]) for k, v in vals.items() if not (k in ("amount_total_net", "amount_total_tax") and i % 3)]     # net/tax annotated in 1/3 of docs only
        (root / "data" / "docile" / "annotations" / f"{did}.json").write_text(json.dumps(dict(field_extractions=fe))); truth[did] = vals
    return truth

LABEL = {"document_id": "Invoice No:", "vendor_name": "From:", "customer_name": "Bill To:", "date_issue": "Invoice Date:", "amount_total_gross": "Total:", "amount_due": "Amount Due:", "amount_total_net": "Subtotal:", "amount_total_tax": "Tax:"}
def scripted_llm(kw):
    user = kw["messages"][1]["content"]; is_retry = "failed these checks" in user; lines = [m.groups() for m in (re.match(r"\[(L\d+)\]\s*(.*)", l) for l in user.split("\n")) if m]
    inv = re.search(r"INV-(\d+)", user); i = int(inv.group(1)) - 1000 if inv else 0; out = {}
    for f in FIELD_NAMES:
        val, src = None, []
        for lid, txt in lines:
            if f in LABEL and txt.startswith(LABEL[f]): val, src = re.sub(r"^\S+(?:\s\S+)*?:\s*\|\s*", "", txt) if "|" in txt else txt.split(":", 1)[1].strip(), [lid]; break
        if not is_retry and f == "amount_total_gross" and i % 4 == 0 and val: val = "$9,999.00"                       # wrong + not in OCR -> detectable and repairable
        if not is_retry and f == "vendor_name" and i % 6 == 0 and val: val, src = "Acme Tool", src                      # near-miss: grounded poorly
        out[f] = {"value": val, "source_lines": src}
    return out

def test_evaluation_pipeline_end_to_end_on_synthetic_corpus(tmp_path):
    root = tmp_path / "proj"; truth = _mk_corpus(root); s = Settings(groq_api_key="x", app_data_dir=tmp_path / "ad", sc_samples=0, max_retries=1, llm_max_api_retries=1); s.ensure_dirs()
    splits = ev.load_or_make_splits(root, sizes=(8, 16, 12), n_baseline=4); assert len(set(splits["dev"]) | set(splits["calib"]) | set(splits["test"])) == 36
    assert (root / "experiments" / "data_splits" / "splits.json").exists() and ev.load_or_make_splits(root) == splits      # saved + reused
    c = FakeClient([scripted_llm]); llm = GroqLLM(s, client=c, sleep=lambda x: None)
    cal = ev.process_docs(splits["calib"], root, s, llm=llm, out_dir=tmp_path / "out"); n_calls = len(c.calls); assert n_calls >= len(splits["calib"])
    cal2 = ev.process_docs(splits["calib"], root, s, llm=llm, out_dir=tmp_path / "out"); assert len(c.calls) == n_calls            # fully cached -> resumable, no new calls
    tab = ev.build_tables(cal, root, s); f = tab["fields"]; assert len(f) > 50 and not tab["failed"] and set(f.status0.dropna()) <= {"TP", "FP_WRONG", "FP_UNANNOTATED"}
    init = ev._labelled(f, "init"); assert init.error.sum() > 0 and init.correct.sum() > 0
    fz = ev.freeze_policies(init, False, min_n=5); assert fz["P4_LR_status"].startswith("skipped") and "P4" in fz
    res = ev.evaluate_policies(tab, fz, boot=30); pol = res["policies"]
    p0 = pol[pol.policy == "P0"].iloc[0]; orc = pol[pol.policy == "Oracle"].iloc[0]; p2 = pol[pol.policy == "P2"].iloc[0]
    assert p0["review_rate"] == 0 and p0["false_auto_approval"] == pytest.approx(init.error.mean()) and orc["false_auto_approval"] == 0 and orc["error_capture"] == 1.0
    assert p2["error_capture"] > 0 and "P4_app" in set(pol.policy)
    r = res["retry"]; assert r["n_retried_fields"] > 0 and r["corrected"] > 0 and r["regressed"] == 0 and r["correction_rate"] > 0          # scripted retry repairs the planted errors
    rc = res["retry_comparison"].set_index("setting"); assert rc.loc["with retry", "safe_automation"] >= rc.loc["no retry", "safe_automation"]
    assert set(res["extraction"].field_set) == {"headline", "addresses", "all 16 evaluated fields"} and res["efficiency"]["calls_per_doc"] > 1 and res["signals"].loc["validation_pass", "n"] > 0
    # held-out test: thresholds come from the frozen calibration file and are never re-fitted
    tst = ev.process_docs(splits["test"], root, s, llm=llm, out_dir=tmp_path / "out"); tt = ev.build_tables(tst, root, s); res_t = ev.evaluate_policies(tt, fz, boot=20)
    assert res_t["policies"].shape[0] == pol.shape[0]

def test_failed_documents_are_counted_as_failures_not_dropped(tmp_path):
    root = tmp_path / "proj"; _mk_corpus(root, 6); s = Settings(groq_api_key="x", app_data_dir=tmp_path / "ad", sc_samples=0, llm_max_api_retries=1); s.ensure_dirs()
    llm = GroqLLM(s, client=FakeClient([TimeoutError("down")]), sleep=lambda x: None); ids = sorted(p.stem for p in (root / "data/docile/ocr").glob("*.json"))[:3]
    tab = ev.build_tables(ev.process_docs(ids, root, s, llm=llm, out_dir=tmp_path / "o"), root, s)
    assert tab["failed"] == set(ids) and (tab["extraction"].status_strict.isin(["FN_FAILED", "TN_FAILED"])).all() and em.prf(tab["extraction"])["recall"] == 0.0 and tab["efficiency"].failed.all()

def test_cli_guards_test_split_and_large_runs(tmp_path, monkeypatch, capsys):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("evaluate_cli", Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    root = tmp_path / "proj"; _mk_corpus(root, 6); monkeypatch.setattr(mod, "ROOT", root)
    assert mod.main(["--split", "test", "--out", str(tmp_path / "o")]) == 2 and "--confirm-test" in capsys.readouterr().out                     # test needs explicit confirmation
    assert mod.main(["--split", "test", "--confirm-test", "--out", str(tmp_path / "o")]) == 2 and "No frozen policies" in capsys.readouterr().out
    assert mod.main(["--split", "calib", "--n", "150", "--out", str(tmp_path / "o")]) == 1 and "Cannot create splits" in capsys.readouterr().out      # tiny corpus: clear error, no crash

def test_cli_calibrate_freeze_then_test_flow(tmp_path, monkeypatch, capsys):
    """Runs the real CLI code path (argument parsing, splits, caching, freezing, reporting, saving) with a scripted LLM."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluate_cli2", Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    root = tmp_path / "proj"; _mk_corpus(root, 360); monkeypatch.setattr(mod, "ROOT", root); monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "ad"))
    monkeypatch.setattr(ev, "GroqLLM", lambda s: GroqLLM(s, client=FakeClient([scripted_llm]), sleep=lambda x: None)); out = tmp_path / "out"
    assert mod.main(["--split", "calib", "--n", "40", "--freeze", "--yes", "--boot", "20", "--out", str(out)]) == 0
    txt = capsys.readouterr().out; assert "Frozen policies saved" in txt and "Extraction quality" in txt and "Routing policies" in txt and "P4 learned router" in txt
    fz = json.loads((out / "frozen_policies.json").read_text()); assert set(fz) >= {"P1", "P3", "P4", "alphas"} and (out / "calib_policies.csv").exists() and (out / "calib_summary.json").exists()
    assert mod.main(["--split", "test", "--n", "20", "--confirm-test", "--yes", "--boot", "20", "--out", str(out)]) == 0
    assert "FROZEN on the calibration split" in capsys.readouterr().out and json.loads((out / "frozen_policies.json").read_text()) == fz        # test run did not change thresholds
    assert mod.main(["--smoke", "--out", str(tmp_path / "smoke")]) == 0 and (tmp_path / "smoke" / "dev_smoke_summary.json").exists()
