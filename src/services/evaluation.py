"""Ground-truth evaluation of routing policies P0-P4 (with / without retry) on DocILE documents.
Reuses the application's core pipeline, so measured behaviour equals application behaviour. Caching makes runs resumable."""
from __future__ import annotations
import itertools, json, logging, random, time
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
from ..config import Settings
from ..extraction.extractor import Extractor
from ..extraction.llm_client import GroqLLM
from ..ocr.engine import OCRError, ProvidedOCREngine
from ..routing.router import ACCEPT
from .core import CoreResult, evaluate_attempt, run_core
from .eval_metrics import ADDRESS, HEADLINE, bootstrap_docs, choose_threshold, load_gt, prf, risk_coverage, router_metrics, score_row, signal_auc

log = logging.getLogger(__name__)
ALL_EVAL_FIELDS = HEADLINE + ADDRESS + ["vendor_email", "vendor_tax_id", "po_number", "currency", "amount_paid", "payment_reference", "payment_terms"]

def jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8"); tmp.replace(path)

def load_or_make_splits(root: Path, sizes=(50, 150, 150), seed: int = 42, n_baseline: int = 20) -> dict:
    """Uses experiments/data_splits/splits.json from the research notebook when present (identical ids); otherwise builds the same split."""
    sp = root / "experiments" / "data_splits" / "splits.json"
    if sp.exists(): return json.loads(sp.read_text())
    ocr_dir, ann_dir, pdf_dir = (root / "data" / "docile" / x for x in ("ocr", "annotations", "pdfs"))
    ids = sorted(p.stem for p in ann_dir.glob("*.json") if (ocr_dir / f"{p.stem}.json").exists())
    pdf_ids = sorted(p.stem for p in pdf_dir.glob("*.pdf")) if pdf_dir.exists() else ids
    if len(ids) < sum(sizes): raise ValueError(f"Only {len(ids)} annotated documents with OCR found under data/docile; {sum(sizes)} are needed for the {sizes} split.")
    base = [i for i in pdf_ids[:n_baseline] if i in set(ids)]; rest = [i for i in ids if i not in set(base)]; random.Random(seed).shuffle(rest); k = sizes[0] - len(base)
    out = dict(seed=seed, baseline=base, dev=base + rest[:k], calib=rest[k:k + sizes[1]], test=rest[k + sizes[1]:k + sizes[1] + sizes[2]])
    assert not (set(out["dev"]) & set(out["calib"])) and not (set(out["dev"]) & set(out["test"])) and not (set(out["calib"]) & set(out["test"]))
    jdump(out, sp); return out

def process_docs(doc_ids, root: Path, s: Settings, llm=None, out_dir: Path | None = None, progress=None) -> dict:
    """Runs the app's core pipeline on DocILE docs (provided OCR). LLM outputs are cached per (doc, config) -> resumable. Returns {doc_id: (ocr, CoreResult)|error}."""
    eng = ProvidedOCREngine(root / "data" / "docile" / "ocr"); ex = Extractor(s, llm or GroqLLM(s), cache_dir=(out_dir or root / "experiments" / "app_eval") / "llm_cache"); res = {}
    for i, d in enumerate(doc_ids):
        try: ocr = eng.load(d)
        except OCRError as e: res[d] = ("ocr_error", str(e)); continue
        if progress: progress(i, len(doc_ids), d)
        res[d] = (ocr, run_core(ocr, ex, s, cache_key=d, retry_enabled=True))
    return res

def build_tables(processed: dict, root: Path, s: Settings) -> dict:
    """Field-instance table (initial, no-retry state + final, with-retry state), extraction rows, and document-level bookkeeping."""
    rows, ext, failed, eff = [], [], set(), []
    for d, item in processed.items():
        gt = load_gt(json.loads((root / "data" / "docile" / "annotations" / f"{d}.json").read_text()))
        if item[0] == "ocr_error" or item[1].status == "failed":
            failed.add(d); [ext.append(dict(doc_id=d, field=f, pred=None, **score_row(f, None, gt, True))) for f in ALL_EVAL_FIELDS]
            if item[0] != "ocr_error": eff.append(dict(doc_id=d, **item[1].usage(), failed=True))
            continue
        ocr, res = item; samples = [a.fields for a in res.samples if a.ok]
        init0 = evaluate_attempt(res.history[0].attempt, ocr, samples, s, 0)        # initial attempt routed WITHOUT retry (RETRY -> REVIEW)
        init1 = res.history[0]                                                       # initial attempt routed WITH retry available (shows what would be retried)
        fin = res.final; u = res.usage(); eff.append(dict(doc_id=d, **u, failed=False, retried=any(h.role == "retry" for h in res.history)))
        for f in ALL_EVAL_FIELDS:
            ext.append(dict(doc_id=d, field=f, pred=fin.values[f], **score_row(f, fin.values[f], gt)))
        for f in HEADLINE + ADDRESS:
            v0, vf = init0.values[f], fin.values[f]
            if v0 is None and vf is None: continue
            sg = init0.signals[f]; o, l = sg.ocr, sg.llm
            r0 = score_row(f, v0, gt) if v0 is not None else None; rf = score_row(f, vf, gt) if vf is not None else None
            rows.append(dict(doc_id=d, field=f, group=f, pred0=v0, predf=vf,
                             status0=r0["status_strict"] if r0 else None, statusf=rf["status_strict"] if rf else None,
                             ocr_conf_mean=o.conf_mean if o else np.nan, ocr_conf_min=o.conf_min if o else np.nan, ocr_status=o.status if o else None,
                             llm_status=l.status if l else None, g_cited=(l.g_cited if l and l.g_cited is not None else np.nan), g_any=l.g_any if l else np.nan,
                             sc_agree=(l.sc_agree if l and l.sc_agree is not None else np.nan), v_fail=int(sg.validation["n_fail"] > 0), v_applicable=int(sg.validation["n_applicable"] > 0),
                             app_decision0=init0.routing.fields[f].decision, app_would_retry=init1.routing.fields[f].decision == "RETRY",
                             app_decisionf=fin.routing.fields[f].decision, retried=any(h.role == "retry" for h in res.history), retry_outcome=(res.retry_outcomes[-1] if res.retry_outcomes else None)))
    return dict(fields=pd.DataFrame(rows), extraction=pd.DataFrame(ext), failed=failed, efficiency=pd.DataFrame(eff))

def _labelled(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """state 'init' or 'final': correctness of the prediction in that state; unannotated predictions are excluded (reported separately)."""
    st, pr = ("status0", "pred0") if state == "init" else ("statusf", "predf")
    d = df[df[pr].notna() & (df[st] != "FP_UNANNOTATED")].copy(); d["correct"] = (d[st] == "TP").astype(int); d["error"] = 1 - d["correct"]
    return d[d.field.isin(HEADLINE)].reset_index(drop=True)

# ---------- policies (operate on the INITIAL, no-retry extraction) ----------
def p1_score(df: pd.DataFrame, sc_on: bool) -> np.ndarray: return (df.sc_agree if sc_on else df.ocr_conf_mean).fillna(-1.0).values
def p4_accept(df: pd.DataFrame, conf_mean: float, conf_min: float, sc_on: bool, min_sc: float = 1.0) -> np.ndarray:
    a = (df.llm_status == "SUPPORTED") & (df.ocr_status == "HIGH") & (df.ocr_conf_mean.fillna(0) >= conf_mean) & (df.ocr_conf_min.fillna(0) >= conf_min) & (df.v_fail == 0)
    if sc_on: a &= df.sc_agree.fillna(0) >= min_sc
    return a.values

P4_GRID = [(m, n) for m in (0.0, 0.7, 0.8, 0.85, 0.9, 0.95) for n in (0.0, 0.3, 0.5, 0.7)]

def freeze_policies(cal: pd.DataFrame, sc_on: bool, alphas=(0.05, 0.10), min_n: int = 30, mode: str = "point") -> dict:
    """Thresholds chosen on CALIBRATION data only. P4 grid: maximise accepted fields subject to calibration risk <= alpha."""
    fz = dict(sc_on=sc_on, alphas=list(alphas), mode=mode, min_n=min_n, P1={}, P3={}, P4={}, P4_LR=None)
    for a in alphas:
        fz["P1"][str(a)] = choose_threshold(p1_score(cal, sc_on), cal.error, a, None, mode, min_n)
        fz["P3"][str(a)] = choose_threshold(p1_score(cal, sc_on), cal.error, a, cal.v_fail.values == 0, mode, min_n)
        best = None
        for m, n in P4_GRID:
            acc = p4_accept(cal, m, n, sc_on); k = int(acc.sum())
            if k >= min_n and cal.error.values[acc].mean() <= a and (best is None or (k, m, n) > best[0]): best = ((k, m, n), (m, n))
        fz["P4"][str(a)] = None if best is None else list(best[1])
    # optional learned router: only with enough labelled examples AND both classes; otherwise the transparent rule router above is used
    n_pos, n_neg = int(cal.correct.sum()), int(cal.error.sum())
    fz["P4_LR_status"] = "fitted" if (len(cal) >= 200 and min(n_pos, n_neg) >= 30) else f"skipped (needs >=200 labelled fields and >=30 of each class; have {len(cal)} with {n_pos} correct / {n_neg} errors)"
    return fz

def apply_policy(df: pd.DataFrame, name: str, fz: dict, alpha: float, sc_on: bool) -> np.ndarray:
    a = str(alpha)
    if name == "P0": return np.ones(len(df), bool)
    if name == "P2": return df.v_fail.values == 0
    if name == "Oracle": return df.error.values == 0
    if name == "P1":
        t = fz["P1"].get(a); s = p1_score(df, sc_on); return np.zeros(len(df), bool) if t is None else s >= t
    if name == "P3":
        t = fz["P3"].get(a); s = p1_score(df, sc_on); return np.zeros(len(df), bool) if t is None else (s >= t) & (df.v_fail.values == 0)
    if name == "P4":
        p = fz["P4"].get(a); return np.zeros(len(df), bool) if p is None else p4_accept(df, p[0], p[1], sc_on)
    if name == "P4_app": return (df.app_decision0 == ACCEPT).values           # the application's configured router (settings thresholds), no retry
    raise ValueError(name)

POLICIES = ["P0", "P1", "P2", "P3", "P4", "P4_app", "Oracle"]

def evaluate_policies(tables: dict, fz: dict, boot: int = 300) -> dict:
    sc_on = fz["sc_on"]; f = tables["fields"]; nfail = len(tables["failed"]); out = {}
    init = _labelled(f, "init"); rows = []
    for a in fz["alphas"]:
        for p in POLICIES:
            if p in ("P0", "P2", "P4_app", "Oracle") and a != fz["alphas"][0] and p != "P4": continue
            acc = apply_policy(init, p, fz, a, sc_on); m = router_metrics(init, acc, nfail, boot)
            rows.append(dict(alpha=a if p in ("P1", "P3", "P4") else None, policy=p, **m))
    out["policies"] = pd.DataFrame(rows)
    # retry: only fields the application router would retry
    fr = f[f.app_would_retry].copy(); rr = {}
    if len(fr):
        was_ok = fr.status0 == "TP"; now_ok = fr.statusf == "TP"
        rr = dict(n_retried_fields=int(len(fr)), corrected=int((~was_ok & now_ok).sum()), regressed=int((was_ok & ~now_ok).sum()), still_wrong=int((~was_ok & ~now_ok).sum()), unchanged_correct=int((was_ok & now_ok).sum()),
                  correction_rate=float((~was_ok & now_ok).mean()), regression_rate=float((was_ok & ~now_ok).mean()))
    else: rr = dict(n_retried_fields=0)
    out["retry"] = rr
    # with vs without retry (application router): decisions after retry judged on FINAL values
    fin = _labelled(f, "final"); init_app = _labelled(f, "init")
    wr = pd.DataFrame([dict(setting="no retry", **router_metrics(init_app, (init_app.app_decision0 == ACCEPT).values, nfail, boot)),
                       dict(setting="with retry", **router_metrics(fin, (fin.app_decisionf == ACCEPT).values, nfail, boot))])
    out["retry_comparison"] = wr
    out["signals"] = pd.DataFrame({n: signal_auc(v, init.error) for n, v in {"ocr_conf_mean": init.ocr_conf_mean.values, "ocr_conf_min": init.ocr_conf_min.values, "g_any": init.g_any.values,
                                    "sc_agree": init.sc_agree.values, "validation_pass": (1 - init.v_fail).values, "llm_supported": (init.llm_status == "SUPPORTED").astype(float).values}.items()}).T
    out["risk_coverage"] = {n: risk_coverage(p1_score(init, sc_on) if n == "P1" else init.g_any.values, init.error.values) for n in ("P1", "grounding")}
    ext = tables["extraction"]; er = []
    for nm, flds in (("headline", HEADLINE), ("addresses", ADDRESS), ("all 16 evaluated fields", ALL_EVAL_FIELDS)):
        sub = ext[ext.field.isin(flds)]
        for col, lab in (("status_strict", "strict"), ("status_norm", "normalized")):
            m = prf(sub, col); pt, lo, hi = bootstrap_docs(sub, lambda x, c=col: prf(x, c)["f1"], boot or 200)
            er.append(dict(field_set=nm, matching=lab, precision=m["precision"], recall=m["recall"], f1=m["f1"], f1_ci_lo=lo, f1_ci_hi=hi, exact_match_acc=m["exact_match_acc"], FP_unannotated=m["FP_unannotated"]))
    out["extraction"] = pd.DataFrame(er)
    e = tables["efficiency"]
    out["efficiency"] = dict(docs=len(f.doc_id.unique()) + nfail, failed_docs=nfail, calls_per_doc=float(e.calls.mean()) if len(e) else None, prompt_tokens_per_doc=float(e.prompt.mean()) if len(e) else None,
                             completion_tokens_per_doc=float(e.completion.mean()) if len(e) else None, latency_s_per_doc=float(e.latency_s.mean()) if len(e) else None, retried_docs=int(e.get("retried", pd.Series(dtype=bool)).sum()) if len(e) else 0,
                             note="cost = tokens x current Groq price for the model (not hard-coded)")
    out["labelled_fields"] = int(len(init)); out["error_rate"] = float(init.error.mean()) if len(init) else None
    return out
