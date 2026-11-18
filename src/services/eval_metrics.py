"""Pure evaluation maths (no I/O): field-level P/R/F1 against ALL ground-truth instances, router metrics, thresholds, bootstrap CIs."""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np, pandas as pd
from ..extraction.schema import DOCILE_TO_APP, KIND
from ..utils.normalize import date_interps, norm_amount, norm_text, strict_key, tok_f1

HEADLINE = ["document_id", "date_issue", "date_due", "vendor_name", "customer_name", "amount_total_gross", "amount_due", "amount_total_net", "amount_total_tax"]
ADDRESS = ["vendor_address", "customer_address"]

def load_gt(annotation: dict) -> dict:
    """DocILE annotation -> {app_field: [ALL distinct ground-truth strings]} (any instance is acceptable)."""
    gt: dict = defaultdict(list)
    for f in annotation.get("field_extractions", []):
        app = DOCILE_TO_APP.get(f.get("fieldtype")); t = (f.get("text") or "").strip()
        if app and t and t not in gt[app]: gt[app].append(t)
    return dict(gt)

def match_strict(pred, gts) -> bool: return any(strict_key(pred) == strict_key(g) for g in gts)
def match_norm(field: str, pred, gts) -> bool:
    k = KIND[field]
    for g in gts:
        if strict_key(pred) == strict_key(g): return True
        if k == "amount":
            a = norm_amount(pred)
            if a is not None and a == norm_amount(g): return True
        elif k == "date":
            if date_interps(pred) & date_interps(g): return True
        elif k == "address":
            if tok_f1(pred, g) >= 0.8: return True
        elif norm_text(pred) == norm_text(g): return True
    return False

def score_row(field: str, pred, gt: dict, failed: bool = False) -> dict:
    """Statuses: TP FP_WRONG FN TN FP_UNANNOTATED FN_FAILED TN_FAILED. A wrong value is both an FP and an FN."""
    gts = gt.get(field, [])
    if failed: s = "FN_FAILED" if gts else "TN_FAILED"; return dict(status_strict=s, status_norm=s, reason=None)
    if pred is None: s = "FN" if gts else "TN"; return dict(status_strict=s, status_norm=s, reason=None)
    if not gts:
        reason = next((f"value_equals_gt_of:{o}" for o, gl in gt.items() if o != field and match_norm(field, pred, gl)), "no_annotation")
        return dict(status_strict="FP_UNANNOTATED", status_norm="FP_UNANNOTATED", reason=reason)
    return dict(status_strict="TP" if match_strict(pred, gts) else "FP_WRONG", status_norm="TP" if match_norm(field, pred, gts) else "FP_WRONG", reason=None)

def prf(df: pd.DataFrame, col: str = "status_strict", unannotated: str = "fp") -> dict:
    s = df[col]; tp = int((s == "TP").sum()); wrong = int((s == "FP_WRONG").sum()); un = int((s == "FP_UNANNOTATED").sum())
    fn = int((s == "FN").sum() + (s == "FN_FAILED").sum()) + wrong; fp = wrong + (un if unannotated == "fp" else 0)
    p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if not math.isnan(p) and not math.isnan(r) and (p + r) else float("nan")
    return dict(precision=p, recall=r, f1=f1, TP=tp, FP_wrong=wrong, FP_unannotated=un, FN=fn, exact_match_acc=(tp / (tp + wrong + int((s == "FN").sum() + (s == "FN_FAILED").sum())) if (tp + wrong + int((s == "FN").sum())) else float("nan")))

def bootstrap_docs(df: pd.DataFrame, fn, B: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed); groups = {d: g for d, g in df.groupby("doc_id")}; ids = list(groups); pt = fn(df); vals = []
    if not ids: return pt, float("nan"), float("nan")
    for _ in range(B):
        v = fn(pd.concat([groups[ids[i]] for i in rng.integers(0, len(ids), len(ids))]))
        if v is not None and not (isinstance(v, float) and math.isnan(v)): vals.append(v)
    return (pt, *np.percentile(vals, [2.5, 97.5])) if vals else (pt, float("nan"), float("nan"))

# ---------------- thresholds and router metrics ----------------
def choose_threshold(score, error, alpha: float, mask=None, mode: str = "point", min_n: int = 30, delta: float = 0.10):
    """Highest-coverage t with risk(accept = score>=t & mask) <= alpha; mode 'ucb' uses a one-sided Clopper-Pearson bound. None = infeasible."""
    from scipy.stats import beta
    s = np.nan_to_num(np.asarray(score, float), nan=-1e9); e = np.asarray(error, int); m = np.ones(len(s), bool) if mask is None else np.asarray(mask, bool); best = None
    for t in sorted(set(s[m]), reverse=True):
        acc = (s >= t) & m; n = int(acc.sum()); k = int(e[acc].sum())
        if n < min_n: continue
        risk = k / n if mode == "point" else (1.0 if k == n else beta.ppf(1 - delta, k + 1, n - k))
        if risk <= alpha: best = t
    return None if best is None else float(best)

def doc_counts(df: pd.DataFrame, accept) -> pd.DataFrame:
    d = df.assign(acc=np.asarray(accept).astype(int)); d["acc_err"] = d.acc * d.error; d["acc_ok"] = d.acc * d.correct; d["rej"] = 1 - d.acc
    g = d.groupby("doc_id").agg(n=("acc", "size"), n_acc=("acc", "sum"), n_err=("error", "sum"), n_err_acc=("acc_err", "sum"), n_ok_acc=("acc_ok", "sum"), n_rej=("rej", "sum"))
    g["doc_approved"] = (g.n_rej == 0).astype(int); g["doc_approved_err"] = ((g.n_rej == 0) & (g.n_err > 0)).astype(int); return g

def _metrics(g: pd.DataFrame, n_failed_docs: int = 0) -> dict:
    n, na, ne, nea, noa = g.n.sum(), g.n_acc.sum(), g.n_err.sum(), g.n_err_acc.sum(), g.n_ok_acc.sum(); nd = len(g) + n_failed_docs; da = g.doc_approved.sum()
    return dict(n_pred=int(n), accepted=int(na), review_rate=1 - na / n if n else float("nan"), false_auto_approval=nea / na if na else float("nan"),
                error_capture=1 - nea / ne if ne else float("nan"), safe_automation=noa / n if n else float("nan"),
                doc_review_rate=1 - da / nd if nd else float("nan"), doc_false_auto_approval=g.doc_approved_err.sum() / da if da else float("nan"))

def router_metrics(df: pd.DataFrame, accept, n_failed_docs: int = 0, boot: int = 0, seed: int = 0) -> dict:
    g = doc_counts(df, accept); m = _metrics(g, n_failed_docs)
    if boot:
        rng = np.random.default_rng(seed); arr, cols, res = g.values, list(g.columns), defaultdict(list)
        for _ in range(boot):
            for k, v in _metrics(pd.DataFrame(arr[rng.integers(0, len(arr), len(arr))], columns=cols), n_failed_docs).items():
                if k not in ("n_pred", "accepted"): res[k].append(v)
        for k, v in res.items():
            v = np.array(v, float); v = v[~np.isnan(v)]; m[k + "_lo"], m[k + "_hi"] = (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (float("nan"), float("nan"))
    return m

def signal_auc(score, error) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    s = np.asarray(score, float); e = np.asarray(error, int); m = ~np.isnan(s)
    if m.sum() < 5 or e[m].min() == e[m].max(): return dict(n=int(m.sum()), auroc=float("nan"), auprc=float("nan"))
    return dict(n=int(m.sum()), auroc=float(roc_auc_score(e[m], -s[m])), auprc=float(average_precision_score(e[m], -s[m])))

def risk_coverage(score, error):
    s = np.nan_to_num(np.asarray(score, float), nan=-1e9); e = np.asarray(error, int); cov, risk = [], []
    for t in sorted(set(s), reverse=True): a = s >= t; cov.append(float(a.mean())); risk.append(float(e[a].mean()))
    return cov, risk
