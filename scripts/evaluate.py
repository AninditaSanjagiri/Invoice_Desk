"""Ground-truth evaluation of routing policies on DocILE (existing local copy; nothing is downloaded).

  python scripts/evaluate.py --smoke                                  # 5 dev documents, ~5-10 real Groq calls
  python scripts/evaluate.py --split dev --n 20                       # configurable sample
  python scripts/evaluate.py --split calib --n 150 --freeze --yes     # choose thresholds on CALIBRATION, save frozen_policies.json
  python scripts/evaluate.py --split test --n 150 --confirm-test --yes   # held-out test: uses the frozen thresholds, never tunes

LLM calls are cached in experiments/app_eval/llm_cache (resumable). Runs that would need > 100 calls require --yes."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.config import get_settings
from src.services import evaluation as ev

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "calib", "test"], default="dev"); ap.add_argument("--n", type=int, default=None, help="number of documents (default: whole split)")
    ap.add_argument("--smoke", action="store_true", help="5 development documents, self-consistency off"); ap.add_argument("--sc", type=int, default=None, help="self-consistency samples per document")
    ap.add_argument("--freeze", action="store_true", help="(calib) save thresholds chosen on this split"); ap.add_argument("--confirm-test", action="store_true", help="required to touch the held-out test split")
    ap.add_argument("--yes", action="store_true", help="allow runs estimated at > 100 LLM calls"); ap.add_argument("--boot", type=int, default=300); ap.add_argument("--out", default=str(ROOT / "experiments" / "app_eval"))
    a = ap.parse_args(argv); s = get_settings()
    if a.sc is not None: s.sc_samples = a.sc
    if a.smoke: a.split, a.n, s.sc_samples = "dev", 5, 0
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "data" / "docile" / "annotations").exists(): print("DocILE not found at data/docile/{annotations,ocr}. Place the existing copy there (do not re-download)."); return 1
    if not s.has_api_key: print("GROQ_API_KEY is not set (.env). Cached results are still usable, but uncached documents cannot be processed."); 
    fz_path = out / "frozen_policies.json"
    if a.split == "test":
        if not a.confirm_test: print("Refusing to run the TEST split without --confirm-test (thresholds must be frozen on calibration first)."); return 2
        if not fz_path.exists(): print("No frozen policies. Run: python scripts/evaluate.py --split calib --freeze"); return 2
    try: splits = ev.load_or_make_splits(ROOT)
    except ValueError as e: print(f"Cannot create splits: {e}"); return 1
    ids = splits[a.split][: a.n] if a.n else splits[a.split]
    est = len(ids) * (1 + s.sc_samples) * 1.3
    print(f"{a.split}: {len(ids)} documents, self-consistency={s.sc_samples}, max_retries={s.max_retries}  -> about {est:.0f} LLM calls (cached calls are free)")
    if est > 100 and not a.yes: print("Estimated > 100 calls. Re-run with --yes to confirm."); return 2
    processed = ev.process_docs(ids, ROOT, s, out_dir=out, progress=lambda i, n, d: print(f"  [{i + 1}/{n}] {d}", flush=True))
    tables = ev.build_tables(processed, ROOT, s)
    cal_init = ev._labelled(tables["fields"], "init")
    if a.split == "test": fz = json.loads(fz_path.read_text()); note = "thresholds FROZEN on the calibration split"
    else:
        fz = ev.freeze_policies(cal_init, s.sc_samples > 0, min_n=30 if a.split == "calib" else 10)
        note = "thresholds fitted on THIS split - for development only, not reportable" if a.split != "calib" else "thresholds chosen on calibration (use --freeze to save them)"
        if a.freeze and a.split == "calib": ev.jdump(fz, fz_path); print(f"Frozen policies saved to {fz_path}")
    res = ev.evaluate_policies(tables, fz, a.boot)
    tag = f"{a.split}{'_smoke' if a.smoke else ''}"
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(f"\n=== {tag}: {res['labelled_fields']} labelled non-null headline predictions, error rate {res['error_rate']:.1%}; {note} ===")
    print("\nExtraction quality (strict = primary; normalized = secondary):"); print(res["extraction"].round(3).to_string(index=False))
    print("\nSignal quality (AUROC/AUPRC for detecting errors):"); print(res["signals"].round(3))
    cols = ["alpha", "policy", "accepted", "review_rate", "false_auto_approval", "error_capture", "safe_automation", "doc_review_rate", "doc_false_auto_approval"]
    print("\nRouting policies (initial extraction, no retry):"); print(res["policies"][cols].round(3).to_string(index=False))
    print("\nRetry (fields the application router would retry):", res["retry"]); print(res["retry_comparison"][["setting"] + cols[2:]].round(3).to_string(index=False))
    print("\nEfficiency:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res["efficiency"].items()}); print("P4 learned router:", fz["P4_LR_status"])
    for k in ("extraction", "signals", "policies", "retry_comparison"): res[k].round(4).to_csv(out / f"{tag}_{k}.csv")
    tables["fields"].to_csv(out / f"{tag}_field_rows.csv", index=False)
    ev.jdump(dict(split=a.split, n_docs=len(ids), note=note, retry=res["retry"], efficiency=res["efficiency"], error_rate=res["error_rate"], p4_lr=fz["P4_LR_status"], failed_docs=sorted(tables["failed"])), out / f"{tag}_summary.json")
    print(f"\nSaved to {out}"); return 0

if __name__ == "__main__": sys.exit(main())
