"""Signal 3: transparent deterministic validation. Every rule returns PASS / FAIL / NOT_APPLICABLE with a human-readable message
and the fields it affects. A rule is NOT_APPLICABLE (never FAIL) when the information it needs is missing."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from decimal import Decimal
from ..extraction.schema import FIELD_BY_NAME, KIND, FieldOut
from ..utils.normalize import parse_amount, date_interps, tok_f1, money_tolerance, currency_code

PASS, FAIL, NA = "PASS", "FAIL", "NOT_APPLICABLE"

@dataclass
class RuleResult:
    rule: str; status: str; fields: list; message: str; critical: bool = True
    def to_dict(self): return asdict(self)

def _v(vals, f): return vals.get(f)
def _amt(vals, f):
    v = _v(vals, f); return parse_amount(v) if v is not None else None
def _fmt(x: Decimal) -> str: return f"{x:,.2f}"

def validate(values: dict, required: tuple = (), evidence: dict | None = None, min_grounding: float = 0.99) -> list[RuleResult]:
    """values: {field: value|None}. evidence (optional): {field: dict(cited_exist, has_citation, g_cited)} from Signal 2 for the evidence-consistency rule."""
    R: list[RuleResult] = []
    for f in required:
        present = _v(values, f) is not None
        R.append(RuleResult("required_field_present", PASS if present else FAIL, [f], f"Required field '{f}' is {'present' if present else 'missing'}.", True))
    for f, k in KIND.items():
        v = _v(values, f)
        if v is None: continue
        if k == "amount":
            a = _amt(values, f); ok = a is not None and a >= 0
            R.append(RuleResult("amount_parse", PASS if ok else FAIL, [f], f"'{v}' parses as {_fmt(a)}." if ok else f"'{v}' is not a valid non-negative amount.", True))
        elif k == "date":
            ok = bool(date_interps(v))
            R.append(RuleResult("date_parse", PASS if ok else FAIL, [f], f"'{v}' is a valid date." if ok else f"'{v}' is not a valid date.", True))
    di, dd = _v(values, "date_issue"), _v(values, "date_due")
    if di and dd and date_interps(di) and date_interps(dd):
        a, b = date_interps(di, True), date_interps(dd, True); ok = any(y2 >= y1 for y1 in a for y2 in b)
        R.append(RuleResult("due_after_issue", PASS if ok else FAIL, ["date_issue", "date_due"], "Due date is on/after the issue date." if ok else f"Due date '{dd}' is before issue date '{di}'.", True))
    else: R.append(RuleResult("due_after_issue", NA, ["date_issue", "date_due"], "Needs two parsable dates.", True))
    g, due, n, t, paid = (_amt(values, k) for k in ("amount_total_gross", "amount_due", "amount_total_net", "amount_total_tax", "amount_paid"))
    cur = currency_code(_v(values, "currency")) or next((currency_code(_v(values, k)) for k in ("amount_total_gross", "amount_due") if currency_code(_v(values, k))), None)
    flds = ["amount_total_net", "amount_total_tax", "amount_total_gross"]
    if None not in (g, n, t):
        tol = money_tolerance(g, cur); ok = abs(n + t - g) <= tol
        R.append(RuleResult("net_plus_tax_eq_gross", PASS if ok else FAIL, flds, f"{_fmt(n)} + {_fmt(t)} = {_fmt(n + t)} vs gross {_fmt(g)} (tolerance {_fmt(tol)})." if True else "", True))
    else: R.append(RuleResult("net_plus_tax_eq_gross", NA, flds, "Needs net, tax and gross.", True))
    dflds = ["amount_due", "amount_total_gross", "amount_paid"]
    if g is not None and due is not None:
        tol = money_tolerance(g, cur)
        if paid is not None:
            ok = abs(g - paid - due) <= tol; msg = f"gross {_fmt(g)} - paid {_fmt(paid)} = {_fmt(g - paid)} vs amount due {_fmt(due)}."
        else:
            ok = due <= g + tol; msg = f"Amount due {_fmt(due)} {'does not exceed' if ok else 'exceeds'} the total {_fmt(g)} (late fees or adjustments could explain an excess)."
        R.append(RuleResult("amount_due_vs_total", PASS if ok else FAIL, dflds, msg, True))
    else: R.append(RuleResult("amount_due_vs_total", NA, dflds, "Needs amount due and gross total.", True))
    if g is not None and n is not None:
        ok = n <= g + money_tolerance(g, cur)
        R.append(RuleResult("net_le_gross", PASS if ok else FAIL, ["amount_total_net", "amount_total_gross"], "Net does not exceed gross." if ok else f"Net {_fmt(n)} exceeds gross {_fmt(g)}.", True))
    else: R.append(RuleResult("net_le_gross", NA, ["amount_total_net", "amount_total_gross"], "Needs net and gross.", True))
    vn, cn = _v(values, "vendor_name"), _v(values, "customer_name")
    if vn and cn:
        ok = tok_f1(vn, cn) < 0.8
        R.append(RuleResult("vendor_customer_distinct", PASS if ok else FAIL, ["vendor_name", "customer_name"], "Vendor and customer names differ." if ok else "Vendor and customer names are (nearly) identical - possible field confusion.", True))
    else: R.append(RuleResult("vendor_customer_distinct", NA, ["vendor_name", "customer_name"], "Needs both names.", True))
    va, ca = _v(values, "vendor_address"), _v(values, "customer_address")
    if va and ca:
        ok = tok_f1(va, ca) < 0.8
        R.append(RuleResult("vendor_customer_address_distinct", PASS if ok else FAIL, ["vendor_address", "customer_address"], "Addresses differ." if ok else "Vendor and customer addresses are (nearly) identical - possible field confusion.", False))
    cs = currency_code(_v(values, "currency"))
    if cs:
        for f in ("amount_total_gross", "amount_due", "amount_total_net", "amount_total_tax", "amount_paid"):
            ac = currency_code(_v(values, f)) if _v(values, f) and any(ch in str(_v(values, f)) for ch in "$€£₹¥") else None
            if ac and ac != cs: R.append(RuleResult("currency_consistent", FAIL, ["currency", f], f"{f} uses {ac} but the currency field says {cs}.", True))
    if evidence:
        for f, e in evidence.items():
            if _v(values, f) is None: continue
            if not e["has_citation"]: R.append(RuleResult("source_evidence_consistent", NA, [f], "No cited OCR lines to compare against.", False))
            elif not e["cited_exist"]: R.append(RuleResult("source_evidence_consistent", FAIL, [f], "The model cited OCR lines that do not exist.", True))
            else:
                ok = e["g_cited"] is not None and e["g_cited"] >= min_grounding
                R.append(RuleResult("source_evidence_consistent", PASS if ok else FAIL, [f], "Value is supported by the cited OCR lines." if ok else "Value is not found in the cited OCR lines.", True))
    return R

def per_field(results: list[RuleResult]) -> dict:
    """field -> dict(n_applicable, n_fail, fails=[rule names], messages=[...]) - failures are attributable to the fields they affect."""
    out: dict = {}
    for r in results:
        if r.status == NA: continue
        for f in r.fields:
            d = out.setdefault(f, dict(n_applicable=0, n_fail=0, fails=[], messages=[]))
            d["n_applicable"] += 1
            if r.status == FAIL: d["n_fail"] += 1; d["fails"].append(r.rule); d["messages"].append(r.message)
    return out
