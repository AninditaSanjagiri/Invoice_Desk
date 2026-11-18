"""Routing engine: field-level ACCEPT / RETRY / REVIEW and a document-level decision, with visible reasons. Pure functions (unit-testable)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from ..config import Settings
from ..reliability.signals import FieldSignals

ACCEPT, RETRY, REVIEW, ABSENT = "ACCEPT", "RETRY", "REVIEW", "ABSENT"
NONCRITICAL_RULES = {"vendor_customer_address_distinct"}
REPAIRABLE_PREFIXES = ("UNSUPPORTED", "INVALID_CITATION", "MISCITED", "UNCITED", "RULE:", "SC_DISAGREE", "REQUIRED_MISSING")

@dataclass
class FieldDecision:
    decision: str; issues: list = field(default_factory=list); reasons: list = field(default_factory=list); repairable: bool = False
    def to_dict(self): return asdict(self)

@dataclass
class RoutingResult:
    fields: dict; doc_decision: str; doc_reasons: list; retry_fields: dict = field(default_factory=dict)     # retry_fields: name -> [issue messages]
    review_fields: list = field(default_factory=list); issue_set: set = field(default_factory=set)
    def to_dict(self): return dict(doc_decision=self.doc_decision, doc_reasons=self.doc_reasons, review_fields=self.review_fields, fields={k: v.to_dict() for k, v in self.fields.items()})

def field_issues(sig: FieldSignals, s: Settings, sc_enabled: bool) -> list[tuple[str, str]]:
    """(code, human message) for every reliability problem of a non-null field, across all three signals."""
    iss = []
    o, l, v = sig.ocr, sig.llm, sig.validation
    if l is not None:
        if l.status == "UNSUPPORTED": iss.append(("UNSUPPORTED", l.reason))
        elif l.status == "INVALID_CITATION": iss.append(("INVALID_CITATION", l.reason))
        elif l.status == "MISCITED": iss.append(("MISCITED", l.reason))
        elif l.status == "UNCITED_FOUND" and not s.allow_uncited: iss.append(("UNCITED", l.reason + " Citations are required for acceptance."))
        if sc_enabled and l.sc_agree is not None and l.sc_agree < s.min_sc_agreement: iss.append(("SC_DISAGREE", f"Only {l.sc_agree:.0%} of {l.n_samples} independent samples agree with this value."))
    if o is not None:
        if o.status == "LOW": iss.append(("LOW_OCR", "OCR quality of the supporting text is low: " + o.reason))
        elif o.status == "UNAVAILABLE" and (l is None or l.status in ("SUPPORTED", "UNCITED_FOUND")):
            iss.append(("NO_OCR_EVIDENCE", o.reason))     # (an unsupported value is already explained - and repairable - via Signal 2)
    for rule, msg in zip(v.get("fails", []), v.get("messages", [])):
        if rule != "source_evidence_consistent": iss.append((f"RULE:{rule}", msg))     # evidence is already judged by Signal 2 (avoid double counting)
    return iss

def route_fields(signals: dict[str, FieldSignals], s: Settings, retry_remaining: int, sc_enabled: bool | None = None) -> RoutingResult:
    sc_enabled = (s.sc_samples > 0) if sc_enabled is None else sc_enabled
    decisions, retry_fields, review_fields, issue_set = {}, {}, [], set()
    for name, sig in signals.items():
        required = name in s.required_fields
        if sig.value is None:
            if required:
                code, msg = "REQUIRED_MISSING", f"Required field '{name}' was not extracted."
                issue_set.add((name, code))
                if retry_remaining > 0: decisions[name] = FieldDecision(RETRY, [code], [msg], True); retry_fields[name] = [msg]
                else: decisions[name] = FieldDecision(REVIEW, [code], [msg], True); review_fields.append(name)
            else: decisions[name] = FieldDecision(ABSENT, [], ["No value extracted (field not present in the document)."])
            continue
        iss = field_issues(sig, s, sc_enabled)
        if not iss: decisions[name] = FieldDecision(ACCEPT, [], ["Evidence supports the value, OCR quality is adequate and applicable rules pass."]); continue
        codes = [c for c, _ in iss]; msgs = [m for _, m in iss]
        for c in codes: issue_set.add((name, c))
        repairable = all(c.startswith(REPAIRABLE_PREFIXES) for c in codes)
        if repairable and retry_remaining > 0: decisions[name] = FieldDecision(RETRY, codes, msgs, True); retry_fields[name] = msgs
        else:
            why = msgs + ([] if repairable else ["Not repairable by re-extraction (OCR evidence problem)."]) + (["Retry limit reached."] if repairable else [])
            decisions[name] = FieldDecision(REVIEW, codes, why, repairable); review_fields.append(name)
    critical_review = [n for n in review_fields if n in s.required_fields or any(c.startswith("RULE:") and c[5:] not in NONCRITICAL_RULES for c in decisions[n].issues)]
    if critical_review: doc, why = REVIEW, [f"Unresolved critical field(s): {', '.join(critical_review)}."]
    elif retry_fields: doc, why = RETRY, [f"{len(retry_fields)} field(s) look repairable: {', '.join(retry_fields)}."]
    elif review_fields: doc, why = REVIEW, [f"{len(review_fields)} field(s) need human verification: {', '.join(review_fields)} (the other fields passed all checks)."]
    else: doc, why = ACCEPT, ["All extracted fields passed the reliability checks and all required fields are present."]
    return RoutingResult(decisions, doc, why, retry_fields, review_fields, issue_set)

def compare_attempts(before: RoutingResult, before_vals: dict, after: RoutingResult, after_vals: dict) -> str:
    """'improved' | 'no_improvement' | 'regressed' - regression = a previously clean field now has issues or changed value."""
    clean_before = {n for n, d in before.fields.items() if d.decision == ACCEPT}
    broke = [n for n in clean_before if after_vals.get(n) != before_vals.get(n) or any(i[0] == n for i in after.issue_set)]
    if broke or len(after.issue_set) > len(before.issue_set): return "regressed"
    return "improved" if len(after.issue_set) < len(before.issue_set) else "no_improvement"
