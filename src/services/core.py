"""Core reliability pipeline for ONE OCR document: extract -> (self-consistency) -> 3 signals -> route -> controlled retry.
Used by the Streamlit app AND the evaluation script, so measured behaviour equals application behaviour."""
from __future__ import annotations
import logging, time
from dataclasses import dataclass, field
from ..config import Settings
from ..extraction.extractor import Attempt, Extractor
from ..extraction.schema import FIELD_NAMES
from ..ocr.models import OCRDocument
from ..reliability.signals import assess_fields, evidence_for_rules
from ..routing.router import RoutingResult, FieldDecision, ACCEPT, RETRY, REVIEW, ABSENT, route_fields, compare_attempts
from ..validation.rules import validate, per_field

log = logging.getLogger(__name__)

@dataclass
class Evaluated:
    """One extraction attempt together with its three-signal assessment and routing."""
    attempt: Attempt; signals: dict; rules: list; routing: RoutingResult | None; values: dict = field(default_factory=dict); role: str = "initial"

@dataclass
class CoreResult:
    status: str                                            # 'completed' | 'failed' (processing failure, e.g. LLM/API/parse failure)
    error: str | None = None
    history: list = field(default_factory=list)            # every Evaluated attempt in order (initial, retry 1, ...)
    final: Evaluated | None = None                         # the result that is kept (a regressed retry is NOT kept)
    samples: list = field(default_factory=list)            # self-consistency Attempts (real independent API calls)
    retry_outcomes: list = field(default_factory=list)     # 'improved' | 'no_improvement' | 'regressed' | 'failed'
    failed_attempt: Attempt | None = None
    seconds: float = 0.0
    @property
    def all_attempts(self) -> list: return [e.attempt for e in self.history] + self.samples + ([self.failed_attempt] if self.failed_attempt else [])
    def usage(self) -> dict:
        a = self.all_attempts
        return dict(prompt=sum(x.usage["prompt"] for x in a), completion=sum(x.usage["completion"] for x in a), calls=len(a),
                    latency_s=sum(x.latency_s for x in a), api_attempts=sum(x.api_attempts for x in a))

def evaluate_attempt(attempt: Attempt, ocr: OCRDocument, samples: list, s: Settings, retry_remaining: int, role: str = "initial") -> Evaluated:
    vals = {n: attempt.fields[n].value for n in FIELD_NAMES}
    rules = validate(vals, s.required_fields, evidence_for_rules(ocr, attempt.fields), s.min_grounding)
    sigs = assess_fields(ocr, attempt.fields, samples, per_field(rules), s)
    return Evaluated(attempt, sigs, rules, route_fields(sigs, s, retry_remaining), vals, role)

def run_core(ocr: OCRDocument, extractor: Extractor, s: Settings, cache_key: str | None = None, retry_enabled: bool = True, progress=None) -> CoreResult:
    t0 = time.time(); say = progress or (lambda m: None)
    say("Extracting fields with the LLM...")
    a0 = extractor.extract(ocr, cache_key, apply_heuristics=False)
    if not a0.ok:
        return CoreResult("failed", f"LLM extraction failed ({a0.status}): {a0.error}", failed_attempt=a0, seconds=time.time() - t0)
    res = CoreResult("completed"); samples = []
    for i in range(1, s.sc_samples + 1):
        say(f"Self-consistency sample {i}/{s.sc_samples}...")
        sm = extractor.sample(ocr, i, cache_key); res.samples.append(sm)
        if sm.ok: samples.append(sm.fields)
    allowed = s.max_retries if retry_enabled else 0
    say("Computing reliability signals and routing...")
    cur = evaluate_attempt(a0, ocr, samples, s, allowed); res.history.append(cur); res.final = cur; used = 0
    while cur.routing.doc_decision == RETRY and used < allowed:
        used += 1; say(f"Retry {used}/{allowed}: re-extracting flagged fields...")
        ra = extractor.retry(ocr, cur.attempt.fields, cur.routing.retry_fields, cache_key, used)
        if not ra.ok:                                       # failed retry: keep the previous result, remaining RETRY fields become REVIEW
            res.history.append(Evaluated(ra, {}, [], None, {}, "retry")); res.retry_outcomes.append("failed")
            res.final = evaluate_attempt(cur.attempt, ocr, samples, s, 0, cur.role); break
        nxt = evaluate_attempt(ra, ocr, samples, s, allowed - used, "retry")
        outcome = compare_attempts(cur.routing, cur.values, nxt.routing, nxt.values)
        res.history.append(nxt); res.retry_outcomes.append(outcome)
        if outcome == "regressed": res.final = evaluate_attempt(cur.attempt, ocr, samples, s, 0, cur.role); break
        cur = nxt; res.final = nxt
    res.seconds = time.time() - t0
    return res


def run_simple_core(ocr: OCRDocument, extractor: Extractor, s: Settings, cache_key: str | None = None, progress=None) -> CoreResult:
    """Single-pass baseline: one extraction, deterministic validation, manual review.

    OCR/grounding signals are retained for display and diagnosis, but do not drive a
    learned/three-signal router. No self-consistency samples or automatic retries.
    """
    t0 = time.time(); say = progress or (lambda m: None)
    say("Extracting fields with the LLM...")
    attempt = extractor.extract(ocr, cache_key, apply_heuristics=True)
    if not attempt.ok:
        return CoreResult("failed", f"LLM extraction failed ({attempt.status}): {attempt.error}", failed_attempt=attempt, seconds=time.time()-t0)
    say("Running deterministic validation...")
    values = {n: attempt.fields[n].value for n in FIELD_NAMES}
    rules = validate(values, s.required_fields, evidence_for_rules(ocr, attempt.fields), s.min_grounding)
    rule_by_field = per_field(rules)
    signals = assess_fields(ocr, attempt.fields, [], rule_by_field, s)  # display-only diagnostics
    decisions = {}; review_fields = []; critical = []
    for name in FIELD_NAMES:
        sig = signals[name]
        if sig.value is None:
            if name in s.required_fields:
                decisions[name] = FieldDecision(REVIEW, ["REQUIRED_MISSING"], [f"Required field '{name}' was not extracted."])
                review_fields.append(name); critical.append(name)
            else:
                decisions[name] = FieldDecision(ABSENT, [], ["No value extracted; field may not be present on this invoice."])
            continue
        failed = list(sig.validation.get("fails", []))
        # Source-evidence checks are handled by the reviewer; other failed deterministic rules flag review.
        important = [r for r in failed if r != "source_evidence_consistent"]
        if important:
            msgs = [m for r, m in zip(sig.validation.get("fails", []), sig.validation.get("messages", [])) if r in important]
            decisions[name] = FieldDecision(REVIEW, [f"RULE:{r}" for r in important], msgs or ["Deterministic validation requires review."])
            review_fields.append(name)
            if name in s.required_fields or any(r not in ("vendor_customer_address_distinct",) for r in important): critical.append(name)
        else:
            decisions[name] = FieldDecision(ACCEPT, [], ["No deterministic validation failure; verify against the original invoice before approval."])
    doc_decision = REVIEW if review_fields else ACCEPT
    reasons = ([f"Manual review required for: {', '.join(dict.fromkeys(review_fields))}."] if review_fields else ["Required fields were extracted and deterministic checks passed; human approval is still required."])
    # Conservative document-type warning: do not silently treat order worksheets or cancellation records as ordinary invoices.
    heading_text = ocr.full_text[:6000].casefold()
    non_invoice_markers = ("order worksheet", "order printout", "advertising order", "purchase order", "credit memo", "credit note", "statement of account")
    if any(marker in heading_text for marker in non_invoice_markers):
        doc_decision = REVIEW
        reasons.append("Document-type check: this file appears to be an order worksheet, purchase order, credit document, or statement rather than a standard invoice. Confirm the document type before approval.")
    routing = RoutingResult(fields=decisions, doc_decision=doc_decision, doc_reasons=reasons, review_fields=review_fields)
    evaluated = Evaluated(attempt=attempt, signals=signals, rules=rules, routing=routing, values=values, role="initial")
    return CoreResult("completed", history=[evaluated], final=evaluated, seconds=time.time()-t0)
