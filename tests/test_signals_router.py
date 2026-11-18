from conftest import good_ocr, good_llm_fields, line_id, make_export
from src.config import Settings
from src.extraction.schema import FIELD_NAMES, FieldOut, parse_extraction
from src.ocr.serialize import ocr_from_export
from src.reliability.signals import ocr_signal, llm_signal, assess_fields, evidence_for_rules
from src.routing.router import route_fields, compare_attempts, ACCEPT, RETRY, REVIEW, ABSENT
from src.services.core import evaluate_attempt
from src.extraction.extractor import Attempt

S = Settings(groq_api_key="x", sc_samples=0)

def fields_from(d): return parse_extraction(d)

def test_ocr_signal_is_per_field_not_document_mean():
    ocr = good_ocr([("Invoice No:", "INV-1", 0.99), ("Total:", "$5.00", 0.40), ("From:", "Acme", 0.99)])
    fo = FieldOut(value="$5.00", source_lines=[line_id(ocr, "Total:")])
    sig = ocr_signal("amount_total_gross", fo, ocr, S)
    assert sig.status == "LOW" and abs(sig.conf_mean - 0.40) < 1e-6 and ocr.mean_conf > 0.7        # document mean would have hidden the low field
    hi = ocr_signal("document_id", FieldOut(value="INV-1", source_lines=[line_id(ocr, "Invoice No:")]), ocr, S)
    assert hi.status == "HIGH" and hi.n_words >= 1 and hi.bbox is not None

def test_ocr_signal_unavailable_is_distinct_from_low():
    ocr = good_ocr(); sig = ocr_signal("vendor_name", FieldOut(value="Totally Different Corp", source_lines=[]), ocr, S)
    assert sig.status == "UNAVAILABLE" and sig.conf_mean is None and sig.located is False
    assert ocr_signal("vendor_name", FieldOut(value=None), ocr, S) is None

def test_llm_signal_statuses():
    ocr = good_ocr(); L = line_id
    assert llm_signal("document_id", FieldOut(value="INV-2041", source_lines=[L(ocr, "Invoice No:")]), ocr, [], S).status == "SUPPORTED"
    assert llm_signal("document_id", FieldOut(value="INV-2041", source_lines=[]), ocr, [], S).status == "UNCITED_FOUND"
    assert llm_signal("document_id", FieldOut(value="INV-2041", source_lines=["L999"]), ocr, [], S).status == "INVALID_CITATION"
    assert llm_signal("document_id", FieldOut(value="INV-2041", source_lines=[L(ocr, "Tax:")]), ocr, [], S).status == "MISCITED"
    assert llm_signal("document_id", FieldOut(value="INV-9999", source_lines=[L(ocr, "Tax:")]), ocr, [], S).status == "UNSUPPORTED"
    assert llm_signal("amount_total_gross", FieldOut(value="1100", source_lines=[L(ocr, "Total:")]), ocr, [], S).status == "SUPPORTED"      # normalised match

def test_self_consistency_is_computed_from_independent_samples():
    ocr = good_ocr(); fo = FieldOut(value="INV-2041", source_lines=[line_id(ocr, "Invoice No:")])
    other = {n: FieldOut(value=None) for n in FIELD_NAMES}; same = dict(other, document_id=FieldOut(value="INV-2041")); diff = dict(other, document_id=FieldOut(value="INV-2040"))
    sig = llm_signal("document_id", fo, ocr, [same, same, diff], S)
    assert abs(sig.sc_agree - 2 / 3) < 1e-9 and sig.n_samples == 3 and llm_signal("document_id", fo, ocr, [], S).sc_agree is None

def _eval(d, ocr=None, remaining=1, s=S):
    ocr = ocr or good_ocr(); f = fields_from(d); att = Attempt("initial", "ok", f)
    return evaluate_attempt(att, ocr, [], s, remaining), ocr

def test_clean_extraction_is_accepted():
    ev, _ = _eval(good_llm_fields())
    assert ev.routing.doc_decision == ACCEPT and ev.routing.fields["vendor_name"].decision == ACCEPT and ev.routing.fields["vendor_email"].decision == ABSENT

def test_valid_json_alone_is_not_enough_unsupported_value_goes_to_retry_then_review():
    d = good_llm_fields(); d["vendor_name"] = {"value": "Invented Vendor GmbH", "source_lines": ["L001"]}
    ev, _ = _eval(d, remaining=1); assert ev.routing.fields["vendor_name"].decision == RETRY and ev.routing.doc_decision == RETRY and "vendor_name" in ev.routing.retry_fields
    ev0, _ = _eval(d, remaining=0); assert ev0.routing.fields["vendor_name"].decision == REVIEW and ev0.routing.doc_decision == REVIEW and "Retry limit reached." in ev0.routing.fields["vendor_name"].reasons

def test_validation_failure_is_attributed_and_repairable():
    d = good_llm_fields(); d["amount_total_gross"] = {"value": "$1,500.00", "source_lines": d["amount_total_gross"]["source_lines"]}     # not in OCR AND breaks net+tax=gross
    ev, _ = _eval(d); r = ev.routing.fields
    assert r["amount_total_gross"].decision == RETRY and any(c.startswith("RULE:net_plus_tax") for c in r["amount_total_gross"].issues)
    assert any(c.startswith("RULE:net_plus_tax") for c in r["amount_total_tax"].issues)      # attributable to every field the rule affects

def test_low_ocr_quality_goes_straight_to_review_even_if_value_is_grounded():
    ocr = good_ocr([("Invoice No:", "INV-2041", 0.30), ("From:", "Acme Tools Ltd", 0.99), ("Total:", "$1,100.00", 0.99)]); L = lambda t: [line_id(ocr, t)]
    d = {n: {"value": None, "source_lines": []} for n in FIELD_NAMES}
    d.update(document_id={"value": "INV-2041", "source_lines": L("Invoice No:")}, vendor_name={"value": "Acme Tools Ltd", "source_lines": L("From:")}, amount_total_gross={"value": "$1,100.00", "source_lines": L("Total:")})
    ev, _ = _eval(d, ocr, remaining=1); f = ev.routing.fields
    assert f["document_id"].decision == REVIEW and "LOW_OCR" in f["document_id"].issues and f["vendor_name"].decision != REVIEW
    assert ev.routing.doc_decision == REVIEW           # document_id is a required (critical) field

def test_required_field_missing_retries_once_then_reviews():
    d = good_llm_fields(); d["document_id"] = {"value": None, "source_lines": []}
    ev, _ = _eval(d, remaining=1); assert ev.routing.fields["document_id"].decision == RETRY
    ev0, _ = _eval(d, remaining=0); assert ev0.routing.fields["document_id"].decision == REVIEW and ev0.routing.doc_decision == REVIEW

def test_noncritical_field_problem_does_not_force_other_fields_into_review():
    ocr = good_ocr(good_ocr().lines and [("From:", "Acme Tools Ltd", 0.98), ("Invoice No:", "INV-2041", 0.99), ("Invoice Date:", "03/25/2024", 0.97), ("Total:", "$1,100.00", 0.99), ("Address:", "12 Main Street Springfield", 0.45)])
    L = lambda t: [line_id(ocr, t)]; d = {n: {"value": None, "source_lines": []} for n in FIELD_NAMES}
    d.update(vendor_name={"value": "Acme Tools Ltd", "source_lines": L("From:")}, document_id={"value": "INV-2041", "source_lines": L("Invoice No:")}, date_issue={"value": "03/25/2024", "source_lines": L("Invoice Date:")},
             amount_total_gross={"value": "$1,100.00", "source_lines": L("Total:")}, vendor_address={"value": "12 Main Street Springfield", "source_lines": L("Address:")})
    ev, _ = _eval(d, ocr, remaining=0); f = ev.routing.fields
    assert f["vendor_address"].decision == REVIEW and f["vendor_name"].decision == ACCEPT and f["amount_total_gross"].decision == ACCEPT
    assert ev.routing.doc_decision == REVIEW and "vendor_address" in ev.routing.review_fields and "critical" not in " ".join(ev.routing.doc_reasons).lower()

def test_self_consistency_disagreement_triggers_retry_when_enabled():
    s = Settings(groq_api_key="x", sc_samples=2); ocr = good_ocr(); f = fields_from(good_llm_fields(ocr)); bad = dict(f, vendor_name=FieldOut(value="Acme Tool"))
    ev = evaluate_attempt(Attempt("initial", "ok", f), ocr, [bad, bad], s, 1)
    assert ev.routing.fields["vendor_name"].decision == RETRY and "SC_DISAGREE" in ev.routing.fields["vendor_name"].issues and ev.routing.fields["document_id"].decision == ACCEPT

def test_compare_attempts_improved_regressed_none():
    d = good_llm_fields(); d["vendor_name"] = {"value": "Invented", "source_lines": ["L001"]}
    before, _ = _eval(d); fixed, _ = _eval(good_llm_fields())
    assert compare_attempts(before.routing, before.values, fixed.routing, fixed.values) == "improved"
    assert compare_attempts(fixed.routing, fixed.values, before.routing, before.values) == "regressed"
    assert compare_attempts(before.routing, before.values, before.routing, before.values) == "no_improvement"
