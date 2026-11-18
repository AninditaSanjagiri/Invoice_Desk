from decimal import Decimal
from src.utils.normalize import parse_amount, norm_amount, date_interps, canon, tok_f1, currency_code, money_tolerance
from src.validation.rules import validate, per_field, PASS, FAIL, NA
from src.reliability.grounding import ground_score

def res(vals, **kw):
    return {(r.rule, tuple(r.fields)): r for r in validate(vals, **kw)}

def test_amount_parsing_formats():
    assert parse_amount("$21,000.00") == Decimal("21000.00") and parse_amount("21.000,00") == Decimal("21000.00")
    assert parse_amount("#21,000.00") == Decimal("21000.00") and parse_amount("1,50") == Decimal("1.50")
    assert parse_amount("1.234.567") == Decimal("1234567") and parse_amount("abc") is None and parse_amount(None) is None
    assert parse_amount("(12.00)") == Decimal("-12.00")

def test_dates_and_canon():
    assert date_interps("25 March 1999") & date_interps("03/25/99")
    assert not (date_interps("03/25/99") & date_interps("04/25/99")) and not date_interps("13/45/2020")
    assert canon("amount", "$1,000.5") == "1000.50" and tok_f1("Acme Tools", "ACME tools ltd") > 0.7

def test_compact_month_name_dates_and_calendar_validation():
    # Compact OCR format used by the advertising worksheet.
    assert date_interps("SEP30/20") == {(20, 9, 30)}
    assert date_interps("SEP30/20", full=True) == {(2020, 9, 30)}
    assert date_interps("Sep 30, 2020", full=True) == {(2020, 9, 30)}
    assert ground_score("date", "SEP30/20", "Document date SEP30/20") == 1.0
    assert date_interps("FEB31/20") == set()
    assert date_interps("02/29/2020", full=True) == {(2020, 2, 29)}
    assert date_interps("02/29/2019", full=True) == set()


def test_missing_information_is_not_applicable_not_fail():
    r = res({"vendor_name": "A"})
    assert r[("net_plus_tax_eq_gross", ("amount_total_net", "amount_total_tax", "amount_total_gross"))].status == NA
    assert r[("due_after_issue", ("date_issue", "date_due"))].status == NA
    assert r[("amount_due_vs_total", ("amount_due", "amount_total_gross", "amount_paid"))].status == NA

def test_net_tax_gross_pass_fail_and_tolerance():
    ok = res({"amount_total_net": "$1,000.00", "amount_total_tax": "$100.00", "amount_total_gross": "$1,100.00"})
    assert ok[("net_plus_tax_eq_gross", ("amount_total_net", "amount_total_tax", "amount_total_gross"))].status == PASS
    near = res({"amount_total_net": "1000.00", "amount_total_tax": "100.01", "amount_total_gross": "1100.00"})
    assert near[("net_plus_tax_eq_gross", ("amount_total_net", "amount_total_tax", "amount_total_gross"))].status == PASS   # within rounding tolerance
    bad = res({"amount_total_net": "1000.00", "amount_total_tax": "100.00", "amount_total_gross": "1500.00"})
    r = bad[("net_plus_tax_eq_gross", ("amount_total_net", "amount_total_tax", "amount_total_gross"))]
    assert r.status == FAIL and "1,100.00" in r.message and set(per_field(list(bad.values()))["amount_total_tax"]["fails"]) == {"net_plus_tax_eq_gross"}

def test_tolerance_is_currency_aware_minor_unit_floor():
    assert money_tolerance(Decimal(0), "USD") == Decimal("0.02") and money_tolerance(Decimal(0), "JPY") == Decimal(2)      # JPY has no minor unit below 1
    assert money_tolerance(Decimal(1000), "USD") == Decimal("5.02") and currency_code("€ 5") == "EUR"

def test_due_vs_total_with_and_without_payments():
    k = ("amount_due_vs_total", ("amount_due", "amount_total_gross", "amount_paid"))
    assert res({"amount_due": "1100", "amount_total_gross": "1100"})[k].status == PASS
    assert res({"amount_due": "1500", "amount_total_gross": "1100"})[k].status == FAIL
    assert res({"amount_due": "600", "amount_total_gross": "1100", "amount_paid": "500"})[k].status == PASS
    assert res({"amount_due": "1100", "amount_total_gross": "1100", "amount_paid": "500"})[k].status == FAIL

def test_dates_order_vendor_customer_and_required():
    assert res({"date_issue": "03/25/2024", "date_due": "04/24/2024"})[("due_after_issue", ("date_issue", "date_due"))].status == PASS
    assert res({"date_issue": "03/25/2024", "date_due": "01/20/2024"})[("due_after_issue", ("date_issue", "date_due"))].status == FAIL   # unambiguous (D/M reading is impossible)
    assert res({"date_issue": "03/25/2024", "date_due": "01/10/2024"})[("due_after_issue", ("date_issue", "date_due"))].status == PASS   # ambiguous date: rule does not flag it
    assert res({"vendor_name": "Acme Ltd", "customer_name": "Acme Ltd"})[("vendor_customer_distinct", ("vendor_name", "customer_name"))].status == FAIL
    assert res({"vendor_name": "Acme Ltd", "customer_name": "Contoso"})[("vendor_customer_distinct", ("vendor_name", "customer_name"))].status == PASS
    r = res({}, required=("document_id",)); assert r[("required_field_present", ("document_id",))].status == FAIL
    assert res({"document_id": "1"}, required=("document_id",))[("required_field_present", ("document_id",))].status == PASS

def test_parse_failures_and_currency_mismatch():
    r = res({"amount_total_gross": "abc", "date_issue": "99/99/9999"})
    assert r[("amount_parse", ("amount_total_gross",))].status == FAIL and r[("date_parse", ("date_issue",))].status == FAIL
    c = [x for x in validate({"currency": "USD", "amount_total_gross": "€100.00"}) if x.rule == "currency_consistent"]
    assert c and c[0].status == FAIL

def test_evidence_rule_uses_signal2_inputs():
    ev = {"vendor_name": dict(has_citation=True, cited_exist=True, g_cited=0.4), "document_id": dict(has_citation=False, cited_exist=True, g_cited=None)}
    r = res({"vendor_name": "A", "document_id": "1"}, evidence=ev)
    assert r[("source_evidence_consistent", ("vendor_name",))].status == FAIL and r[("source_evidence_consistent", ("document_id",))].status == NA


def test_dates_embedded_in_header_text_and_common_formats():
    # Header timestamps commonly append a time after the date in OCR output.
    assert date_interps("SEP30/20 13.08") == {(20, 9, 30)}
    assert date_interps("Date SEP30/20 13:08", full=True) == {(2020, 9, 30)}
    assert date_interps("30-Sep-20", full=True) == {(2020, 9, 30)}
    assert date_interps("2020-09-30", full=True) == {(2020, 9, 30)}
    assert date_interps("2020/9/30", full=True) == {(2020, 9, 30)}
    assert date_interps("Invoice date: 09/30/2020", full=True) == {(2020, 9, 30)}
    assert date_interps("Invoice date: 30/09/2020", full=True) == {(2020, 9, 30)}
    assert date_interps("FEB 29 2024", full=True) == {(2024, 2, 29)}
    assert date_interps("FEB 29 2023", full=True) == set()


def test_date_parser_does_not_accept_time_as_a_date():
    assert date_interps("13:08") == set()
    assert date_interps("13.08") == set()
