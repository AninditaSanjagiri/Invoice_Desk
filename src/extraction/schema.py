"""Field catalogue, strict JSON schema and Pydantic models for LLM output."""
from __future__ import annotations
from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError, create_model

@dataclass(frozen=True)
class FieldSpec:
    name: str; label: str; kind: str; definition: str; docile: str | None     # kind: id|date|amount|text|address|email|currency|terms

FIELDS: list[FieldSpec] = [
    FieldSpec("document_id", "Invoice / document no.", "id", "the invoice / document number of THIS document (not a customer number, PO number or account number)", "document_id"),
    FieldSpec("vendor_name", "Vendor name", "text", "the full name of the company that issued the invoice, as printed (keep legal suffixes such as Inc., Ltd, GmbH)", "vendor_name"),
    FieldSpec("vendor_address", "Vendor address", "address", "the complete postal address (all lines, in order) of the vendor", "vendor_address"),
    FieldSpec("vendor_email", "Vendor email", "email", "the vendor's e-mail address", "vendor_email"),
    FieldSpec("vendor_tax_id", "Vendor tax ID", "id", "the vendor's tax / VAT / GST identification number", "vendor_tax_id"),
    FieldSpec("customer_name", "Customer name", "text", "the name of the party being billed (the recipient/buyer), not the vendor", "customer_billing_name"),
    FieldSpec("customer_address", "Customer address", "address", "the complete postal address of the billed customer", "customer_billing_address"),
    FieldSpec("date_issue", "Issue date", "date", "the date the invoice was issued", "date_issue"),
    FieldSpec("date_due", "Due date", "date", "the payment due date, only if one is explicitly stated", "date_due"),
    FieldSpec("po_number", "Purchase order no.", "id", "the customer's purchase order number", "customer_order_id"),
    FieldSpec("currency", "Currency", "currency", "the currency of the amounts, as printed (symbol or ISO code)", "currency_code_amount_due"),
    FieldSpec("amount_total_net", "Net amount", "amount", "the total before tax, only if explicitly shown as net / subtotal; do not repeat the gross total as net when no net or subtotal is printed", "amount_total_net"),
    FieldSpec("amount_total_tax", "Tax amount", "amount", "the total tax amount, only if explicitly shown", "amount_total_tax"),
    FieldSpec("amount_total_gross", "Gross / total", "amount", "the total of the invoice including tax", "amount_total_gross"),
    FieldSpec("amount_due", "Amount due", "amount", "the amount still to be paid. If one printed figure is both the total and the amount due, return it for both fields", "amount_due"),
    FieldSpec("amount_paid", "Amount paid", "amount", "the amount already paid, only if explicitly shown", "amount_paid"),
    FieldSpec("payment_reference", "Payment reference", "id", "the payment reference / remittance reference to quote when paying", "payment_reference"),
    FieldSpec("payment_terms", "Payment terms", "terms", "the payment terms as printed (e.g. 'Net 30')", "payment_terms"),
]
FIELD_BY_NAME = {f.name: f for f in FIELDS}
FIELD_NAMES = [f.name for f in FIELDS]
KIND = {f.name: f.kind for f in FIELDS}
DOCILE_TO_APP = {f.docile: f.name for f in FIELDS if f.docile}

def json_schema(fields: list[str] = FIELD_NAMES) -> dict:
    item = {"type": "object", "properties": {"value": {"type": ["string", "null"]}, "source_lines": {"type": "array", "items": {"type": "string"}}},
            "required": ["value", "source_lines"], "additionalProperties": False}
    return {"type": "object", "properties": {f: item for f in fields}, "required": list(fields), "additionalProperties": False}

class FieldOut(BaseModel):
    value: str | None = None
    source_lines: list[str] = Field(default_factory=list)

def build_output_model(fields: list[str] = FIELD_NAMES):
    return create_model("InvoiceExtraction", **{f: (FieldOut, FieldOut()) for f in fields})

InvoiceExtraction = build_output_model()

class ParseError(ValueError): pass

def parse_extraction(parsed: dict, fields: list[str] = FIELD_NAMES) -> dict[str, FieldOut]:
    """Validate against the Pydantic schema; empty strings become null; unknown keys are ignored."""
    if not isinstance(parsed, dict): raise ParseError("LLM output is not a JSON object")
    try: model = build_output_model(fields).model_validate({f: parsed.get(f, {}) for f in fields})
    except ValidationError as e: raise ParseError(f"LLM output failed schema validation: {str(e)[:300]}") from e
    out = {}
    for f in fields:
        fo = getattr(model, f); v = fo.value.strip() if isinstance(fo.value, str) else fo.value
        out[f] = FieldOut(value=v or None, source_lines=[s.strip() for s in fo.source_lines if s and s.strip()] if v else [])
    return out
