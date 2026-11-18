"""Shared test helpers. NOTE: the fake OCR/LLM objects below are used ONLY for isolated unit / local-integration tests;
they are never used by the application at runtime and prove nothing about real OCR or real Groq behaviour."""
import io, json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Settings
from src.extraction.llm_client import GroqLLM
from src.extraction.schema import FIELD_NAMES
from src.ocr.models import OCRDocument
from src.ocr.serialize import ocr_from_export

def make_export(rows, page_w=1.0):
    """rows: list of (label, value, conf) -> DocTR-format export dict; each row is one visual line (label, gap, value)."""
    lines, y = [], 0.05
    for label, value, conf in rows:
        words, x = [], 0.05
        for w in str(label).split():
            words.append(dict(value=w, confidence=0.99, geometry=[[x, y], [x + 0.012 * len(w), y + 0.012]])); x += 0.012 * len(w) + 0.01
        if value:
            x += 0.05
            for w in str(value).split():
                words.append(dict(value=w, confidence=conf, geometry=[[x, y], [x + 0.012 * len(w), y + 0.012]])); x += 0.012 * len(w) + 0.01
        lines.append(dict(geometry=[[0, y], [1, y + 0.012]], words=words)); y += 0.04
    return dict(pages=[dict(page_idx=0, dimensions=[1000, 800], blocks=[dict(geometry=[[0, 0], [1, 1]], lines=lines)])])

GOOD_ROWS = [("From:", "Acme Tools Ltd", 0.98), ("Invoice No:", "INV-2041", 0.99), ("Invoice Date:", "03/25/2024", 0.97), ("Due Date:", "04/24/2024", 0.97),
             ("Bill To:", "Contoso Pharma", 0.98), ("Subtotal:", "$1,000.00", 0.99), ("Tax:", "$100.00", 0.99), ("Total:", "$1,100.00", 0.99), ("Amount Due:", "$1,100.00", 0.99)]
def good_ocr(rows=None) -> OCRDocument: return ocr_from_export(make_export(rows or GOOD_ROWS), "test")

def line_id(ocr, text_part):
    return next(l.id for l in ocr.lines if text_part in l.text)

def good_llm_fields(ocr=None):
    ocr = ocr or good_ocr(); L = lambda t: [line_id(ocr, t)]
    vals = dict(vendor_name=("Acme Tools Ltd", "From:"), document_id=("INV-2041", "Invoice No:"), date_issue=("03/25/2024", "Invoice Date:"), date_due=("04/24/2024", "Due Date:"),
                customer_name=("Contoso Pharma", "Bill To:"), amount_total_net=("$1,000.00", "Subtotal:"), amount_total_tax=("$100.00", "Tax:"),
                amount_total_gross=("$1,100.00", "Total:"), amount_due=("$1,100.00", "Amount Due:"))
    return {n: ({"value": vals[n][0], "source_lines": L(vals[n][1])} if n in vals else {"value": None, "source_lines": []}) for n in FIELD_NAMES}

class _Msg:
    def __init__(s, c): s.content = c
class _Ch:
    def __init__(s, c): s.message = _Msg(c); s.finish_reason = "stop"
class _U: prompt_tokens = 1000; completion_tokens = 200
class _R:
    def __init__(s, c): s.choices = [_Ch(c)]; s.usage = _U()

class FakeClient:
    """script: list of items consumed per call; item = dict/str (returned as content) | Exception (raised) | callable(kwargs)->item. Records every call."""
    def __init__(self, script): self.script, self.calls = list(script), []; self.chat = self; self.completions = self
    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if callable(item): item = item(kw)
        if isinstance(item, Exception): raise item
        return _R(item if isinstance(item, str) else json.dumps(item))

class StatusError(Exception):
    def __init__(self, msg, status_code): super().__init__(msg); self.status_code = status_code

@pytest.fixture
def settings(tmp_path):
    s = Settings(groq_api_key="test-key-not-real", app_data_dir=tmp_path / "app_data", llm_max_api_retries=3, max_retries=1, sc_samples=0)
    s.ensure_dirs(); return s

@pytest.fixture
def llm_factory(settings):
    def make(script): c = FakeClient(script); return GroqLLM(settings, client=c, sleep=lambda x: None), c
    return make

def png_bytes(color=(255, 255, 255)):
    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", (64, 64), color).save(b, "PNG"); return b.getvalue()

class FakeOCREngine:
    """Returns a fixed OCRDocument (isolated tests only)."""
    def __init__(self, ocr=None, error=None): self.ocr, self.error, self.calls = ocr, error, 0
    def run(self, path, max_pages):
        self.calls += 1
        if self.error: raise self.error
        return self.ocr or good_ocr()
