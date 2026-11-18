"""Parsing / normalisation helpers shared by validation, signals and evaluation."""
from __future__ import annotations
import re
from datetime import date as calendar_date
from decimal import Decimal, InvalidOperation

def norm_text(s) -> str: return re.sub(r"[^\w]+", " ", str(s).casefold()).strip()
def strict_key(s) -> str: return re.sub(r"\s+", " ", str(s).strip().casefold())

def parse_amount(s) -> Decimal | None:
    """'$21,000.00' -> Decimal('21000.00'); '21.000,00' -> Decimal('21000.00'); None if unparsable."""
    if s is None: return None
    raw = str(s); t = re.sub(r"[^\d.,\-]", "", raw)
    if not re.search(r"\d", t): return None
    neg = t.startswith("-") or ("(" in raw and ")" in raw); t = t.replace("-", "")
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif "," in t:
        t = t.replace(",", ".") if (t.count(",") == 1 and re.search(r",\d{1,2}$", t)) else t.replace(",", "")
    elif t.count(".") > 1: t = t.replace(".", "")
    try: v = Decimal(t)
    except InvalidOperation: return None
    return -v if neg else v

def norm_amount(s) -> str | None:
    v = parse_amount(s); return None if v is None else f"{v:.2f}"

MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
_MONTH_NAMES = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

def _year_value(raw_year: str, full: bool) -> int:
    y = int(raw_year)
    if len(raw_year) >= 4:
        return y if full else y % 100
    expanded = 2000 + y if y <= 69 else 1900 + y
    return expanded if full else expanded % 100

def _valid_date_tuple(year: int, month: int, day: int, full: bool) -> bool:
    # Validation uses a four-digit year, even when the returned tuple uses a 2-digit year.
    y = year if full or year >= 100 else (2000 + year if year <= 69 else 1900 + year)
    try:
        calendar_date(y, month, day)
        return True
    except ValueError:
        return False

def date_interps(s, full: bool = False) -> set:
    """Return valid possible (year, month, day) readings found in a date string.

    Handles common numeric and month-name formats, compact OCR dates (SEP30/20),
    and date values embedded in surrounding header/time text. Ambiguous numeric
    dates preserve both valid interpretations rather than guessing locale.
    """
    text = str(s or "").strip().lower()
    if not text:
        return set()
    out: set[tuple[int, int, int]] = set()

    def add(raw_year: str, month: int, day: int) -> None:
        year = _year_value(raw_year, full)
        if _valid_date_tuple(year, month, day, full):
            out.add((year, month, day))

    month_pat = "|".join(sorted(_MONTH_NAMES, key=len, reverse=True))

    # Month-first: SEP30/20, Sep 30, 2020, September 30 2020.
    month_first = re.compile(
        rf"(?<![a-z])(?P<month>{month_pat})\.?\s*(?P<day>\d{{1,2}})"
        rf"(?:\s*[,./-]\s*|\s+)(?P<year>\d{{2,4}})(?!\d)", re.I
    )
    for m in month_first.finditer(text):
        add(m.group("year"), _MONTH_NAMES[m.group("month").lower()], int(m.group("day")))

    # Day-first with month name: 30 Sep 20, 30-September-2020.
    day_first = re.compile(
        rf"(?<!\d)(?P<day>\d{{1,2}})[\s,./-]+(?P<month>{month_pat})\.?[\s,./-]*(?P<year>\d{{2,4}})(?!\d)", re.I
    )
    for m in day_first.finditer(text):
        add(m.group("year"), _MONTH_NAMES[m.group("month").lower()], int(m.group("day")))

    # ISO / year-first: 2020-09-30, 2020/9/30.
    iso = re.compile(r"(?<!\d)(?P<year>\d{4})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})(?!\d)")
    for m in iso.finditer(text):
        add(m.group("year"), int(m.group("month")), int(m.group("day")))

    # Numeric day/month/year or month/day/year. Keep both interpretations if valid.
    numeric = re.compile(r"(?<!\d)(?P<a>\d{1,2})[-/.](?P<b>\d{1,2})[-/.](?P<year>\d{2,4})(?!\d)")
    for m in numeric.finditer(text):
        a, b, year = int(m.group("a")), int(m.group("b")), m.group("year")
        add(year, a, b)  # month/day/year
        add(year, b, a)  # day/month/year

    return out

def canon(kind: str, val) -> str | None:
    """Canonical form for equality checks, by field kind."""
    if val is None: return None
    if kind == "amount": return norm_amount(val) or norm_text(val)
    if kind == "date":
        it = date_interps(val); return "|".join(sorted(f"{y:02d}-{m:02d}-{d:02d}" for y, m, d in it)) if it else norm_text(val)
    return norm_text(val)

def tok_f1(p, g) -> float:
    p, g = norm_text(p).split(), norm_text(g).split()
    if not p or not g: return 0.0
    c = sum(min(p.count(w), g.count(w)) for w in set(p))
    if not c: return 0.0
    pr, rc = c / len(p), c / len(g); return 2 * pr * rc / (pr + rc)

CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}
CURRENCY_DECIMALS = {"JPY": 0, "KRW": 0}
def currency_code(value) -> str | None:
    if not value: return None
    v = str(value).strip()
    for sym, code in CURRENCY_SYMBOLS.items():
        if sym in v: return code
    m = re.search(r"\b([A-Za-z]{3})\b", v)
    return m.group(1).upper() if m else None

def money_tolerance(amount: Decimal, currency: str | None) -> Decimal:
    """Currency-aware tolerance: two minor units plus 0.5% of the amount (rounded line taxes)."""
    minor = Decimal(1) if CURRENCY_DECIMALS.get(currency or "", 2) == 0 else Decimal("0.01")
    return minor * 2 + abs(amount) * Decimal("0.005")
