"""Environment-driven settings (no secrets in source code)."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

def _b(name, default): return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")
def _i(name, default): return int(os.getenv(name, default))
def _f(name, default): return float(os.getenv(name, default))
def _model_name():
    explicit = os.getenv("LLM_MODEL", "").strip()
    if explicit: return explicit
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "groq": return os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

@dataclass
class Settings:
    # Select the LLM provider with LLM_PROVIDER=gemini or groq.
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").strip().lower())
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""), repr=False)
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""), repr=False)
    model: str = field(default_factory=_model_name)
    reasoning_effort: str = field(default_factory=lambda: os.getenv("REASONING_EFFORT", "low"))
    max_completion_tokens: int = field(default_factory=lambda: _i("MAX_COMPLETION_TOKENS", 8192))
    llm_timeout_s: float = field(default_factory=lambda: _f("LLM_TIMEOUT_S", 90))
    llm_max_api_retries: int = field(default_factory=lambda: _i("LLM_MAX_API_RETRIES", 5))
    max_prompt_chars: int = field(default_factory=lambda: _i("MAX_PROMPT_CHARS", 30000))
    # reliability / routing
    max_retries: int = field(default_factory=lambda: _i("MAX_RETRIES", 1))
    sc_samples: int = field(default_factory=lambda: _i("SELF_CONSISTENCY_SAMPLES", 0))
    sc_temperature: float = field(default_factory=lambda: _f("SELF_CONSISTENCY_TEMPERATURE", 0.7))
    min_ocr_conf_mean: float = field(default_factory=lambda: _f("MIN_OCR_CONF_MEAN", 0.85))
    min_ocr_conf_min: float = field(default_factory=lambda: _f("MIN_OCR_CONF_MIN", 0.50))
    min_grounding: float = field(default_factory=lambda: _f("MIN_GROUNDING", 0.99))
    min_sc_agreement: float = field(default_factory=lambda: _f("MIN_SC_AGREEMENT", 1.0))
    allow_uncited: bool = field(default_factory=lambda: _b("ALLOW_UNCITED", False))
    required_fields: tuple = field(default_factory=lambda: tuple(x.strip() for x in os.getenv("REQUIRED_FIELDS", "document_id,vendor_name,date_issue,amount_total_gross").split(",") if x.strip()))
    # uploads / storage
    max_file_mb: int = field(default_factory=lambda: _i("MAX_FILE_MB", 20))
    max_pages: int = field(default_factory=lambda: _i("MAX_PAGES", 10))
    app_data_dir: Path = field(default_factory=lambda: ROOT / os.getenv("APP_DATA_DIR", "app_data"))

    @property
    def upload_dir(self) -> Path: return self.app_data_dir / "uploads"
    @property
    def ocr_cache_dir(self) -> Path: return self.app_data_dir / "ocr_cache"
    @property
    def db_path(self) -> Path: return self.app_data_dir / "invoices.db"
    @property
    def has_api_key(self) -> bool:
        provider = self.llm_provider
        return bool(self.gemini_api_key if provider == "gemini" else self.groq_api_key if provider == "groq" else "")

    def ensure_dirs(self):
        for d in (self.app_data_dir, self.upload_dir, self.ocr_cache_dir): d.mkdir(parents=True, exist_ok=True)

    def llm_config_dict(self) -> dict:      # stored with every run (no secrets)
        return dict(provider=self.llm_provider, model=self.model, reasoning_effort=self.reasoning_effort, max_completion_tokens=self.max_completion_tokens,
                    sc_samples=self.sc_samples, sc_temperature=self.sc_temperature, max_prompt_chars=self.max_prompt_chars)

def get_settings() -> Settings:
    s = Settings(); s.ensure_dirs(); return s
