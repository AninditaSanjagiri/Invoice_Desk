"""Groq chat-completions wrapper: structured output, retry/backoff, graceful failure (never raises, never logs prompts or keys)."""
from __future__ import annotations
import json, logging, time
from dataclasses import dataclass, field
from ..config import Settings

log = logging.getLogger(__name__)

@dataclass
class LLMResult:
    status: str                                   # 'ok' | 'failed'
    parsed: dict | None = None
    raw: str | None = None
    usage: dict = field(default_factory=lambda: dict(prompt=0, completion=0))
    latency_s: float = 0.0
    attempts: int = 0
    error: str | None = None
    json_mode: str | None = None
    finish_reason: str | None = None

class GroqLLM:
    def __init__(self, settings: Settings, client=None, sleep=time.sleep):
        self.s, self._client, self._sleep = settings, client, sleep
        self._use_reasoning_effort = True

    def _get_client(self):
        if self._client is None:
            if not self.s.groq_api_key: raise RuntimeError("GROQ_API_KEY is not configured (copy .env.example to .env).")
            from groq import Groq
            self._client = Groq(api_key=self.s.groq_api_key, timeout=self.s.llm_timeout_s, max_retries=0)
        return self._client

    def complete(self, system: str, user: str, schema: dict, temperature: float = 0.0) -> LLMResult:
        t0 = time.time(); usage = dict(prompt=0, completion=0); json_obj = False; last = None
        for attempt in range(1, self.s.llm_max_api_retries + 1):
            try:
                client = self._get_client()
                sys_msg = system + ("\n\nReturn a single JSON object with exactly these keys: " + json.dumps(list(schema["properties"])) if json_obj else "")
                kw = dict(model=self.s.model, temperature=temperature, max_completion_tokens=self.s.max_completion_tokens,
                          messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
                          response_format=({"type": "json_object"} if json_obj else {"type": "json_schema", "json_schema": {"name": "invoice_extraction", "strict": True, "schema": schema}}))
                if self._use_reasoning_effort and self.s.reasoning_effort: kw["reasoning_effort"] = self.s.reasoning_effort
                r = client.chat.completions.create(**kw)
                u = getattr(r, "usage", None); usage["prompt"] += getattr(u, "prompt_tokens", 0) or 0; usage["completion"] += getattr(u, "completion_tokens", 0) or 0
                raw = r.choices[0].message.content
                parsed = json.loads(raw)
                if not isinstance(parsed, dict): raise ValueError("model returned JSON that is not an object")
                return LLMResult("ok", parsed, raw, usage, time.time() - t0, attempt, None, "json_object" if json_obj else "json_schema", getattr(r.choices[0], "finish_reason", None))
            except RuntimeError as e:                                   # missing key: retrying cannot help
                return LLMResult("failed", None, None, usage, time.time() - t0, attempt, str(e))
            except Exception as e:
                name = type(e).__name__; msg = str(e).lower(); last = f"{name}: {str(e)[:200]}".replace(self.s.groq_api_key or "\0", "***")
                status = getattr(e, "status_code", None)
                if status in (401, 403) or "invalid api key" in msg or "authentication" in name.lower():
                    return LLMResult("failed", None, None, usage, time.time() - t0, attempt, "Authentication failed - check GROQ_API_KEY.")
                if "reasoning_effort" in msg and any(w in msg for w in ("unsupported", "invalid", "unknown", "not support")): self._use_reasoning_effort = False; continue
                if "json_validate_failed" in msg or "failed to generate json" in msg or isinstance(e, json.JSONDecodeError) or isinstance(e, ValueError):
                    if attempt >= 2: json_obj = True                    # fall back to JSON-object mode, validated locally by Pydantic
                wait = min(60, 2 ** attempt) + (10 if status == 429 or "rate" in msg else 0)
                log.warning("LLM call failed (attempt %d/%d): %s", attempt, self.s.llm_max_api_retries, name)
                if attempt < self.s.llm_max_api_retries: self._sleep(wait)
        return LLMResult("failed", None, None, usage, time.time() - t0, self.s.llm_max_api_retries, last or "unknown error")


class GeminiLLM:
    """Gemini API wrapper using the current google-genai SDK and JSON-schema output."""
    def __init__(self, settings: Settings, client=None, sleep=time.sleep):
        self.s, self._client, self._sleep = settings, client, sleep

    def _get_client(self):
        if self._client is None:
            if not self.s.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not configured (add it to .env).")
            from google import genai
            self._client = genai.Client(api_key=self.s.gemini_api_key, http_options={"timeout": int(self.s.llm_timeout_s * 1000)})
        return self._client

    @staticmethod
    def _gemini_schema(schema: dict) -> dict:
        """Convert JSON Schema nullable type arrays to Gemini-supported anyOf form."""
        if isinstance(schema, list):
            return [GeminiLLM._gemini_schema(x) for x in schema]
        if not isinstance(schema, dict):
            return schema
        out = {k: GeminiLLM._gemini_schema(v) for k, v in schema.items()}
        typ = out.get("type")
        if isinstance(typ, list) and "null" in typ:
            non_null = [t for t in typ if t != "null"]
            if len(non_null) == 1:
                rest = {k: v for k, v in out.items() if k != "type"}
                out = {"anyOf": [{"type": non_null[0], **rest}, {"type": "null"}]}
        return out

    def complete(self, system: str, user: str, schema: dict, temperature: float = 0.0) -> LLMResult:
        t0 = time.time(); usage = dict(prompt=0, completion=0); last = None
        for attempt in range(1, self.s.llm_max_api_retries + 1):
            try:
                client = self._get_client()
                from google.genai import types
                response = client.models.generate_content(
                    model=self.s.model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                        max_output_tokens=self.s.max_completion_tokens,
                        response_mime_type="application/json",
                        response_json_schema=self._gemini_schema(schema),
                    ),
                )
                meta = getattr(response, "usage_metadata", None)
                usage["prompt"] = getattr(meta, "prompt_token_count", 0) or 0
                usage["completion"] = getattr(meta, "candidates_token_count", 0) or 0
                raw = getattr(response, "text", None)
                if not raw:
                    raise ValueError("Gemini returned an empty response; check quota, safety blocks, or output limits.")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("Gemini returned JSON that is not an object.")
                return LLMResult("ok", parsed, raw, usage, time.time() - t0, attempt, None, "json_schema", None)
            except RuntimeError as e:
                return LLMResult("failed", None, None, usage, time.time() - t0, attempt, str(e))
            except Exception as e:
                name = type(e).__name__; msg = str(e).lower()
                last = f"{name}: {str(e)[:200]}".replace(self.s.gemini_api_key or "\0", "***")
                status = getattr(e, "status_code", None)
                if status in (401, 403) or "api key" in msg and ("invalid" in msg or "not valid" in msg):
                    return LLMResult("failed", None, None, usage, time.time() - t0, attempt, "Gemini authentication failed - check GEMINI_API_KEY.")
                if attempt < self.s.llm_max_api_retries:
                    wait = min(20, 2 ** attempt) + (5 if status == 429 or "quota" in msg or "rate" in msg else 0)
                    log.warning("Gemini call failed (attempt %d/%d): %s", attempt, self.s.llm_max_api_retries, name)
                    self._sleep(wait)
        return LLMResult("failed", None, None, usage, time.time() - t0, self.s.llm_max_api_retries, last or "unknown Gemini API error")


def create_llm(settings: Settings):
    """Build the configured provider. Unknown providers fail clearly instead of silently switching."""
    provider = settings.llm_provider.strip().lower()
    if provider == "gemini":
        return GeminiLLM(settings)
    if provider == "groq":
        return GroqLLM(settings)
    raise ValueError("LLM_PROVIDER must be either 'gemini' or 'groq'.")
