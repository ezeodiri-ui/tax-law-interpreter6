"""
Model-agnostic answer generation.

Select the model with environment variables only; no code changes:

  LLM_PROVIDER = groq | gemini | openrouter | ollama | openai_compat | anthropic
  LLM_MODEL    = model name as the provider spells it (overrides the preset default)
  LLM_API_KEY  = key for that provider (or the provider's own variable, e.g. GROQ_API_KEY)
  LLM_BASE_URL = only for openai_compat / self-hosted servers (vLLM, LM Studio, ...)

Every provider except 'anthropic' is reached through the OpenAI-compatible
/chat/completions protocol, so any new model or host that speaks it works immediately.
"""
from __future__ import annotations
import json
import os
import httpx
from .prompts import SYSTEM, build_user

# Defaults are starting points. Providers rename and retire free models often:
# check the provider's model list and set LLM_MODEL if a default stops working.
PRESETS = {
    "groq":       dict(base="https://api.groq.com/openai/v1", model="llama-3.3-70b-versatile", key_env="GROQ_API_KEY"),
    "gemini":     dict(base="https://generativelanguage.googleapis.com/v1beta/openai", model="gemini-2.0-flash", key_env="GEMINI_API_KEY"),
    "openrouter": dict(base="https://openrouter.ai/api/v1", model="meta-llama/llama-3.3-70b-instruct:free", key_env="OPENROUTER_API_KEY"),
    "ollama":     dict(base="http://localhost:11434/v1", model="qwen2.5:7b-instruct", key_env=None),
    "openai_compat": dict(base="", model="", key_env="LLM_API_KEY"),
}


def _cfg() -> dict | None:
    prov = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not prov:  # backwards compatible: a bare Anthropic key still works
        prov = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else ""
    if not prov:
        return None
    if prov == "anthropic":
        key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        return dict(provider=prov, model=os.getenv("LLM_MODEL") or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"), key=key) if key else None
    p = PRESETS.get(prov)
    if p is None:
        return None
    key = os.getenv("LLM_API_KEY") or (os.getenv(p["key_env"]) if p["key_env"] else "ollama")
    base = os.getenv("LLM_BASE_URL") or p["base"]
    model = os.getenv("LLM_MODEL") or p["model"]
    if not (key and base and model):
        return None
    return dict(provider=prov, model=model, key=key, base=base.rstrip("/"))


def available() -> bool:
    return _cfg() is not None


def describe() -> str:
    c = _cfg()
    return f"{c['provider']}:{c['model']}" if c else ""


def list_models() -> list[str]:
    """Ask the provider which models this key can use (OpenAI-compatible /models)."""
    c = _cfg()
    if c is None or c["provider"] == "anthropic":
        return []
    r = httpx.get(f"{c['base']}/models", headers={"Authorization": f"Bearer {c['key']}"}, timeout=30)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []))


def _messages(query, sources, history):
    return [{"role": "system", "content": SYSTEM}] + list(history or []) + [{"role": "user", "content": build_user(query, sources)}]


def _stream_openai_compat(c, query, sources, history):
    body = {"model": c["model"], "messages": _messages(query, sources, history), "temperature": 0, "stream": True, "max_tokens": 1800}
    headers = {"Authorization": f"Bearer {c['key']}", "Content-Type": "application/json"}
    with httpx.stream("POST", f"{c['base']}/chat/completions", json=body, headers=headers, timeout=120) as r:
        if r.status_code >= 400:
            r.read()
            msg = f"{c['provider']} returned {r.status_code}: {r.text[:300]}"
            if r.status_code in (400, 404) and "model" in r.text.lower():
                try:
                    msg += " | Models available to your key: " + ", ".join(list_models()[:40]) + \
                           " | Set LLM_MODEL to one of these chat models and restart."
                except Exception:
                    pass
            raise RuntimeError(msg)
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
            if delta:
                yield delta


def _stream_anthropic(c, query, sources, history):
    import anthropic
    client = anthropic.Anthropic(api_key=c["key"])
    msgs = list(history or []) + [{"role": "user", "content": build_user(query, sources)}]
    with client.messages.stream(model=c["model"], max_tokens=1800, temperature=0, system=SYSTEM, messages=msgs) as s:
        for t in s.text_stream:
            yield t


def stream_answer(query: str, sources: list[dict], history: list[dict] | None = None):
    c = _cfg()
    if c is None:
        raise RuntimeError("No LLM configured")
    fn = _stream_anthropic if c["provider"] == "anthropic" else _stream_openai_compat
    yield from fn(c, query, sources, history)
