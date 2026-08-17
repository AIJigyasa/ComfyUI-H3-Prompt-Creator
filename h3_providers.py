"""HTTP clients for every LLM backend the H3 nodes can talk to.

Deliberately dependency-free: everything goes through urllib, matching the rest
of this node pack. A ComfyUI custom node that required four vendor SDKs would
force four pip installs into the user's ComfyUI environment.

Each provider gets the same call signature and returns plain text, so the H3
nodes never branch on provider beyond picking a name.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

OLLAMA = "Ollama (Local)"
OPENAI = "OpenAI"
ANTHROPIC = "Anthropic"
OPENROUTER = "OpenRouter"
GEMINI = "Google Gemini"
DETERMINISTIC = "Built-in deterministic"

PROVIDERS = [OLLAMA, OPENAI, ANTHROPIC, OPENROUTER, GEMINI, DETERMINISTIC]

CLOUD_PROVIDERS = {OPENAI, ANTHROPIC, OPENROUTER, GEMINI}

# Env vars are checked when the api_key widget is blank. Preferred: a workflow
# JSON carrying a pasted key can be shared by accident.
ENV_KEYS = {
    OPENAI: ("OPENAI_API_KEY",),
    ANTHROPIC: ("ANTHROPIC_API_KEY",),
    OPENROUTER: ("OPENROUTER_API_KEY",),
    GEMINI: ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

DEFAULT_MODELS = {
    OPENAI: "gpt-4o",
    ANTHROPIC: "claude-opus-5",
    OPENROUTER: "anthropic/claude-sonnet-5",
    GEMINI: "gemini-2.0-flash",
}

# Anthropic models that reject temperature/top_p/top_k with a 400. Sending the
# parameter to any of these fails the request outright.
_ANTHROPIC_NO_SAMPLING = ("claude-opus-5", "claude-sonnet-5", "claude-opus-4-8",
                          "claude-opus-4-7", "claude-fable-5", "claude-mythos-5")


def api_key_for(provider: str, supplied: str) -> str:
    """Widget value wins, else the provider's environment variable."""
    supplied = (supplied or "").strip()
    if supplied:
        return supplied
    for name in ENV_KEYS.get(provider, ()):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def default_model(provider: str) -> str:
    return DEFAULT_MODELS.get(provider, "")


def _post(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_error(provider: str, exc: urllib.error.HTTPError) -> Tuple[str, RuntimeError]:
    detail = exc.read().decode("utf-8", errors="replace")
    hint = ""
    if exc.code in (401, 403):
        hint = " (check the API key)"
    elif exc.code == 404:
        hint = " (check the model name)"
    elif exc.code == 429:
        hint = " (rate limited or out of quota)"
    return detail, RuntimeError(f"{provider} HTTP {exc.code}{hint}: {detail[:500]}")


def _wrap_transport(provider: str, url: str, exc: Exception) -> RuntimeError:
    if isinstance(exc, urllib.error.URLError):
        return RuntimeError(f"Cannot reach {provider} at {url}: {exc.reason}")
    if isinstance(exc, TimeoutError):
        return RuntimeError(f"{provider} request timed out.")
    return RuntimeError(f"{provider} request failed: {exc}")


# ---------------------------------------------------------------------------
# OpenAI-compatible (OpenAI, OpenRouter)
# ---------------------------------------------------------------------------

def _openai_style(
    provider: str, url: str, api_key: str, model: str, system: str, user: str,
    images: List[str], temperature: float, max_tokens: int, timeout: int,
    want_json: bool, extra_headers: Optional[Dict[str, str]] = None,
) -> str:
    content: List[Dict[str, Any]] = [{"type": "text", "text": user}]
    for b64 in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if want_json:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if extra_headers:
        headers.update(extra_headers)

    # Newer OpenAI models renamed max_tokens and reject non-default temperature.
    # Drop whichever parameter the API names in its 400 and retry once each.
    for _ in range(3):
        try:
            obj = _post(url, payload, headers, timeout)
            break
        except urllib.error.HTTPError as exc:
            detail, error = _http_error(provider, exc)
            low = detail.lower()
            if exc.code == 400 and "max_completion_tokens" in low and "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
                continue
            if exc.code == 400 and "temperature" in low and "temperature" in payload:
                payload.pop("temperature")
                continue
            if exc.code == 400 and "response_format" in low and "response_format" in payload:
                payload.pop("response_format")
                continue
            raise error from exc
        except Exception as exc:
            raise _wrap_transport(provider, url, exc) from exc
    else:
        raise RuntimeError(f"{provider}: request rejected after parameter retries.")

    choices = obj.get("choices") or []
    if not choices:
        raise RuntimeError(f"{provider} returned no choices: {json.dumps(obj)[:300]}")
    message = choices[0].get("message") or {}
    text = message.get("content") or ""
    if isinstance(text, list):  # some gateways return content parts
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    if not text and choices[0].get("finish_reason") == "length":
        raise RuntimeError(f"{provider} hit the token limit before producing any text.")
    return text.strip()


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------

def _anthropic(
    api_key: str, model: str, system: str, user: str, images: List[str],
    temperature: float, max_tokens: int, timeout: int,
) -> str:
    content: List[Dict[str, Any]] = []
    for b64 in images:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    content.append({"type": "text", "text": user})

    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "system": system,
        "messages": [{"role": "user", "content": content}],
    }
    # temperature/top_p/top_k return 400 on current Claude models, and thinking
    # is on by default there (max_tokens covers thinking + text, so it needs
    # headroom rather than a tight cap).
    if not any(name in model for name in _ANTHROPIC_NO_SAMPLING):
        payload["temperature"] = float(temperature)

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    url = "https://api.anthropic.com/v1/messages"
    try:
        obj = _post(url, payload, headers, timeout)
    except urllib.error.HTTPError as exc:
        detail, error = _http_error(ANTHROPIC, exc)
        if exc.code == 400 and "temperature" in detail.lower() and "temperature" in payload:
            payload.pop("temperature")
            try:
                obj = _post(url, payload, headers, timeout)
            except Exception as inner:
                raise error from inner
        else:
            raise error from exc
    except Exception as exc:
        raise _wrap_transport(ANTHROPIC, url, exc) from exc

    if obj.get("stop_reason") == "refusal":
        raise RuntimeError("Anthropic declined this request (stop_reason: refusal).")
    parts = [b.get("text", "") for b in (obj.get("content") or []) if b.get("type") == "text"]
    text = "".join(parts).strip()
    if not text and obj.get("stop_reason") == "max_tokens":
        raise RuntimeError(
            "Anthropic hit max_tokens before writing any text. Raise max_output_tokens — "
            "thinking and the reply share that budget on current Claude models."
        )
    return text


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

def _gemini(
    api_key: str, model: str, system: str, user: str, images: List[str],
    temperature: float, max_tokens: int, timeout: int, want_json: bool,
) -> str:
    parts: List[Dict[str, Any]] = [{"text": user}]
    for b64 in images:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})

    generation: Dict[str, Any] = {
        "temperature": float(temperature),
        "maxOutputTokens": int(max_tokens),
    }
    if want_json:
        generation["responseMimeType"] = "application/json"
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": generation,
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    try:
        obj = _post(url, payload, {"Content-Type": "application/json"}, timeout)
    except urllib.error.HTTPError as exc:
        _, error = _http_error(GEMINI, exc)
        raise error from exc
    except Exception as exc:
        raise _wrap_transport(GEMINI, "generativelanguage.googleapis.com", exc) from exc

    candidates = obj.get("candidates") or []
    if not candidates:
        blocked = (obj.get("promptFeedback") or {}).get("blockReason")
        raise RuntimeError(f"Gemini returned no candidates{f' (blocked: {blocked})' if blocked else ''}.")
    candidate = candidates[0]
    chunks = [p.get("text", "") for p in ((candidate.get("content") or {}).get("parts") or [])]
    text = "".join(chunks).strip()
    if not text and candidate.get("finishReason") == "MAX_TOKENS":
        raise RuntimeError("Gemini hit maxOutputTokens before producing any text.")
    return text


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def cloud_chat(
    provider: str, api_key: str, model: str, system: str, user: str,
    images: Optional[List[str]] = None, temperature: float = 0.25,
    max_tokens: int = 4096, timeout: int = 300, want_json: bool = True,
) -> str:
    """One text reply from a hosted provider. Raises RuntimeError on failure."""
    images = images or []
    model = (model or "").strip() or default_model(provider)
    if not api_key:
        names = " or ".join(ENV_KEYS.get(provider, ("an API key",)))
        raise RuntimeError(f"{provider} needs an API key — set the api_key widget or ${names}.")

    if provider == OPENAI:
        return _openai_style(
            OPENAI, "https://api.openai.com/v1/chat/completions", api_key, model,
            system, user, images, temperature, max_tokens, timeout, want_json,
        )
    if provider == OPENROUTER:
        return _openai_style(
            OPENROUTER, "https://openrouter.ai/api/v1/chat/completions", api_key, model,
            system, user, images, temperature, max_tokens, timeout, want_json,
            extra_headers={
                "HTTP-Referer": "https://github.com/comfyui-h3-prompt-creator",
                "X-Title": "ComfyUI H3 Prompt Creator",
            },
        )
    if provider == ANTHROPIC:
        return _anthropic(api_key, model, system, user, images, temperature, max_tokens, timeout)
    if provider == GEMINI:
        return _gemini(api_key, model, system, user, images, temperature, max_tokens, timeout, want_json)
    raise RuntimeError(f"Unknown provider: {provider}")


def preflight(provider: str, api_key: str, model: str, timeout: int = 20) -> str:
    """Cheap reachability/auth check so a bad key fails before the slow calls."""
    model = (model or "").strip() or default_model(provider)
    try:
        if provider == ANTHROPIC:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            )
        elif provider == GEMINI:
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            )
        elif provider == OPENROUTER:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        else:
            req = urllib.request.Request(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except urllib.error.HTTPError as exc:
        detail, error = _http_error(provider, exc)
        raise error from exc
    except Exception as exc:
        raise _wrap_transport(provider, provider, exc) from exc
    return f"{provider} reachable, model: {model}"
