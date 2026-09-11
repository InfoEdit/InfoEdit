"""OpenAI-compatible LLM access for InfoEdit (code-level branch).

This branch talks to an OpenAI-compatible proxy instead of Vertex AI, so every
model — Gemini, Claude, GPT — is reached through one client and differs only by
the ``model`` string.

Configure with two environment variables (see env.example.sh):

    OPENAI_API_KEY    key issued by the proxy
    OPENAI_BASE_URL   proxy endpoint, e.g. https://…/v1

Everything this branch needs is one call shape — a prompt, optionally with
images attached, returning text:

    chat_text(model, prompt, images=[...]) -> str

That covers both users of the API here: the judge (original + edited image in,
verdict JSON out) and the code-level editors (HTML/PPTX source in, edited
source out).

Check a model before launching a run:

    python llm_client.py --probe gemini-3.5-flash
"""

from __future__ import annotations

import base64
import io
import os
import threading
from typing import Iterable, Sequence

from openai import OpenAI

__all__ = ["get_client", "chat_text", "describe"]

_client = None
_lock = threading.Lock()


def get_client() -> OpenAI:
    """Process-wide OpenAI client aimed at the proxy (created once)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                key = os.environ.get("OPENAI_API_KEY")
                base = os.environ.get("OPENAI_BASE_URL")
                if not key or not base:
                    raise RuntimeError(
                        "OPENAI_API_KEY and OPENAI_BASE_URL must be set — copy "
                        "env.example.sh to env.sh, fill it in, then: source env.sh"
                    )
                _client = OpenAI(api_key=key, base_url=base)
    return _client


def describe() -> str:
    return f"OpenAI-compatible proxy at {os.environ.get('OPENAI_BASE_URL', '(unset)')}"


def _png_b64(image) -> str:
    """PIL.Image | bytes | path -> base64 PNG (no data: prefix)."""
    if isinstance(image, (bytes, bytearray)):
        return base64.b64encode(image).decode()
    if isinstance(image, str):
        with open(image, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _messages(prompt: str, images: Sequence, system: str | None = None) -> list:
    head = [{"role": "system", "content": system}] if system else []
    if not images:
        return head + [{"role": "user", "content": prompt}]
    parts = [{"type": "text", "text": prompt}]
    for img in images:
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_png_b64(img)}"},
        })
    return head + [{"role": "user", "content": parts}]


def chat_text(
    model: str,
    prompt: str,
    images: Iterable | None = None,
    *,
    system: str | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> str:
    """Send a prompt (optionally with images) and return the text reply.

    Raises whatever the OpenAI SDK raises; callers own the retry policy.
    """
    kwargs = {"model": model, "messages": _messages(prompt, list(images or []), system)}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    resp = get_client().chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def _probe(model: str) -> None:
    from PIL import Image

    print(f"proxy : {describe()}")
    print(f"model : {model}\n")

    try:
        out = chat_text(model, "Reply with the single word: ok", max_tokens=8)
        print(f"  text         ✅  {out[:40]!r}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  text         ❌  {type(exc).__name__}: {exc}")
        print("\n  -> this model is unusable on this key; nothing else will work.")
        return

    swatch = Image.new("RGB", (64, 64), (200, 80, 80))
    try:
        out = chat_text(model, "What colour is this image? One word.", [swatch], max_tokens=8)
        print(f"  image input  ✅  {out[:40]!r}  -> usable as a judge")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  image input  ❌  {type(exc).__name__}: {exc}")
        print("                   -> fine for code-level editing, but cannot judge")

    try:
        out = chat_text(model, 'Reply with JSON: {"ok": true}', json_mode=True, max_tokens=32)
        print(f"  json mode    ✅  {out[:40]!r}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  json mode    ⚠️  {type(exc).__name__}: {exc}")
        print("                   -> judge falls back to fenced-JSON parsing")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Check what this proxy supports.")
    ap.add_argument("--probe", metavar="MODEL", required=True,
                    help="model id to test, e.g. gemini-3.5-flash")
    _probe(ap.parse_args().probe)
