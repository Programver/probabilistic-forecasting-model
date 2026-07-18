"""
Optional LLM polish for the insight layer.

This is the *only* place the system may touch a network, and it is used solely by
the app/CLI insight command — **never** by the scored offline pipeline. It takes
the grounded, deterministic driver report + narrative sections and asks an LLM to
rewrite them as a fluent executive briefing, under strict instructions to use
only the supplied figures (no fabrication) and to communicate uncertainty.

Provider precedence (auto): Gemini → OpenAI → Anthropic, chosen by which SDK is
installed and which API key is present. Any failure — missing key, no SDK, no
network, timeout, bad response — returns ``None`` so the caller falls back to the
fully-offline template narrative. It never raises.

For Gemini either SDK works: the supported ``google-genai`` (preferred) or the
EOL ``google-generativeai`` (legacy fallback). See :func:`_gemini_sdk`.

Environment:
    AIGNITION_LLM_PROVIDER   auto | gemini | openai | anthropic   (default: auto)
    GEMINI_API_KEY / GOOGLE_API_KEY,  OPENAI_API_KEY,  ANTHROPIC_API_KEY
    AIGNITION_GEMINI_MODEL   (default: gemini-2.5-flash)
    AIGNITION_OPENAI_MODEL   (default: gpt-4o-mini)
    AIGNITION_ANTHROPIC_MODEL(default: claude-3-5-haiku-latest)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from logging_utils import get_logger

log = get_logger("llm")

SYSTEM_PROMPT = (
    "You are a senior performance-marketing analyst at a digital agency, briefing "
    "an e-commerce client's leadership on a probabilistic revenue forecast. You are "
    "rigorous and concise. Strict rules: use ONLY the figures provided in the "
    "context — never invent numbers, campaigns, or events. Always communicate "
    "uncertainty (reference the P10–P90 range, not just the point estimate). Be "
    "concrete and action-oriented. Attribute changes to the drivers given "
    "(seasonality, trend, spend & mix). Do not claim causality beyond what the "
    "attribution supports; frame it as 'primarily associated with'. Output clean "
    "markdown with the sections requested."
)

USER_TEMPLATE = """Write an executive forecast briefing from the structured facts below.

Sections to produce (markdown H2 headers):
1. Executive Summary (3-4 sentences; lead with the headline revenue & ROAS and the P10-P90 range)
2. What's Driving the Forecast (bullet the seasonality / trend / spend-&-mix attribution and any calendar events)
3. Channel Outlook (one line per channel: forecast, change vs trailing, ROAS)
4. Risks to Watch (bullets; include uncertainty and any low/declining ROAS)
5. Recommended Actions (bullets; use the marginal-ROAS opportunities to say where to add or trim budget)

Keep it under ~350 words. Round money sensibly. Here are the grounded facts (JSON):

```json
{facts}
```

And here is a deterministic draft you may improve on (do not contradict its numbers):

{draft}
"""


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str


def _get(*names: str) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def _gemini_sdk() -> Optional[str]:
    """Which Gemini SDK is importable: the supported one, the legacy one, or neither.

    ``google-genai`` is the supported successor; ``google-generativeai`` is EOL
    (it prints a FutureWarning on import and is no longer maintained). We accept
    either so the briefing works whichever one is installed.
    """
    try:
        import google.genai  # noqa: F401
        return "genai"
    except Exception:
        pass
    try:
        import google.generativeai  # noqa: F401
        return "legacy"
    except Exception:
        return None


def available_provider() -> Optional[str]:
    """Return the provider that can actually be used right now, or None."""
    pref = os.environ.get("AIGNITION_LLM_PROVIDER", "auto").lower()
    order = [pref] if pref != "auto" else ["gemini", "openai", "anthropic"]
    for prov in order:
        if prov == "gemini" and _get("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if _gemini_sdk() is not None:
                return "gemini"
            continue
        if prov == "openai" and _get("OPENAI_API_KEY"):
            try:
                import openai  # noqa: F401
                return "openai"
            except Exception:
                continue
        if prov == "anthropic" and _get("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # noqa: F401
                return "anthropic"
            except Exception:
                continue
    return None


def _facts_json(report) -> str:
    d = report.to_dict()
    # keep the payload compact
    return json.dumps(d, default=str, indent=None)


def generate(report, draft: str, timeout: int = 30) -> Optional[LLMResult]:
    """Return an LLM-polished briefing, or None to fall back to ``draft``."""
    provider = available_provider()
    if provider is None:
        log.info("no LLM provider available (no key/SDK) — using offline narrative")
        return None
    prompt = USER_TEMPLATE.format(facts=_facts_json(report), draft=draft)
    try:
        if provider == "gemini":
            return _gemini(prompt, timeout)
        if provider == "openai":
            return _openai(prompt, timeout)
        if provider == "anthropic":
            return _anthropic(prompt, timeout)
    except Exception as exc:  # never raise into the app
        log.warning("LLM generation failed (%s) — falling back to offline narrative: %s",
                    provider, exc)
    return None


def _gemini(prompt: str, timeout: int) -> Optional[LLMResult]:
    model_name = os.environ.get("AIGNITION_GEMINI_MODEL", "gemini-2.5-flash")
    sdk = _gemini_sdk()

    if sdk == "genai":  # supported SDK — HTTP/httpx, thread-safe
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=_get("GEMINI_API_KEY", "GOOGLE_API_KEY"))
        resp = client.models.generate_content(
            model=model_name, contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=0.4,
                http_options=types.HttpOptions(timeout=timeout * 1000),  # ms
            ),
        )
        text = (getattr(resp, "text", "") or "").strip()
        return LLMResult(text, "gemini", model_name) if text else None

    if sdk == "legacy":  # EOL SDK, still supported here as a fallback
        import google.generativeai as genai
        # transport="rest": the legacy default gRPC transport uses a background
        # async DNS resolver that can crash the whole process when called off the
        # main thread (e.g. Streamlit's ScriptRunner thread). REST avoids that.
        genai.configure(api_key=_get("GEMINI_API_KEY", "GOOGLE_API_KEY"), transport="rest")
        gm = genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)
        resp = gm.generate_content(prompt, request_options={"timeout": timeout})
        text = (getattr(resp, "text", "") or "").strip()
        return LLMResult(text, "gemini", model_name) if text else None

    return None


def _openai(prompt: str, timeout: int) -> Optional[LLMResult]:
    import openai
    client = openai.OpenAI(api_key=_get("OPENAI_API_KEY"), timeout=timeout)
    model_name = os.environ.get("AIGNITION_OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        temperature=0.4,
    )
    text = (resp.choices[0].message.content or "").strip()
    return LLMResult(text, "openai", model_name) if text else None


def _anthropic(prompt: str, timeout: int) -> Optional[LLMResult]:
    import anthropic
    client = anthropic.Anthropic(api_key=_get("ANTHROPIC_API_KEY"), timeout=timeout)
    model_name = os.environ.get("AIGNITION_ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    resp = client.messages.create(
        model=model_name, max_tokens=1200, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(getattr(b, "text", "") for b in resp.content).strip()
    return LLMResult(text, "anthropic", model_name) if text else None
