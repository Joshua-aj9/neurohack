"""LM Studio (OpenAI-compatible) suggestion client with timeout and static fallback.

`suggest()` always returns exactly n strings and never raises: on timeout / server down / invalid JSON
it falls back to a static phrase + n-gram list and reports `source="fallback"`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass


@dataclass
class Suggestions:
    items: list[str]
    source: str  # "llm" | "fallback"
    latency_s: float
    error: str | None = None
    raw: str | None = None


def parse_suggestions(text: str) -> list[str]:
    """Extract a list of strings from model output: {"suggestions": [...]} or a bare JSON array."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    candidates = [text]
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidates.append(m.group(0))
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            obj = obj.get(
                "suggestions", next((v for v in obj.values() if isinstance(v, list)), None)
            )
        if isinstance(obj, list) and all(isinstance(s, str) for s in obj):
            return [s.strip() for s in obj if s.strip()]
    raise ValueError(f"no JSON suggestions in: {text[:200]!r}")


def clean(items: list[str], partial: str) -> list[str]:
    """Drop duplicates / empties / echoes of the already-typed text; keep order."""
    seen, out = set(), []
    p = partial.strip().lower()
    for s in items:
        s = s.strip().strip('"').strip()
        if p and s.lower().startswith(p):
            s = s[len(p) :].strip()
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s)
    return out


def fallback_suggestions(partial: str, n: int, fb_cfg: dict) -> list[str]:
    """Static fallback: phrases that extend the partial text, then sentence starters."""
    p = partial.strip().lower()
    out = []
    for ph in fb_cfg["phrases"]:
        low = ph.lower()
        if p and low.startswith(p) and len(low) > len(p):
            out.append(ph[len(p) :].strip())
        elif not p:
            out.append(ph)
    if not p:
        out = fb_cfg["starters"] + out
    elif not out:
        out = list(fb_cfg["continuations"])  # text is a complete phrase: offer ways to continue it
    out = clean(out, "")
    for filler in fb_cfg["continuations"] + fb_cfg["phrases"] + fb_cfg["starters"]:
        if len(out) >= n:
            break
        if filler not in out:
            out.append(filler)
    return out[:n]


class LLMClient:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._client = None
        self.model = cfg.get("model")

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.cfg["base_url"],
                api_key=self.cfg["api_key"],
                timeout=self.cfg["timeout_s"],
                max_retries=0,
            )
        if not self.model:
            models = self._client.models.list()
            self.model = models.data[0].id
        return self._client

    def build_messages(self, context: str, partial: str, n: int) -> list[dict]:
        user = (
            f'Caregiver said: "{context}"\nReply typed so far: "{partial}"\n'
            f"Give exactly {n} different suggestions."
        )
        return [
            {"role": "system", "content": self.cfg["system_prompt"]},
            {"role": "user", "content": user},
        ]

    def suggest(self, context: str, partial: str, n: int | None = None) -> Suggestions:
        n = n or self.cfg["n_suggestions"]
        t0 = time.perf_counter()
        raw = None
        try:
            client = self._ensure()
            kwargs = {}
            if self.cfg.get("structured_output"):
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "suggestions",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["suggestions"],
                            "properties": {
                                "suggestions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": n,
                                    "maxItems": n,
                                }
                            },
                        },
                    },
                }
            resp = client.chat.completions.create(
                model=self.model,
                messages=self.build_messages(context, partial, n),
                temperature=self.cfg["temperature"],
                max_tokens=self.cfg["max_tokens"],
                **kwargs,
            )
            raw = resp.choices[0].message.content or ""
            items = clean(parse_suggestions(raw), partial)
            if len(items) < n:
                items += [
                    s
                    for s in fallback_suggestions(partial, n, self.cfg["fallback"])
                    if s not in items
                ]
                return Suggestions(
                    items[:n], "llm+fallback", time.perf_counter() - t0, "short list", raw
                )
            return Suggestions(items[:n], "llm", time.perf_counter() - t0, None, raw)
        except Exception as e:  # never let the UI die on LLM problems
            return Suggestions(
                fallback_suggestions(partial, n, self.cfg["fallback"]),
                "fallback",
                time.perf_counter() - t0,
                f"{type(e).__name__}: {e}",
                raw,
            )
