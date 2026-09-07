"""Minimal shared LLM helper for the training builders."""
from __future__ import annotations

import json
import re

import httpx

from api.agent import _call_claude
from api.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from api.net import get_client


def available() -> bool:
    return bool(ANTHROPIC_API_KEY)


def _text(resp: dict) -> str:
    return "\n".join(b.get("text", "") for b in resp.get("content", [])
                     if b.get("type") == "text").strip()


async def ask(system: str, user: str, *, max_tokens: int = 1400) -> str:
    client: httpx.AsyncClient = get_client()
    resp = await _call_claude(client, [{"role": "user", "content": user}],
                              use_tools=False, system=system, max_tokens=max_tokens)
    return _text(resp)


def json_array(text: str) -> list:
    """Pull the first JSON array out of an LLM reply."""
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    raw = fence.group(1) if fence else text
    start = raw.find("[")
    if start == -1:
        return []
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        depth += (ch == "[") - (ch == "]")
        if depth == 0:
            try:
                out = json.loads(raw[start:i + 1])
                return out if isinstance(out, list) else []
            except json.JSONDecodeError:
                return []
    return []


MODEL = CLAUDE_MODEL
