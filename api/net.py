"""Shared HTTP client and a small TTL cache for upstream retrieval calls.

Two performance aids used by the retrieval layer:

* one long-lived ``httpx.AsyncClient`` so TLS/TCP connections to PubMed, openFDA,
  RxNorm, ClinicalTrials.gov, … are kept alive and reused across queries, and
* an in-process TTL cache keyed on (tool name, args) so a repeated lookup — the
  same drug across queries, or a retry inside the LLM tool loop — skips the
  network entirely.
"""
from __future__ import annotations

import asyncio
import json
import ssl
import time
from typing import Awaitable, Callable

import httpx

from .config import CONTACT, REQUEST_TIMEOUT, RETRIEVAL_CACHE_TTL

_client: httpx.AsyncClient | None = None

# Transient transport/TLS failures worth one more try with a fresh connection:
# a stale pooled keepalive socket, a mid-stream "bad record mac", a reset.
_TRANSIENT_ERRORS = (httpx.TransportError, ssl.SSLError)


async def with_retry(factory: Callable[[], Awaitable], *, attempts: int = 3, base_delay: float = 0.4):
    """Await ``factory()``, retrying on transient transport/TLS errors with a
    short backoff. ``factory`` must return a fresh awaitable each call. Only
    connection-level failures are retried (the response was never received, so a
    retried POST cannot double-apply); HTTP status errors pass straight through.
    """
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return await factory()
        except _TRANSIENT_ERRORS as exc:
            last = exc
            if i == attempts - 1:
                break
            await asyncio.sleep(base_delay * (2 ** i))
    raise last


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": f"clinical-agent/1.0 ({CONTACT})"},
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=40, max_keepalive_connections=20, keepalive_expiry=30.0
            ),
        )
    return _client


async def aclose_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


# --- TTL response cache ---------------------------------------------------------

_MAX_ENTRIES = 512
_cache: dict[str, tuple[float, object]] = {}
_stats = {"hits": 0, "misses": 0}


def _key(name: str, params: dict) -> str:
    return name + "|" + json.dumps(params, sort_keys=True, default=str)


async def cached_call(name: str, params: dict, factory: Callable[[], Awaitable]):
    """Return a cached (raw_citations, source_api, note) tuple if fresh, else call
    `factory()` and cache its result. `factory` must produce a new awaitable each
    time it is invoked."""
    if RETRIEVAL_CACHE_TTL <= 0:
        return await with_retry(factory)

    key = _key(name, params)
    now = time.time()
    hit = _cache.get(key)
    if hit is not None and now - hit[0] < RETRIEVAL_CACHE_TTL:
        _stats["hits"] += 1
        return hit[1]

    _stats["misses"] += 1
    value = await with_retry(factory)
    _cache[key] = (now, value)
    if len(_cache) > _MAX_ENTRIES:  # evict the oldest quarter
        for old in sorted(_cache, key=lambda k: _cache[k][0])[: _MAX_ENTRIES // 4]:
            _cache.pop(old, None)
    return value


def cache_stats() -> dict:
    total = _stats["hits"] + _stats["misses"]
    return {
        "hits": _stats["hits"],
        "misses": _stats["misses"],
        "entries": len(_cache),
        "hit_rate": round(_stats["hits"] / total, 3) if total else 0.0,
        "ttl_seconds": RETRIEVAL_CACHE_TTL,
    }


def clear_cache() -> int:
    n = len(_cache)
    _cache.clear()
    return n
