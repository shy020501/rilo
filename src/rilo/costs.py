"""Auditable estimates from API usage, never an assertion about an invoice.

Rates verified 2026-10-28 against the official model and prompt-caching pages.
Input, cache reads, cache writes, and output are disjoint billing buckets.
Reasoning tokens are already included in output_tokens and are not added again.
"""
from __future__ import annotations

import re
from typing import Any


PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-6-astra"
CACHING_SOURCE = "https://developers.openai.com/api/docs/guides/prompt-caching"
PRICING_VERIFIED_ON = "2026-10-28"

# Standard, short-context USD per million tokens. Source for each entry is its
# official model page; the Sol cache-read discount differs from Astra's.
_MODEL_RATES = {
    "gpt-6-astra": {"input": 10.0, "cached": 1.0, "output": 50.0},
    "gpt-6.1-sol": {"input": 2.0, "cached": .10, "output": 10.0},
}


def _priced_model(model: str) -> str | None:
    for name in _MODEL_RATES:
        if re.fullmatch(re.escape(name) + r"(?:-\d{4}-\d{2}-\d{2})?", model):
            return name
    return None


def _count(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    return value if type(value) is int and value >= 0 else None


def token_counts(usage: dict[str, Any] | None) -> dict[str, int] | None:
    """Normalize known counters while preserving the raw object in the ledger."""
    if not isinstance(usage, dict):
        return None
    input_tokens = _count(usage.get("input_tokens"))
    output_tokens = _count(usage.get("output_tokens"))
    details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    if input_tokens is None or output_tokens is None or not isinstance(details, dict) or not isinstance(output_details, dict):
        return None
    cached = _count(details.get("cached_tokens"), 0)
    writes = _count(details.get("cache_write_tokens"), 0)
    reasoning = _count(output_details.get("reasoning_tokens"), 0)
    total = _count(usage.get("total_tokens"), input_tokens + output_tokens)
    if any(value is None for value in (cached, writes, reasoning, total)):
        return None
    if cached + writes > input_tokens or reasoning > output_tokens or total != input_tokens + output_tokens:
        return None
    write_details = details.get("cache_write_tokens_details")
    if write_details is None:
        # The policy uses the documented default 30-minute cache lifetime.
        short_writes, long_writes = writes, 0
    elif isinstance(write_details, dict):
        short_writes = _count(write_details.get("ttl_30m_tokens"), 0)
        long_writes = _count(write_details.get("ttl_12h_tokens"), 0)
        if short_writes is None or long_writes is None or short_writes + long_writes != writes:
            return None
    else:
        return None
    return {
        "input_tokens": input_tokens, "ordinary_input_tokens": input_tokens - cached - writes,
        "cached_input_tokens": cached, "cache_write_tokens": writes,
        "cache_write_30m_tokens": short_writes, "cache_write_12h_tokens": long_writes,
        "output_tokens": output_tokens, "reasoning_tokens": reasoning, "total_tokens": total,
    }


def estimate_cost(model: str, usage: dict[str, Any] | None, *, service_tier: str = "default",
                  official_endpoint: bool = True) -> dict[str, Any]:
    """Return null for unknown prices/usage rather than silently charging zero.

Only verified standard models and dated snapshots are priced. A custom gateway
may bill differently, so official OpenAI rates are not applied to that endpoint.
"""
    counts = token_counts(usage)
    priced_model = _priced_model(model)
    reason = None
    if counts is None:
        reason = "usage_missing_or_invalid"
    elif not official_endpoint:
        reason = "custom_endpoint_pricing_unknown"
    elif priced_model is None:
        reason = "model_pricing_unknown"
    elif service_tier not in ("default", "standard"):
        reason = "service_tier_pricing_unknown"
    if reason:
        return {"tokens": counts, "estimated_cost_usd": None, "pricing": None,
                "cost_unknown": True, "cost_unknown_reason": reason}

    long_context = counts["input_tokens"] > 272_000
    base = _MODEL_RATES[priced_model]
    input_rate = base["input"] * (2 if long_context else 1)
    rates = {"ordinary_input": input_rate,
             "cached_input": base["cached"] * (2 if long_context else 1),
             "cache_write_30m": input_rate * 1.25, "cache_write_12h": input_rate * 2.5,
             "output": base["output"] * (1.5 if long_context else 1)}
    components = {
        "ordinary_input": counts["ordinary_input_tokens"] * rates["ordinary_input"] / 1_000_000,
        "cached_input": counts["cached_input_tokens"] * rates["cached_input"] / 1_000_000,
        "cache_write_30m": counts["cache_write_30m_tokens"] * rates["cache_write_30m"] / 1_000_000,
        "cache_write_12h": counts["cache_write_12h_tokens"] * rates["cache_write_12h"] / 1_000_000,
        "output": counts["output_tokens"] * rates["output"] / 1_000_000,
    }
    return {
        "tokens": counts, "estimated_cost_usd": sum(components.values()),
        "cost_unknown": False, "cost_unknown_reason": None,
        "pricing": {"source": f"https://developers.openai.com/api/docs/models/{priced_model}",
                    "caching_source": CACHING_SOURCE,
                    "verified_on": PRICING_VERIFIED_ON, "service_tier": "standard",
                    "context": "long" if long_context else "short", "currency": "USD",
                    "rates_per_million_tokens": rates, "components_usd": components,
                    "estimate_only": True, "default_cache_write_ttl": "30m"},
    }
