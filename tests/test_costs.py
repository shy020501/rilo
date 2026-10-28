from __future__ import annotations

import pytest

from rilo.costs import estimate_cost


def usage(*, input_tokens=12000, output_tokens=2000, cached=4000, writes=7000, write_details=None):
    details = {"cached_tokens": cached, "cache_write_tokens": writes}
    if write_details is not None:
        details["cache_write_tokens_details"] = write_details
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "input_tokens_details": details, "output_tokens_details": {"reasoning_tokens": 1500},
            "total_tokens": input_tokens + output_tokens}


def test_standard_cost_counts_each_token_once_and_reasoning_is_already_output():
    result = estimate_cost("gpt-6-astra", usage())
    assert result["tokens"]["ordinary_input_tokens"] == 1000
    # 1000*10 + 4000*1 + 7000*12.5 + 2000*50 dollars per million.
    assert result["estimated_cost_usd"] == pytest.approx(0.2015)
    assert result["cost_unknown"] is False
    assert result["pricing"]["rates_per_million_tokens"]["output"] == 50


def test_mixed_ttls_have_separate_write_costs():
    result = estimate_cost("gpt-6-astra", usage(write_details={"ttl_30m_tokens": 3000, "ttl_12h_tokens": 4000}))
    assert result["estimated_cost_usd"] == pytest.approx(0.2515)
    assert result["pricing"]["components_usd"]["cache_write_12h"] == pytest.approx(0.1)


def test_long_context_rates_apply_to_entire_request_with_exact_boundary():
    short = estimate_cost("gpt-6-astra", usage(input_tokens=272000, cached=0, writes=0))
    long = estimate_cost("gpt-6-astra-2026-10-01", usage(input_tokens=272001, cached=0, writes=0))
    assert short["estimated_cost_usd"] == pytest.approx(2.82)
    assert long["estimated_cost_usd"] == pytest.approx(5.59002)
    assert long["pricing"]["context"] == "long"


@pytest.mark.parametrize("changes", [
    {"model": "unknown"}, {"model": "gpt-6-astra-pro"},
    {"service_tier": "priority"}, {"official_endpoint": False}, {"usage": None},
    {"usage": {}}, {"usage": usage(cached=10000, writes=7000)},
    {"usage": usage(write_details={"ttl_30m_tokens": 1, "ttl_12h_tokens": 1})},
    {"usage": usage(input_tokens=True)},
])
def test_unknown_prices_or_invalid_usage_are_not_reported_as_zero(changes):
    arguments = {"model": "gpt-6-astra", "usage": usage(), **changes}
    result = estimate_cost(**arguments)
    assert result["estimated_cost_usd"] is None
    assert result["cost_unknown"] is True
    assert result["cost_unknown_reason"]


def test_zero_usage_is_valid_zero_cost_when_explicitly_reported():
    result = estimate_cost("gpt-6-astra", {"input_tokens": 0, "output_tokens": 0})
    assert result["estimated_cost_usd"] == 0
    assert result["cost_unknown"] is False


def test_sol_pricing_uses_its_own_cache_discount_and_write_rates():
    result = estimate_cost("gpt-6.1-sol", usage())
    # 1000*2 + 4000*.1 + 7000*2.5 + 2000*10 dollars per million.
    assert result["estimated_cost_usd"] == pytest.approx(.0399)
    assert result["pricing"]["rates_per_million_tokens"]["cached_input"] == .1
    assert result["pricing"]["source"].endswith("/gpt-6.1-sol")
    result = estimate_cost("gpt-6.1-sol-2026-10-01", usage(input_tokens=272001, cached=1000, writes=1000,
        write_details={"ttl_30m_tokens": 500, "ttl_12h_tokens": 500}))
    assert result["pricing"]["context"] == "long"
    assert result["pricing"]["rates_per_million_tokens"] == {
        "ordinary_input": 4.0, "cached_input": .2, "cache_write_30m": 5.0,
        "cache_write_12h": 10.0, "output": 15.0}
