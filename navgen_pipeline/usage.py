"""Secret-free token, latency, duration and cost accounting.

Ark Responses exposes token usage for text calls while Seedance task
responses usually expose video metadata (and may expose provider-specific
usage).  This module keeps those measurements separate so a large provider
counter is never mistaken for billable LLM tokens.  Cost is deliberately an
estimate unless the caller supplies the current rates from the account
console.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _as_nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _first_int(source: Mapping[str, Any], names: Iterable[str]) -> int:
    for name in names:
        value = _as_nonnegative_int(source.get(name))
        if value is not None:
            return value
    return 0


def normalize_token_usage(raw: Any) -> Dict[str, int]:
    """Normalize common Responses/Chat/Seedance usage field names."""

    if not isinstance(raw, Mapping):
        return {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    input_tokens = _first_int(raw, ("input_tokens", "prompt_tokens", "input_token_count"))
    output_tokens = _first_int(raw, ("output_tokens", "completion_tokens", "output_token_count"))
    reasoning_tokens = _first_int(raw, ("reasoning_tokens", "reasoning_token_count"))
    # Responses commonly nests reasoning/cache details under *_tokens_details.
    # Read those aliases without treating cached tokens as additional input.
    output_details = raw.get("output_tokens_details")
    if isinstance(output_details, Mapping) and reasoning_tokens == 0:
        reasoning_tokens = _first_int(output_details, ("reasoning_tokens", "reasoning_token_count"))
    total_tokens = _first_int(raw, ("total_tokens", "total_token_count"))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def _numeric_cost(raw: Any) -> Optional[float]:
    """Find an explicitly returned CNY cost, if a provider supplies one."""

    if not isinstance(raw, Mapping):
        return None
    for key in ("cost_cny", "estimated_cost_cny", "cost", "amount_cny"):
        value = raw.get(key)
        if isinstance(value, Mapping):
            value = value.get("cny", value.get("amount"))
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number) and number >= 0:
            return number
    for key in ("billing", "bill", "usage"):
        nested = raw.get(key)
        found = _numeric_cost(nested)
        if found is not None:
            return found
    return None


def _rate_cost(tokens: Mapping[str, int], input_rate: Optional[float], output_rate: Optional[float]) -> Optional[float]:
    if input_rate is None or output_rate is None:
        return None
    if input_rate < 0 or output_rate < 0:
        return None
    return (tokens.get("input_tokens", 0) / 1000.0) * input_rate + (
        tokens.get("output_tokens", 0) / 1000.0
    ) * output_rate


def text_event(
    stage: str,
    model: str,
    call: Optional[Mapping[str, Any]],
    *,
    input_rate_cny_per_1k: Optional[float] = None,
    output_rate_cny_per_1k: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert one ``ArkResponsesClient.last_call`` record into an event."""

    call = dict(call or {})
    raw_usage = call.get("usage", {})
    tokens = normalize_token_usage(raw_usage)
    provider_cost = _numeric_cost(raw_usage)
    estimated = _rate_cost(tokens, input_rate_cny_per_1k, output_rate_cny_per_1k)
    cost = provider_cost if provider_cost is not None else estimated
    if provider_cost is not None:
        cost_status, cost_basis = "provider_reported", "provider_usage"
    elif estimated is not None:
        cost_status, cost_basis = "estimated", "configured_token_rates"
    elif input_rate_cny_per_1k is not None or output_rate_cny_per_1k is not None:
        # A half-configured token rate is useful for diagnostics, but must not
        # be presented as a complete bill.  Keep cost_cny null in this case.
        cost_status, cost_basis = "partial", "incomplete_token_rates"
    else:
        cost_status, cost_basis = "unpriced", "rates_not_configured"
    event: Dict[str, Any] = {
        "kind": "text",
        "stage": stage,
        "model": model,
        "status": call.get("status", "unknown"),
        "latency_s": float(call.get("latency_s", 0.0) or 0.0),
        **tokens,
        "provider_usage": dict(raw_usage) if isinstance(raw_usage, Mapping) else {},
        "cost_cny": round(cost, 8) if cost is not None else None,
        "cost_status": cost_status,
        "cost_basis": cost_basis,
    }
    if call.get("error_type"):
        event["error_type"] = str(call["error_type"])
    return event


def video_event(
    model: str,
    task_response: Optional[Mapping[str, Any]],
    *,
    requested_duration_s: Optional[float],
    actual_duration_s: Optional[float],
    latency_s: float,
    resolution: Optional[str] = None,
    ratio: Optional[str] = None,
    size_bytes: Optional[int] = None,
    per_second_cny: Optional[float] = None,
    per_video_cny: Optional[float] = None,
    per_1k_tokens_cny: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert one completed Seedance task into a usage event."""

    response = dict(task_response or {})
    raw_usage = response.get("usage", {})
    provider_tokens = normalize_token_usage(raw_usage)
    provider_cost = _numeric_cost(response)
    if provider_cost is None:
        provider_cost = _numeric_cost(raw_usage)
    duration = actual_duration_s if actual_duration_s is not None else requested_duration_s
    estimated: Optional[float] = None
    cost_basis = "rates_not_configured"
    if provider_cost is None and duration is not None and per_second_cny is not None and per_second_cny >= 0:
        estimated = float(duration) * float(per_second_cny)
        cost_basis = "configured_video_seconds_rate"
    elif provider_cost is None and per_video_cny is not None and per_video_cny >= 0:
        estimated = float(per_video_cny)
        cost_basis = "configured_video_flat_rate"
    elif provider_cost is None and per_1k_tokens_cny is not None and per_1k_tokens_cny >= 0:
        # Some gateways expose a provider token counter for video.  It is kept
        # separate from text tokens and only used for an estimate when the
        # operator explicitly supplies a token rate.
        estimated = (provider_tokens["total_tokens"] / 1000.0) * float(per_1k_tokens_cny)
        cost_basis = "configured_video_provider_token_rate"
    cost = provider_cost if provider_cost is not None else estimated
    event: Dict[str, Any] = {
        "kind": "video",
        "stage": "seedance",
        "model": model,
        "status": str(response.get("status", "unknown")),
        "latency_s": round(max(0.0, float(latency_s)), 6),
        "requested_duration_s": float(requested_duration_s) if requested_duration_s is not None else None,
        "actual_duration_s": float(actual_duration_s) if actual_duration_s is not None else None,
        "resolution": resolution,
        "ratio": ratio,
        "size_bytes": int(size_bytes) if size_bytes is not None and int(size_bytes) >= 0 else None,
        "provider_usage": dict(raw_usage) if isinstance(raw_usage, Mapping) else {},
        "provider_input_tokens": provider_tokens["input_tokens"],
        "provider_output_tokens": provider_tokens["output_tokens"],
        "provider_total_tokens": provider_tokens["total_tokens"],
        "cost_cny": round(cost, 8) if cost is not None else None,
        "cost_status": "provider_reported" if provider_cost is not None else ("estimated" if estimated is not None else ("partial" if any(rate is not None for rate in (per_second_cny, per_video_cny, per_1k_tokens_cny)) else "unpriced")),
        "cost_basis": "provider_usage" if provider_cost is not None else cost_basis,
    }
    if response.get("id"):
        event["task_id"] = str(response["id"])
    return event


def aggregate_usage(events: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-stage events for a sample or a complete run."""

    items = [dict(event) for event in events]
    text_items = [event for event in items if event.get("kind") == "text"]
    video_items = [event for event in items if event.get("kind") == "video"]
    cost_values = [float(event["cost_cny"]) for event in items if event.get("cost_cny") is not None]
    priced_statuses = {"provider_reported", "estimated", "configured"}
    priced = [event for event in items if event.get("cost_cny") is not None and str(event.get("cost_status")) in priced_statuses]
    not_applicable = [event for event in items if str(event.get("cost_status")) == "not_applicable"]
    unpriced = [
        event
        for event in items
        if str(event.get("cost_status", "unpriced")) in {"unpriced", "partial"}
        or (
            event.get("cost_cny") is None
            and str(event.get("cost_status")) not in {"not_applicable"}
        )
    ]
    if not items:
        status = "unpriced"
    elif len(not_applicable) == len(items):
        status = "not_applicable"
    elif unpriced or len(priced) + len(not_applicable) != len(items):
        status = "partial" if priced or not_applicable else "unpriced"
    elif priced and not_applicable:
        status = "partial"
    else:
        status = "configured"
    return {
        "request_count": len(items),
        "text_request_count": len(text_items),
        "video_request_count": len(video_items),
        "input_tokens": sum(int(event.get("input_tokens", 0) or 0) for event in text_items),
        "output_tokens": sum(int(event.get("output_tokens", 0) or 0) for event in text_items),
        "reasoning_tokens": sum(int(event.get("reasoning_tokens", 0) or 0) for event in text_items),
        "total_tokens": sum(int(event.get("total_tokens", 0) or 0) for event in text_items),
        "video_provider_total_tokens": sum(int(event.get("provider_total_tokens", 0) or 0) for event in video_items),
        "video_provider_input_tokens": sum(int(event.get("provider_input_tokens", 0) or 0) for event in video_items),
        "video_provider_output_tokens": sum(int(event.get("provider_output_tokens", 0) or 0) for event in video_items),
        "latency_s": round(sum(float(event.get("latency_s", 0.0) or 0.0) for event in items), 6),
        "text_latency_s": round(sum(float(event.get("latency_s", 0.0) or 0.0) for event in text_items), 6),
        "video_latency_s": round(sum(float(event.get("latency_s", 0.0) or 0.0) for event in video_items), 6),
        "video_requested_duration_s": round(
            sum(float(event.get("requested_duration_s", 0.0) or 0.0) for event in video_items), 6
        ),
        "video_actual_duration_s": round(
            sum(float(event.get("actual_duration_s", 0.0) or 0.0) for event in video_items), 6
        ),
        "video_size_bytes": sum(int(event.get("size_bytes", 0) or 0) for event in video_items),
        "estimated_cost_cny": round(sum(cost_values), 8) if cost_values else (0.0 if status == "not_applicable" else None),
        "billable_cost_cny": round(sum(float(event["cost_cny"]) for event in priced), 8) if priced else 0.0 if status == "not_applicable" else None,
        "cost_status": status,
        "unpriced_event_count": len(unpriced),
        "priced_event_count": len(priced),
        "not_applicable_event_count": len(not_applicable),
        "events": items,
    }
