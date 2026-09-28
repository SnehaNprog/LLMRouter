import logging
import re

from app.router.deployment import Deployment
from app.routing.selector import RoutingChoice

logger = logging.getLogger("app.routing.decisions")

_UNSAFE = re.compile(r"[^A-Za-z0-9._:/@+-]")
_MAX_VALUE = 100
_LOGGED_CANDIDATES = 5  # the ranking is logged in full detail only this far down


def _clean(value: str) -> str:
    """Make a configured name safe to put in a header: ASCII, no spaces or control characters."""
    return _UNSAFE.sub("_", value)[:_MAX_VALUE]


def routing_headers(choice: RoutingChoice, served: Deployment | None) -> dict[str, str]:
    """Headers that explain an ``auto`` request: what answered it, and why that model.

    Only fixed vocabulary and operator-chosen model names appear here. Nothing from the
    caller's prompt is ever echoed.
    """
    answered = served.alias if served is not None else choice.alias
    tier = served.config.tier if served is not None else choice.tier
    reason = "pinned" if choice.pinned else (",".join(choice.complexity.top(3)) or "none")
    headers = {
        "x-llmrouter-model": _clean(answered),
        "x-llmrouter-tier": tier.value if tier is not None else "none",
        "x-llmrouter-complexity": f"{choice.complexity.value:.2f}",
        "x-llmrouter-required-tier": choice.required_tier.value,
        "x-llmrouter-reason": reason,
        "x-llmrouter-escalated": "true" if answered != choice.alias else "false",
    }
    if choice.policy is not None:
        headers["x-llmrouter-policy"] = _clean(choice.policy)  # the organisation's policy applied
    if not choice.clears_bar:
        headers["x-llmrouter-degraded"] = "true"  # nothing as strong as the request needed
    return headers


def log_decision(choice: RoutingChoice, *, requested: str, tenant_id: str, key_id: str) -> None:
    """Record why a model was chosen, before the request is sent anywhere."""
    logger.info(
        "routing decision",
        extra={
            "requested_model": requested,
            "chosen": choice.alias,
            "tier": choice.tier.value if choice.tier is not None else None,
            "policy": choice.policy,
            "pinned": choice.pinned,
            "required_tier": choice.required_tier.value,
            "clears_bar": choice.clears_bar,
            "complexity": choice.complexity.value,
            "signals": {
                name: choice.complexity.contributions[name] for name in choice.complexity.top(3)
            },
            "candidates": [
                {
                    "alias": c.alias,
                    "tier": c.tier.value if c.tier is not None else None,
                    "expected_cost_usd": round(c.expected_cost_usd, 9),
                    "expected_latency_s": c.expected_latency_s,
                    "latency_measured": c.latency_measured,
                    "utility": c.utility,
                }
                for c in choice.candidates[:_LOGGED_CANDIDATES]
            ],
            "tenant_id": tenant_id,
            "key_id": key_id,
        },
    )


def log_outcome(
    choice: RoutingChoice, served: Deployment | None, *, tenant_id: str, key_id: str
) -> None:
    """Record which model finally answered, which differs from the choice after an escalation."""
    answered = served.alias if served is not None else None
    logger.info(
        "request routed",
        extra={
            "chosen": choice.alias,
            "served": answered,
            "served_tier": None
            if served is None or served.config.tier is None
            else served.config.tier.value,
            "escalated": answered is not None and answered != choice.alias,
            "tenant_id": tenant_id,
            "key_id": key_id,
        },
    )
