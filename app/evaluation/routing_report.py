from pydantic import BaseModel

from app.evaluation.dataset import Benchmark
from app.evaluation.runner import EvalRecord


class RoutingBreakdown(BaseModel):
    """How the router spread a benchmark run over the model tiers.

    Built from the ``x-llmrouter-*`` headers the gateway puts on every ``auto`` answer, so it
    describes what actually served each request (after any escalation), not what was planned.
    """

    answered: int  # answers that carried routing headers at all
    by_tier: dict[str, int]  # tier of the model that served the answer
    escalated: int  # served by a different model than the first choice
    degraded: int  # nothing as strong as the request needed was available
    policy_applied: dict[str, int]  # answers shaped by the organisation's routing policy
    mean_complexity: float | None
    by_difficulty: dict[str, dict[str, int]]  # the author's difficulty label -> tier -> answers
    by_category: dict[str, dict[str, int]]

    @property
    def policy_shaped(self) -> int:
        return sum(self.policy_applied.values())


def _bump(table: dict[str, dict[str, int]], row: str, column: str) -> None:
    cells = table.setdefault(row, {})
    cells[column] = cells.get(column, 0) + 1


def routing_breakdown(records: list[EvalRecord], benchmark: Benchmark) -> RoutingBreakdown:
    """Count where an ``auto`` run's answers came from, overall and per difficulty and category."""
    by_tier: dict[str, int] = {}
    by_difficulty: dict[str, dict[str, int]] = {}
    by_category: dict[str, dict[str, int]] = {}
    policies: dict[str, int] = {}
    complexities: list[float] = []
    escalated = degraded = answered = 0
    for record in records:
        if not record.ok or "x-llmrouter-tier" not in record.routing:
            continue
        answered += 1
        headers = record.routing
        tier = headers["x-llmrouter-tier"]
        by_tier[tier] = by_tier.get(tier, 0) + 1
        item = benchmark.get(record.item_id)
        _bump(by_difficulty, item.difficulty, tier)
        _bump(by_category, item.category, tier)
        escalated += headers.get("x-llmrouter-escalated") == "true"
        degraded += headers.get("x-llmrouter-degraded") == "true"
        if policy := headers.get("x-llmrouter-policy"):
            policies[policy] = policies.get(policy, 0) + 1
        try:
            complexities.append(float(headers["x-llmrouter-complexity"]))
        except (KeyError, ValueError):
            pass
    return RoutingBreakdown(
        answered=answered,
        by_tier=dict(sorted(by_tier.items())),
        escalated=escalated,
        degraded=degraded,
        policy_applied=dict(sorted(policies.items())),
        mean_complexity=sum(complexities) / len(complexities) if complexities else None,
        by_difficulty={k: dict(sorted(v.items())) for k, v in sorted(by_difficulty.items())},
        by_category={k: dict(sorted(v.items())) for k, v in sorted(by_category.items())},
    )


def format_breakdown(breakdown: RoutingBreakdown) -> str:
    def row(counts: dict[str, int]) -> str:
        return ", ".join(f"{tier} {n}" for tier, n in counts.items())

    lines = [
        f"routing ({breakdown.answered} answers): {row(breakdown.by_tier) or 'no routing headers'}",
        f"  escalated {breakdown.escalated}, degraded {breakdown.degraded}"
        + (
            f", mean complexity {breakdown.mean_complexity:.2f}"
            if breakdown.mean_complexity is not None
            else ""
        ),
    ]
    for name, table in (
        ("difficulty", breakdown.by_difficulty),
        ("category", breakdown.by_category),
    ):
        for key, counts in table.items():
            lines.append(f"  {name} {key:<10} {row(counts)}")
    return "\n".join(lines)
