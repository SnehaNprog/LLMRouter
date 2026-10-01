from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from app.evaluation.dataset import Benchmark
from app.evaluation.quality import ScoreRecord
from app.evaluation.routing_report import RoutingBreakdown, routing_breakdown
from app.evaluation.runner import EvalRecord, percentile
from app.evaluation.stats import Interval, bootstrap_ci

ClaimStatus = Literal["established", "point_estimate_only", "not_met", "unknown"]
SMALL_SAMPLE = 200  # below this many items, intervals are wide enough to say so
WEAK_BASELINE = 0.7  # a baseline scoring below this makes "quality retained" hard to read


@dataclass(frozen=True)
class Targets:
    """What the evaluation is checked against (the figures quoted on the résumé)."""

    cost_saved: float = 0.70
    quality_retained: float = 0.93


class Metric(BaseModel):
    value: float | None
    ci: Interval | None = None  # 95%, by bootstrap over the benchmark's items


class GroupResult(BaseModel):
    cost_pairs: int
    quality_pairs: int
    cost_saved: float | None
    baseline_quality: float | None
    router_quality: float | None
    quality_retained: float | None


class LatencyResult(BaseModel):
    pairs: int
    baseline_p50_s: float | None
    router_p50_s: float | None
    baseline_p95_s: float | None
    router_p95_s: float | None

    @property
    def p50_speedup(self) -> float | None:
        if not self.baseline_p50_s or not self.router_p50_s:
            return None
        return self.baseline_p50_s / self.router_p50_s


class RunFacts(BaseModel):
    label: str
    items: int
    failed: int
    unscored: int
    served_by: dict[str, int]
    judge_models: list[str]


class Claim(BaseModel):
    name: str
    target: float
    value: float | None
    ci_low: float | None
    status: ClaimStatus


class EvaluationReport(BaseModel):
    benchmark_items: int
    mix: dict[str, float]
    baseline: RunFacts
    router: RunFacts
    cost_pairs: int
    quality_pairs: int
    baseline_cost_usd: float
    router_cost_usd: float
    cost_saved: Metric
    baseline_quality: float | None
    router_quality: float | None
    quality_retained: Metric
    quality_delta: Metric  # router minus baseline, in score points (0 to 1)
    parity_rate: float | None  # share of items where the router did at least as well
    latency: LatencyResult
    routing: RoutingBreakdown | None
    by_difficulty: dict[str, GroupResult]
    by_category: dict[str, GroupResult]
    claims: list[Claim]
    warnings: list[str]


def judge_claim(name: str, target: float, metric: Metric) -> Claim:
    """Whether a measured figure meets its target, and how firmly.

    ``established`` means even the low end of the 95% interval meets it; ``point_estimate_only``
    means the best estimate does but the interval reaches below, so the data cannot rule out that
    the true figure falls short.
    """
    low = metric.ci.low if metric.ci is not None else None
    if metric.value is None:
        status: ClaimStatus = "unknown"
    elif low is not None and low >= target:
        status = "established"
    elif metric.value >= target:
        status = "point_estimate_only"
    else:
        status = "not_met"
    return Claim(name=name, target=target, value=metric.value, ci_low=low, status=status)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _facts(records: list[EvalRecord], scores: list[ScoreRecord]) -> RunFacts:
    served: dict[str, int] = {}
    for record in records:
        if record.ok:
            key = record.served_model or "unknown"
            served[key] = served.get(key, 0) + 1
    return RunFacts(
        label=records[0].run if records else "",
        items=len(records),
        failed=sum(1 for r in records if not r.ok),
        unscored=sum(1 for s in scores if s.score is None),
        served_by=dict(sorted(served.items())),
        judge_models=sorted({s.judge_model for s in scores if s.judge_model}),
    )


def saved_ratio(baseline: list[float], router: list[float], idx: list[int]) -> float | None:
    base = sum(baseline[i] for i in idx)
    return None if base <= 0 else 1 - sum(router[i] for i in idx) / base


def retained_ratio(baseline: list[float], router: list[float], idx: list[int]) -> float | None:
    base = sum(baseline[i] for i in idx)
    return None if base <= 0 else sum(router[i] for i in idx) / base


def _delta(baseline: list[float], router: list[float], idx: list[int]) -> float | None:
    if not idx:
        return None
    return (sum(router[i] for i in idx) - sum(baseline[i] for i in idx)) / len(idx)


def evaluate(
    benchmark: Benchmark,
    baseline: list[EvalRecord],
    router: list[EvalRecord],
    baseline_scores: list[ScoreRecord],
    router_scores: list[ScoreRecord],
    *,
    targets: Targets | None = None,
    seed: int = 0,
    resamples: int = 2000,
) -> EvaluationReport:
    """Compare the router with the always-frontier baseline, item by item.

    Every figure is *paired*: an item counts only if both runs have a usable result for it, so
    the two sides always describe the same prompts. Cost is compared over the items both runs
    answered and priced; quality over the items both runs have a score for (an answer that
    failed counts as a 0 there, since a failure is the worst quality). The report says how many
    items each comparison rests on and warns where that falls short of the benchmark.
    """
    targets = targets or Targets()
    known = {item.id for item in benchmark}
    for source in (baseline, router, baseline_scores, router_scores):
        for entry in source:
            if entry.item_id not in known:
                raise ValueError(
                    f"the results contain '{entry.item_id}', which is not in the benchmark"
                )
    base_rec, route_rec = {r.item_id: r for r in baseline}, {r.item_id: r for r in router}
    base_sc = {s.item_id: s for s in baseline_scores}
    route_sc = {s.item_id: s for s in router_scores}
    shared = [i.id for i in benchmark if i.id in base_rec and i.id in route_rec]

    cost_ids = [
        i
        for i in shared
        if base_rec[i].ok
        and route_rec[i].ok
        and base_rec[i].cost_usd is not None
        and route_rec[i].cost_usd is not None
    ]
    quality_ids = [
        i
        for i in shared
        if i in base_sc
        and i in route_sc
        and base_sc[i].score is not None
        and route_sc[i].score is not None
    ]
    latency_ids = [
        i
        for i in shared
        if base_rec[i].ok
        and route_rec[i].ok
        and base_rec[i].latency_s is not None
        and route_rec[i].latency_s is not None
    ]

    base_cost = [base_rec[i].cost_usd or 0.0 for i in cost_ids]
    route_cost = [route_rec[i].cost_usd or 0.0 for i in cost_ids]
    base_q = [base_sc[i].score or 0.0 for i in quality_ids]
    route_q = [route_sc[i].score or 0.0 for i in quality_ids]

    all_idx = list(range(len(cost_ids)))
    q_idx = list(range(len(quality_ids)))
    cost_saved = Metric(
        value=saved_ratio(base_cost, route_cost, all_idx),
        ci=bootstrap_ci(
            len(cost_ids),
            lambda idx: saved_ratio(base_cost, route_cost, idx),
            resamples=resamples,
            seed=seed,
        )
        if cost_ids
        else None,
    )
    retained = Metric(
        value=retained_ratio(base_q, route_q, q_idx),
        ci=bootstrap_ci(
            len(quality_ids),
            lambda idx: retained_ratio(base_q, route_q, idx),
            resamples=resamples,
            seed=seed,
        )
        if quality_ids
        else None,
    )
    delta = Metric(
        value=_delta(base_q, route_q, q_idx),
        ci=bootstrap_ci(
            len(quality_ids),
            lambda idx: _delta(base_q, route_q, idx),
            resamples=resamples,
            seed=seed,
        )
        if quality_ids
        else None,
    )

    def group(key: Literal["difficulty", "category"]) -> dict[str, GroupResult]:
        names = sorted({getattr(benchmark.get(i), key) for i in shared})
        out: dict[str, GroupResult] = {}
        for name in names:
            cidx = [n for n, i in enumerate(cost_ids) if getattr(benchmark.get(i), key) == name]
            qidx = [n for n, i in enumerate(quality_ids) if getattr(benchmark.get(i), key) == name]
            out[name] = GroupResult(
                cost_pairs=len(cidx),
                quality_pairs=len(qidx),
                cost_saved=saved_ratio(base_cost, route_cost, cidx),
                baseline_quality=_mean([base_q[n] for n in qidx]),
                router_quality=_mean([route_q[n] for n in qidx]),
                quality_retained=retained_ratio(base_q, route_q, qidx),
            )
        return out

    base_lat = [base_rec[i].latency_s or 0.0 for i in latency_ids]
    route_lat = [route_rec[i].latency_s or 0.0 for i in latency_ids]
    latency = LatencyResult(
        pairs=len(latency_ids),
        baseline_p50_s=percentile(base_lat, 0.5),
        router_p50_s=percentile(route_lat, 0.5),
        baseline_p95_s=percentile(base_lat, 0.95),
        router_p95_s=percentile(route_lat, 0.95),
    )

    breakdown = routing_breakdown(router, benchmark)
    base_facts, route_facts = _facts(baseline, baseline_scores), _facts(router, router_scores)
    report = EvaluationReport(
        benchmark_items=len(benchmark),
        mix={
            "easy": benchmark.share("easy"),
            "medium": benchmark.share("medium"),
            "hard": benchmark.share("hard"),
        },
        baseline=base_facts,
        router=route_facts,
        cost_pairs=len(cost_ids),
        quality_pairs=len(quality_ids),
        baseline_cost_usd=sum(base_cost),
        router_cost_usd=sum(route_cost),
        cost_saved=cost_saved,
        baseline_quality=_mean(base_q),
        router_quality=_mean(route_q),
        quality_retained=retained,
        quality_delta=delta,
        parity_rate=(
            sum(1 for b, r in zip(base_q, route_q, strict=True) if r >= b - 1e-9) / len(base_q)
            if base_q
            else None
        ),
        latency=latency,
        routing=breakdown if breakdown.answered else None,
        by_difficulty=group("difficulty"),
        by_category=group("category"),
        claims=[
            judge_claim("cost saved", targets.cost_saved, cost_saved),
            judge_claim("quality retained", targets.quality_retained, retained),
        ],
        warnings=[],
    )
    report.warnings = _warnings(report, len(shared), len(benchmark))
    return report


def _warnings(report: EvaluationReport, shared: int, total: int) -> list[str]:
    notes: list[str] = []
    base, route = report.baseline, report.router
    if shared < total:
        notes.append(
            f"only {shared} of the benchmark's {total} items have an answer in both runs; "
            "the comparison covers those"
        )
    for facts in (base, route):
        if facts.failed:
            notes.append(f"run '{facts.label}' has {facts.failed} failed request(s)")
        if facts.unscored:
            notes.append(
                f"run '{facts.label}' has {facts.unscored} answer(s) with no quality score "
                "(left out of the quality figures, not counted as 0)"
            )
    if report.cost_pairs < shared:
        notes.append(
            f"cost is compared on {report.cost_pairs} of {shared} items; the rest were not "
            "answered and priced in both runs"
        )
    if report.quality_pairs < shared:
        notes.append(f"quality is compared on {report.quality_pairs} of {shared} items")
    if len(base.served_by) > 1:
        notes.append(
            f"the baseline was served by {len(base.served_by)} different models "
            f"{list(base.served_by)}, so it is not an always-one-model baseline"
        )
    if report.routing is not None and report.routing.policy_shaped:
        notes.append(
            f"{report.routing.policy_shaped} router answer(s) were shaped by an organisation "
            "routing policy, so this is not the router on its own"
        )
    judges = set(base.judge_models) | set(route.judge_models)
    if len(judges) > 1:
        notes.append(f"the two runs were graded by different judges {sorted(judges)}")
    used = set(base.served_by) | set(route.served_by)
    if judges & used:
        notes.append(
            f"judge model(s) {sorted(judges & used)} also wrote answers being compared; a model "
            "tends to grade its own style generously"
        )
    if report.baseline_quality is not None and report.baseline_quality < WEAK_BASELINE:
        notes.append(
            f"the baseline itself scores only {report.baseline_quality:.2f}, so a retained-quality "
            "ratio is measured against a weak yardstick"
        )
    if report.benchmark_items < SMALL_SAMPLE:
        notes.append(
            f"{report.benchmark_items} prompts is a small sample; read the confidence intervals, "
            "not just the point estimates"
        )
    return notes


def format_report(report: EvaluationReport) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    def metric(m: Metric) -> str:
        if m.value is None:
            return "n/a"
        if m.ci is None:
            return pct(m.value)
        return f"{pct(m.value)} (95% CI {pct(m.ci.low)} to {pct(m.ci.high)})"

    def seconds(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}s"

    mix = report.mix
    lines = [
        f"EVALUATION  router '{report.router.label}' vs baseline '{report.baseline.label}'",
        f"  benchmark: {report.benchmark_items} prompts "
        f"({pct(mix['easy'])} easy, {pct(mix['medium'])} medium, {pct(mix['hard'])} hard)",
        "",
        f"  cost     ${report.baseline_cost_usd:.6f} -> ${report.router_cost_usd:.6f}   "
        f"saved {metric(report.cost_saved)}   [{report.cost_pairs} items]",
    ]
    b, r = report.baseline_quality, report.router_quality
    lines.append(
        f"  quality  baseline {'n/a' if b is None else f'{b:.3f}'}, "
        f"router {'n/a' if r is None else f'{r:.3f}'}   "
        f"retained {metric(report.quality_retained)}   [{report.quality_pairs} items]"
    )
    if report.parity_rate is not None:
        lines.append(
            f"           router matched or beat the baseline on {pct(report.parity_rate)} of items"
        )
    lat = report.latency
    speed = lat.p50_speedup
    lines.append(
        f"  latency  p50 {seconds(lat.baseline_p50_s)} -> {seconds(lat.router_p50_s)}"
        + (f" ({speed:.2f}x)" if speed is not None else "")
        + f", p95 {seconds(lat.baseline_p95_s)} -> {seconds(lat.router_p95_s)}"
        + f"   [{lat.pairs} items]"
    )
    if report.routing is not None:
        tiers = ", ".join(f"{t} {n}" for t, n in report.routing.by_tier.items())
        lines.append(
            f"  routing  {tiers}; escalated {report.routing.escalated}, "
            f"degraded {report.routing.degraded}"
        )
    for title, table in (("difficulty", report.by_difficulty), ("category", report.by_category)):
        lines.append("")
        lines.append(f"  by {title}:")
        for key, g in table.items():
            lines.append(
                f"    {key:<11} saved {pct(g.cost_saved):>7}"
                f"   retained {pct(g.quality_retained):>7}"
                f"   (n={g.quality_pairs})"
            )
    lines.append("")
    lines.append("  against the targets:")
    for claim in report.claims:
        low = "" if claim.ci_low is None else f", CI lower bound {pct(claim.ci_low)}"
        lines.append(
            f"    {claim.name} >= {pct(claim.target)}: {claim.status.replace('_', ' ').upper()} "
            f"(measured {pct(claim.value)}{low})"
        )
    if report.warnings:
        lines.append("")
        lines.append("  read with care:")
        lines.extend(f"    - {w}" for w in report.warnings)
    return "\n".join(lines)
