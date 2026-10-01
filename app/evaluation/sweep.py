from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import BaseModel

from app.core.model_config import load_config
from app.core.tiers import TIERS, Tier
from app.evaluation.dataset import Benchmark
from app.evaluation.evaluate import (
    EvaluationReport,
    Targets,
    evaluate,
    format_report,
    retained_ratio,
    saved_ratio,
)
from app.evaluation.pricing import AnyKey
from app.evaluation.quality import ScoreRecord, load_scores
from app.evaluation.runner import EvalRecord, load_records, percentile
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry
from app.routing.policy import RoutingPolicy
from app.routing.score import ComplexityScore, ComplexityWeights, score_request
from app.routing.selector import NoRoutableModelError, choose_model

STEP = 0.05
DEFAULT_TARGET = 0.93


class SweepDataError(Exception):
    """The runs the sweep replays are missing or incomplete. The message lists every gap."""


@dataclass(frozen=True)
class Answer:
    """What one model did with one benchmark item: all the sweep needs to replay a choice."""

    record: EvalRecord
    score: ScoreRecord

    @property
    def cost(self) -> float:
        assert self.record.cost_usd is not None
        return self.record.cost_usd

    @property
    def quality(self) -> float:
        assert self.score.score is not None
        return self.score.score


Runs = dict[str, dict[str, Answer]]  # alias -> item id -> answer


class Setup:
    """The gateway's routing configuration, loaded without its API keys or any provider."""

    def __init__(self, config: Path) -> None:
        loaded = load_config(config, AnyKey())
        self.configs = loaded.models
        self.policy: RoutingPolicy = loaded.policy
        self.weights = ComplexityWeights.from_overrides(loaded.complexity)
        self.registry = ModelRegistry(
            [Deployment.create(c, FakeProvider(c.name)) for c in self.configs]
        )

    def tiered(self) -> dict[str, Tier]:
        return {alias: tier for tier, aliases in self.registry.tiers().items() for alias in aliases}

    def frontier_aliases(self) -> list[str]:
        return self.registry.aliases_in_tier(Tier.FRONTIER)


def run_path(runs_dir: Path, alias: str) -> Path:
    return runs_dir / f"model-{alias}.jsonl"


def load_runs(
    runs_dir: Path, aliases: list[str], benchmark: Benchmark, scores_of: Callable[[Path], Path]
) -> Runs:
    """Every tiered model's answer to every benchmark item, with its score.

    The sweep replays routing choices from these, so a gap would silently skew a result: every
    item must have been answered, priced and scored for every model, or it refuses to run and
    lists what to redo.
    """
    problems: list[str] = []
    runs: Runs = {}
    for alias in aliases:
        path = run_path(runs_dir, alias)
        try:
            records = {r.item_id: r for r in load_records(path)}
            scores = {s.item_id: s for s in load_scores(scores_of(path))}
        except FileNotFoundError as exc:
            problems.append(f"{alias}: {exc.filename} not found")
            continue
        answers: dict[str, Answer] = {}
        for item in benchmark:
            record, score = records.get(item.id), scores.get(item.id)
            if record is None or not record.ok:
                problems.append(f"{alias}: no answer for {item.id}")
            elif record.cost_usd is None:
                problems.append(f"{alias}: {item.id} could not be priced")
            elif score is None or score.score is None:
                problems.append(f"{alias}: {item.id} has no quality score")
            else:
                answers[item.id] = Answer(record, score)
        runs[alias] = answers
    if problems:
        shown = problems[:20]
        more = f"\n  - ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise SweepDataError(
            "the model runs are incomplete (rerun the missing ones with --resume and score "
            "them):\n  - " + "\n  - ".join(shown) + more
        )
    return runs


def threshold_grid(step: float = STEP) -> list[tuple[float, float]]:
    """Every (slm ceiling, mid ceiling) pair on the grid, with the slm ceiling strictly lower."""
    steps = round(1 / step)
    values = [round(n * step, 4) for n in range(1, steps + 1)]
    return [(low, high) for low in values for high in values if low < high]


def split_items(benchmark: Benchmark) -> tuple[list[str], list[str]]:
    """Share the items between a tuning half and a held-out half, evenly within each category.

    Thresholds are picked on the first and judged on the second, so the figure quoted for the
    chosen thresholds is not flattered by having been fitted to the very prompts it is measured
    on. The split is fixed by the ids alone: the same benchmark always splits the same way.
    """
    tune: list[str] = []
    holdout: list[str] = []
    seen: dict[str, int] = {}
    for item in benchmark:
        position = seen.get(item.category, 0)
        seen[item.category] = position + 1
        (tune if position % 2 == 0 else holdout).append(item.id)
    return tune, holdout


class SweepPoint(BaseModel):
    slm_ceiling: float
    mid_ceiling: float
    tiers: dict[str, int]  # how many benchmark items were routed to each tier
    saved_all: float | None
    retained_all: float | None
    saved_tune: float | None
    retained_tune: float | None
    saved_holdout: float | None
    retained_holdout: float | None


class SweepReport(BaseModel):
    baseline_model: str
    models: dict[str, str]  # alias -> tier
    items: int
    tune_items: int
    holdout_items: int
    target_retained: float
    grid_points: int
    points: list[SweepPoint]
    pareto: list[SweepPoint]
    default_point: SweepPoint | None
    selected: SweepPoint | None
    holdout: EvaluationReport | None  # the selected thresholds, judged on the held-out half
    notes: list[str]


class Replay:
    """Works out what the router would have done at any thresholds, from stored answers."""

    def __init__(self, setup: Setup, benchmark: Benchmark, runs: Runs, baseline_model: str) -> None:
        self.setup, self.benchmark, self.runs = setup, benchmark, runs
        self.baseline_model = baseline_model
        self.ids = [item.id for item in benchmark]
        weights = setup.weights
        self.complexity = {
            item.id: score_request(item.to_request("auto"), weights).value for item in benchmark
        }
        self._by_tier = self._choices_by_tier()
        self.base_cost = [runs[baseline_model][i].cost for i in self.ids]
        self.base_quality = [runs[baseline_model][i].quality for i in self.ids]

    def _tracker(self) -> LatencyTracker:
        """Latencies as measured in the runs, so the choice among a tier's models is the one the
        live router would make once it had seen some traffic."""
        tracker = LatencyTracker()
        for alias, answers in self.runs.items():
            seen = [a.record.latency_s for a in answers.values() if a.record.latency_s is not None]
            median = percentile(seen, 0.5)
            if median is not None:
                for deployment in self.setup.registry.get(alias):
                    tracker.observe(deployment.id, median)
        return tracker

    def _choices_by_tier(self) -> dict[str, dict[Tier, str]]:
        """For each item, the model the selector picks when a given tier is required.

        What a prompt needs is the only thing the thresholds change; which model of that tier is
        then picked does not depend on them. So the choice is worked out once per item and tier
        (forcing the required tier) and looked up for every setting of the thresholds.
        """
        forced = {
            Tier.SLM: replace(self.setup.policy, slm_ceiling=1.0, mid_ceiling=1.0),
            Tier.MID: replace(self.setup.policy, slm_ceiling=-1.0, mid_ceiling=1.0),
            Tier.FRONTIER: replace(self.setup.policy, slm_ceiling=-1.0, mid_ceiling=-0.5),
        }
        tracker = self._tracker()
        table: dict[str, dict[Tier, str]] = {}
        for item in self.benchmark:
            request = item.to_request("auto")
            table[item.id] = {}
            for tier in TIERS:
                try:
                    choice = choose_model(
                        request,
                        self.setup.registry,
                        tracker,
                        policy=forced[tier],
                        weights=self.setup.weights,
                        score=ComplexityScore(value=0.5, contributions={}, small_talk=False),
                    )
                except NoRoutableModelError as exc:
                    raise SweepDataError(f"{item.id}: nothing can serve it ({exc})") from None
                table[item.id][tier] = choice.alias
        return table

    def assignment(self, policy: RoutingPolicy) -> list[str]:
        """The model chosen for each item, in benchmark order, under these thresholds."""
        return [self._by_tier[i][policy.required_tier(self.complexity[i])] for i in self.ids]

    def point(self, slm: float, mid: float, tune: list[int], holdout: list[int]) -> SweepPoint:
        policy = replace(self.setup.policy, slm_ceiling=slm, mid_ceiling=mid)
        chosen = self.assignment(policy)
        cost = [self.runs[a][i].cost for a, i in zip(chosen, self.ids, strict=True)]
        quality = [self.runs[a][i].quality for a, i in zip(chosen, self.ids, strict=True)]
        everything = list(range(len(self.ids)))
        tiers: dict[str, int] = {}
        for alias in chosen:
            tier = self.setup.tiered()[alias].value
            tiers[tier] = tiers.get(tier, 0) + 1

        def measures(idx: list[int]) -> tuple[float | None, float | None]:
            return (
                saved_ratio(self.base_cost, cost, idx),
                retained_ratio(self.base_quality, quality, idx),
            )

        saved_all, retained_all = measures(everything)
        saved_tune, retained_tune = measures(tune)
        saved_holdout, retained_holdout = measures(holdout)
        return SweepPoint(
            slm_ceiling=slm,
            mid_ceiling=mid,
            tiers=dict(sorted(tiers.items())),
            saved_all=saved_all,
            retained_all=retained_all,
            saved_tune=saved_tune,
            retained_tune=retained_tune,
            saved_holdout=saved_holdout,
            retained_holdout=retained_holdout,
        )

    def simulated_run(
        self, slm: float, mid: float, ids: list[str]
    ) -> tuple[list[EvalRecord], list[ScoreRecord]]:
        """The records and scores the router would have produced at these thresholds."""
        policy = replace(self.setup.policy, slm_ceiling=slm, mid_ceiling=mid)
        chosen = dict(zip(self.ids, self.assignment(policy), strict=True))
        tiers = self.setup.tiered()
        records, scores = [], []
        for item_id in ids:
            alias = chosen[item_id]
            answer = self.runs[alias][item_id]
            routing = {
                "x-llmrouter-model": alias,
                "x-llmrouter-tier": tiers[alias].value,
                "x-llmrouter-escalated": "false",
                "x-llmrouter-complexity": f"{self.complexity[item_id]:.2f}",
            }
            records.append(
                answer.record.model_copy(
                    update={"run": "sweep", "served_model": alias, "routing": routing}
                )
            )
            scores.append(answer.score.model_copy(update={"run": "sweep"}))
        return records, scores


def pareto_front(points: list[SweepPoint]) -> list[SweepPoint]:
    """The settings no other setting beats on both cost saved and quality retained."""
    usable = [p for p in points if p.saved_all is not None and p.retained_all is not None]

    def beaten(p: SweepPoint) -> bool:
        assert p.saved_all is not None and p.retained_all is not None
        return any(
            q.saved_all is not None
            and q.retained_all is not None
            and q.saved_all >= p.saved_all
            and q.retained_all >= p.retained_all
            and (q.saved_all > p.saved_all or q.retained_all > p.retained_all)
            for q in usable
        )

    front: list[SweepPoint] = []
    seen: set[tuple[float, float]] = set()
    for p in sorted(usable, key=lambda p: (-(p.saved_all or 0), p.slm_ceiling, p.mid_ceiling)):
        key = (round(p.saved_all or 0, 9), round(p.retained_all or 0, 9))
        if not beaten(p) and key not in seen:
            seen.add(key)
            front.append(p)
    return front


def select_operating_point(points: list[SweepPoint], target: float) -> SweepPoint | None:
    """The cheapest thresholds that retain enough quality *on the tuning half*.

    "Cheapest" is the most cost saved; ties go to the higher retained quality, then to the
    lower thresholds, so the choice is deterministic.
    """
    eligible = [
        p
        for p in points
        if p.retained_tune is not None and p.saved_tune is not None and p.retained_tune >= target
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda p: (-(p.saved_tune or 0), -(p.retained_tune or 0), p.slm_ceiling, p.mid_ceiling),
    )


def sweep(
    setup: Setup,
    benchmark: Benchmark,
    runs: Runs,
    baseline_model: str,
    *,
    target_retained: float = DEFAULT_TARGET,
    step: float = STEP,
    seed: int = 0,
    resamples: int = 2000,
) -> SweepReport:
    """Try every threshold setting on the grid and report the trade-off.

    Each setting is replayed from stored answers (no model is called), the settings no other
    beats are listed, and one is chosen using only the tuning half of the benchmark and then
    measured, with confidence intervals, on the held-out half.
    """
    if baseline_model not in runs:
        raise SweepDataError(f"there is no run for the baseline model '{baseline_model}'")
    replay = Replay(setup, benchmark, runs, baseline_model)
    tune_ids, holdout_ids = split_items(benchmark)
    position = {item_id: n for n, item_id in enumerate(replay.ids)}
    tune = [position[i] for i in tune_ids]
    holdout = [position[i] for i in holdout_ids]

    points = [replay.point(slm, mid, tune, holdout) for slm, mid in threshold_grid(step)]
    default = setup.policy
    default_point = replay.point(default.slm_ceiling, default.mid_ceiling, tune, holdout)
    selected = select_operating_point(points, target_retained)

    holdout_report: EvaluationReport | None = None
    if selected is not None:
        records, scores = replay.simulated_run(
            selected.slm_ceiling, selected.mid_ceiling, holdout_ids
        )
        base_records = [
            runs[baseline_model][i].record.model_copy(update={"run": "baseline"})
            for i in holdout_ids
        ]
        base_scores = [
            runs[baseline_model][i].score.model_copy(update={"run": "baseline"})
            for i in holdout_ids
        ]
        holdout_report = evaluate(
            benchmark.only(holdout_ids),
            base_records,
            records,
            base_scores,
            scores,
            targets=Targets(quality_retained=target_retained),
            seed=seed,
            resamples=resamples,
        )

    return SweepReport(
        baseline_model=baseline_model,
        models={alias: tier.value for alias, tier in setup.tiered().items()},
        items=len(benchmark),
        tune_items=len(tune_ids),
        holdout_items=len(holdout_ids),
        target_retained=target_retained,
        grid_points=len(points),
        points=points,
        pareto=pareto_front(points),
        default_point=default_point,
        selected=selected,
        holdout=holdout_report,
        notes=_notes(selected, default_point, holdout_report, target_retained, len(holdout_ids)),
    )


def _notes(
    selected: SweepPoint | None,
    default_point: SweepPoint | None,
    holdout: EvaluationReport | None,
    target: float,
    holdout_items: int,
) -> list[str]:
    notes = [
        "the sweep replays stored answers: it assumes each prompt is served by the model the "
        "router picks for its tier, and leaves out escalation after failures and live latency "
        "feedback. Confirm a chosen setting by running the real router with it "
        "(`make eval-router`)",
        f"thresholds were chosen on the tuning half only; the held-out half has {holdout_items} "
        "prompts, so its intervals are wide",
    ]
    if selected is None:
        notes.append(f"no thresholds retain {target:.0%} of the quality on the tuning half")
    elif holdout is not None and holdout.quality_retained.value is not None:
        if holdout.quality_retained.value < target:
            notes.append(
                f"the chosen thresholds retained {holdout.quality_retained.value:.1%} on the "
                f"held-out half, below the {target:.0%} target they reached on the tuning half: "
                "they did not generalise"
            )
    return notes


def format_sweep(report: SweepReport) -> str:
    def pct(value: float | None) -> str:
        return "   n/a" if value is None else f"{value * 100:5.1f}%"

    def tiers(point: SweepPoint) -> str:
        return " ".join(f"{t[:3]}{n}" for t, n in point.tiers.items())

    models = ", ".join(f"{a} ({t})" for a, t in report.models.items())
    lines = [
        f"SWEEP  {report.grid_points} threshold settings, baseline '{report.baseline_model}'",
        f"  models: {models}",
        f"  items: {report.items} "
        f"({report.tune_items} for tuning, {report.holdout_items} held out)",
        "",
        "  settings no other setting beats, and how each does on all items / the tuning half /",
        "  the held-out half (cost saved, quality retained), with the tiers the items go to:",
        "    slm   mid |     all items    |  tuning half    |  held-out half  | routed",
    ]
    for p in report.pareto:
        lines.append(
            f"    {p.slm_ceiling:.2f}  {p.mid_ceiling:.2f} |"
            f" {pct(p.saved_all)} {pct(p.retained_all)} |"
            f" {pct(p.saved_tune)} {pct(p.retained_tune)} |"
            f" {pct(p.saved_holdout)} {pct(p.retained_holdout)} | {tiers(p)}"
        )
    default = report.default_point
    if default is not None:
        lines += [
            "",
            f"  current defaults ({default.slm_ceiling:.2f} / {default.mid_ceiling:.2f}): "
            f"saved {pct(default.saved_all).strip()}, retained {pct(default.retained_all).strip()}"
            f" (held out: saved {pct(default.saved_holdout).strip()}, "
            f"retained {pct(default.retained_holdout).strip()})   {tiers(default)}",
        ]
    lines.append("")
    if report.selected is None:
        lines.append(
            f"  no setting retains {report.target_retained:.0%} of the quality on the tuning half"
        )
    else:
        s = report.selected
        lines += [
            f"  chosen on the tuning half for >= {report.target_retained:.0%} retained quality: "
            f"slm_ceiling {s.slm_ceiling:.2f}, mid_ceiling {s.mid_ceiling:.2f}",
            "  to use it, put this in config.yaml:",
            "    routing:",
            "      tier_ceilings:",
            f"        slm: {s.slm_ceiling:.2f}",
            f"        mid: {s.mid_ceiling:.2f}",
        ]
        if report.holdout is not None:
            lines += ["", "  measured on the held-out half it was not tuned on:"]
            lines += ["  " + line for line in format_report(report.holdout).splitlines()]
    if report.notes:
        lines += ["", "  read with care:"]
        lines += [f"    - {n}" for n in report.notes]
    return "\n".join(lines)
