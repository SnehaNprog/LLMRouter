import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.evaluation import cli
from app.evaluation.dataset import load_benchmark
from app.evaluation.evaluate import (
    EvaluationReport,
    Metric,
    Targets,
    evaluate,
    format_report,
    judge_claim,
)
from app.evaluation.quality import ScoreRecord
from app.evaluation.runner import EvalRecord
from app.evaluation.stats import Interval, bootstrap_ci

BENCH = load_benchmark()
IDS = [i.id for i in BENCH]


# ---------- bootstrap ----------


def mean_of(values: list[float]) -> Any:
    return lambda idx: sum(values[i] for i in idx) / len(idx)


def test_the_same_seed_gives_the_same_interval() -> None:
    values = [random.Random(1).random() * (i + 1) for i in range(40)]
    first = bootstrap_ci(40, mean_of(values), seed=3)
    again = bootstrap_ci(40, mean_of(values), seed=3)
    other = bootstrap_ci(40, mean_of(values), seed=4)
    assert first == again
    assert first != other


def test_the_interval_is_the_middle_of_the_resampled_values() -> None:
    draws = iter(range(1000))  # 1000 resamples whose statistics are 0, 1, ..., 999
    ci = bootstrap_ci(5, lambda idx: float(next(draws)), resamples=1000, level=0.90)
    assert ci == Interval(low=50.0, high=949.0)  # 5% of the draws lie below, 5% above


def test_the_interval_is_the_same_whatever_order_the_draws_come_in() -> None:
    values = list(range(1000))
    random.Random(2).shuffle(values)
    draws = iter(values)
    ci = bootstrap_ci(5, lambda idx: float(next(draws)), resamples=1000, level=0.90)
    assert ci == Interval(low=50.0, high=949.0)


def test_the_interval_brackets_the_estimate() -> None:
    values = [float(i % 2) for i in range(100)]  # a mean of exactly 0.5
    ci = bootstrap_ci(100, mean_of(values), resamples=1000)
    assert ci is not None and ci.low < 0.5 < ci.high
    assert 0.35 < ci.low and ci.high < 0.65


def test_a_constant_statistic_has_a_zero_width_interval() -> None:
    ci = bootstrap_ci(10, lambda idx: 0.7)
    assert ci == Interval(low=0.7, high=0.7)


def test_more_data_narrows_the_interval() -> None:
    def width(n: int) -> float:
        values = [float(i % 2) for i in range(n)]
        ci = bootstrap_ci(n, mean_of(values), resamples=1000)
        assert ci is not None
        return ci.high - ci.low

    assert width(400) < width(25) / 2


def test_a_level_changes_how_wide_it_is() -> None:
    values = [float(i % 5) for i in range(60)]
    narrow = bootstrap_ci(60, mean_of(values), level=0.5)
    wide = bootstrap_ci(60, mean_of(values), level=0.99)
    assert narrow is not None and wide is not None
    assert wide.low <= narrow.low and wide.high >= narrow.high
    assert wide.high - wide.low > narrow.high - narrow.low


def test_resamples_where_the_statistic_is_undefined_are_skipped() -> None:
    calls = iter(range(1_000_000))
    ci = bootstrap_ci(5, lambda idx: None if next(calls) % 2 else 1.0, resamples=100)
    assert ci == Interval(low=1.0, high=1.0)


def test_with_no_defined_resample_there_is_no_interval() -> None:
    assert bootstrap_ci(5, lambda idx: None, resamples=20) is None


@pytest.mark.parametrize(
    "kwargs", [{"n": 0}, {"n": 5, "resamples": 0}, {"n": 5, "level": 0}, {"n": 5, "level": 1}]
)
def test_nonsense_arguments_are_refused(kwargs: dict[str, Any]) -> None:
    n = kwargs.pop("n")
    with pytest.raises(ValueError):
        bootstrap_ci(n, lambda idx: 1.0, **kwargs)


def test_an_interval_knows_what_it_contains() -> None:
    assert Interval(low=1, high=2).contains(1.5)
    assert not Interval(low=1, high=2).contains(2.5)


# ---------- building synthetic runs ----------


def run(
    label: str,
    *,
    cost: float | Mapping[str, float] = 1.0,
    latency: float | Mapping[str, float] = 1.0,
    served: str = "smart",
    routing: dict[str, str] | None = None,
    ids: list[str] | None = None,
    failed: set[str] = frozenset(),  # type: ignore[assignment]
) -> list[EvalRecord]:
    records = []
    for item_id in ids or IDS:
        ok = item_id not in failed
        records.append(
            EvalRecord(
                item_id=item_id,
                run=label,
                requested_model="x",
                served_model=served if ok else None,
                ok=ok,
                reply="r" if ok else None,
                error=None if ok else "boom",
                cost_usd=(cost[item_id] if isinstance(cost, dict) else cost) if ok else None,
                latency_s=(latency[item_id] if isinstance(latency, dict) else latency)
                if ok
                else None,
                routing=routing or {},
            )
        )
    return records


def scores(
    label: str,
    value: float | Mapping[str, float | None] = 1.0,
    *,
    judge: str | None = "grader",
    ids: list[str] | None = None,
) -> list[ScoreRecord]:
    out = []
    for item_id in ids or IDS:
        score = value[item_id] if isinstance(value, dict) else value
        out.append(
            ScoreRecord(
                item_id=item_id,
                run=label,
                method="rubric" if judge else "number",
                score=score,
                judge_model=judge,
            )
        )
    return out


def compare(
    baseline: list[EvalRecord],
    router: list[EvalRecord],
    base_scores: list[ScoreRecord] | None = None,
    route_scores: list[ScoreRecord] | None = None,
    **kwargs: Any,
) -> EvaluationReport:
    return evaluate(
        BENCH,
        baseline,
        router,
        base_scores if base_scores is not None else scores("baseline"),
        route_scores if route_scores is not None else scores("router"),
        resamples=200,
        **kwargs,
    )


# ---------- cost ----------


def test_cost_saved_is_one_minus_the_ratio_of_totals() -> None:
    report = compare(run("baseline", cost=1.0), run("router", cost=0.3))
    assert report.cost_saved.value == pytest.approx(0.7)
    assert report.baseline_cost_usd == pytest.approx(90.0)
    assert report.router_cost_usd == pytest.approx(27.0)
    assert report.cost_pairs == 90
    assert report.cost_saved.ci is not None
    assert report.cost_saved.ci.low == pytest.approx(0.7)
    assert report.cost_saved.ci.high == pytest.approx(0.7)


def test_it_is_a_ratio_of_totals_not_an_average_of_ratios() -> None:
    base = {i: 1.0 for i in IDS} | {IDS[0]: 100.0}
    cheap = {i: 1.0 for i in IDS} | {IDS[0]: 100.0}
    cheap[IDS[1]] = 0.0  # one cheap item saved entirely, one dear item not at all
    report = compare(run("baseline", cost=base), run("router", cost=cheap))
    assert report.cost_saved.value == pytest.approx(1 / 189)  # 189 spent before, 188 after


def test_a_router_that_costs_more_saves_a_negative_amount() -> None:
    report = compare(run("baseline", cost=1.0), run("router", cost=1.5))
    assert report.cost_saved.value == pytest.approx(-0.5)


def test_a_failed_router_answer_is_left_out_of_the_cost_comparison() -> None:
    failed = {IDS[0], IDS[1]}
    report = compare(run("baseline", cost=1.0), run("router", cost=0.5, failed=failed))
    assert report.cost_pairs == 88
    assert report.cost_saved.value == pytest.approx(0.5)  # not flattered by the two it skipped
    assert any("cost is compared on 88 of 90" in w for w in report.warnings)
    assert any("1 failed" in w or "2 failed" in w for w in report.warnings)


def test_unpriced_answers_are_left_out_of_the_cost_comparison() -> None:
    router = run("router", cost=0.5)
    router[0] = router[0].model_copy(update={"cost_usd": None})
    report = compare(run("baseline", cost=1.0), router)
    assert report.cost_pairs == 89


def test_a_free_baseline_leaves_cost_saved_undefined() -> None:
    report = compare(run("baseline", cost=0.0), run("router", cost=0.0))
    assert report.cost_saved.value is None and report.cost_saved.ci is None


# ---------- quality ----------


def test_quality_retained_is_the_ratio_of_mean_scores() -> None:
    base = scores("baseline", {i: 1.0 for i in IDS})
    half = {i: 1.0 for i in IDS} | {i: 0.5 for i in IDS[:10]}
    report = compare(run("baseline"), run("router"), base, scores("router", half))
    assert report.baseline_quality == pytest.approx(1.0)
    assert report.router_quality == pytest.approx(85 / 90)
    assert report.quality_retained.value == pytest.approx(85 / 90)
    assert report.quality_delta.value == pytest.approx(-5 / 90)
    assert report.parity_rate == pytest.approx(80 / 90)
    assert report.quality_pairs == 90


def test_a_better_router_retains_more_than_all_of_it() -> None:
    base = scores("baseline", 0.5)
    report = compare(run("baseline"), run("router"), base, scores("router", 0.75))
    assert report.quality_retained.value == pytest.approx(1.5)
    assert report.parity_rate == 1.0


def test_parity_counts_ties_as_matching() -> None:
    report = compare(run("baseline"), run("router"), scores("baseline", 0.6), scores("router", 0.6))
    assert report.parity_rate == 1.0


def test_unscored_answers_are_left_out_not_counted_as_zero() -> None:
    values: dict[str, float | None] = {i: 1.0 for i in IDS} | {IDS[0]: None, IDS[1]: None}
    report = compare(run("baseline"), run("router"), None, scores("router", values))
    assert report.quality_pairs == 88
    assert report.quality_retained.value == pytest.approx(1.0)
    assert any("2 answer(s) with no quality score" in w for w in report.warnings)


def test_a_failed_answer_counts_as_a_zero_in_quality() -> None:
    values = {i: 1.0 for i in IDS} | {IDS[0]: 0.0}  # what scoring gives a failed request
    report = compare(
        run("baseline"), run("router", failed={IDS[0]}), None, scores("router", values)
    )
    assert report.quality_pairs == 90
    assert report.quality_retained.value == pytest.approx(89 / 90)


def test_a_baseline_of_zero_quality_leaves_retention_undefined() -> None:
    report = compare(run("baseline"), run("router"), scores("baseline", 0.0), scores("router", 1.0))
    assert report.quality_retained.value is None


def test_items_missing_from_one_run_are_excluded_and_reported() -> None:
    few = IDS[:60]
    report = compare(run("baseline"), run("router", ids=few), None, scores("router", ids=few))
    assert report.cost_pairs == report.quality_pairs == 60
    assert any(
        "only 60 of the benchmark's 90 items have an answer in both runs" in w
        for w in report.warnings
    )


def test_items_the_baseline_lacks_are_excluded_too() -> None:
    few = IDS[:60]
    report = compare(
        run("baseline", ids=few), run("router"), scores("baseline", ids=few), scores("router")
    )
    assert report.cost_pairs == report.quality_pairs == 60
    assert any("only 60 of the benchmark's 90" in w for w in report.warnings)


# ---------- latency ----------


def test_latency_is_compared_on_paired_answered_items() -> None:
    base = {i: 2.0 for i in IDS}
    fast = {i: 1.0 for i in IDS} | {IDS[0]: 9.0}
    report = compare(
        run("baseline", latency=base),
        run("router", latency=fast, failed={IDS[1]}),
    )
    lat = report.latency
    assert lat.pairs == 89
    assert (lat.baseline_p50_s, lat.router_p50_s) == (2.0, 1.0)
    assert lat.router_p95_s == 1.0 and lat.baseline_p95_s == 2.0
    assert lat.p50_speedup == pytest.approx(2.0)


def test_speedup_is_undefined_without_latencies() -> None:
    report = compare(run("baseline", failed=set(IDS)), run("router", failed=set(IDS)))
    assert report.latency.pairs == 0 and report.latency.p50_speedup is None


# ---------- breakdowns ----------


def test_results_are_broken_down_by_difficulty_and_category() -> None:
    cheap = {i: (0.0 if BENCH.get(i).difficulty == "easy" else 1.0) for i in IDS}
    report = compare(run("baseline", cost=1.0), run("router", cost=cheap))
    assert report.by_difficulty["easy"].cost_saved == pytest.approx(1.0)
    assert report.by_difficulty["hard"].cost_saved == pytest.approx(0.0)
    assert report.by_difficulty["easy"].cost_pairs == 42
    assert set(report.by_category) == {"chat", "extraction", "qa", "math", "code", "reasoning"}
    assert sum(g.quality_pairs for g in report.by_category.values()) == 90


def test_group_quality_is_worked_out_inside_the_group() -> None:
    values = {i: (1.0 if BENCH.get(i).category == "math" else 0.5) for i in IDS}
    report = compare(
        run("baseline"), run("router"), scores("baseline", 1.0), scores("router", values)
    )
    assert report.by_category["math"].quality_retained == pytest.approx(1.0)
    assert report.by_category["chat"].quality_retained == pytest.approx(0.5)
    assert report.by_category["chat"].router_quality == pytest.approx(0.5)


def test_the_mix_of_the_benchmark_is_reported() -> None:
    report = compare(run("baseline"), run("router"))
    assert report.mix == {"easy": 42 / 90, "medium": 23 / 90, "hard": 25 / 90}


def test_routing_is_summarised_from_the_routers_headers() -> None:
    headers = {"x-llmrouter-tier": "slm", "x-llmrouter-escalated": "false"}
    report = compare(run("baseline"), run("router", served="cheap", routing=headers))
    assert report.routing is not None and report.routing.by_tier == {"slm": 90}
    assert compare(run("baseline"), run("router")).routing is None


# ---------- claims ----------


@pytest.mark.parametrize(
    ("metric", "status"),
    [
        (Metric(value=0.80, ci=Interval(low=0.72, high=0.88)), "established"),
        (Metric(value=0.70, ci=Interval(low=0.70, high=0.75)), "established"),  # on the line
        (Metric(value=0.74, ci=Interval(low=0.66, high=0.82)), "point_estimate_only"),
        (Metric(value=0.70, ci=Interval(low=0.60, high=0.80)), "point_estimate_only"),
        (Metric(value=0.69, ci=Interval(low=0.60, high=0.78)), "not_met"),
        (Metric(value=0.80, ci=None), "point_estimate_only"),
        (Metric(value=0.60, ci=None), "not_met"),
        (Metric(value=None, ci=None), "unknown"),
    ],
)
def test_a_claim_is_established_only_if_the_whole_interval_meets_the_target(
    metric: Metric, status: str
) -> None:
    assert judge_claim("cost saved", 0.70, metric).status == status


def test_the_report_checks_both_headline_claims_against_the_targets() -> None:
    report = compare(run("baseline", cost=1.0), run("router", cost=0.2))
    names = {c.name: c for c in report.claims}
    assert names["cost saved"].status == "established" and names["cost saved"].target == 0.70
    assert names["quality retained"].status == "established"


def test_targets_can_be_changed() -> None:
    report = compare(run("baseline", cost=1.0), run("router", cost=0.2), targets=Targets(0.9, 0.5))
    assert report.claims[0].status == "not_met" and report.claims[0].target == 0.9


# ---------- warnings ----------


def test_a_clean_comparison_warns_only_about_the_small_sample() -> None:
    report = compare(run("baseline"), run("router"))
    assert len(report.warnings) == 1 and "small sample" in report.warnings[0]


def test_a_baseline_served_by_several_models_is_flagged() -> None:
    base = run("baseline")
    base[0] = base[0].model_copy(update={"served_model": "other"})
    assert any(
        "not an always-one-model baseline" in w for w in compare(base, run("router")).warnings
    )


def test_policy_shaped_routing_is_flagged() -> None:
    headers = {"x-llmrouter-tier": "slm", "x-llmrouter-policy": "pinned"}
    report = compare(run("baseline"), run("router", routing=headers))
    assert any("not the router on its own" in w for w in report.warnings)


def test_different_judges_are_flagged() -> None:
    report = compare(
        run("baseline"), run("router"), scores("baseline", judge="a"), scores("router", judge="b")
    )
    assert any("graded by different judges" in w for w in report.warnings)


def test_a_judge_that_also_wrote_answers_is_flagged() -> None:
    report = compare(
        run("baseline", served="smart"),
        run("router"),
        scores("baseline", judge="smart"),
        scores("router", judge="smart"),
    )
    assert any("also wrote answers" in w for w in report.warnings)


def test_a_weak_baseline_is_flagged() -> None:
    report = compare(run("baseline"), run("router"), scores("baseline", 0.5), scores("router", 0.5))
    assert any("weak yardstick" in w for w in report.warnings)


# ---------- the report ----------


def test_the_report_reads_clearly() -> None:
    report = compare(run("baseline", cost=1.0), run("router", cost=0.25, latency=0.5))
    text = format_report(report)
    assert "router 'router' vs baseline 'baseline'" in text
    assert "90 prompts (46.7% easy, 25.6% medium, 27.8% hard)" in text
    assert "saved 75.0% (95% CI 75.0% to 75.0%)" in text
    assert "retained 100.0% (95% CI 100.0% to 100.0%)" in text
    assert "p50 1.00s -> 0.50s (2.00x)" in text
    assert "by difficulty:" in text and "by category:" in text
    assert "cost saved >= 70.0%: ESTABLISHED" in text
    assert "read with care:" in text and "small sample" in text


def test_undefined_figures_print_as_not_available() -> None:
    report = compare(run("baseline", cost=0.0), run("router", cost=0.0), scores("baseline", 0.0))
    text = format_report(report)
    assert "saved n/a" in text and "retained n/a" in text
    assert "cost saved >= 70.0%: UNKNOWN" in text


def test_a_point_estimate_only_claim_is_labelled_as_such() -> None:
    mixed = {i: (0.0 if n % 2 else 1.0) for n, i in enumerate(IDS)}  # half the items saved fully
    report = compare(run("baseline", cost=1.0), run("router", cost=mixed))
    assert "POINT ESTIMATE ONLY" in format_report(report) or "NOT MET" in format_report(report)


def test_the_report_survives_a_round_trip_through_json() -> None:
    report = compare(run("baseline", cost=1.0), run("router", cost=0.4))
    again = EvaluationReport.model_validate_json(report.model_dump_json())
    assert again == report


def test_the_same_seed_gives_the_same_report() -> None:
    mixed = {i: (0.2 + 0.1 * (n % 5)) for n, i in enumerate(IDS)}
    first = compare(run("baseline", cost=1.0), run("router", cost=mixed), seed=5)
    again = compare(run("baseline", cost=1.0), run("router", cost=mixed), seed=5)
    other = compare(run("baseline", cost=1.0), run("router", cost=mixed), seed=6)
    assert first.cost_saved == again.cost_saved
    assert first.cost_saved.ci != other.cost_saved.ci


# ---------- the command ----------


def write_run(tmp_path: Path, label: str, records: list[EvalRecord], sc: list[ScoreRecord]) -> Path:
    path = tmp_path / f"{label}.jsonl"
    path.write_text("".join(r.model_dump_json() + "\n" for r in records))
    cli.scores_path(path).write_text("".join(s.model_dump_json() + "\n" for s in sc))
    return path


def run_evaluate(tmp_path: Path, *extra: str, cost: float = 0.25) -> int:
    baseline = write_run(tmp_path, "baseline", run("baseline", cost=1.0), scores("baseline"))
    router = write_run(tmp_path, "router", run("router", cost=cost), scores("router"))
    return cli.main(["evaluate", "--baseline", str(baseline), "--router", str(router), *extra])


def test_the_evaluate_command_prints_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_evaluate(tmp_path) == 0
    out = capsys.readouterr().out
    assert "saved 75.0%" in out and "quality" in out


def test_the_evaluate_command_needs_no_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLMROUTER_EVAL_KEY", raising=False)
    assert run_evaluate(tmp_path) == 0


def test_the_report_can_be_written_as_json(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "report.json"
    assert run_evaluate(tmp_path, "--json", str(out)) == 0
    data = json.loads(out.read_text())
    assert data["cost_saved"]["value"] == pytest.approx(0.75)
    assert data["benchmark_items"] == 90


def test_require_point_passes_when_the_estimates_meet_the_targets(tmp_path: Path) -> None:
    assert run_evaluate(tmp_path, "--require", "point") == 0
    assert run_evaluate(tmp_path, "--require", "established") == 0


def test_a_point_estimate_that_the_data_cannot_firmly_support(tmp_path: Path) -> None:
    """72% saved on 90 prompts, but the interval reaches below 70%: --require tells them apart."""
    noisy = {i: (1.0 if n < 25 else 0.0) for n, i in enumerate(IDS)}
    baseline = write_run(tmp_path, "baseline", run("baseline", cost=1.0), scores("baseline"))
    router = write_run(tmp_path, "router", run("router", cost=noisy), scores("router"))
    args = ["evaluate", "--baseline", str(baseline), "--router", str(router)]

    assert cli.main([*args, "--require", "point"]) == 0
    assert cli.main([*args, "--require", "established"]) == 1


def test_require_fails_when_a_target_is_missed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_evaluate(tmp_path, "--require", "point", cost=0.6) == 1
    assert "not met (point): cost saved" in capsys.readouterr().err


def test_targets_are_adjustable_from_the_command_line(tmp_path: Path) -> None:
    assert run_evaluate(tmp_path, "--require", "point", "--target-cost-saved", "0.9") == 1
    assert run_evaluate(tmp_path, "--require", "point", "--target-cost-saved", "0.5") == 0


def test_missing_scores_say_what_to_run_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = write_run(tmp_path, "baseline", run("baseline"), scores("baseline"))
    router = tmp_path / "router.jsonl"
    router.write_text("".join(r.model_dump_json() + "\n" for r in run("router")))
    assert cli.main(["evaluate", "--baseline", str(baseline), "--router", str(router)]) == 2
    err = capsys.readouterr().err
    assert "router.scores.jsonl" in err and "score" in err


def test_results_for_an_unknown_item_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stranger = [
        EvalRecord(item_id="qa-99", run="router", requested_model="x", ok=True, cost_usd=1.0)
    ]
    baseline = write_run(tmp_path, "baseline", run("baseline"), scores("baseline"))
    router = write_run(tmp_path, "router", stranger, scores("router", ids=["chat-01"]))
    assert cli.main(["evaluate", "--baseline", str(baseline), "--router", str(router)]) == 2
    assert capsys.readouterr().err.startswith("error:")


@pytest.mark.parametrize("where", ["baseline", "router", "baseline_scores", "router_scores"])
def test_results_for_items_outside_the_benchmark_are_refused(where: str) -> None:
    baseline, router = run("baseline"), run("router")
    base_scores, route_scores = scores("baseline"), scores("router")
    stranger = run("x", ids=["chat-01"])[0].model_copy(update={"item_id": "qa-99"})
    stranger_score = scores("x", ids=["chat-01"])[0].model_copy(update={"item_id": "qa-99"})
    if where == "baseline":
        baseline.append(stranger)
    elif where == "router":
        router.append(stranger)
    elif where == "baseline_scores":
        base_scores.append(stranger_score)
    else:
        route_scores.append(stranger_score)

    with pytest.raises(ValueError, match="'qa-99'.*not in the benchmark"):
        evaluate(BENCH, baseline, router, base_scores, route_scores)


# ---------- the whole pipeline ----------


async def test_run_score_and_evaluate_compose_through_the_real_gateway() -> None:
    """Baseline run, router run, judging and the report, end to end on fake providers."""
    import re

    import httpx

    from app.core.config import Settings
    from app.core.tiers import Tier
    from app.evaluation.judge import Judge
    from app.evaluation.pricing import PriceBook
    from app.evaluation.quality import score_records
    from app.evaluation.runner import Runner
    from app.main import create_app
    from app.providers.fake import FakeProvider
    from app.router.deployment import Deployment
    from app.router.registry import ModelRegistry
    from tests.conftest import model_config, seed_credentials

    cheap_cfg = model_config(
        name="cheap", tier=Tier.SLM, input_cost_per_mtok=1.0, output_cost_per_mtok=2.0
    )
    smart_cfg = model_config(
        name="smart", tier=Tier.FRONTIER, input_cost_per_mtok=10.0, output_cost_per_mtok=30.0
    )
    cheap, smart = FakeProvider(reply="42"), FakeProvider(reply="42")

    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry(
            [Deployment.create(cheap_cfg, cheap), Deployment.create(smart_cfg, smart)]
        )

    def judge_reply(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][1]["content"]
        count = len(re.findall(r"^\d+\. ", prompt.split("CRITERIA:")[1], re.MULTILINE))
        verdicts = [{"criterion": n, "met": True, "reason": "ok"} for n in range(1, count + 1)]
        content = json.dumps({"verdicts": verdicts})
        message = {"role": "assistant", "content": content}
        body = {
            "id": "j",
            "object": "chat.completion",
            "model": "j",
            "choices": [{"index": 0, "message": message}],
        }
        return httpx.Response(200, json=body)

    app = create_app(Settings(_env_file=None, retry_base_delay=0.0), factory)
    book = PriceBook([cheap_cfg, smart_cfg])
    async with app.router.lifespan_context(app):
        secret = await seed_credentials(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gw",
            headers={"Authorization": f"Bearer {secret}"},
        ) as client:
            runner = Runner(client, book)
            baseline = await runner.run(BENCH, model="smart", label="baseline")
            smart_for_baseline = smart.calls
            routed = await runner.run(BENCH, model="auto", label="router")
    transport = httpx.MockTransport(judge_reply)
    async with httpx.AsyncClient(transport=transport, base_url="http://j") as judge_client:
        judge = Judge(judge_client, "grader")
        base_scores = await score_records(baseline, BENCH, judge)
        route_scores = await score_records(routed, BENCH, judge)

    report = evaluate(BENCH, baseline, routed, base_scores, route_scores, resamples=200)

    # every answer carries the fake provider's 10 prompt and 5 completion tokens
    per_cheap, per_smart = 10 * 1 + 5 * 2, 10 * 10 + 5 * 30
    routed_to_smart = smart.calls - smart_for_baseline
    expected = 1 - (cheap.calls * per_cheap + routed_to_smart * per_smart) / (90 * per_smart)
    assert smart_for_baseline == 90
    assert report.cost_pairs == report.quality_pairs == 90
    assert report.baseline.served_by == {"smart": 90}
    assert report.cost_saved.value == pytest.approx(expected)
    assert cheap.calls > 0  # the router really did use the cheaper model
    assert report.routing is not None and sum(report.routing.by_tier.values()) == 90
    assert report.baseline.failed == report.router.failed == 0
    # both runs gave the same replies, so quality is identical by construction
    assert report.quality_retained.value == pytest.approx(1.0)
