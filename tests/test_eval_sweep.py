import json
import random
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.core.tiers import Tier
from app.evaluation import cli
from app.evaluation.dataset import BenchmarkItem, DatasetError, load_benchmark
from app.evaluation.quality import ScoreRecord
from app.evaluation.runner import EvalRecord, load_records
from app.evaluation.sweep import (
    Answer,
    Replay,
    Setup,
    SweepDataError,
    SweepPoint,
    SweepReport,
    format_sweep,
    load_runs,
    pareto_front,
    run_path,
    select_operating_point,
    split_items,
    sweep,
    threshold_grid,
)
from app.routing.selector import choose_model

BENCH = load_benchmark()
IDS = [i.id for i in BENCH]

CONFIG = """
models:
  - name: fast
    tier: slm
    provider: openai_compatible
    model: small
    base_url: https://a.example.com/v1
    api_key: os.environ/NOT_SET_A
    input_cost_per_mtok: 0.05
    output_cost_per_mtok: 0.08
  - name: balanced
    tier: mid
    provider: openai_compatible
    model: medium
    base_url: https://b.example.com/v1
    api_key: os.environ/NOT_SET_B
    input_cost_per_mtok: 0.6
    output_cost_per_mtok: 0.8
  - name: smart
    tier: frontier
    provider: anthropic
    model: big
    api_key: os.environ/NOT_SET_C
    input_cost_per_mtok: 3.0
    output_cost_per_mtok: 15.0
"""
COST = {"fast": 1.0, "fast2": 1.2, "balanced": 4.0, "smart": 10.0}
LATENCY = {"fast": 0.4, "fast2": 0.4, "balanced": 1.0, "smart": 2.5}


def quality(alias: str, item: BenchmarkItem) -> float:
    """Each model is at least as good as the one below it on every item."""
    table = {
        "fast": {"easy": 1.0, "medium": 0.5, "hard": 0.0},
        "fast2": {"easy": 1.0, "medium": 0.5, "hard": 0.0},
        "balanced": {"easy": 1.0, "medium": 1.0, "hard": 0.5},
        "smart": {"easy": 1.0, "medium": 1.0, "hard": 1.0},
    }
    return table[alias][item.difficulty]


def write_config(tmp_path: Path, text: str = CONFIG) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def write_runs(
    runs_dir: Path,
    *,
    score_of: Callable[[str, BenchmarkItem], float | None] = quality,
    aliases: tuple[str, ...] = ("fast", "balanced", "smart"),
    skip: Callable[[str, str], bool] = lambda alias, item_id: False,
    latency: dict[str, float] | None = None,
) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    for alias in aliases:
        records, scores = [], []
        for item in BENCH:
            if skip(alias, item.id):
                continue
            records.append(
                EvalRecord(
                    item_id=item.id,
                    run=f"model-{alias}",
                    requested_model=alias,
                    served_model=alias,
                    ok=True,
                    reply="r",
                    cost_usd=COST[alias],
                    latency_s=(latency or LATENCY)[alias],
                )
            )
            scores.append(
                ScoreRecord(
                    item_id=item.id,
                    run=f"model-{alias}",
                    method="number",
                    score=score_of(alias, item),
                )
            )
        path = run_path(runs_dir, alias)
        path.write_text("".join(r.model_dump_json() + "\n" for r in records))
        cli.scores_path(path).write_text("".join(s.model_dump_json() + "\n" for s in scores))


@pytest.fixture
def world(tmp_path: Path) -> tuple[Setup, Path]:
    setup = Setup(write_config(tmp_path))
    runs_dir = tmp_path / "runs"
    write_runs(runs_dir)
    return setup, runs_dir


def replay_for(setup: Setup, runs_dir: Path) -> Replay:
    runs = load_runs(runs_dir, list(setup.tiered()), BENCH, cli.scores_path)
    return Replay(setup, BENCH, runs, "smart")


# ---------- the grid and the split ----------


def test_the_default_grid_has_every_valid_pair_of_ceilings() -> None:
    grid = threshold_grid()
    assert len(grid) == 190
    assert all(0 < low < high <= 1 for low, high in grid)
    assert (0.35, 0.65) in grid  # the shipped defaults are on the grid
    assert len(set(grid)) == len(grid)


def test_a_coarser_step_gives_a_smaller_grid() -> None:
    assert threshold_grid(0.25) == [
        (0.25, 0.5),
        (0.25, 0.75),
        (0.25, 1.0),
        (0.5, 0.75),
        (0.5, 1.0),
        (0.75, 1.0),
    ]


def test_the_split_shares_every_item_between_the_two_halves() -> None:
    tune, holdout = split_items(BENCH)
    assert sorted(tune + holdout) == sorted(IDS)
    assert not set(tune) & set(holdout)
    assert len(tune) == 48 and len(holdout) == 42


def test_each_category_is_shared_out_evenly() -> None:
    tune, holdout = split_items(BENCH)
    for category in {i.category for i in BENCH}:
        a = sum(1 for i in tune if i.startswith(category + "-"))
        b = sum(1 for i in holdout if i.startswith(category + "-"))
        assert abs(a - b) <= 1


def test_the_split_alternates_starting_with_the_tuning_half() -> None:
    tune, holdout = split_items(BENCH)
    assert "chat-01" in tune and "chat-02" in holdout and "chat-03" in tune


def test_the_split_depends_only_on_the_benchmark() -> None:
    assert split_items(BENCH) == split_items(BENCH)


# ---------- loading the runs ----------


def test_complete_runs_load(world: tuple[Setup, Path]) -> None:
    setup, runs_dir = world
    runs = load_runs(runs_dir, list(setup.tiered()), BENCH, cli.scores_path)
    assert set(runs) == {"fast", "balanced", "smart"}
    assert all(len(answers) == 90 for answers in runs.values())
    assert runs["fast"]["math-01"].cost == 1.0


def test_every_gap_in_the_runs_is_reported_at_once(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    write_runs(
        runs_dir,
        score_of=lambda alias, item: None if (alias == "fast" and item.id == "qa-01") else 1.0,
        skip=lambda alias, item_id: alias == "balanced" and item_id == "chat-01",
        aliases=("fast", "balanced"),
    )
    failed = load_records(run_path(runs_dir, "fast"))
    failed[0] = failed[0].model_copy(update={"ok": False, "reply": None, "cost_usd": None})
    run_path(runs_dir, "fast").write_text("".join(r.model_dump_json() + "\n" for r in failed))
    unpriced = load_records(run_path(runs_dir, "balanced"))
    unpriced[1] = unpriced[1].model_copy(update={"cost_usd": None})
    run_path(runs_dir, "balanced").write_text("".join(r.model_dump_json() + "\n" for r in unpriced))

    with pytest.raises(SweepDataError) as caught:
        load_runs(runs_dir, ["fast", "balanced", "smart"], BENCH, cli.scores_path)

    message = str(caught.value)
    assert "fast: no answer for chat-01" in message  # the failed request
    assert "fast: qa-01 has no quality score" in message
    assert "balanced: no answer for chat-01" in message  # never run
    assert "balanced: chat-03 could not be priced" in message
    assert "smart: " in message and "not found" in message  # the missing run


def test_empty_run_files_list_every_missing_item(tmp_path: Path) -> None:
    path = run_path(tmp_path, "fast")
    path.write_text("")
    cli.scores_path(path).write_text("")
    with pytest.raises(SweepDataError) as caught:
        load_runs(tmp_path, ["fast"], BENCH, cli.scores_path)
    assert "and 70 more" in str(caught.value)
    assert str(caught.value).count("no answer for") == 20


# ---------- the setup ----------


def test_the_setup_loads_the_gateway_config_without_its_keys(world: tuple[Setup, Path]) -> None:
    setup, _ = world
    assert setup.tiered() == {"fast": Tier.SLM, "balanced": Tier.MID, "smart": Tier.FRONTIER}
    assert setup.frontier_aliases() == ["smart"]
    assert setup.policy.slm_ceiling == 0.35


# ---------- the replay is faithful to the real router ----------


def test_the_replay_chooses_what_the_real_selector_chooses(world: tuple[Setup, Path]) -> None:
    """The heart of the sweep's validity: same prompts, same thresholds, same model."""
    setup, runs_dir = world
    replay = replay_for(setup, runs_dir)
    tracker = replay._tracker()
    rng = random.Random(4)
    for _ in range(25):
        slm = round(rng.uniform(0.02, 0.9), 2)
        mid = round(rng.uniform(slm + 0.01, 1.0), 2)
        policy = replace(setup.policy, slm_ceiling=slm, mid_ceiling=mid)
        chosen = replay.assignment(policy)
        for item, alias in zip(BENCH, chosen, strict=True):
            real = choose_model(
                item.to_request("auto"),
                setup.registry,
                tracker,
                policy=policy,
                weights=setup.weights,
            )
            assert alias == real.alias, (slm, mid, item.id)


def test_at_the_extremes_everything_goes_to_one_tier(world: tuple[Setup, Path]) -> None:
    setup, runs_dir = world
    replay = replay_for(setup, runs_dir)
    everything_small = replace(setup.policy, slm_ceiling=1.0, mid_ceiling=1.0)
    everything_big = replace(setup.policy, slm_ceiling=-1.0, mid_ceiling=-0.5)
    assert set(replay.assignment(everything_small)) == {"fast"}
    assert set(replay.assignment(everything_big)) == {"smart"}


def test_routing_everything_to_the_baseline_saves_nothing_and_loses_nothing(
    world: tuple[Setup, Path],
) -> None:
    setup, runs_dir = world
    replay = replay_for(setup, runs_dir)
    tune, holdout = (list(range(0, 90, 2)), list(range(1, 90, 2)))
    point = replay.point(-1.0, -0.5, tune, holdout)
    assert point.tiers == {"frontier": 90}
    assert point.saved_all == pytest.approx(0.0)
    assert point.retained_all == pytest.approx(1.0)


def test_routing_everything_to_the_smallest_model_gives_hand_computed_figures(
    world: tuple[Setup, Path],
) -> None:
    setup, runs_dir = world
    replay = replay_for(setup, runs_dir)
    point = replay.point(1.0, 1.0, list(range(90)), [])
    expected_quality = sum(quality("fast", item) for item in BENCH) / 90
    assert point.tiers == {"slm": 90}
    assert point.saved_all == pytest.approx(1 - 1.0 / 10.0)
    assert point.retained_all == pytest.approx(expected_quality)
    assert point.saved_holdout is None and point.retained_holdout is None


def test_a_point_reports_the_halves_separately(world: tuple[Setup, Path]) -> None:
    setup, runs_dir = world
    replay = replay_for(setup, runs_dir)
    tune_ids, holdout_ids = split_items(BENCH)
    position = {i: n for n, i in enumerate(replay.ids)}
    point = replay.point(
        1.0, 1.0, [position[i] for i in tune_ids], [position[i] for i in holdout_ids]
    )
    expect_tune = sum(quality("fast", BENCH.get(i)) for i in tune_ids) / len(tune_ids)
    expect_holdout = sum(quality("fast", BENCH.get(i)) for i in holdout_ids) / len(holdout_ids)
    assert point.retained_tune == pytest.approx(expect_tune)
    assert point.retained_holdout == pytest.approx(expect_holdout)


def test_raising_a_ceiling_never_costs_more_and_never_improves_quality(
    world: tuple[Setup, Path],
) -> None:
    """With models that are each at least as good as the one below, more routing down can only
    save more and lose more: a property of the replay, checked over the whole grid."""
    setup, runs_dir = world
    replay = replay_for(setup, runs_dir)
    points = {
        (p.slm_ceiling, p.mid_ceiling): p
        for p in (replay.point(s, m, [], []) for s, m in threshold_grid())
    }
    for (slm, mid), p in points.items():
        for other in (
            points.get((round(slm + 0.05, 2), mid)),
            points.get((slm, round(mid + 0.05, 2))),
        ):
            if other is not None:
                assert p.saved_all is not None and other.saved_all is not None
                assert p.retained_all is not None and other.retained_all is not None
                assert other.saved_all >= p.saved_all - 1e-9
                assert other.retained_all <= p.retained_all + 1e-9


def test_the_tiers_a_point_routes_to_add_up(world: tuple[Setup, Path]) -> None:
    setup, runs_dir = world
    replay = replay_for(setup, runs_dir)
    for slm, mid in threshold_grid(0.25):
        assert sum(replay.point(slm, mid, [], []).tiers.values()) == 90


def test_the_simulated_run_carries_the_routing_headers(world: tuple[Setup, Path]) -> None:
    setup, runs_dir = world
    replay = replay_for(setup, runs_dir)
    records, scores = replay.simulated_run(0.35, 0.65, ["chat-01", "reasoning-05"])
    assert [r.item_id for r in records] == ["chat-01", "reasoning-05"]
    assert all(r.run == "sweep" and r.routing["x-llmrouter-tier"] for r in records)
    assert all(r.served_model == r.routing["x-llmrouter-model"] for r in records)
    assert all(s.run == "sweep" for s in scores)


# ---------- the settings no other setting beats ----------


def pt(
    saved: float | None, retained: float | None, slm: float = 0.1, mid: float = 0.2, **kw: Any
) -> SweepPoint:
    fields: dict[str, Any] = {
        "slm_ceiling": slm,
        "mid_ceiling": mid,
        "tiers": {},
        "saved_all": saved,
        "retained_all": retained,
        "saved_tune": saved,
        "retained_tune": retained,
        "saved_holdout": saved,
        "retained_holdout": retained,
    }
    return SweepPoint(**{**fields, **kw})


def test_a_dominated_setting_is_left_off() -> None:
    best, worse, other = pt(0.8, 0.95, 0.1), pt(0.7, 0.9, 0.2), pt(0.9, 0.80, 0.3)
    front = pareto_front([worse, best, other])
    assert [p.slm_ceiling for p in front] == [0.3, 0.1]  # best saving first; "worse" is beaten


def test_settings_that_tie_with_each_other_are_not_beaten_by_each_other() -> None:
    a, b = pt(0.8, 0.9, 0.1), pt(0.8, 0.9, 0.2)
    assert len(pareto_front([a, b])) == 1  # identical results are listed once


def test_a_setting_better_on_one_axis_survives() -> None:
    a, b = pt(0.8, 0.9, 0.1), pt(0.8, 0.95, 0.2)
    assert [p.slm_ceiling for p in pareto_front([a, b])] == [0.2]


def test_settings_without_figures_are_ignored() -> None:
    assert pareto_front([pt(None, None), pt(0.5, None), pt(None, 0.9)]) == []


def test_the_front_of_nothing_is_nothing() -> None:
    assert pareto_front([]) == []


# ---------- choosing a setting ----------


def test_the_chosen_setting_saves_the_most_among_those_that_keep_enough_quality() -> None:
    points = [
        pt(0.5, 0.99, 0.1),
        pt(0.7, 0.94, 0.2),  # the best saving that still keeps 93%
        pt(0.9, 0.90, 0.3),
    ]
    chosen = select_operating_point(points, 0.93)
    assert chosen is not None and chosen.slm_ceiling == 0.2


def test_the_choice_looks_at_the_tuning_half_only() -> None:
    fooling = pt(
        0.9, 0.99, 0.3, saved_tune=0.9, retained_tune=0.80
    )  # great elsewhere, poor on tune
    honest = pt(0.5, 0.95, 0.1)
    chosen = select_operating_point([fooling, honest], 0.93)
    assert chosen is not None and chosen.slm_ceiling == 0.1


def test_the_target_itself_is_enough() -> None:
    chosen = select_operating_point([pt(0.6, 0.93)], 0.93)
    assert chosen is not None


def test_ties_go_to_better_quality_then_lower_thresholds() -> None:
    a = pt(0.7, 0.95, 0.3, 0.6)
    b = pt(0.7, 0.97, 0.2, 0.5)
    c = pt(0.7, 0.97, 0.1, 0.4)
    assert select_operating_point([a, b, c], 0.9) == c


def test_when_nothing_reaches_the_target_nothing_is_chosen() -> None:
    assert select_operating_point([pt(0.9, 0.5), pt(0.1, 0.9)], 0.93) is None
    assert select_operating_point([], 0.93) is None
    assert select_operating_point([pt(None, None)], 0.93) is None


# ---------- the whole sweep ----------


def run_sweep(world: tuple[Setup, Path], **kwargs: Any) -> SweepReport:
    setup, runs_dir = world
    runs = load_runs(runs_dir, list(setup.tiered()), BENCH, cli.scores_path)
    return sweep(setup, BENCH, runs, "smart", resamples=200, **kwargs)


def test_the_sweep_covers_the_whole_grid_and_reports_the_defaults(
    world: tuple[Setup, Path],
) -> None:
    report = run_sweep(world)
    assert report.grid_points == len(report.points) == 190
    assert report.default_point is not None
    assert (report.default_point.slm_ceiling, report.default_point.mid_ceiling) == (0.35, 0.65)
    assert (report.items, report.tune_items, report.holdout_items) == (90, 48, 42)
    assert report.models == {"fast": "slm", "balanced": "mid", "smart": "frontier"}
    assert 1 <= len(report.pareto) <= len(report.points)


def test_the_chosen_setting_is_measured_on_the_held_out_half_only(
    world: tuple[Setup, Path],
) -> None:
    report = run_sweep(world)
    assert report.selected is not None and report.selected.retained_tune is not None
    assert report.selected.retained_tune >= report.target_retained
    assert report.holdout is not None
    assert report.holdout.benchmark_items == 42
    assert report.holdout.cost_pairs == report.holdout.quality_pairs == 42
    assert report.holdout.baseline.label == "baseline" and report.holdout.router.label == "sweep"
    # its figures agree with the sweep's own numbers for the same half
    assert report.holdout.cost_saved.value == pytest.approx(report.selected.saved_holdout)
    assert report.holdout.quality_retained.value == pytest.approx(report.selected.retained_holdout)
    assert report.holdout.cost_saved.ci is not None


def test_a_higher_target_chooses_safer_thresholds(world: tuple[Setup, Path]) -> None:
    loose, strict = run_sweep(world, target_retained=0.80), run_sweep(world, target_retained=0.99)
    assert loose.selected is not None and strict.selected is not None
    assert loose.selected.saved_tune is not None and strict.selected.saved_tune is not None
    assert loose.selected.saved_tune >= strict.selected.saved_tune


def test_thresholds_that_do_not_carry_over_are_called_out(tmp_path: Path) -> None:
    """The small model is perfect on the tuning half and hopeless on the held-out half."""
    tune_ids, _ = split_items(BENCH)

    def lopsided(alias: str, item: BenchmarkItem) -> float:
        if alias != "fast":
            return 1.0
        return 1.0 if item.id in tune_ids else 0.0

    setup = Setup(write_config(tmp_path))
    runs_dir = tmp_path / "runs"
    write_runs(runs_dir, score_of=lopsided)
    runs = load_runs(runs_dir, list(setup.tiered()), BENCH, cli.scores_path)

    report = sweep(setup, BENCH, runs, "smart", resamples=200)

    assert report.selected is not None
    assert report.selected.retained_tune == pytest.approx(1.0)
    assert report.holdout is not None and report.holdout.quality_retained.value is not None
    assert report.holdout.quality_retained.value < 0.93
    assert any("did not generalise" in n for n in report.notes)


def test_when_no_setting_keeps_enough_quality_nothing_is_chosen(tmp_path: Path) -> None:
    setup = Setup(write_config(tmp_path))
    runs_dir = tmp_path / "runs"
    write_runs(runs_dir, score_of=lambda alias, item: 0.0 if alias == "fast" else 1.0)
    runs = load_runs(runs_dir, list(setup.tiered()), BENCH, cli.scores_path)

    report = sweep(setup, BENCH, runs, "smart", target_retained=1.0, resamples=100)

    # the smallest ceiling (0.05) still sends the plainest prompts to the useless small model
    assert report.selected is None and report.holdout is None
    assert any("no thresholds retain 100%" in n for n in report.notes)


def test_an_unknown_baseline_is_refused(world: tuple[Setup, Path]) -> None:
    setup, runs_dir = world
    runs = load_runs(runs_dir, list(setup.tiered()), BENCH, cli.scores_path)
    with pytest.raises(SweepDataError, match="no run for the baseline model 'nope'"):
        sweep(setup, BENCH, runs, "nope")


def test_the_report_survives_a_round_trip_through_json(world: tuple[Setup, Path]) -> None:
    report = run_sweep(world)
    assert SweepReport.model_validate_json(report.model_dump_json()) == report


def test_the_same_inputs_give_the_same_report(world: tuple[Setup, Path]) -> None:
    assert run_sweep(world, seed=3) == run_sweep(world, seed=3)


def test_the_report_reads_clearly(world: tuple[Setup, Path]) -> None:
    text = format_sweep(run_sweep(world))
    assert "SWEEP  190 threshold settings, baseline 'smart'" in text
    assert "items: 90 (48 for tuning, 42 held out)" in text
    assert "settings no other setting beats" in text
    assert "current defaults (0.35 / 0.65)" in text
    assert "to use it, put this in config.yaml:" in text and "tier_ceilings:" in text
    assert "measured on the held-out half it was not tuned on:" in text
    assert "EVALUATION" in text
    assert "read with care:" in text and "replays stored answers" in text


def test_a_report_with_no_chosen_setting_says_so(tmp_path: Path) -> None:
    setup = Setup(write_config(tmp_path))
    runs_dir = tmp_path / "runs"
    write_runs(runs_dir, score_of=lambda alias, item: 0.0 if alias == "fast" else 1.0)
    runs = load_runs(runs_dir, list(setup.tiered()), BENCH, cli.scores_path)
    text = format_sweep(sweep(setup, BENCH, runs, "smart", target_retained=1.0, resamples=100))
    assert "no setting retains 100% of the quality on the tuning half" in text
    assert "tier_ceilings" not in text


# ---------- the commands ----------


def test_the_sweep_command_prints_the_report_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    write_runs(tmp_path / "runs")
    out = tmp_path / "sweep.json"

    code = cli.main(
        [
            "sweep",
            "--config",
            str(config),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--json",
            str(out),
            "--resamples",
            "100",
            "--step",
            "0.1",
        ]
    )

    assert code == 0
    assert "SWEEP  45 threshold settings" in capsys.readouterr().out
    assert SweepReport.model_validate_json(out.read_text()).grid_points == 45


def test_the_sweep_command_needs_no_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLMROUTER_EVAL_KEY", raising=False)
    config = write_config(tmp_path)
    write_runs(tmp_path / "runs")
    args = [
        "sweep",
        "--config",
        str(config),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--resamples",
        "50",
    ]
    assert cli.main(args) == 0


def test_missing_runs_are_reported_with_the_way_to_fix_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    assert cli.main(["sweep", "--config", str(config), "--runs-dir", str(tmp_path / "none")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "model-fast.jsonl" in err and "rerun" in err


def test_two_frontier_models_need_an_explicit_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path,
        CONFIG
        + """
  - name: smarter
    tier: frontier
    provider: anthropic
    model: bigger
    api_key: os.environ/NOT_SET_D
""",
    )
    assert cli.main(["sweep", "--config", str(config), "--runs-dir", str(tmp_path)]) == 2
    assert "--baseline-model" in capsys.readouterr().err


def test_an_unknown_baseline_model_is_refused_by_the_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    assert cli.main(["sweep", "--config", str(config), "--baseline-model", "zzz"]) == 2
    assert "not a tiered model" in capsys.readouterr().err


def test_a_bad_config_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["sweep", "--config", str(tmp_path / "missing.yaml")]) == 2
    assert capsys.readouterr().err.startswith("error:")


# ---------- the models command ----------


class Gateway:
    def __init__(self) -> None:
        self.models: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.models.append(json.loads(request.content)["model"])
        body = {
            "id": "x",
            "object": "chat.completion",
            "model": "upstream",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        return httpx.Response(200, json=body)


@pytest.fixture
def mock_gateway(monkeypatch: pytest.MonkeyPatch) -> Gateway:
    real = httpx.AsyncClient
    gateway = Gateway()

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(gateway), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return gateway


def run_models(tmp_path: Path, *extra: str) -> int:
    return cli.main(
        [
            "models",
            "--api-key",
            "k",
            "--config",
            str(write_config(tmp_path)),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--limit",
            "3",
            *extra,
        ]
    )


def test_the_models_command_runs_each_tiered_model_in_turn(
    tmp_path: Path, mock_gateway: Gateway, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_models(tmp_path) == 0
    assert sorted(set(mock_gateway.models)) == ["balanced", "fast", "smart"]
    assert mock_gateway.models.count("fast") == 3
    for alias in ("fast", "balanced", "smart"):
        records = load_records(run_path(tmp_path / "runs", alias))
        assert len(records) == 3 and all(r.run == f"model-{alias}" for r in records)
    assert "run 'model-fast': 3/3 answered" in capsys.readouterr().out


def test_the_models_command_can_be_narrowed_to_some_models(
    tmp_path: Path, mock_gateway: Gateway
) -> None:
    assert run_models(tmp_path, "--alias", "fast") == 0
    assert set(mock_gateway.models) == {"fast"}
    assert not run_path(tmp_path / "runs", "smart").exists()


def test_an_unknown_model_is_refused_by_the_models_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_models(tmp_path, "--alias", "nope") == 2
    assert "not tiered models in the config" in capsys.readouterr().err


def test_the_models_command_fails_if_a_model_could_not_be_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = httpx.AsyncClient

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        def refuse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"error": {"message": "no"}})

        return real(transport=httpx.MockTransport(refuse), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    assert run_models(tmp_path) == 1
    assert "failed for 'fast'" in capsys.readouterr().err
    assert run_models(tmp_path, "--allow-failures") == 0


def test_the_models_command_resumes(tmp_path: Path, mock_gateway: Gateway) -> None:
    assert run_models(tmp_path) == 0
    before = len(mock_gateway.models)
    assert run_models(tmp_path, "--resume") == 0
    assert len(mock_gateway.models) == before


def test_what_the_models_command_writes_is_what_the_sweep_reads(
    tmp_path: Path, mock_gateway: Gateway
) -> None:
    run_models(tmp_path)
    path = run_path(tmp_path / "runs", "fast")
    assert path.name == "model-fast.jsonl"
    assert cli.scores_path(path).name == "model-fast.scores.jsonl"  # where `score` writes them


def test_only_helps_pick_a_half_of_a_benchmark() -> None:
    half = BENCH.only(["chat-02", "chat-01"])
    assert [i.id for i in half] == ["chat-01", "chat-02"]
    with pytest.raises(DatasetError, match="not in the benchmark"):
        BENCH.only(["chat-01", "nope-01"])


def test_answers_expose_their_cost_and_quality(world: tuple[Setup, Path]) -> None:
    setup, runs_dir = world
    runs = load_runs(runs_dir, list(setup.tiered()), BENCH, cli.scores_path)
    answer = runs["balanced"]["chat-01"]
    assert isinstance(answer, Answer)
    assert (answer.cost, answer.quality) == (4.0, 1.0)


TWO_SMALL = (
    CONFIG
    + """
  - name: fast2
    tier: slm
    provider: openai_compatible
    model: small-too
    base_url: https://d.example.com/v1
    api_key: os.environ/NOT_SET_D
    input_cost_per_mtok: 0.06
    output_cost_per_mtok: 0.09
"""
)


def test_among_a_tiers_models_the_measured_speeds_decide(tmp_path: Path) -> None:
    """fast is a little cheaper but, as measured, far slower: the live router would prefer
    fast2 once it had seen the latencies, so the replay has to as well."""
    setup = Setup(write_config(tmp_path, TWO_SMALL))
    runs_dir = tmp_path / "runs"
    write_runs(
        runs_dir,
        aliases=("fast", "fast2", "balanced", "smart"),
        latency={"fast": 20.0, "fast2": 0.05, "balanced": 1.0, "smart": 2.5},
    )
    runs = load_runs(runs_dir, list(setup.tiered()), BENCH, cli.scores_path)
    replay = Replay(setup, BENCH, runs, "smart")

    everything_small = replace(setup.policy, slm_ceiling=1.0, mid_ceiling=1.0)

    assert set(replay.assignment(everything_small)) == {"fast2"}
    # and that is what the real selector does with the same measurements
    tracker = replay._tracker()
    for item in list(BENCH)[:10]:
        real = choose_model(
            item.to_request("auto"),
            setup.registry,
            tracker,
            policy=everything_small,
            weights=setup.weights,
        )
        assert real.alias == "fast2"
