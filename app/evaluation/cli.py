import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from app.core.model_config import ConfigError
from app.evaluation.dataset import DEFAULT_DATASET, Benchmark, DatasetError, load_benchmark
from app.evaluation.evaluate import Targets, evaluate, format_report
from app.evaluation.judge import Judge
from app.evaluation.pricing import PriceBook
from app.evaluation.quality import (
    QualitySummary,
    ScoreRecord,
    load_scores,
    negative_control,
    quality_summary,
    score_records,
)
from app.evaluation.routing_report import format_breakdown, routing_breakdown
from app.evaluation.runner import EvalRecord, Runner, RunSummary, load_records, summarize
from app.evaluation.sweep import (
    DEFAULT_TARGET,
    STEP,
    Setup,
    SweepDataError,
    format_sweep,
    load_runs,
    run_path,
    sweep,
)
from app.routing.policy import AUTO_MODEL

DEFAULT_URL = "http://localhost:8000"


def format_summary(summary: RunSummary) -> str:
    def seconds(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}s"

    served = ", ".join(f"{alias} x{count}" for alias, count in summary.served_by.items())
    lines = [
        f"run '{summary.run}': {summary.succeeded}/{summary.items} answered, "
        f"{summary.failed} failed, {summary.unpriced} unpriced",
        f"  tokens   {summary.prompt_tokens} prompt + {summary.completion_tokens} completion",
        f"  cost     ${summary.total_cost_usd:.6f} total, ${summary.mean_cost_usd:.6f} per answer",
        f"  latency  p50 {seconds(summary.latency_p50_s)}, p95 {seconds(summary.latency_p95_s)}",
        f"  served   {served or 'nothing'}",
    ]
    return "\n".join(lines)


def format_quality(summary: QualitySummary) -> str:
    mean = "n/a" if summary.mean is None else f"{summary.mean:.3f}"
    lines = [
        f"quality of '{summary.run}': mean {mean} over {summary.scored}/{summary.items} items "
        f"({summary.unscored} unscored, {summary.no_answer} without an answer, counted as 0)"
    ]
    for name, groups in (
        ("method", summary.by_method),
        ("difficulty", summary.by_difficulty),
        ("category", summary.by_category),
    ):
        for key, group in groups.items():
            lines.append(f"  {name:<10} {key:<10} {group.mean:.3f}  (n={group.n})")
    return "\n".join(lines)


def scores_path(results: Path) -> Path:
    """Where the scores of a results file are written: next to it, ``<name>.scores.jsonl``."""
    return results.with_name(results.stem + ".scores.jsonl")


def choose_frontier(prices: PriceBook, requested: str | None) -> str:
    """The model the always-frontier baseline asks: the one named, else the config's only one."""
    if requested is not None:
        if requested not in prices:
            raise ValueError(f"'{requested}' is not a model in the config: {prices.aliases()}")
        return requested
    frontier = prices.frontier_aliases()
    if len(frontier) != 1:
        raise ValueError(
            f"the config has {len(frontier)} frontier-tier models {frontier}; "
            "say which one the baseline should use with --model"
        )
    return frontier[0]


async def _run(
    benchmark: Benchmark,
    prices: PriceBook,
    *,
    base_url: str,
    api_key: str,
    model: str,
    label: str,
    out: Path,
    resume: bool,
    concurrency: int,
    request_timeout: float,
) -> list[EvalRecord]:
    async with httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=request_timeout,
    ) as client:
        runner = Runner(client, prices, concurrency=concurrency)
        done = 0

        def progress(record: EvalRecord) -> None:
            nonlocal done
            done += 1
            mark = "ok" if record.ok else f"FAILED ({record.error})"
            print(f"[{done}/{len(benchmark)}] {record.item_id}: {mark}", file=sys.stderr)

        return await runner.run(
            benchmark, model=model, label=label, out=out, resume=resume, on_record=progress
        )


def _add_run_arguments(
    sub: argparse.ArgumentParser, default_label: str, *, one_run: bool = True
) -> None:
    if one_run:
        sub.add_argument("--label", default=default_label, help="name of this run in the results")
        sub.add_argument(
            "--out", type=Path, help="results file (default: eval_results/<label>.jsonl)"
        )
    else:
        sub.add_argument(
            "--runs-dir", type=Path, default=Path("eval_results"), help="where results are written"
        )
    sub.add_argument("--base-url", default=os.environ.get("LLMROUTER_EVAL_URL", DEFAULT_URL))
    sub.add_argument("--api-key", default=os.environ.get("LLMROUTER_EVAL_KEY"))
    sub.add_argument("--config", type=Path, default=Path("config.yaml"), help="for prices")
    sub.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    sub.add_argument("--resume", action="store_true", help="skip items already answered in --out")
    sub.add_argument("--concurrency", type=int, default=4)
    sub.add_argument("--timeout", type=float, default=120.0, help="seconds per request")
    sub.add_argument("--limit", type=int, help="only the first N items (a quick trial)")
    sub.add_argument("--category", action="append", help="only these categories")
    sub.add_argument("--difficulty", action="append", help="only these difficulties")
    sub.add_argument(
        "--allow-failures", action="store_true", help="exit 0 even if some items failed"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation", description="Measure the router against a baseline"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser(
        "baseline",
        help="ask every benchmark prompt of the frontier model: the always-frontier cost",
    )
    baseline.add_argument("--model", help="the frontier alias (default: the config's only one)")
    _add_run_arguments(baseline, "baseline")
    router = commands.add_parser(
        "router", help="ask every benchmark prompt as model=auto and let the router choose"
    )
    router.add_argument(
        "--allow-policy",
        action="store_true",
        help="accept answers shaped by the key's organisation policy (a pin or tier cap)",
    )
    _add_run_arguments(router, "router")
    score = commands.add_parser(
        "score", help="grade the answers in one or more results files for quality"
    )
    score.add_argument("results", nargs="+", type=Path, help="results file(s) from a run")
    score.add_argument(
        "--judge-model",
        required=True,
        help="the gateway model that grades open-ended answers (ideally not one being compared)",
    )
    score.add_argument("--base-url", default=os.environ.get("LLMROUTER_EVAL_URL", DEFAULT_URL))
    score.add_argument("--api-key", default=os.environ.get("LLMROUTER_EVAL_KEY"))
    score.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    score.add_argument("--concurrency", type=int, default=4)
    score.add_argument("--timeout", type=float, default=120.0, help="seconds per request")
    score.add_argument("--resume", action="store_true", help="keep scores already in the file")
    score.add_argument(
        "--negative-control",
        action="store_true",
        help="also grade a non-answer for every open-ended item, to see how generous the judge is",
    )
    score.add_argument(
        "--allow-unscored", action="store_true", help="exit 0 even if some answers got no score"
    )
    models = commands.add_parser(
        "models",
        help="ask every benchmark prompt of each tiered model in turn (what the sweep replays)",
    )
    models.add_argument("--alias", action="append", help="only these models (default: all tiered)")
    _add_run_arguments(models, "", one_run=False)
    sw = commands.add_parser(
        "sweep",
        help="replay the router at many thresholds from the per-model runs: cost against quality",
    )
    sw.add_argument("--runs-dir", type=Path, default=Path("eval_results"))
    sw.add_argument("--config", type=Path, default=Path("config.yaml"))
    sw.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    sw.add_argument("--baseline-model", help="the frontier alias (default: the config's only one)")
    sw.add_argument("--target-retained", type=float, default=DEFAULT_TARGET)
    sw.add_argument("--step", type=float, default=STEP, help="grid spacing for both ceilings")
    sw.add_argument("--seed", type=int, default=0)
    sw.add_argument("--resamples", type=int, default=2000)
    sw.add_argument("--json", type=Path, dest="json_out", help="also write the report as JSON")
    ev = commands.add_parser(
        "evaluate",
        help="compare a router run with the baseline: cost saved, quality retained, latency",
    )
    ev.add_argument("--baseline", type=Path, default=Path("eval_results/baseline.jsonl"))
    ev.add_argument("--router", type=Path, default=Path("eval_results/router.jsonl"))
    ev.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ev.add_argument("--json", type=Path, dest="json_out", help="also write the report as JSON")
    ev.add_argument("--seed", type=int, default=0, help="fixes the bootstrap draws")
    ev.add_argument("--resamples", type=int, default=2000, help="bootstrap resamples")
    ev.add_argument("--target-cost-saved", type=float, default=Targets().cost_saved)
    ev.add_argument("--target-quality-retained", type=float, default=Targets().quality_retained)
    ev.add_argument(
        "--require",
        choices=["point", "established"],
        help="exit 1 unless both targets are met by the best estimate (point) or even by the "
        "low end of the 95%% interval (established)",
    )
    return parser


def _main_sweep(args: argparse.Namespace) -> int:
    try:
        setup = Setup(args.config)
        benchmark = load_benchmark(args.dataset)
        aliases = list(setup.tiered())
        baseline_model = args.baseline_model
        if baseline_model is None:
            frontier = setup.frontier_aliases()
            if len(frontier) != 1:
                raise ValueError(
                    f"the config has {len(frontier)} frontier-tier models {frontier}; "
                    "say which one is the baseline with --baseline-model"
                )
            baseline_model = frontier[0]
        elif baseline_model not in aliases:
            raise ValueError(f"'{baseline_model}' is not a tiered model in the config: {aliases}")
        runs = load_runs(args.runs_dir, aliases, benchmark, scores_path)
        report = sweep(
            setup,
            benchmark,
            runs,
            baseline_model,
            target_retained=args.target_retained,
            step=args.step,
            seed=args.seed,
            resamples=args.resamples,
        )
    except (ConfigError, DatasetError, SweepDataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_sweep(report))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"report written to {args.json_out}")
    return 0


def _main_models(args: argparse.Namespace) -> int:
    try:
        setup = Setup(args.config)
        prices = PriceBook(setup.configs)
        benchmark = load_benchmark(args.dataset)
        if args.category or args.difficulty or args.limit:
            benchmark = benchmark.subset(
                categories=set(args.category) if args.category else None,
                difficulties=set(args.difficulty) if args.difficulty else None,
                limit=args.limit,
            )
        aliases = list(setup.tiered())
        if args.alias:
            unknown = [a for a in args.alias if a not in aliases]
            if unknown:
                raise ValueError(f"not tiered models in the config: {unknown} (have {aliases})")
            aliases = args.alias
    except (ConfigError, DatasetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    status = 0
    for alias in aliases:
        out = run_path(args.runs_dir, alias)
        records = asyncio.run(
            _run(
                benchmark,
                prices,
                base_url=args.base_url,
                api_key=args.api_key,
                model=alias,
                label=f"model-{alias}",
                out=out,
                resume=args.resume,
                concurrency=args.concurrency,
                request_timeout=args.timeout,
            )
        )
        summary = summarize(records)
        print(format_summary(summary))
        print(f"results written to {out}")
        if summary.failed and not args.allow_failures:
            print(f"error: {summary.failed} item(s) failed for '{alias}'", file=sys.stderr)
            status = 1
    return status


def _main_evaluate(args: argparse.Namespace) -> int:
    try:
        benchmark = load_benchmark(args.dataset)
        baseline = load_records(args.baseline)
        router = load_records(args.router)
        baseline_scores = load_scores(scores_path(args.baseline))
        router_scores = load_scores(scores_path(args.router))
    except FileNotFoundError as exc:
        print(
            f"error: {exc.filename} not found; run the runs and `score` first "
            "(see `make eval-baseline eval-router eval-score`)",
            file=sys.stderr,
        )
        return 2
    except (DatasetError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        report = evaluate(
            benchmark,
            baseline,
            router,
            baseline_scores,
            router_scores,
            targets=Targets(args.target_cost_saved, args.target_quality_retained),
            seed=args.seed,
            resamples=args.resamples,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_report(report))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"report written to {args.json_out}")
    if args.require:
        accepted = (
            {"established"}
            if args.require == "established"
            else {"established", "point_estimate_only"}
        )
        failing = [c.name for c in report.claims if c.status not in accepted]
        if failing:
            print(f"error: not met ({args.require}): {', '.join(failing)}", file=sys.stderr)
            return 1
    return 0


async def _score(
    benchmark: Benchmark,
    runs: list[tuple[Path, list[EvalRecord]]],
    *,
    base_url: str,
    api_key: str,
    judge_model: str,
    concurrency: int,
    request_timeout: float,
    resume: bool,
    control: bool,
) -> tuple[list[tuple[Path, list[ScoreRecord]]], list[ScoreRecord]]:
    async with httpx.AsyncClient(
        base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=request_timeout
    ) as client:
        judge = Judge(client, judge_model)
        scored = []
        for path, records in runs:
            scores = await score_records(
                records,
                benchmark,
                judge,
                concurrency=concurrency,
                out=scores_path(path),
                resume=resume,
            )
            scored.append((path, scores))
        floor = await negative_control(benchmark, judge, concurrency=concurrency) if control else []
        return scored, floor


def _main_score(args: argparse.Namespace) -> int:
    try:
        benchmark = load_benchmark(args.dataset)
        runs = [(path, load_records(path)) for path in args.results]
    except (DatasetError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for path, records in runs:
        if any(r.served_model == args.judge_model for r in records):
            print(
                f"warning: the judge model '{args.judge_model}' also wrote answers in {path}; "
                "a model tends to grade its own style generously, so prefer another judge",
                file=sys.stderr,
            )
    try:
        scored, floor = asyncio.run(
            _score(
                benchmark,
                runs,
                base_url=args.base_url,
                api_key=args.api_key,
                judge_model=args.judge_model,
                concurrency=args.concurrency,
                request_timeout=args.timeout,
                resume=args.resume,
                control=args.negative_control,
            )
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    status = 0
    for path, scores in scored:
        summary = quality_summary(scores, benchmark)
        print(format_quality(summary))
        print(f"scores written to {scores_path(path)}")
        if summary.unscored and not args.allow_unscored:
            print(
                f"error: {summary.unscored} answer(s) in {path} could not be graded, so their "
                "quality is unknown. Rerun with --resume, or pass --allow-unscored.",
                file=sys.stderr,
            )
            status = 1
    if args.negative_control:
        values = [s.score for s in floor if s.score is not None]
        generous = [s.item_id for s in floor if s.score]
        mean = sum(values) / len(values) if values else float("nan")
        print(
            f"negative control: the judge gave a non-answer {mean:.3f} on average over "
            f"{len(values)} open-ended items; {len(generous)} scored above zero {generous}"
        )
    return status


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        return _main_evaluate(args)
    if args.command == "sweep":
        return _main_sweep(args)

    if not args.api_key:
        print("error: give an API key with --api-key or LLMROUTER_EVAL_KEY", file=sys.stderr)
        return 2
    if args.command == "score":
        return _main_score(args)
    if args.command == "models":
        return _main_models(args)
    try:
        prices = PriceBook.from_file(args.config)
        benchmark = load_benchmark(args.dataset)
        if args.category or args.difficulty or args.limit:
            benchmark = benchmark.subset(
                categories=set(args.category) if args.category else None,
                difficulties=set(args.difficulty) if args.difficulty else None,
                limit=args.limit,
            )
        model = choose_frontier(prices, args.model) if args.command == "baseline" else AUTO_MODEL
    except (ConfigError, DatasetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = args.out or Path("eval_results") / f"{args.label}.jsonl"
    records = asyncio.run(
        _run(
            benchmark,
            prices,
            base_url=args.base_url,
            api_key=args.api_key,
            model=model,
            label=args.label,
            out=out,
            resume=args.resume,
            concurrency=args.concurrency,
            request_timeout=args.timeout,
        )
    )
    summary = summarize(records)
    print(format_summary(summary))
    status = 0
    if args.command == "router":
        breakdown = routing_breakdown(records, benchmark)
        print(format_breakdown(breakdown))
        if breakdown.policy_shaped and not args.allow_policy:
            print(
                f"error: {breakdown.policy_shaped} answer(s) were shaped by the key's organisation "
                f"routing policy {list(breakdown.policy_applied)}, so this is not the router on "
                "its own. Use a key from an organisation with no pin or tier cap "
                "(or pass --allow-policy).",
                file=sys.stderr,
            )
            status = 1
    print(f"results written to {out}")
    if summary.failed and not args.allow_failures:
        print(
            f"error: {summary.failed} item(s) failed; numbers from an incomplete run are not "
            "comparable. Fix the cause and rerun with --resume (or pass --allow-failures).",
            file=sys.stderr,
        )
        status = 1
    return status
