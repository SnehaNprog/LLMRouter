import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from app.core.model_config import ConfigError
from app.evaluation.dataset import DEFAULT_DATASET, Benchmark, DatasetError, load_benchmark
from app.evaluation.pricing import PriceBook
from app.evaluation.routing_report import format_breakdown, routing_breakdown
from app.evaluation.runner import EvalRecord, Runner, RunSummary, summarize
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


def _add_run_arguments(sub: argparse.ArgumentParser, default_label: str) -> None:
    sub.add_argument("--label", default=default_label, help="name of this run in the results")
    sub.add_argument("--base-url", default=os.environ.get("LLMROUTER_EVAL_URL", DEFAULT_URL))
    sub.add_argument("--api-key", default=os.environ.get("LLMROUTER_EVAL_KEY"))
    sub.add_argument("--config", type=Path, default=Path("config.yaml"), help="for prices")
    sub.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    sub.add_argument("--out", type=Path, help="results file (default: eval_results/<label>.jsonl)")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.api_key:
        print("error: give an API key with --api-key or LLMROUTER_EVAL_KEY", file=sys.stderr)
        return 2
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
