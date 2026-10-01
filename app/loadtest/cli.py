import argparse
import asyncio
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn

from app.loadtest.harness import HarnessError, StackConfig, running_stack
from app.loadtest.scenarios import (
    ScenarioResult,
    default_specs,
    paced_specs,
    prewarm,
    run_outage,
    run_rate_limit,
    run_spec,
    sweep_specs,
)
from app.loadtest.stub import StubSettings, create_stub

ALL = ("direct", "gateway", "limited", "auto", "stream", "sweep", "paced", "outage", "rate_limit")


def format_results(results: list[ScenarioResult]) -> str:
    def ms(value: float | None) -> str:
        return "-" if value is None else f"{value:.1f}"

    lines = [
        f"{'scenario':<14}{'req/s':>9}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}"
        f"{'err %':>8}{'gw CPU %':>10}{'upstream':>10}",
    ]
    for r in results:
        s = r.summary
        if s is None:
            continue
        lines.append(
            f"{r.name:<14}{s.throughput_rps:>9.1f}{ms(s.latency_ms.p50):>9}{ms(s.latency_ms.p95):>9}"
            f"{ms(s.latency_ms.p99):>9}{s.error_rate * 100:>8.2f}"
            f"{'-' if r.gateway_cpu_percent is None else f'{r.gateway_cpu_percent:.0f}':>10}"
            f"{'-' if r.upstream_requests is None else r.upstream_requests:>10}"
        )
        if s.ttfb_ms is not None:
            lines.append(f"{'':<14}  time to first chunk: p50 {ms(s.ttfb_ms.p50)} ms, "
                         f"p95 {ms(s.ttfb_ms.p95)} ms")  # fmt: skip
        if s.behind_ms is not None:
            lines.append(f"{'':<14}  generator lateness: p95 {ms(s.behind_ms.p95)} ms, "
                         f"max {ms(s.behind_ms.max)} ms")  # fmt: skip
        if s.errors:
            lines.append(f"{'':<14}  errors: {s.errors}")
    for r in results:
        if r.summary is not None:
            continue
        lines.append("")
        lines.append(r.title)
        if "phases" in r.details:
            for p in r.details["phases"]:
                lines.append(
                    f"  {p['phase']:<9} sent {p['sent']:>5}  ok {p['ok']:>5}  "
                    f"reached the upstream {p['reached_upstream']:>5}  errors {p['errors']}"
                )
        else:
            lines.append("  " + ", ".join(f"{k} {v}" for k, v in r.details.items()))
    return "\n".join(lines)


def environment(config: StackConfig, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": platform.machine(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "redis": "yes" if config.redis_url else "no (quotas fail open)",
        "database": (config.database_url or "sqlite file").split(":")[0],
        "stub_latency_ms": config.stub.latency_ms,
        "concurrency": args.concurrency,
        "generators": args.generators,
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "prewarm_s": args.prewarm,
    }


async def _run_all(args: argparse.Namespace, config: StackConfig) -> list[ScenarioResult]:
    chosen = list(ALL) if args.scenarios == "all" else args.scenarios.split(",")
    unknown = [name for name in chosen if name not in ALL]
    if unknown:
        raise ValueError(f"unknown scenario(s) {unknown}; choose from {list(ALL)}")
    results: list[ScenarioResult] = []
    with running_stack(config) as stack:
        key = stack.make_key()
        if args.prewarm > 0:
            await prewarm(
                stack,
                key,
                concurrency=args.concurrency,
                generators=args.generators,
                seconds=args.prewarm,
            )
        specs = default_specs(
            concurrency=args.concurrency, duration=args.duration, warmup=args.warmup
        )
        for name in chosen:
            print(f"running {name} ...", file=sys.stderr)
            if name == "limited":
                # generous limits that are never hit, so what is measured is the cost of checking
                limited = stack.make_key(
                    rpm_limit=1_000_000, tpm_limit=1_000_000_000, monthly_budget_usd=100_000
                )
                spec = replace(
                    specs["gateway"], name="limited", title="Key with RPM, TPM and a budget"
                )
                results.append(await run_spec(stack, spec, key=limited, generators=args.generators))
            elif name in specs:
                results.append(
                    await run_spec(stack, specs[name], key=key, generators=args.generators)
                )
            elif name == "sweep":
                levels = [int(n) for n in args.sweep.split(",")]
                for spec in sweep_specs(levels, duration=args.duration, warmup=args.warmup):
                    results.append(await run_spec(stack, spec, key=key, generators=args.generators))
            elif name == "paced":
                rates = [float(n) for n in args.rates.split(",")]
                for spec in paced_specs(rates, duration=args.duration, warmup=args.warmup):
                    results.append(await run_spec(stack, spec, key=key, generators=args.generators))
            elif name == "outage":
                results.append(
                    await run_outage(stack, key, rate=args.outage_rate, phase_seconds=args.phase)
                )
            elif name == "rate_limit":
                results.append(await run_rate_limit(stack, limit=60, requests=300, concurrency=100))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.loadtest", description="Load-test the gateway against a stand-in model"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    stub = commands.add_parser("stub", help="run just the stand-in model server")
    stub.add_argument("--port", type=int, default=9100)
    stub.add_argument("--host", default="127.0.0.1", help="0.0.0.0 inside a container")
    stub.add_argument("--latency-ms", type=float, default=50.0)
    stub.add_argument("--jitter-ms", type=float, default=0.0)
    stub.add_argument("--error-rate", type=float, default=0.0)
    stub.add_argument("--completion-tokens", type=int, default=50)

    run = commands.add_parser("run", help="start a stub and a gateway, and measure the gateway")
    run.add_argument("--scenarios", default="all", help=f"comma list of {', '.join(ALL)}, or all")
    run.add_argument("--concurrency", type=int, default=50)
    run.add_argument("--duration", type=float, default=10.0, help="seconds measured per scenario")
    run.add_argument("--warmup", type=float, default=2.0, help="seconds not counted, per scenario")
    run.add_argument("--sweep", default="1,10,50,100,200,400", help="concurrency levels")
    run.add_argument("--rates", default="100,300,600", help="open-loop arrival rates (req/s)")
    run.add_argument("--outage-rate", type=float, default=100.0)
    run.add_argument("--phase", type=float, default=5.0, help="seconds per outage phase")
    run.add_argument("--stub-latency-ms", type=float, default=50.0)
    run.add_argument(
        "--prewarm",
        type=float,
        default=3.0,
        help="seconds of unrecorded traffic first, so no scenario pays for a cold start",
    )
    run.add_argument(
        "--generators",
        type=int,
        default=3,
        help="load-generating processes (one Python process tops out near 600 req/s)",
    )
    run.add_argument("--redis-url", help="use this Redis (default: none, so quotas fail open)")
    run.add_argument("--database-url", help="use this database (default: a SQLite file)")
    run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="extra gateway setting, e.g. --set LLMROUTER_BREAKER_SHARED=false",
    )
    run.add_argument("--json", type=Path, dest="json_out", help="also write the results as JSON")
    args = parser.parse_args(argv)

    if args.command == "stub":
        settings = StubSettings(
            latency_ms=args.latency_ms,
            jitter_ms=args.jitter_ms,
            error_rate=args.error_rate,
            completion_tokens=args.completion_tokens,
        )
        uvicorn.run(create_stub(settings), host=args.host, port=args.port, log_level="warning")
        return 0

    extra: dict[str, str] = {}
    for item in args.set:
        name, separator, value = item.partition("=")
        if not separator or not name.startswith("LLMROUTER_"):
            print(f"error: --set wants LLMROUTER_NAME=value, not '{item}'", file=sys.stderr)
            return 2
        extra[name] = value
    config = StackConfig(
        stub=StubSettings(latency_ms=args.stub_latency_ms),
        redis_url=args.redis_url,
        database_url=args.database_url,
        gateway_env=extra,
    )
    try:
        results = asyncio.run(_run_all(args, config))
    except (HarnessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_results(results))
    if args.json_out:
        report = {
            "environment": environment(config, args) | {"gateway_settings": extra},
            "results": [r.model_dump() for r in results],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"results written to {args.json_out}")
    return 0
