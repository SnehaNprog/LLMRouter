import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import update

from app.core.config import Settings
from app.core.tiers import Tier
from app.db.models import Tenant
from app.evaluation import cli
from app.evaluation.dataset import load_benchmark
from app.evaluation.pricing import PriceBook
from app.evaluation.routing_report import format_breakdown, routing_breakdown
from app.evaluation.runner import EvalRecord, Runner, load_records, summarize
from app.main import create_app
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from tests.conftest import model_config, seed_credentials

BENCH = load_benchmark()


def rec(item_id: str, tier: str | None = "slm", **headers: str) -> EvalRecord:
    routing = {} if tier is None else {"x-llmrouter-tier": tier, **headers}
    return EvalRecord(
        item_id=item_id, run="router", requested_model="auto", ok=True, routing=routing
    )


# ---------- the breakdown ----------


def test_answers_are_counted_by_the_tier_that_served_them() -> None:
    records = [rec("chat-01"), rec("chat-02"), rec("code-01", "mid"), rec("math-10", "frontier")]
    breakdown = routing_breakdown(records, BENCH)
    assert breakdown.answered == 4
    assert breakdown.by_tier == {"frontier": 1, "mid": 1, "slm": 2}


def test_it_is_broken_down_by_the_difficulty_and_category_of_the_item() -> None:
    records = [
        rec("chat-01"),  # easy
        rec("math-10", "frontier"),  # hard
        rec("math-11", "mid"),  # hard
        rec("math-01"),  # easy
    ]
    breakdown = routing_breakdown(records, BENCH)
    assert breakdown.by_difficulty == {
        "easy": {"slm": 2},
        "hard": {"frontier": 1, "mid": 1},
    }
    assert breakdown.by_category == {
        "chat": {"slm": 1},
        "math": {"frontier": 1, "mid": 1, "slm": 1},
    }


def test_escalations_degradations_and_complexity_are_totalled() -> None:
    records = [
        rec("chat-01", **{"x-llmrouter-escalated": "true", "x-llmrouter-complexity": "0.20"}),
        rec("chat-02", **{"x-llmrouter-escalated": "false", "x-llmrouter-complexity": "0.40"}),
        rec("chat-03", **{"x-llmrouter-degraded": "true", "x-llmrouter-complexity": "oops"}),
        rec("chat-04", **{"x-llmrouter-escalated": "true", "x-llmrouter-complexity": "0.30"}),
    ]
    breakdown = routing_breakdown(records, BENCH)
    assert (breakdown.escalated, breakdown.degraded) == (2, 1)
    assert breakdown.mean_complexity == pytest.approx(0.30)  # the unreadable value is skipped


def test_policy_shaped_answers_are_counted_by_policy() -> None:
    records = [
        rec("chat-01", **{"x-llmrouter-policy": "pinned"}),
        rec("chat-02", **{"x-llmrouter-policy": "pinned"}),
        rec("chat-03", **{"x-llmrouter-policy": "max-tier:mid"}),
        rec("chat-04"),
    ]
    breakdown = routing_breakdown(records, BENCH)
    assert breakdown.policy_applied == {"max-tier:mid": 1, "pinned": 2}
    assert breakdown.policy_shaped == 3


def test_failures_and_unrouted_answers_are_left_out() -> None:
    failed = rec("chat-01").model_copy(update={"ok": False})
    baseline_style = rec("chat-02", tier=None)  # no routing headers: a direct request
    breakdown = routing_breakdown([failed, baseline_style, rec("chat-03")], BENCH)
    assert breakdown.answered == 1


def test_an_empty_run_has_an_empty_breakdown() -> None:
    breakdown = routing_breakdown([], BENCH)
    assert breakdown.answered == 0 and breakdown.mean_complexity is None
    assert breakdown.by_tier == {} and breakdown.policy_shaped == 0


def test_the_breakdown_reads_clearly() -> None:
    text = format_breakdown(
        routing_breakdown(
            [rec("chat-01", **{"x-llmrouter-complexity": "0.10"}), rec("math-10", "frontier")],
            BENCH,
        )
    )
    assert "routing (2 answers): frontier 1, slm 1" in text
    assert "escalated 0, degraded 0, mean complexity 0.10" in text
    assert "difficulty easy" in text and "category math" in text


def test_a_run_without_routing_headers_says_so() -> None:
    assert "no routing headers" in format_breakdown(routing_breakdown([], BENCH))


# ---------- through the real gateway ----------


@pytest.fixture
async def gateway() -> AsyncIterator[tuple[httpx.AsyncClient, FastAPI, FakeProvider, FakeProvider]]:
    cheap, smart = FakeProvider(reply="cheap says hi"), FakeProvider(reply="smart says hi")

    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry(
            [
                Deployment.create(
                    model_config(
                        name="cheap",
                        tier=Tier.SLM,
                        input_cost_per_mtok=1.0,
                        output_cost_per_mtok=2.0,
                    ),
                    cheap,
                ),
                Deployment.create(
                    model_config(
                        name="smart",
                        tier=Tier.FRONTIER,
                        input_cost_per_mtok=10.0,
                        output_cost_per_mtok=30.0,
                    ),
                    smart,
                ),
            ]
        )

    app = create_app(Settings(_env_file=None, retry_base_delay=0.0), factory)
    async with app.router.lifespan_context(app):
        secret = await seed_credentials(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway",
            headers={"Authorization": f"Bearer {secret}"},
        ) as client:
            yield client, app, cheap, smart


BOOK = PriceBook(
    [
        model_config(
            name="cheap", tier=Tier.SLM, input_cost_per_mtok=1.0, output_cost_per_mtok=2.0
        ),
        model_config(
            name="smart", tier=Tier.FRONTIER, input_cost_per_mtok=10.0, output_cost_per_mtok=30.0
        ),
    ]
)


async def test_the_breakdown_matches_what_the_providers_actually_served(
    gateway: tuple[httpx.AsyncClient, FastAPI, FakeProvider, FakeProvider],
) -> None:
    client, _, cheap, smart = gateway

    records = await Runner(client, BOOK).run(BENCH, model="auto", label="router")

    breakdown = routing_breakdown(records, BENCH)
    assert breakdown.answered == len(BENCH) == cheap.calls + smart.calls
    assert breakdown.by_tier.get("slm", 0) == cheap.calls
    assert breakdown.by_tier.get("frontier", 0) == smart.calls
    assert breakdown.policy_shaped == 0
    assert sum(sum(row.values()) for row in breakdown.by_difficulty.values()) == len(BENCH)
    assert sum(sum(row.values()) for row in breakdown.by_category.values()) == len(BENCH)


async def test_routing_is_cheaper_than_always_asking_the_frontier_model(
    gateway: tuple[httpx.AsyncClient, FastAPI, FakeProvider, FakeProvider],
) -> None:
    client, _, _, _ = gateway
    runner = Runner(client, BOOK)

    baseline = summarize(await runner.run(BENCH, model="smart", label="baseline"))
    routed = summarize(await runner.run(BENCH, model="auto", label="router"))

    assert baseline.failed == routed.failed == 0
    assert routed.total_cost_usd < baseline.total_cost_usd  # every cheap answer saves money
    assert routed.served_by.get("cheap", 0) > 0


async def test_a_pinned_organisation_is_visible_in_the_breakdown(
    gateway: tuple[httpx.AsyncClient, FastAPI, FakeProvider, FakeProvider],
) -> None:
    client, app, _, smart = gateway
    async with app.state.db_session_factory() as session:
        await session.execute(update(Tenant).values(routing_pinned_model="smart"))
        await session.commit()

    records = await Runner(client, BOOK).run(BENCH.subset(limit=10), model="auto", label="router")

    breakdown = routing_breakdown(records, BENCH)
    assert breakdown.policy_applied == {"pinned": 10}
    assert smart.calls == 10


# ---------- the command ----------


class Scripted:
    def __init__(self, response: Callable[[httpx.Request], httpx.Response]) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response(request)


def answer(**headers: str) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = {
            "id": "x",
            "object": "chat.completion",
            "model": "upstream",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        return httpx.Response(200, json=body, headers=headers)

    return respond


@pytest.fixture
def mock_gateway(monkeypatch: pytest.MonkeyPatch) -> Callable[[Scripted], None]:
    real = httpx.AsyncClient

    def install(fake: Scripted) -> None:
        def factory(**kwargs: Any) -> httpx.AsyncClient:
            return real(transport=httpx.MockTransport(fake), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    return install


def write_config(tmp_path: Path, *, with_frontier: bool = True) -> Path:
    path = tmp_path / "config.yaml"
    frontier = (
        """
  - name: smart
    tier: frontier
    provider: openai_compatible
    model: big
    base_url: https://api.example.com/v1
    input_cost_per_mtok: 10.0
    output_cost_per_mtok: 30.0
"""
        if with_frontier
        else ""
    )
    path.write_text(
        """
models:
  - name: cheap
    tier: slm
    provider: openai_compatible
    model: small
    base_url: https://api.example.com/v1
    input_cost_per_mtok: 1.0
    output_cost_per_mtok: 2.0
"""
        + frontier
    )
    return path


def run_router(tmp_path: Path, *extra: str, with_frontier: bool = True) -> int:
    return cli.main(
        [
            "router",
            "--api-key",
            "k",
            "--config",
            str(write_config(tmp_path, with_frontier=with_frontier)),
            "--out",
            str(tmp_path / "router.jsonl"),
            "--limit",
            "4",
            *extra,
        ]
    )


ROUTED = {"x-llmrouter-model": "cheap", "x-llmrouter-tier": "slm", "x-llmrouter-complexity": "0.12"}


def test_the_router_command_asks_for_auto_and_reports_the_routing(
    tmp_path: Path, mock_gateway: Callable[[Scripted], None], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = Scripted(answer(**ROUTED))
    mock_gateway(fake)

    code = run_router(tmp_path)

    assert code == 0
    assert {json.loads(r.content)["model"] for r in fake.requests} == {"auto"}
    out = capsys.readouterr().out
    assert "run 'router': 4/4 answered" in out
    assert "routing (4 answers): slm 4" in out
    assert "cheap x4" in out
    records = load_records(tmp_path / "router.jsonl")
    assert all(r.run == "router" and r.served_model == "cheap" for r in records)
    assert records[0].cost_usd == pytest.approx((100 * 1 + 50 * 2) / 1e6)


def test_the_router_command_needs_no_frontier_model_in_the_config(
    tmp_path: Path, mock_gateway: Callable[[Scripted], None]
) -> None:
    mock_gateway(Scripted(answer(**ROUTED)))
    assert run_router(tmp_path, with_frontier=False) == 0


def test_an_answer_shaped_by_an_organisation_policy_fails_the_run(
    tmp_path: Path,
    mock_gateway: Callable[[Scripted], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_gateway(Scripted(answer(**ROUTED, **{"x-llmrouter-policy": "pinned"})))

    assert run_router(tmp_path) == 1

    err = capsys.readouterr().err
    assert "routing policy" in err and "pinned" in err
    assert run_router(tmp_path, "--allow-policy") == 0


def test_failures_fail_the_router_run_too(
    tmp_path: Path, mock_gateway: Callable[[Scripted], None]
) -> None:
    mock_gateway(Scripted(lambda r: httpx.Response(400, json={"error": {"message": "no"}})))
    assert run_router(tmp_path) == 1
    assert run_router(tmp_path, "--allow-failures") == 0


def test_the_router_run_can_be_resumed(
    tmp_path: Path, mock_gateway: Callable[[Scripted], None]
) -> None:
    mock_gateway(Scripted(answer(**ROUTED)))
    assert run_router(tmp_path) == 0
    again = Scripted(answer(**ROUTED))
    mock_gateway(again)
    assert run_router(tmp_path, "--resume") == 0
    assert again.requests == []


def test_the_baseline_command_does_not_take_the_policy_flag() -> None:
    with pytest.raises(SystemExit):
        cli.main(["baseline", "--allow-policy", "--api-key", "k"])


def test_the_two_commands_use_different_default_labels_and_files() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["baseline"]).label == "baseline"
    assert parser.parse_args(["router"]).label == "router"
