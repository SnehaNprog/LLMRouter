import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.core.model_config import ConfigError
from app.core.tiers import Tier
from app.evaluation import cli
from app.evaluation.dataset import Benchmark, load_benchmark, parse_benchmark
from app.evaluation.pricing import PriceBook
from app.evaluation.runner import (
    EvalRecord,
    Runner,
    load_records,
    percentile,
    summarize,
)
from app.main import create_app
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from tests.conftest import model_config, seed_credentials


def prices(**per_alias: tuple[float, float]) -> PriceBook:
    return PriceBook(
        model_config(name=alias, input_cost_per_mtok=cin, output_cost_per_mtok=cout)
        for alias, (cin, cout) in per_alias.items()
    )


PRICES = prices(cheap=(1.0, 2.0), smart=(10.0, 30.0))


def small(n: int = 3) -> Benchmark:
    lines = [
        json.dumps(
            {
                "id": f"qa-{i:02d}",
                "category": "qa",
                "difficulty": "easy",
                "prompt": f"Question {i}?",
                "scoring": {"kind": "contains", "any_of": ["x"]},
                "max_tokens": 50,
            }
        )
        for i in range(1, n + 1)
    ]
    return parse_benchmark("\n".join(lines))


def completion(
    content: str | None = "an answer", usage: dict[str, int] | None = None, **body: Any
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "x",
        "object": "chat.completion",
        "model": "upstream",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": usage if usage is not None else {"prompt_tokens": 100, "completion_tokens": 50},
    }
    return {**payload, **body}


class Fake:
    """A scripted endpoint: replies come from a queue of callables or responses."""

    def __init__(
        self, *replies: httpx.Response | Callable[[httpx.Request], httpx.Response]
    ) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply(request) if callable(reply) else reply


def ok(**kwargs: Any) -> httpx.Response:
    headers = kwargs.pop("headers", None)
    return httpx.Response(200, json=completion(**kwargs), headers=headers)


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.calls.append(seconds)


class Ticks:
    """A clock that advances 0.25 s every time it is read."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.25
        return self.now


def runner_for(
    handler: Callable[[httpx.Request], httpx.Response] | Callable[[httpx.Request], Any],
    pricing: PriceBook = PRICES,
    **kwargs: Any,
) -> tuple[Runner, Sleeps, httpx.AsyncClient]:
    sleeps = Sleeps()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://gw")
    options = {"sleep": sleeps.sleep, "clock": Ticks(), **kwargs}
    return Runner(client, pricing, **options), sleeps, client


# ---------- one request ----------


async def test_the_request_asks_the_chosen_model_the_items_question() -> None:
    fake = Fake(ok())
    runner, _, client = runner_for(fake)
    item = small(1).items[0]

    await runner.ask(item, model="smart", label="baseline")

    sent = fake.requests[0]
    assert sent.url.path == "/v1/chat/completions"
    body = json.loads(sent.content)
    assert body["model"] == "smart"
    assert body["messages"] == [{"role": "user", "content": "Question 1?"}]
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 50
    await client.aclose()


async def test_a_successful_answer_is_recorded_with_its_cost_and_latency() -> None:
    runner, _, client = runner_for(Fake(ok()))

    record = await runner.ask(small(1).items[0], model="smart", label="baseline")

    assert record.ok and record.status_code == 200
    assert record.reply == "an answer"
    assert (record.prompt_tokens, record.completion_tokens) == (100, 50)
    assert record.cost_usd == pytest.approx((100 * 10 + 50 * 30) / 1e6)
    assert record.latency_s == 0.25  # one tick between sending and receiving
    assert record.served_model == "smart"
    assert record.attempts == 1
    await client.aclose()


async def test_an_auto_answer_is_priced_as_the_model_that_served_it() -> None:
    headers = {
        "x-llmrouter-model": "cheap",
        "x-llmrouter-tier": "slm",
        "x-llmrouter-escalated": "false",
    }
    runner, _, client = runner_for(Fake(ok(headers=headers)))

    record = await runner.ask(small(1).items[0], model="auto", label="router")

    assert record.served_model == "cheap"
    assert record.cost_usd == pytest.approx((100 * 1 + 50 * 2) / 1e6)
    assert record.routing["x-llmrouter-tier"] == "slm"
    assert record.routing["x-llmrouter-escalated"] == "false"
    await client.aclose()


async def test_an_answer_from_an_unknown_model_is_unpriced_not_free() -> None:
    runner, _, client = runner_for(Fake(ok(headers={"x-llmrouter-model": "mystery"})))
    record = await runner.ask(small(1).items[0], model="auto", label="router")
    assert record.ok and record.cost_usd is None
    await client.aclose()


async def test_missing_usage_leaves_the_cost_unknown() -> None:
    body = completion()
    del body["usage"]
    runner, _, client = runner_for(Fake(httpx.Response(200, json=body)))
    record = await runner.ask(small(1).items[0], model="smart", label="b")
    assert record.ok and record.cost_usd is None and record.prompt_tokens is None
    await client.aclose()


@pytest.mark.parametrize("usage", [{"prompt_tokens": 10}, {"completion_tokens": 10}, {}])
async def test_partial_usage_leaves_the_cost_unknown(usage: dict[str, int]) -> None:
    runner, _, client = runner_for(Fake(ok(usage=usage)))
    record = await runner.ask(small(1).items[0], model="smart", label="b")
    assert record.ok and record.cost_usd is None
    await client.aclose()


async def test_an_empty_string_reply_is_still_an_answer() -> None:
    runner, _, client = runner_for(Fake(ok(content="")))
    record = await runner.ask(small(1).items[0], model="smart", label="b")
    assert record.ok and record.reply == ""
    await client.aclose()


async def test_a_reply_with_no_content_is_a_failure() -> None:
    runner, _, client = runner_for(Fake(ok(content=None)))
    record = await runner.ask(small(1).items[0], model="smart", label="b")
    assert not record.ok and "no content" in (record.error or "")
    await client.aclose()


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"choices": []}'])
async def test_a_malformed_response_is_a_failure(body: bytes) -> None:
    runner, _, client = runner_for(Fake(httpx.Response(200, content=body)))
    record = await runner.ask(small(1).items[0], model="smart", label="b")
    assert not record.ok and record.error == "malformed completion response"
    await client.aclose()


# ---------- failures and retries ----------


async def test_a_bad_request_is_not_retried() -> None:
    error = {"error": {"message": "model not allowed", "type": "x"}}
    fake = Fake(httpx.Response(403, json=error))
    runner, sleeps, client = runner_for(fake)

    record = await runner.ask(small(1).items[0], model="smart", label="b")

    assert not record.ok and record.status_code == 403
    assert record.error == "HTTP 403: model not allowed"
    assert len(fake.requests) == 1 and sleeps.calls == []
    await client.aclose()


@pytest.mark.parametrize("status", [429, 502, 503, 504])
async def test_transient_failures_are_retried(status: int) -> None:
    fake = Fake(httpx.Response(status, json={"error": {"message": "later"}}), ok())
    runner, sleeps, client = runner_for(fake)

    record = await runner.ask(small(1).items[0], model="smart", label="b")

    assert record.ok and record.attempts == 2
    assert sleeps.calls == [1.0]
    assert record.latency_s == 0.25  # the answering attempt only, not the failed one before it
    await client.aclose()


async def test_retry_after_is_honoured_but_capped() -> None:
    fake = Fake(
        httpx.Response(429, headers={"Retry-After": "7"}, json={}),
        httpx.Response(429, headers={"Retry-After": "9999"}, json={}),
        ok(),
    )
    runner, sleeps, client = runner_for(fake, max_wait=30)

    await runner.ask(small(1).items[0], model="smart", label="b")

    assert sleeps.calls == [7.0, 30.0]
    await client.aclose()


async def test_the_wait_grows_without_a_retry_after() -> None:
    fake = Fake(*[httpx.Response(503, json={})] * 3, ok())
    runner, sleeps, client = runner_for(fake)
    await runner.ask(small(1).items[0], model="smart", label="b")
    assert sleeps.calls == [1.0, 2.0, 4.0]
    await client.aclose()


async def test_an_item_that_keeps_failing_is_recorded_as_failed() -> None:
    fake = Fake(httpx.Response(503, json={"error": {"message": "down"}}))
    runner, _, client = runner_for(fake, retries=2)

    record = await runner.ask(small(1).items[0], model="smart", label="b")

    assert not record.ok and record.status_code == 503
    assert record.attempts == 3 and len(fake.requests) == 3
    await client.aclose()


async def test_network_errors_are_retried_then_recorded() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    runner, sleeps, client = runner_for(boom, retries=1)

    record = await runner.ask(small(1).items[0], model="smart", label="b")

    assert not record.ok and record.status_code is None
    assert "ConnectError" in (record.error or "")
    assert record.attempts == 2 and sleeps.calls == [1.0]
    await client.aclose()


async def test_no_retries_means_one_attempt() -> None:
    fake = Fake(httpx.Response(503, json={}))
    runner, _, client = runner_for(fake, retries=0)
    record = await runner.ask(small(1).items[0], model="smart", label="b")
    assert record.attempts == 1 and len(fake.requests) == 1
    await client.aclose()


def test_nonsense_runner_settings_are_rejected() -> None:
    client = httpx.AsyncClient()
    with pytest.raises(ValueError):
        Runner(client, PRICES, concurrency=0)
    with pytest.raises(ValueError):
        Runner(client, PRICES, retries=-1)


# ---------- a whole run ----------


async def test_records_come_back_in_benchmark_order() -> None:
    async def slow_for_first(request: httpx.Request) -> httpx.Response:
        question = json.loads(request.content)["messages"][0]["content"]
        await asyncio.sleep(0.05 if "1" in question else 0)
        return ok(content=question)

    runner, _, client = runner_for(slow_for_first, concurrency=3)

    records = await runner.run(small(3), model="smart", label="b")

    assert [r.item_id for r in records] == ["qa-01", "qa-02", "qa-03"]
    assert [r.reply for r in records] == ["Question 1?", "Question 2?", "Question 3?"]
    await client.aclose()


async def test_no_more_requests_are_in_flight_than_the_concurrency() -> None:
    running = peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return ok()

    runner, _, client = runner_for(handler, concurrency=2)
    await runner.run(small(8), model="smart", label="b")

    assert peak == 2
    await client.aclose()


async def test_every_record_is_written_to_the_results_file(tmp_path: Path) -> None:
    runner, _, client = runner_for(Fake(ok()))
    out = tmp_path / "runs" / "b.jsonl"

    records = await runner.run(small(3), model="smart", label="b", out=out)

    assert [r.item_id for r in load_records(out)] == [r.item_id for r in records]
    assert len(out.read_text().splitlines()) == 3
    await client.aclose()


async def test_a_new_run_replaces_the_old_results_file(tmp_path: Path) -> None:
    runner, _, client = runner_for(Fake(ok()))
    out = tmp_path / "b.jsonl"
    await runner.run(small(3), model="smart", label="b", out=out)
    await runner.run(small(2), model="smart", label="b", out=out)
    assert len(out.read_text().splitlines()) == 2
    await client.aclose()


async def test_resume_asks_only_what_is_missing_or_failed(tmp_path: Path) -> None:
    out = tmp_path / "b.jsonl"
    first = Fake(ok(), httpx.Response(400, json={"error": {"message": "no"}}), ok())
    runner, _, client = runner_for(first, concurrency=1)
    await runner.run(small(3), model="smart", label="b", out=out)
    await client.aclose()

    second = Fake(ok(content="redone"))
    runner, _, client = runner_for(second, concurrency=1)
    records = await runner.run(small(3), model="smart", label="b", out=out, resume=True)
    await client.aclose()

    assert len(second.requests) == 1  # only the failed item was asked again
    assert [r.ok for r in records] == [True, True, True]
    assert records[1].reply == "redone"
    assert [r.item_id for r in load_records(out)] == ["qa-01", "qa-02", "qa-03"]


async def test_resume_does_not_reuse_another_runs_answers(tmp_path: Path) -> None:
    out = tmp_path / "x.jsonl"
    runner, _, client = runner_for(Fake(ok()))
    await runner.run(small(2), model="smart", label="other", out=out)

    again = Fake(ok())
    runner, _, client2 = runner_for(again)
    await runner.run(small(2), model="smart", label="mine", out=out, resume=True)

    assert len(again.requests) == 2
    await client.aclose()
    await client2.aclose()


async def test_resume_with_no_file_just_runs(tmp_path: Path) -> None:
    runner, _, client = runner_for(Fake(ok()))
    records = await runner.run(
        small(2), model="smart", label="b", out=tmp_path / "n.jsonl", resume=True
    )
    assert len(records) == 2
    await client.aclose()


async def test_the_callback_hears_about_every_item() -> None:
    seen: list[str] = []
    runner, _, client = runner_for(Fake(ok()))
    await runner.run(small(4), model="smart", label="b", on_record=lambda r: seen.append(r.item_id))
    assert sorted(seen) == ["qa-01", "qa-02", "qa-03", "qa-04"]
    await client.aclose()


# ---------- results files ----------


def record(item_id: str, **kwargs: Any) -> EvalRecord:
    fields: dict[str, Any] = {
        "item_id": item_id,
        "run": "b",
        "requested_model": "smart",
        "ok": True,
    }
    return EvalRecord(**{**fields, **kwargs})


def test_when_an_item_appears_twice_the_last_record_wins(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text(
        record("qa-01", ok=False, error="x").model_dump_json()
        + "\n\n"
        + record("qa-01", reply="fixed").model_dump_json()
        + "\n"
    )
    loaded = load_records(path)
    assert len(loaded) == 1 and loaded[0].reply == "fixed"


def test_a_corrupt_results_file_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text(record("qa-01").model_dump_json() + "\n{oops\n")
    with pytest.raises(ValueError, match="line 2"):
        load_records(path)


# ---------- summaries ----------


def test_percentiles_use_the_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert percentile(values, 0.50) == 5.0
    assert percentile(values, 0.95) == 10.0
    assert percentile(values, 0.10) == 1.0
    assert percentile([3.0], 0.95) == 3.0
    assert percentile([], 0.5) is None


def test_the_summary_adds_up_what_was_answered() -> None:
    records = [
        record(
            "qa-01",
            served_model="cheap",
            cost_usd=0.001,
            latency_s=1.0,
            prompt_tokens=10,
            completion_tokens=5,
        ),
        record(
            "qa-02",
            served_model="smart",
            cost_usd=0.003,
            latency_s=3.0,
            prompt_tokens=20,
            completion_tokens=10,
        ),
        record(
            "qa-03",
            served_model="smart",
            cost_usd=None,
            latency_s=2.0,
            prompt_tokens=1,
            completion_tokens=1,
        ),
        record("qa-04", ok=False, error="boom", cost_usd=9.0),
    ]
    summary = summarize(records)

    assert (summary.items, summary.succeeded, summary.failed, summary.unpriced) == (4, 3, 1, 1)
    assert summary.total_cost_usd == pytest.approx(0.004)  # the failure's cost is not counted
    assert summary.mean_cost_usd == pytest.approx(0.002)  # over the two priced answers
    assert (summary.prompt_tokens, summary.completion_tokens) == (31, 16)
    assert summary.latency_p50_s == 2.0
    assert summary.served_by == {"cheap": 1, "smart": 2}


def test_an_empty_summary_does_not_divide_by_zero() -> None:
    summary = summarize([])
    assert summary.items == 0 and summary.mean_cost_usd == 0.0 and summary.latency_p50_s is None


# ---------- prices ----------


def test_an_alias_with_several_deployments_costs_their_weighted_average() -> None:
    book = PriceBook(
        [
            model_config(
                name="m", model="a", weight=3, input_cost_per_mtok=1.0, output_cost_per_mtok=0.0
            ),
            model_config(
                name="m", model="b", weight=1, input_cost_per_mtok=5.0, output_cost_per_mtok=0.0
            ),
        ]
    )
    assert book.cost("m", 1_000_000, 0) == pytest.approx(2.0)  # (3*1 + 1*5) / 4


def test_an_unknown_alias_is_a_key_error_not_a_zero() -> None:
    with pytest.raises(KeyError):
        PRICES.cost("nope", 1, 1)
    assert "nope" not in PRICES and "smart" in PRICES


def test_prices_load_from_a_gateway_config_without_its_api_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
models:
  - name: fast
    tier: slm
    provider: openai_compatible
    model: small
    base_url: https://api.example.com/v1
    api_key: os.environ/SURELY_NOT_SET_ANYWHERE
    input_cost_per_mtok: 0.05
    output_cost_per_mtok: 0.08
  - name: smart
    tier: frontier
    provider: anthropic
    model: big
    api_key: os.environ/ALSO_NOT_SET
    input_cost_per_mtok: 3.0
    output_cost_per_mtok: 15.0
"""
    )
    book = PriceBook.from_file(path)
    assert book.aliases() == ["fast", "smart"]
    assert book.cost("smart", 1_000_000, 1_000_000) == pytest.approx(18.0)
    assert book.frontier_aliases() == ["smart"]
    assert book.tier("fast") is Tier.SLM


def test_a_broken_config_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("models: []\n")
    with pytest.raises(ConfigError):
        PriceBook.from_file(path)


# ---------- choosing the baseline model ----------


def tiered(**tiers: Tier) -> PriceBook:
    return PriceBook(model_config(name=a, tier=t) for a, t in tiers.items())


def test_the_only_frontier_model_is_the_baseline() -> None:
    assert cli.choose_frontier(tiered(a=Tier.SLM, b=Tier.FRONTIER), None) == "b"


def test_several_or_no_frontier_models_need_an_explicit_choice() -> None:
    with pytest.raises(ValueError, match="--model"):
        cli.choose_frontier(tiered(a=Tier.FRONTIER, b=Tier.FRONTIER), None)
    with pytest.raises(ValueError, match="--model"):
        cli.choose_frontier(tiered(a=Tier.SLM), None)
    assert cli.choose_frontier(tiered(a=Tier.FRONTIER, b=Tier.FRONTIER), "a") == "a"


def test_an_unknown_baseline_model_is_refused() -> None:
    with pytest.raises(ValueError, match="not a model in the config"):
        cli.choose_frontier(tiered(a=Tier.FRONTIER), "zzz")


# ---------- through the real gateway ----------


@pytest.fixture
async def gateway() -> AsyncIterator[tuple[httpx.AsyncClient, FakeProvider, FakeProvider]]:
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
            yield client, cheap, smart


async def test_the_baseline_sends_everything_to_the_frontier_model(
    gateway: tuple[httpx.AsyncClient, FakeProvider, FakeProvider],
) -> None:
    client, cheap, smart = gateway
    book = prices(cheap=(1.0, 2.0), smart=(10.0, 30.0))

    records = await Runner(client, book).run(
        load_benchmark().subset(limit=12), model="smart", label="baseline"
    )

    assert all(r.ok for r in records)
    assert smart.calls == 12 and cheap.calls == 0
    assert {r.served_model for r in records} == {"smart"}
    # the fake provider reports 10 prompt and 5 completion tokens per answer
    assert summarize(records).total_cost_usd == pytest.approx(12 * (10 * 10 + 5 * 30) / 1e6)


async def test_auto_answers_are_attributed_and_priced_by_the_model_that_served_them(
    gateway: tuple[httpx.AsyncClient, FakeProvider, FakeProvider],
) -> None:
    client, cheap, smart = gateway
    book = prices(cheap=(1.0, 2.0), smart=(10.0, 30.0))

    records = await Runner(client, book).run(
        load_benchmark().subset(limit=12), model="auto", label="router"
    )

    summary = summarize(records)
    assert summary.failed == 0 and summary.unpriced == 0
    assert sum(summary.served_by.values()) == 12
    assert set(summary.served_by) <= {"cheap", "smart"}
    assert cheap.calls + smart.calls == 12
    expected = cheap.calls * (10 * 1 + 5 * 2) / 1e6 + smart.calls * (10 * 10 + 5 * 30) / 1e6
    assert summary.total_cost_usd == pytest.approx(expected)


async def test_a_model_the_key_cannot_reach_is_a_recorded_failure_not_a_crash(
    gateway: tuple[httpx.AsyncClient, FakeProvider, FakeProvider],
) -> None:
    client, _, _ = gateway
    records = await Runner(client, PRICES).run(small(2), model="no-such-model", label="b")
    assert [r.ok for r in records] == [False, False]
    assert records[0].status_code == 404


# ---------- the command line ----------


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
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
  - name: smart
    tier: frontier
    provider: openai_compatible
    model: big
    base_url: https://api.example.com/v1
    input_cost_per_mtok: 10.0
    output_cost_per_mtok: 30.0
"""
    )
    return path


@pytest.fixture
def mock_gateway(monkeypatch: pytest.MonkeyPatch) -> Callable[[Fake], None]:
    real = httpx.AsyncClient

    def install(fake: Fake) -> None:
        def factory(**kwargs: Any) -> httpx.AsyncClient:
            return real(transport=httpx.MockTransport(fake), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    return install


def run_cli(tmp_path: Path, *extra: str) -> int:
    return cli.main(
        [
            "baseline",
            "--api-key",
            "k",
            "--config",
            str(write_config(tmp_path)),
            "--out",
            str(tmp_path / "out.jsonl"),
            "--limit",
            "4",
            *extra,
        ]
    )


def test_the_baseline_command_runs_and_reports(
    tmp_path: Path, mock_gateway: Callable[[Fake], None], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = Fake(ok())
    mock_gateway(fake)

    code = run_cli(tmp_path)

    assert code == 0
    assert {json.loads(r.content)["model"] for r in fake.requests} == {"smart"}
    assert len(load_records(tmp_path / "out.jsonl")) == 4
    out = capsys.readouterr().out
    assert "run 'baseline': 4/4 answered, 0 failed" in out
    assert "smart x4" in out


def test_failures_make_the_command_fail_unless_allowed(
    tmp_path: Path, mock_gateway: Callable[[Fake], None], capsys: pytest.CaptureFixture[str]
) -> None:
    mock_gateway(Fake(httpx.Response(400, json={"error": {"message": "bad"}})))
    assert run_cli(tmp_path) == 1
    assert "not comparable" in capsys.readouterr().err
    assert run_cli(tmp_path, "--allow-failures") == 0


def test_resume_through_the_command(tmp_path: Path, mock_gateway: Callable[[Fake], None]) -> None:
    mock_gateway(Fake(ok()))
    assert run_cli(tmp_path) == 0
    again = Fake(ok())
    mock_gateway(again)
    assert run_cli(tmp_path, "--resume") == 0
    assert again.requests == []  # everything was already answered


def test_the_command_can_be_narrowed(tmp_path: Path, mock_gateway: Callable[[Fake], None]) -> None:
    fake = Fake(ok())
    mock_gateway(fake)
    code = cli.main(
        [
            "baseline",
            "--api-key",
            "k",
            "--config",
            str(write_config(tmp_path)),
            "--out",
            str(tmp_path / "o.jsonl"),
            "--category",
            "math",
            "--difficulty",
            "easy",
        ]
    )
    assert code == 0
    assert len(fake.requests) == 4  # the four easy math questions


def test_a_missing_api_key_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LLMROUTER_EVAL_KEY", raising=False)
    assert cli.main(["baseline", "--config", str(write_config(tmp_path))]) == 2
    assert "API key" in capsys.readouterr().err


@pytest.mark.parametrize("problem", ["config", "dataset", "model"])
def test_bad_inputs_are_refused_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], problem: str
) -> None:
    args = ["baseline", "--api-key", "k", "--config", str(write_config(tmp_path))]
    if problem == "config":
        args[-1] = str(tmp_path / "missing.yaml")
    elif problem == "dataset":
        args += ["--dataset", str(tmp_path / "missing.jsonl")]
    else:
        args += ["--model", "nope"]
    assert cli.main(args) == 2
    assert capsys.readouterr().err.startswith("error:")
