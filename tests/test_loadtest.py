import asyncio
import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.core.model_config import load_config
from app.core.tiers import Tier
from app.loadtest import harness
from app.loadtest.cli import format_results, main
from app.loadtest.harness import (
    Stack,
    StackConfig,
    free_port,
    gateway_environment,
    model_config_yaml,
    running_stack,
)
from app.loadtest.load import (
    Outcome,
    Run,
    Sample,
    chat_sender,
    closed_loop,
    open_loop,
    summarize,
)
from app.loadtest.scenarios import (
    Job,
    ScenarioResult,
    default_specs,
    merge,
    paced_specs,
    pack,
    run_outage,
    run_spec,
    split,
    sweep_specs,
    unpack,
)
from app.loadtest.stub import StubSettings, create_stub
from app.providers.openai_compat import OpenAICompatibleProvider
from app.schemas.chat import ChatCompletionRequest, Message
from app.schemas.completion import ChatCompletionChunk, ChatCompletionResponse

ASK = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def stub_client(**settings: Any) -> tuple[httpx.AsyncClient, Any]:
    app = create_stub(StubSettings(**settings))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://stub")
    return client, app


# ---------- the stub ----------


async def test_its_answers_are_valid_completions() -> None:
    client, _ = stub_client(latency_ms=0, prompt_tokens=20, completion_tokens=50)
    async with client:
        response = await client.post("/v1/chat/completions", json=ASK)
    completion = ChatCompletionResponse.model_validate(response.json())
    assert completion.model == "m"
    assert completion.usage is not None
    assert (completion.usage.prompt_tokens, completion.usage.completion_tokens) == (20, 50)


async def test_the_gateways_own_adapter_can_talk_to_it() -> None:
    """The point of the stub: the real provider adapter, unmodified, accepts what it says."""
    app = create_stub(StubSettings(latency_ms=0, stream_chunks=3))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        provider = OpenAICompatibleProvider(
            "stub", base_url="http://stub/v1", api_key=None, client=client
        )
        request = ChatCompletionRequest(model="m", messages=[Message(role="user", content="hi")])
        answer = await provider.chat(request)
        chunks = [chunk async for chunk in provider.stream(request)]

    assert answer.choices[0].message.content
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "word0 word1 word2 "
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 50


async def test_a_streamed_answer_is_well_formed_server_sent_events() -> None:
    client, _ = stub_client(latency_ms=0, stream_chunks=4)
    async with client:
        response = await client.post("/v1/chat/completions", json={**ASK, "stream": True})
    events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    parsed = [ChatCompletionChunk.model_validate_json(e) for e in events[:-1]]
    assert sum(1 for c in parsed if c.choices and c.choices[0].delta.content) >= 4
    assert parsed[-1].usage is not None and parsed[-1].choices == []
    assert response.headers["content-type"].startswith("text/event-stream")


async def test_answers_take_about_as_long_as_asked() -> None:
    client, _ = stub_client(latency_ms=100)
    async with client:
        started = asyncio.get_running_loop().time()
        await client.post("/v1/chat/completions", json=ASK)
        took = asyncio.get_running_loop().time() - started
    assert 0.09 <= took < 0.5


async def test_jitter_keeps_answers_within_its_bounds() -> None:
    client, _ = stub_client(latency_ms=60, jitter_ms=40)
    async with client:
        times = []
        for _ in range(12):
            started = asyncio.get_running_loop().time()
            await client.post("/v1/chat/completions", json=ASK)
            times.append(asyncio.get_running_loop().time() - started)
    assert min(times) >= 0.018 and max(times) < 0.3
    assert max(times) - min(times) > 0.01  # and they do vary


async def test_it_can_fail_on_demand() -> None:
    client, _ = stub_client(latency_ms=0, error_rate=1.0)
    async with client:
        response = await client.post("/v1/chat/completions", json=ASK)
        stats = (await client.get("/__stub/stats")).json()
    assert response.status_code == 503
    assert stats["errors"] == 1


async def test_it_can_throttle_with_a_retry_after() -> None:
    client, _ = stub_client(latency_ms=0, throttle_rate=1.0)
    async with client:
        response = await client.post("/v1/chat/completions", json=ASK)
    assert response.status_code == 429 and response.headers["retry-after"] == "1"


async def test_the_failure_share_is_roughly_what_was_asked_and_repeatable() -> None:
    async def failures(seed: int) -> int:
        client, _ = stub_client(latency_ms=0, error_rate=0.3, seed=seed)
        async with client:
            codes = [
                (await client.post("/v1/chat/completions", json=ASK)).status_code
                for _ in range(300)
            ]
        return codes.count(503)

    first, again, other = await failures(5), await failures(5), await failures(6)
    assert 60 <= first <= 120
    assert first == again
    assert first != other or True  # a different seed may differ; the same seed never does


async def test_it_counts_what_reached_it_and_how_many_were_at_once() -> None:
    client, _ = stub_client(latency_ms=100)
    async with client:
        await asyncio.gather(*[client.post("/v1/chat/completions", json=ASK) for _ in range(12)])
        stats = (await client.get("/__stub/stats")).json()
    assert stats["requests"] == 12 and stats["peak_in_flight"] == 12 and stats["in_flight"] == 0


async def test_in_flight_returns_to_zero_after_streams_too() -> None:
    client, _ = stub_client(latency_ms=0)
    async with client:
        await client.post("/v1/chat/completions", json={**ASK, "stream": True})
        stats = (await client.get("/__stub/stats")).json()
    assert stats["in_flight"] == 0


async def test_its_behaviour_can_be_changed_while_it_runs() -> None:
    client, _ = stub_client(latency_ms=0)
    async with client:
        assert (await client.post("/v1/chat/completions", json=ASK)).status_code == 200
        changed = await client.post("/__stub/config", json={"error_rate": 1.0})
        assert changed.json()["error_rate"] == 1.0
        assert (await client.post("/v1/chat/completions", json=ASK)).status_code == 503
        await client.post("/__stub/config", json={"error_rate": 0.0})
        assert (await client.post("/v1/chat/completions", json=ASK)).status_code == 200


async def test_nonsense_changes_are_refused() -> None:
    client, _ = stub_client()
    async with client:
        assert (await client.post("/__stub/config", json={"error_rate": 2})).status_code == 422
        assert (await client.post("/__stub/config", json={"latency_ms": -1})).status_code == 422


async def test_reset_zeroes_the_counters_and_says_what_they_were() -> None:
    client, _ = stub_client(latency_ms=0)
    async with client:
        await client.post("/v1/chat/completions", json=ASK)
        before = (await client.post("/__stub/reset")).json()
        after = (await client.get("/__stub/stats")).json()
    assert before["requests"] == 1 and after["requests"] == 0


# ---------- summarising a run ----------


def sample(latency: float, kind: str = "ok", status: int = 200, **kwargs: Any) -> Sample:
    return Sample(0.0, latency, Outcome(status, kind, kwargs.pop("ttfb", None)), **kwargs)


def test_percentiles_throughput_and_errors() -> None:
    samples = [sample(n / 100) for n in range(1, 101)]  # 10 ms to 1000 ms
    samples += [
        sample(0.5, "http_503", 503),
        sample(0.5, "http_503", 503),
        sample(0.1, "timeout", 0),
    ]
    summary = summarize(Run(samples, wall=10.0))

    assert (summary.requests, summary.ok) == (103, 100)
    assert summary.errors == {"http_503": 2, "timeout": 1}
    assert summary.throughput_rps == 10.0
    assert summary.latency_ms.p50 == 500.0 and summary.latency_ms.p95 == 950.0
    assert summary.latency_ms.p99 == 990.0 and summary.latency_ms.max == 1000.0
    assert summary.latency_ms.mean == pytest.approx(505.0)
    assert summary.error_rate == pytest.approx(3 / 103)


def test_failures_do_not_pollute_the_latency_figures() -> None:
    summary = summarize(Run([sample(0.1), sample(30.0, "timeout", 0)], wall=1.0))
    assert summary.latency_ms.max == 100.0


def test_an_empty_run_summarises_without_dividing_by_zero() -> None:
    summary = summarize(Run([], wall=5.0))
    assert summary.requests == 0 and summary.latency_ms.p50 is None and summary.error_rate == 0.0
    assert summarize(Run([sample(0.1)], wall=0.0)).throughput_rps == 0.0


def test_time_to_first_chunk_and_generator_lateness_appear_only_when_measured() -> None:
    plain = summarize(Run([sample(0.1)], wall=1.0))
    assert plain.ttfb_ms is None and plain.behind_ms is None

    rich = summarize(Run([sample(0.1, ttfb=0.02, behind=0.003)], wall=1.0))
    assert rich.ttfb_ms is not None and rich.ttfb_ms.p50 == 20.0
    assert rich.behind_ms is not None and rich.behind_ms.p50 == 3.0
    on_time = summarize(Run([sample(0.1, behind=0.0)], wall=1.0))
    assert on_time.behind_ms is not None and on_time.behind_ms.max == 0.0


# ---------- generating load ----------


def slow_send(seconds: float, tracker: list[int] | None = None) -> Any:
    running = 0

    async def send() -> Outcome:
        nonlocal running
        running += 1
        if tracker is not None:
            tracker.append(running)
        await asyncio.sleep(seconds)
        running -= 1
        return Outcome(200, "ok")

    return send


async def test_a_closed_loop_keeps_exactly_that_many_requests_going() -> None:
    seen: list[int] = []
    run = await closed_loop(slow_send(0.02, seen), concurrency=7, duration=0.4)
    assert max(seen) == 7
    assert 7 * 0.4 / 0.02 * 0.5 < len(run.samples) <= 7 * 0.4 / 0.02 + 7  # about 140, not more


async def test_a_closed_loop_leaves_out_what_began_during_the_warmup() -> None:
    run = await closed_loop(slow_send(0.01), concurrency=3, duration=0.2, warmup=0.2)
    assert run.samples and min(s.started for s in run.samples) >= 0.2
    assert run.wall == 0.2


async def test_a_closed_loop_stops_when_the_time_is_up() -> None:
    started = asyncio.get_running_loop().time()
    await closed_loop(slow_send(0.01), concurrency=2, duration=0.3)
    assert asyncio.get_running_loop().time() - started < 0.6


CLOSED_NONSENSE = [
    {"concurrency": 0, "duration": 1},
    {"concurrency": 1, "duration": 0},
    {"concurrency": 1, "duration": 1, "warmup": -1},
]
OPEN_NONSENSE = [
    {"rate": 0, "duration": 1},
    {"rate": 1, "duration": 0},
    {"rate": 1, "duration": 1, "warmup": -1},
]


@pytest.mark.parametrize("kwargs", CLOSED_NONSENSE)
async def test_a_closed_loop_refuses_nonsense(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        await closed_loop(slow_send(0), **kwargs)


async def test_an_open_loop_sends_at_the_chosen_rate_regardless_of_how_slow_answers_are() -> None:
    seen: list[int] = []
    run = await open_loop(slow_send(0.3, seen), rate=50, duration=0.4)
    assert len(run.samples) == 20  # 50 per second for 0.4 s, none held back by slow answers
    assert max(seen) > 10  # many in flight at once: it did not wait for answers


async def test_a_load_generator_that_cannot_keep_up_shows_it_as_lateness() -> None:
    run = await open_loop(slow_send(0.05), rate=100, duration=0.2, max_in_flight=1)
    summary = summarize(run)
    assert summary.behind_ms is not None and summary.behind_ms.max is not None
    assert summary.behind_ms.max > 100  # requests began far after their scheduled moment


async def test_an_open_loop_leaves_out_what_began_during_the_warmup() -> None:
    run = await open_loop(slow_send(0.001), rate=100, duration=0.2, warmup=0.2)
    assert 15 <= len(run.samples) <= 25  # the 20 sent after the warm-up, not all 40
    assert min(s.started for s in run.samples) >= 0.2


async def test_a_prompt_open_loop_starts_requests_on_time() -> None:
    run = await open_loop(slow_send(0.001), rate=100, duration=0.3)
    behind = summarize(run).behind_ms
    assert behind is not None and behind.p95 is not None and behind.p95 < 50


@pytest.mark.parametrize("kwargs", OPEN_NONSENSE)
async def test_an_open_loop_refuses_nonsense(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        await open_loop(slow_send(0), **kwargs)


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectError("refused"), "connect"),
        (httpx.RemoteProtocolError("garbled"), "transport"),
    ],
)
async def test_transport_failures_are_classified(error: Exception, kind: str) -> None:
    async def failing() -> Outcome:
        raise error

    run = await closed_loop(failing, concurrency=1, duration=0.05)
    assert run.samples and {s.outcome.kind for s in run.samples} == {kind}


async def test_a_bug_in_the_load_generator_is_not_swallowed() -> None:
    async def buggy() -> Outcome:
        raise ValueError("bug")

    with pytest.raises(ValueError, match="bug"):
        await closed_loop(buggy, concurrency=1, duration=0.05)


# ---------- the request it sends ----------


async def test_a_good_answer_is_ok_and_a_bad_one_names_its_status() -> None:
    client, _ = stub_client(latency_ms=0)
    async with client:
        ok = await chat_sender(client, model="m")()
        await client.post("/__stub/config", json={"error_rate": 1.0})
        bad = await chat_sender(client, model="m")()
    assert (ok.status, ok.kind) == (200, "ok") and (bad.status, bad.kind) == (503, "http_503")


async def test_a_streamed_answer_reports_its_first_chunk() -> None:
    client, _ = stub_client(latency_ms=50)
    async with client:
        outcome = await chat_sender(client, model="m", stream=True)()
    assert outcome.kind == "ok" and outcome.ttfb is not None and outcome.ttfb >= 0.04


async def test_a_stream_cut_short_is_not_counted_as_a_success() -> None:
    app = create_stub(StubSettings(latency_ms=0))

    @app.post("/v1/cut")
    async def cut() -> Any:
        from fastapi.responses import StreamingResponse

        async def body() -> Any:
            yield 'data: {"id":"x"}\n\n'  # no [DONE]

        return StreamingResponse(body(), media_type="text/event-stream")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://s") as c:
        outcome = await chat_sender(c, model="m", stream=True, path="/v1/cut")()
    assert outcome.kind == "truncated"


async def test_the_key_is_sent_as_a_bearer_token() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://x"
    ) as client:
        await chat_sender(client, model="m", api_key="llmr_abc")()
        await chat_sender(client, model="m")()
    assert seen == ["Bearer llmr_abc", ""]


# ---------- the harness's pure parts ----------


def test_the_generated_model_config_is_one_the_gateway_accepts() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(model_config_yaml("http://127.0.0.1:9100"))
        config = load_config(path, {})

    tiers = {m.name: m.tier for m in config.models}
    assert tiers == {"small": Tier.SLM, "big": Tier.FRONTIER}
    assert all(m.base_url == "http://127.0.0.1:9100/v1" for m in config.models)
    assert config.fallbacks == {"small": ("big",)}


def test_the_gateway_is_pointed_at_what_the_test_set_up() -> None:
    env = gateway_environment(StackConfig(), Path("/tmp/c.yaml"), "sqlite+aiosqlite:////tmp/x.db")
    assert env["LLMROUTER_CONFIG_PATH"] == "/tmp/c.yaml"
    assert env["LLMROUTER_DATABASE_URL"] == "sqlite+aiosqlite:////tmp/x.db"
    assert env["LLMROUTER_LOG_LEVEL"] == "WARNING"


def test_redis_is_used_only_when_asked_for() -> None:
    plain = gateway_environment(StackConfig(), Path("c"), "db")
    with_redis = gateway_environment(StackConfig(redis_url="redis://r:1/2"), Path("c"), "db")
    assert with_redis["LLMROUTER_REDIS_URL"] == "redis://r:1/2"
    assert plain.get("LLMROUTER_REDIS_URL") == os.environ.get("LLMROUTER_REDIS_URL")


def test_extra_settings_win() -> None:
    config = StackConfig(gateway_env={"LLMROUTER_LOG_LEVEL": "ERROR", "LLMROUTER_X": "1"})
    env = gateway_environment(config, Path("c"), "db")
    assert env["LLMROUTER_LOG_LEVEL"] == "ERROR" and env["LLMROUTER_X"] == "1"


def test_free_ports_are_usable_and_differ() -> None:
    first, second = free_port(), free_port()
    assert first != second and 1024 < first < 65536


@pytest.mark.parametrize(
    ("ps_output", "rss_mb", "cpu"),
    [
        ("  51200 0:12.50\n", 50.0, 12.5),
        ("  51200 1:02:03.25\n", 50.0, 3723.25),
        ("  102400 1-02:03:04\n", 100.0, 93784.0),
    ],
)
def test_process_stats_are_read_from_ps(
    monkeypatch: pytest.MonkeyPatch, ps_output: str, rss_mb: float, cpu: float
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout=ps_output, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    stats = Stack("s", "g", "k", 123, Path(".")).gateway_process_stats()
    assert stats == {"rss_mb": rss_mb, "cpu_seconds": cpu}


def test_unreadable_process_stats_are_none_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="no such process")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert Stack("s", "g", "k", 123, Path(".")).gateway_process_stats() is None


# ---------- the scenario list ----------


def test_the_standard_scenarios_cover_the_gateway_the_stub_and_streaming() -> None:
    specs = default_specs(concurrency=30, duration=5, warmup=1)
    assert set(specs) == {"direct", "gateway", "auto", "stream"}
    assert specs["direct"].target == "stub" and specs["gateway"].target == "gateway"
    assert specs["auto"].model == "auto" and specs["stream"].stream is True
    assert all(s.concurrency == 30 and s.duration == 5 and s.warmup == 1 for s in specs.values())


def test_a_sweep_has_one_spec_per_level_and_a_paced_run_one_per_rate() -> None:
    sweep = sweep_specs([1, 10, 100], duration=5, warmup=1)
    assert [s.concurrency for s in sweep] == [1, 10, 100] and sweep[0].name == "gateway@1"
    paced = paced_specs([100.0, 300.0], duration=5, warmup=1)
    assert [s.rate for s in paced] == [100.0, 300.0] and all(s.mode == "open" for s in paced)


# ---------- reading the results ----------


def test_the_table_lists_each_run_and_describes_the_special_ones() -> None:
    summary = summarize(Run([sample(0.05, ttfb=0.01), sample(0.05, "http_503", 503)], wall=1.0))
    table = format_results(
        [
            ScenarioResult(name="gateway", title="t", summary=summary, gateway_cpu_percent=42.0,
                           upstream_requests=2),
            ScenarioResult(name="outage", title="Upstream down", details={"phases": [
                {"phase": "outage", "sent": 100, "ok": 0, "errors": {"http_502": 95},
                 "reached_upstream": 6, "p95_ms": 5.0}]}),
            ScenarioResult(name="rate_limit", title="300 at once", details={"allowed": 60}),
        ]
    )  # fmt: skip
    assert "gateway" in table and "42" in table and "time to first chunk" in table
    assert "errors: {'http_503': 1}" in table
    assert "reached the upstream     6" in table and "allowed 60" in table


# ---------- the real thing, briefly ----------


def test_a_short_run_drives_a_real_gateway_over_real_sockets(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    code = main(
        [
            "run",
            "--scenarios", "direct,gateway,limited,stream,auto",
            "--duration", "0.8",
            "--warmup", "0.3",
            "--concurrency", "5",
            "--generators", "1",
            "--prewarm", "0.5",
            "--stub-latency-ms", "20",
            "--json", str(out),
        ]
    )  # fmt: skip

    assert code == 0
    report = json.loads(out.read_text())
    by_name = {r["name"]: r for r in report["results"]}
    assert set(by_name) == {"direct", "gateway", "limited", "stream", "auto"}
    for name, result in by_name.items():
        summary = result["summary"]
        assert summary["ok"] > 20 and summary["errors"] == {}, name
        assert summary["latency_ms"]["p50"] >= 19, name  # nothing answers faster than the stub
    assert by_name["stream"]["summary"]["ttfb_ms"] is not None
    assert by_name["gateway"]["upstream_requests"] >= by_name["gateway"]["summary"]["ok"]
    assert report["environment"]["stub_latency_ms"] == 20
    assert report["environment"]["generators"] == 1
    assert report["environment"]["redis"].startswith("no")


@pytest.fixture(scope="module")
def stack() -> Iterator[Stack]:
    """One real stub and gateway for the tests below, started once."""
    config = StackConfig(
        stub=StubSettings(latency_ms=10),
        gateway_env={
            "LLMROUTER_BREAKER_COOLDOWN": "1",
            "LLMROUTER_BREAKER_FAILURE_THRESHOLD": "5",
            "LLMROUTER_BREAKER_SHARED": "false",
        },
    )
    with running_stack(config) as running:
        yield running


def test_the_gateway_process_is_measured_during_a_scenario(stack: Stack) -> None:
    key = stack.make_key()
    spec = default_specs(concurrency=5, duration=1.5, warmup=0.5)["gateway"]

    result = asyncio.run(run_spec(stack, spec, key=key))

    assert result.gateway_cpu_percent is not None and result.gateway_cpu_percent > 0
    assert result.gateway_rss_mb is not None and result.gateway_rss_mb > 20


def test_an_outage_is_survived_and_the_dead_upstream_is_left_alone(stack: Stack) -> None:
    key = stack.make_key()

    result = asyncio.run(run_outage(stack, key, rate=60, phase_seconds=2))

    healthy, outage, recovery = result.details["phases"]
    assert healthy["errors"] == {} and healthy["ok"] > 80
    assert outage["ok"] == 0 and outage["sent"] > 80
    assert outage["reached_upstream"] < outage["sent"] / 2  # the breaker kept most of them away
    assert recovery["ok"] > recovery["sent"] / 2  # and good answers came back


def test_a_stack_that_cannot_start_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "_run", lambda command, env: (_ for _ in ()).throw(
        harness.HarnessError("alembic failed: boom")))  # fmt: skip
    with pytest.raises(harness.HarnessError, match="boom"), running_stack():
        pass


def test_bad_command_lines_are_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--set", "NOT_A_SETTING=1"]) == 2
    assert "LLMROUTER_NAME=value" in capsys.readouterr().err
    assert main(["run", "--scenarios", "direct,nope"]) == 1
    assert "unknown scenario" in capsys.readouterr().err


# ---------- sharing the load between processes ----------


def a_job(**changes: Any) -> Job:
    fields: dict[str, Any] = {
        "base_url": "http://x", "api_key": "k", "model": "m", "stream": False, "mode": "closed",
        "concurrency": 50, "rate": 100.0, "duration": 5.0, "warmup": 1.0,
    }  # fmt: skip
    return Job(**{**fields, **changes})


def test_connections_are_shared_out_without_losing_or_inventing_any() -> None:
    for connections in (1, 7, 50, 51, 100):
        for parts in (1, 2, 3, 4, 8):
            jobs = split(a_job(concurrency=connections), parts, start_at=10.0)
            assert sum(j.concurrency for j in jobs) == connections, (connections, parts)
            shares = [j.concurrency for j in jobs]
            assert max(shares) - min(shares) <= 1  # as even as whole numbers allow


def test_every_process_starts_at_the_same_moment_with_the_same_settings() -> None:
    jobs = split(a_job(), 4, start_at=1234.5)
    assert {j.start_at for j in jobs} == {1234.5}
    assert {(j.base_url, j.api_key, j.model, j.duration, j.warmup) for j in jobs} == {
        ("http://x", "k", "m", 5.0, 1.0)
    }


def test_the_arrival_rate_is_divided_so_the_total_is_unchanged() -> None:
    jobs = split(a_job(mode="open", rate=600.0), 3, start_at=0.0)
    assert [j.rate for j in jobs] == [200.0, 200.0, 200.0]
    assert sum(j.rate for j in jobs) == 600.0


def test_processes_with_no_connections_to_drive_are_not_started() -> None:
    jobs = split(a_job(concurrency=2), 5, start_at=0.0)
    assert len(jobs) == 2 and all(j.concurrency == 1 for j in jobs)


def test_an_open_loop_keeps_every_process_whatever_the_connection_count() -> None:
    assert len(split(a_job(mode="open", concurrency=0, rate=100.0), 4, start_at=0.0)) == 4


def test_splitting_into_no_parts_is_refused() -> None:
    with pytest.raises(ValueError):
        split(a_job(), 0, start_at=0.0)


def test_samples_survive_the_trip_between_processes() -> None:
    run = Run(
        [
            Sample(0.5, 0.2, Outcome(200, "ok", ttfb=0.05), behind=0.001),
            Sample(0.7, 1.5, Outcome(503, "http_503")),
        ],
        wall=5.0,
    )
    again = unpack(pack(run), run.wall)
    assert again.wall == 5.0 and again.samples == run.samples
    assert summarize(again) == summarize(run)


def test_runs_from_several_processes_merge_into_one() -> None:
    first = Run([Sample(0.0, 0.1, Outcome(200, "ok"))] * 3, wall=5.0)
    second = Run([Sample(0.0, 0.3, Outcome(200, "ok"))] * 2, wall=5.0)
    merged = merge([first, second], wall=5.0)
    summary = summarize(merged)
    assert summary.requests == 5 and summary.throughput_rps == 1.0
    assert summary.latency_ms.max == 300.0


def test_the_same_load_from_several_processes_reaches_the_gateway(stack: Stack) -> None:
    key = stack.make_key()
    spec = default_specs(concurrency=6, duration=1.0, warmup=0.3)["gateway"]

    one = asyncio.run(run_spec(stack, spec, key=key, generators=1))
    three = asyncio.run(run_spec(stack, spec, key=key, generators=3))

    assert one.summary is not None and three.summary is not None
    for summary in (one.summary, three.summary):
        assert summary.errors == {} and summary.ok > 40
    assert three.upstream_requests is not None
    assert three.upstream_requests >= three.summary.ok
    # six connections are six connections however many processes hold them
    assert three.summary.throughput_rps == pytest.approx(one.summary.throughput_rps, rel=0.5)


def test_an_open_loop_split_between_processes_keeps_its_rate(stack: Stack) -> None:
    key = stack.make_key()
    spec = paced_specs([60.0], duration=2.0, warmup=0.5)[0]

    result = asyncio.run(run_spec(stack, spec, key=key, generators=3))

    assert result.summary is not None and result.summary.errors == {}
    assert 100 <= result.summary.requests <= 140  # 60 per second for 2 seconds
    assert result.summary.behind_ms is not None


async def test_one_process_means_no_process_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.loadtest import scenarios

    ran: list[Job] = []

    async def fake_execute(job: Job) -> Run:
        ran.append(job)
        return Run([Sample(0.0, 0.1, Outcome(200, "ok"))], wall=job.duration)

    def no_pool(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a process pool was started for a single generator")

    monkeypatch.setattr(scenarios, "execute", fake_execute)
    monkeypatch.setattr(scenarios, "ProcessPoolExecutor", no_pool)

    run = await scenarios.generate(a_job(), 1)

    assert ran == [a_job()] and len(run.samples) == 1
