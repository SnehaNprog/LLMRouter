import re
import time
from typing import Any

import httpx
import pytest
from openai import OpenAI

from app.core.config import Settings
from app.core.redis import build_redis_client
from app.providers.fake import FakeProvider
from app.quota.headers import RateHeaders, Window, format_duration
from app.quota.limiter import RateLimiter
from app.quota.reservation import NO_RESERVATION
from tests.test_rpm_limit import auth, chat, make_client
from tests.timing import balance_range, refill_since

ALL = {
    f"x-ratelimit-{part}-{kind}"
    for part in ("limit", "remaining", "reset")
    for kind in ("requests", "tokens")
}


def ratelimit_headers(response: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in response.headers.items() if k.startswith("x-ratelimit-")}


# ---------- formatting durations ----------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s"),
        (-5, "0s"),
        (0.0001, "1ms"),
        (0.02, "20ms"),
        (0.0201, "21ms"),  # rounded up, never down
        (0.999, "999ms"),
        (1, "1s"),
        (1.5, "1.5s"),
        (8.64, "8.64s"),
        (59.9999, "1m0s"),
        (60, "1m0s"),
        (360, "6m0s"),
        (90, "1m30s"),
        (3600, "1h0m0s"),
        (3723, "1h2m3s"),
        (90061, "25h1m1s"),
    ],
)
def test_durations_use_the_openai_style(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


# ---------- describing an allowance ----------


def test_a_full_allowance_has_nothing_left_to_wait_for() -> None:
    assert Window.from_balance(60, 60) == Window(limit=60, remaining=60, reset_seconds=0)


def test_a_partly_used_allowance_says_how_long_to_refill() -> None:
    window = Window.from_balance(60, 30)  # 60 a minute: one a second, 30 missing

    assert (window.remaining, window.reset_seconds) == (30, 30)


def test_an_allowance_in_debt_shows_nothing_left_and_a_longer_wait() -> None:
    window = Window.from_balance(60, -60)

    assert (window.remaining, window.reset_seconds) == (0, 120)


def test_only_the_known_allowances_are_reported() -> None:
    nothing = RateHeaders()
    just_requests = RateHeaders(requests=Window(60, 59, 1))
    both = RateHeaders(requests=Window(60, 59, 1), tokens=Window(1000, 900, 6))

    assert nothing.as_headers() == {}
    assert set(just_requests.as_headers()) == {
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
    }
    assert set(both.as_headers()) == ALL
    assert both.as_headers()["x-ratelimit-reset-tokens"] == "6s"


# ---------- on successful responses ----------


def test_responses_carry_the_openai_style_headers() -> None:
    with make_client() as client:
        secret = client.issue_key(rpm_limit=6, tpm_limit=1_000)

        started = time.monotonic()
        response = chat(client, secret, max_tokens=92)  # reserves 100 tokens, uses 15
        low, high = balance_range(15, started)

    headers = ratelimit_headers(response)
    assert set(headers) == ALL
    assert headers["x-ratelimit-limit-requests"] == "6"
    assert headers["x-ratelimit-remaining-requests"] == "5"
    assert headers["x-ratelimit-reset-requests"] == "10s"  # one missing request at 6 a minute
    assert headers["x-ratelimit-limit-tokens"] == "1000"
    assert low <= int(headers["x-ratelimit-remaining-tokens"]) <= high  # 1000 - 15 really used
    assert re.fullmatch(r"\d+ms", headers["x-ratelimit-reset-tokens"])


def test_the_remaining_count_goes_down_with_use() -> None:
    with make_client() as client:
        secret = client.issue_key(rpm_limit=6)

        remaining = [
            int(chat(client, secret).headers["x-ratelimit-remaining-requests"]) for _ in range(4)
        ]
        reset = chat(client, secret).headers["x-ratelimit-reset-requests"]

    assert remaining == [5, 4, 3, 2]
    assert reset == "50s"  # this fifth request leaves one of six: five missing, ten seconds each


def test_token_figures_reflect_what_was_really_used_not_what_was_reserved() -> None:
    heavy = FakeProvider(prompt_tokens=300, completion_tokens=200)  # 500 used per call
    with make_client(heavy) as client:
        secret = client.issue_key(tpm_limit=2_000)

        started = time.monotonic()
        first = chat(client, secret, max_tokens=92)  # reserves only 100, uses 500
        second = chat(client, secret, max_tokens=92)
        after_one = balance_range(500, started, limit=2_000)
        after_two = balance_range(1_000, started, limit=2_000)

    assert after_one[0] <= int(first.headers["x-ratelimit-remaining-tokens"]) <= after_one[1]
    assert after_two[0] <= int(second.headers["x-ratelimit-remaining-tokens"]) <= after_two[1]


def test_an_overrun_shows_zero_remaining_never_a_negative_number() -> None:
    glutton = FakeProvider(prompt_tokens=800, completion_tokens=700)  # 1,500 used
    with make_client(glutton) as client:
        secret = client.issue_key(tpm_limit=1_000)

        response = chat(client, secret, max_tokens=92)

    assert response.headers["x-ratelimit-remaining-tokens"] == "0"
    assert response.headers["x-ratelimit-reset-tokens"].startswith("1m")  # in debt: 90s to clear


def test_only_the_limits_that_exist_are_reported() -> None:
    with make_client() as client:
        requests_only = client.issue_key(rpm_limit=10)
        tokens_only = client.issue_key(tpm_limit=1_000)

        by_requests = ratelimit_headers(chat(client, requests_only))
        by_tokens = ratelimit_headers(chat(client, tokens_only))
        unlimited = ratelimit_headers(chat(client, client.secret))

    assert {k.rsplit("-", 1)[1] for k in by_requests} == {"requests"}
    assert {k.rsplit("-", 1)[1] for k in by_tokens} == {"tokens"}
    assert unlimited == {}


def test_the_gateway_defaults_are_reported_for_keys_without_their_own() -> None:
    with make_client(default_rpm_limit=30, default_tpm_limit=5_000) as client:
        headers = ratelimit_headers(chat(client, client.secret))

    assert headers["x-ratelimit-limit-requests"] == "30"
    assert headers["x-ratelimit-limit-tokens"] == "5000"


def test_each_key_sees_its_own_allowances() -> None:
    with make_client() as client:
        small = client.issue_key(rpm_limit=6)
        large = client.issue_key(rpm_limit=600)
        for _ in range(3):
            chat(client, small)

        small_headers = ratelimit_headers(chat(client, small))
        large_headers = ratelimit_headers(chat(client, large))

    assert small_headers["x-ratelimit-remaining-requests"] == "2"
    assert large_headers["x-ratelimit-remaining-requests"] == "599"


def test_only_chat_responses_carry_them() -> None:
    with make_client() as client:
        secret = client.issue_key(rpm_limit=10)

        models = client.get("/v1/models", headers=client.as_key(secret))

    assert ratelimit_headers(models) == {}


# ---------- streaming ----------


def test_streams_report_the_figures_as_of_the_reservation() -> None:
    with make_client() as client:
        secret = client.issue_key(rpm_limit=6, tpm_limit=1_000)

        response = chat(client, secret, stream=True, max_tokens=92)

    assert response.status_code == 200
    headers = ratelimit_headers(response)
    assert headers["x-ratelimit-remaining-requests"] == "5"
    assert (
        headers["x-ratelimit-remaining-tokens"] == "900"
    )  # 100 reserved; settled after the stream


def test_a_finished_stream_is_settled_before_the_next_request_reports() -> None:
    with make_client() as client:
        secret = client.issue_key(tpm_limit=1_000)
        started = time.monotonic()
        chat(client, secret, stream=True, max_tokens=92)  # reserves 100, really uses about 15

        follow_up = chat(client, secret, max_tokens=92)
        low, high = balance_range(30, started)

    # Had the stream's reservation not been settled, only ~885 would be left.
    assert low <= int(follow_up.headers["x-ratelimit-remaining-tokens"]) <= high


# ---------- on refusals ----------


def test_a_request_rate_refusal_carries_the_headers() -> None:
    with make_client() as client:
        secret = client.issue_key(rpm_limit=2)
        chat(client, secret)
        chat(client, secret)

        response = chat(client, secret)

    assert response.status_code == 429
    headers = ratelimit_headers(response)
    assert headers["x-ratelimit-limit-requests"] == "2"
    assert headers["x-ratelimit-remaining-requests"] == "0"
    assert headers["x-ratelimit-reset-requests"] == "1m0s"
    assert int(response.headers["Retry-After"]) >= 1
    assert "x-ratelimit-limit-tokens" not in headers  # never got as far as tokens


def test_a_token_refusal_reports_both_allowances() -> None:
    heavy = FakeProvider(prompt_tokens=400, completion_tokens=400)
    with make_client(heavy) as client:
        secret = client.issue_key(rpm_limit=6, tpm_limit=1_000)
        started = time.monotonic()
        chat(client, secret)  # leaves ~200 tokens

        response = chat(client, secret)
        low, high = balance_range(800, started)
        refilled = refill_since(started, 1_000)

    assert response.status_code == 429
    headers = ratelimit_headers(response)
    assert set(headers) == ALL
    assert headers["x-ratelimit-remaining-requests"] == "4"  # this attempt used a request
    assert low <= int(headers["x-ratelimit-remaining-tokens"]) <= high
    # About 800 tokens are missing, less whatever has refilled; 1,000 a minute is 16.7 a second.
    reset = re.fullmatch(r"(\d+(?:\.\d+)?)s", headers["x-ratelimit-reset-tokens"])
    assert reset is not None
    assert (
        (800 - refilled - 10) / (1_000 / 60) - 1 <= float(reset.group(1)) <= 810 / (1_000 / 60) + 1
    )
    assert "Retry-After" in response.headers


def test_a_request_too_large_for_the_key_still_reports_the_allowance() -> None:
    with make_client() as client:
        secret = client.issue_key(tpm_limit=300)

        response = chat(client, secret)  # needs ~520

    assert response.json()["error"]["code"] == "request_too_large"
    headers = ratelimit_headers(response)
    assert headers["x-ratelimit-limit-tokens"] == "300"
    assert headers["x-ratelimit-remaining-tokens"] == "300"
    assert headers["x-ratelimit-reset-tokens"] == "0s"


def test_a_budget_refusal_reports_the_request_allowance() -> None:
    from tests.test_budget_limit import make_client as budget_client
    from tests.test_budget_limit import make_key

    with budget_client() as client:
        secret = make_key(client, monthly_budget_usd=0.000001, rpm_limit=6)

        response = chat(client, secret)

    assert response.json()["error"]["code"] == "budget_exceeded"
    assert ratelimit_headers(response)["x-ratelimit-remaining-requests"] == "5"


def test_other_failures_do_not_pretend_to_report_allowances() -> None:
    with make_client(FakeProvider(fail_first=99), retry_max_attempts=1) as client:
        secret = client.issue_key(rpm_limit=6)

        response = chat(client, secret)

    assert response.status_code == 502
    assert ratelimit_headers(response) == {}


# ---------- when there is nothing to report ----------


def test_no_headers_when_redis_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    unreachable = Settings(_env_file=None, redis_url="redis://127.0.0.1:1/0", redis_timeout=0.2)
    monkeypatch.setattr(
        "app.lifespan.build_redis_client", lambda _: build_redis_client(unreachable)
    )

    with make_client() as client:
        secret = client.issue_key(rpm_limit=6, tpm_limit=1_000)

        response = chat(client, secret)

    assert response.status_code == 200  # still served (fails open)
    assert ratelimit_headers(response) == {}  # and honest about not knowing


async def test_the_limiter_returns_the_window_and_the_reservation_keeps_its_balance() -> None:
    import fakeredis

    from app.quota.bucket import BucketStore

    limiter = RateLimiter(BucketStore(fakeredis.FakeAsyncRedis(decode_responses=True)))
    caller = auth(rpm_limit=10)

    window = await limiter.check_request(caller)

    assert window == Window(limit=10, remaining=9, reset_seconds=6.0)
    assert await limiter.check_request(auth(rpm_limit=None)) is None
    assert limiter.token_window(caller, NO_RESERVATION) is None


# ---------- through the OpenAI SDK ----------


def test_the_sdk_can_read_the_headers() -> None:
    with make_client() as client:
        secret = client.issue_key(rpm_limit=6)
        sdk = OpenAI(base_url="http://testserver/v1", api_key=secret, http_client=client)

        raw: Any = sdk.chat.completions.with_raw_response.create(
            model="test-model", messages=[{"role": "user", "content": "hi"}]
        )

    assert raw.headers["x-ratelimit-limit-requests"] == "6"
    assert raw.headers["x-ratelimit-remaining-requests"] == "5"
    assert raw.parse().choices[0].message.content


def test_concurrent_requests_each_get_headers_about_their_own_key() -> None:
    """Header state is per request, so it cannot leak between callers sharing the gateway."""
    from concurrent.futures import ThreadPoolExecutor

    slow = FakeProvider(latency=0.05)  # keeps requests overlapping
    with make_client(slow) as client:
        small = client.issue_key(rpm_limit=600, tpm_limit=100_000)
        large = client.issue_key(rpm_limit=6_000, tpm_limit=900_000)
        callers = [(small, "600", "100000"), (large, "6000", "900000")]

        def call(index: int) -> tuple[int, dict[str, str]]:
            secret, _, _ = callers[index % 2]
            return index, ratelimit_headers(chat(client, secret))

        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(call, range(48)))

    for index, headers in results:
        _, requests_limit, tokens_limit = callers[index % 2]
        assert headers["x-ratelimit-limit-requests"] == requests_limit
        assert headers["x-ratelimit-limit-tokens"] == tokens_limit


@pytest.mark.parametrize("backend", ["fake", "real"])
def test_the_figures_agree_on_a_real_redis(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    if backend == "real":
        url = os.environ.get("LLMROUTER_TEST_REDIS_URL")
        if not url:
            pytest.skip("set LLMROUTER_TEST_REDIS_URL to test against a real Redis")
        real = Settings(_env_file=None, redis_url=url)
        monkeypatch.setattr("app.lifespan.build_redis_client", lambda _: build_redis_client(real))

    with make_client() as client:
        app = client.app
        assert client.portal is not None
        client.portal.call(app.state.redis.flushdb)  # type: ignore[attr-defined]
        secret = client.issue_key(rpm_limit=6, tpm_limit=1_000)

        started = time.monotonic()
        responses = [chat(client, secret, max_tokens=92) for _ in range(3)]
        low, high = balance_range(45, started)
        client.portal.call(app.state.redis.flushdb)  # type: ignore[attr-defined]

    remaining = [int(r.headers["x-ratelimit-remaining-requests"]) for r in responses]
    tokens = [int(r.headers["x-ratelimit-remaining-tokens"]) for r in responses]
    assert remaining == [5, 4, 3]
    assert tokens[0] >= tokens[1] - 1 and low <= tokens[2] <= high  # 15 used per request
