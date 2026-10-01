# Load test notes

What the gateway sustains, what it costs per request, how it behaves when overloaded or when a
provider goes down, and what the load test turned up. Every number here was measured with the
tool in `app/loadtest/` on the machine described below; nothing is estimated.

## How to reproduce

```bash
make redis-up                      # optional: only needed for limits, quotas and shared breakers
python -m app.loadtest run \
    --redis-url redis://localhost:6379/0 \
    --concurrency 50 --duration 10 --generators 4 \
    --set LLMROUTER_BREAKER_COOLDOWN=2 \
    --json loadtest.json
```

`python -m app.loadtest run` starts a stand-in model server (the *stub*), migrates a throwaway
database, starts the real gateway (`uvicorn app.main:app`, one worker) and drives it over real
sockets. No model is called, so nothing is spent and nothing throttles the test. Scenarios:
`direct` (straight to the stub), `gateway`, `limited` (a key with RPM, TPM and a budget), `auto`,
`stream`, `sweep` (rising concurrency), `paced` (fixed arrival rates), `outage` and `rate_limit`.

## What was measured, and on what

| | |
|---|---|
| Machine | Apple M4, 10 cores, macOS; Python 3.12.14 |
| Gateway | one uvicorn worker, `LLMROUTER_LOG_LEVEL=WARNING` |
| Upstream | the stub, one process, each answer takes 50 ms |
| Database | SQLite file unless stated; Postgres 17 in Docker where stated |
| Redis | 7.4 in Docker |
| Load | 4 generator processes; 10 s measured after 2 s of warm-up (and 3 s of unrecorded pre-warm) |
| Repeats | **one run per row.** Treat differences under about 10% as noise. |

Everything runs on one machine, so the gateway, stub and generators compete for the same cores
and the network is loopback. Real deployments will differ.

## Results

### Throughput and latency

| scenario | req/s | p50 ms | p95 ms | p99 ms | errors | gateway CPU |
|---|---:|---:|---:|---:|---:|---:|
| straight to the stub (the ceiling of this setup) | 955 | 52.0 | 56.6 | 61.5 | 0% | n/a |
| gateway, named model | 581 | 64.6 | 145.9 | 598.4 | 0% | 110% |
| gateway, `model=auto` | 545 | 69.1 | 171.0 | 617.2 | 0% | 111% |
| gateway, streamed | 373 | 110.6 | 224.2 | 448.0 | 0% | 110% |

A streamed answer's first chunk arrived at p50 57.5 ms, p95 78.0 ms (the stub's own delay is 50 ms).
"Gateway CPU" is percent of one core, so about 110% means the process has run out of CPU.

### Rising concurrency (closed loop)

| concurrent requests | req/s | p50 ms | p95 ms | p99 ms | errors |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.5 | 64.5 | 68.0 | 79.7 | 0% |
| 10 | 179 | 55.0 | 62.6 | 74.5 | 0% |
| 25 | 426 | 55.4 | 75.9 | 121.0 | 0% |
| 50 | 577 | 64.8 | 154.0 | 611.5 | 0% |
| 100 | 564 | 153.3 | 278.8 | 668.5 | 0% |
| 200 | 541 | 347.0 | 622.6 | 964.1 | 0% |
| 400 | 528 | 737.4 | 1199.4 | 1697.4 | 0% |

**One gateway process tops out near 530 to 580 req/s here, because it is CPU-bound.** Past about
50 concurrent requests, more concurrency buys no more throughput; it only adds queueing delay
(latency grows in proportion, as it should). Throughput stays flat out to 400 concurrent: it
degrades gracefully rather than collapsing, and no request failed.

### Requests arriving at a fixed rate (open loop)

| arrival rate | achieved | p50 ms | p95 ms | p99 ms | errors |
|---:|---:|---:|---:|---:|---:|
| 100 | 100 | 62.3 | 71.7 | 94.3 | 0% |
| 300 | 300 | 61.0 | 80.5 | 110.5 | 0% |
| 500 | 500 | 76.5 | 225.6 | 524.1 | 0% |
| 700 | 679 | 14,907 | 26,457 | 32,213 | 3% (connection errors) |

Below capacity it keeps up. **Above capacity (700 req/s offered against about 560 sustained) the
queue grows without bound**: latencies run to tens of seconds and some connections are refused.
The gateway does not shed load on its own; put a limit in front of it (or run more replicas) if
traffic can exceed capacity.

### The upstream goes down for 5 seconds (100 req/s throughout)

| phase | sent | succeeded | reached the upstream | errors |
|---|---:|---:|---:|---|
| healthy | 500 | 500 | 500 | none |
| outage | 500 | 0 | **12** | 500 x 503 |
| recovery | 500 | 374 | 374 | 126 x 503 |

Once the circuit breaker opened, the dead upstream saw 12 requests instead of 500. The 126 refusals
in the recovery phase are the breaker's 2-second cooldown (set for this run with
`LLMROUTER_BREAKER_COOLDOWN=2`; the default is 30 s).

### A rate limit under a simultaneous burst

300 requests arrived at once (100 at a time) against a key limited to 60 per minute: **exactly 60 were
allowed, 240 got a 429, and the upstream received exactly 60.** The limit held under concurrency.

## What each part costs

Gateway at 100 concurrent, one run each:

| configuration | req/s | p95 ms | p99 ms |
|---|---:|---:|---:|
| SQLite, defaults | 563 | 272 | 652 |
| SQLite, request log off | **743** | 190 | 237 |
| SQLite, request log and metrics off | 763 | 181 | 217 |
| Postgres (in Docker), defaults | 546 | 284 | 361 |
| Postgres (in Docker), request log off | 610 | 248 | 316 |
| no Redis (quotas fail open) | 546 | 290 | 699 |
| ordinary key (no limits) | 554 | 283 | 699 |
| key with RPM, TPM and a budget | **438** | 385 | 887 |

- **The per-request log write is the biggest cost that can be changed**: about 32% of capacity on SQLite
  (a commit per request), about 12% against Postgres. Metrics cost about 3%.
- **Enforcing limits costs about 21% of capacity.** The Redis statistics show why: a limited request
  makes about **five Lua-script calls** (rate check, token reservation, budget reservation, and the
  two settlements). An ordinary key makes none.
- The Postgres figures include a Docker network hop on a laptop, so they overstate what a
  co-located database costs.

## Problems the load test found

1. **A saturated connection pool opened the circuit breaker on a healthy provider (fixed).** With a
   slow provider (8 s answers) and more requests in flight than the gateway's 100 upstream
   connections, requests that waited too long for a free connection were recorded as the
   *provider* timing out. That opened the breaker, which then refused 6,680 requests with a 503
   while the provider reported zero errors; only 100 requests succeeded. Now a pool timeout is its own
   error (`gateway_overloaded`): it is not counted against any breaker, is not retried or sent to a
   fallback (they share the same pool), and returns `503` with `Retry-After: 1`. The same run now
   serves 230 requests, sheds 60, opens no breaker, and the provider still reports zero errors.
2. **Seven plain `def` dependencies, and `/health`, ran in a worker thread on every request (fixed).**
   They only read an attribute, so each request paid for thread hops. They are now `async def`, and
   a test fails if one comes back. Worth about 1 ms per request.
3. **Holding a database connection while the model works** was found by the previous commit's tests,
   not by this tool. What this tool adds is a check that it stays fixed under load: the sweep to 400
   concurrent requests finished with no errors (against the fast stub; it does not exercise 400
   *slow* requests, which the connection-pool tests cover at smaller scale).

## Mistakes to avoid when measuring (all made while producing these numbers)

- **A single load-generator process is the bottleneck above about 600 req/s.** One Python process
  running an HTTP client saturates one core; past roughly 45 connections it also behaves oddly (a
  "direct" run collapsed to 160 req/s with a 1-second tail). The stub was fine: four generator
  processes got 1,906 req/s from it. Always use `--generators`, and check the `direct` row.
- **Closed-loop tests hide overload**; use the open-loop `paced` runs to see queueing.
- **Warm up first**, or whichever scenario runs first looks worse than it is.

## Not explained

At a concurrency of 1, the gateway adds about **10 ms** when the upstream takes 50 ms, but only about
**2 ms** when the upstream answers instantly. The gateway's own handler accounts for 0.4 ms of that
(its metrics show 54.07 ms per request, 53.64 ms of it waiting for the upstream). The extra time is
spent before the handler starts, and grows once the request waits. It does not depend on the web
server or network (the same app run in-process shows it), on the middleware, or on SQLite versus
Postgres, and a trivial FastAPI app does not show it (it adds 2 ms). Removing the request-log write
recovers about 3 ms of it and the thread-hop fix about 1 ms. I could not pin down the rest, so
treat the idle-request overhead as "about 10 ms on this laptop" and measure it on your own hardware.

## What this does not show

- **Real model behaviour.** The stub answers after a fixed delay with a short fixed body. Real
  answers are longer, vary widely in latency and are sometimes throttled.
- **Several gateway replicas.** Everything above is one process. Capacity should scale with
  replicas because the state that must be shared (quotas, breakers) lives in Redis, but that is
  not measured here.
- **A production network, database or Redis.**
- **Long soaks.** Each row is seconds, not hours; there is no data on memory growth over time.
- **Variance.** Each row is one run.

## Leads, not commitments

- Write request logs in batches, off the response path (the largest single saving above).
- Combine a limited request's five Redis round trips into fewer.
- Shed load before the queue grows (a concurrency cap that returns `503` quickly).
- Raise the default `http_max_connections` (100): at 3 s answers it caps one replica near 33 req/s.
