# LLMRouter

An OpenAI-compatible gateway for large language models, written in FastAPI. It puts one API in
front of several providers, gives each customer their own API keys, limits and budgets, and picks
the cheapest model that is strong enough for each request, falling back or escalating when
something fails.

```
                                   ┌──────────────────────────────────────────────┐
  client (OpenAI SDK)              │                  LLMRouter                   │
  POST /v1/chat/completions ─────▶ │  auth ─▶ limits ─▶ routing ─▶ resilience     │ ──▶ Groq / OpenAI / vLLM / Ollama
  Authorization: Bearer llmr_…     │  keys     RPM/TPM/   auto →    retries,      │ ──▶ Anthropic
                                   │  scopes   budget     a model   breakers,     │
                                   └──────┬───────────────┬─────────────┬─────────┘
                                          │               │             │
                                      Postgres/SQLite    Redis       Prometheus
                                      keys, tenants,   quotas, shared  /metrics
                                      request log      breaker state
```

## What it does

| | |
|---|---|
| **One API** | `POST /v1/chat/completions` with streaming, `GET /v1/models`; works with the official OpenAI SDK. Providers: any OpenAI-compatible API (OpenAI, Groq, vLLM, Ollama) and Anthropic. |
| **Customers** | Tenants own virtual API keys (`llmr_…`, stored only as a SHA-256 hash). Keys can expire, be revoked or rotated, and carry scopes, a model allow-list and their own limits. |
| **Limits** | Requests and tokens per minute and a monthly USD budget per key and per tenant, enforced atomically in Redis (Lua scripts), with `x-ratelimit-*` headers. |
| **Smart routing** | `model: "auto"` scores each request's complexity, then picks the cheapest, fastest model of a strong enough tier (small / mid / frontier). Escalates to a stronger model if the first fails. Each tenant can pin a model or cap the tier. |
| **Resilience** | Retries with backoff, fallback chains, circuit breakers per deployment *and* per provider (shared across replicas through Redis), `Retry-After` and cooldown handling on 429s, and routing around anything that is down. |
| **Visibility** | JSON logs with request ids, a request log (metadata only, never prompts or replies), a usage API, and Prometheus metrics. |
| **Measurable** | A benchmark, a quality judge and a paired report that measure what routing actually saves and what it costs in quality (see [Evaluation](#evaluation)). |

## Quick start

With Docker (Postgres, Redis, migrations and the gateway):

```bash
cp .env.example .env                  # provider keys go here, e.g. GROQ_API_KEY=...
cp config.example.yaml config.yaml    # the models you want to offer; edit it
make up                               # build and start everything on :8000

docker compose exec app python -m app.cli create-admin    # prints an admin key, once
curl localhost:8000/ready
```

On your own machine:

```bash
make install                          # pip install -e ".[dev]"
cp config.example.yaml config.yaml
make redis-up                         # Redis for limits and shared breaker state
make migrate                          # create the tables (SQLite by default)
make admin-key                        # prints an admin key, once
make dev                              # uvicorn with reload on :8000
```

Create a tenant's key with the admin key, then call the gateway like OpenAI:

```bash
curl -X POST localhost:8000/admin/tenants/$TENANT/keys \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name": "app", "scopes": ["chat:write", "models:read", "usage:read"]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="llmr_…")
reply = client.chat.completions.with_raw_response.create(
    model="auto",                                   # or a configured model name
    messages=[{"role": "user", "content": "Summarise TCP in one sentence."}],
)
print(reply.headers["x-llmrouter-model"], reply.headers["x-llmrouter-tier"])
print(reply.parse().choices[0].message.content)
```

## Configuration

**Models** live in `config.yaml` (see [config.example.yaml](config.example.yaml)). Secrets are never
written there; an entry names an environment variable instead (`api_key: os.environ/GROQ_API_KEY`).

```yaml
models:
  - name: fast                  # entries with the same name form one model; traffic splits by weight
    tier: slm                   # slm / mid / frontier
    provider: openai_compatible
    model: llama-3.1-8b-instant
    base_url: https://api.groq.com/openai/v1
    api_key: os.environ/GROQ_API_KEY
    input_cost_per_mtok: 0.05   # USD per million tokens: use your vendor's current price list
    output_cost_per_mtok: 0.08
fallbacks:
  fast: [smart]                 # tried in order once every retry on "fast" has failed
```

**Settings** come from `LLMROUTER_*` environment variables (or `.env`); the full list with
explanations is in [app/core/config.py](app/core/config.py). The ones you will most often touch:

| Setting | Default | Meaning |
|---|---|---|
| `LLMROUTER_DATABASE_URL` | SQLite file | Use `postgresql+asyncpg://…` for more than one replica. |
| `LLMROUTER_REDIS_URL` | `redis://localhost:6379/0` | Holds limits, budgets and shared breaker state. |
| `LLMROUTER_QUOTA_FAILURE_MODE` | `open` | If Redis is down: `open` serves unenforced, `closed` refuses limited keys. |
| `LLMROUTER_BREAKER_COOLDOWN` | `30` | Seconds an open circuit breaker waits before trial calls. |
| `LLMROUTER_HTTP_MAX_CONNECTIONS` | `100` | Concurrent upstream requests per process. See the [load-test notes](docs/load-test.md). |
| `LLMROUTER_METRICS_TOKEN` | none | Require this bearer token on `/metrics`. |
| `LLMROUTER_REQUEST_LOG_ENABLED` | `true` | Keep the metadata-only request log. |

## API

| Endpoint | Scope | What it does |
|---|---|---|
| `POST /v1/chat/completions` | `chat:write` | Chat completion; `stream: true` for server-sent events. |
| `GET /v1/models` | `models:read` | The models this key may use (`auto` first when routing is available). |
| `GET /v1/usage`, `GET /v1/usage/requests` | `usage:read` | Your organisation's requests, tokens and cost, and its recent requests. |
| `POST/GET/PATCH/DELETE /admin/tenants` | `admin:tenants` | Manage tenants (budgets, routing policy). |
| `POST/GET /admin/tenants/{id}/keys`, `…/{key}/revoke`, `…/{key}/rotate` | `admin:keys` | Manage a tenant's keys. |
| `GET /admin/tenants/{id}/usage`, `…/usage/requests` | `admin:tenants` | Any tenant's usage. |
| `GET /health`, `GET /ready` | none | Liveness; readiness (database and Redis). |
| `GET /metrics` | metrics token | Prometheus metrics. |

Errors use OpenAI's shape (`{"error": {"message", "type", "code"}}`). Notable codes:
`rate_limit_exceeded`, `budget_exceeded`, `upstream_rate_limited` (with the vendor's `Retry-After`),
`providers_unavailable` (every option is behind an open breaker), `gateway_overloaded`.

## How it works

### Keys, scopes and tenants
A key is 256 random bits behind the `llmr_` prefix; the database keeps only its SHA-256 hash, so a
leaked database does not leak keys. Scopes (`chat:write`, `models:read`, `usage:read`,
`admin:tenants`, `admin:keys`) decide what a key may do, and a key can only grant admin scopes it
holds itself, so a narrow admin key cannot mint a broader one. Revocation takes effect on the next
request. Everything a tenant owns (keys, usage, request logs) is only visible to it.

### Limits
Per-minute request and token limits are token buckets updated inside one Redis Lua script, so
concurrent requests, from any number of replicas, cannot oversubscribe a bucket. A request reserves
the worst case first (prompt plus the maximum completion) and settles against real usage after,
returning what it did not use. A monthly USD budget works the same way, for a key and for its whole
organisation together, all-or-nothing.

### Smart routing
`model: "auto"` runs four steps:
1. **Score the request** (0 to 1) from cheap signals in the request (length of the last message and
   of the context, code, maths, reasoning cues, constraints, multi-part questions, structured output,
   tools, images, expected output length, turns, non-English), combined so that several weak signals
   add up. Greetings and thanks are scored down.
2. **Pick the weakest tier that clears the bar** (`slm_ceiling` 0.35 and `mid_ceiling` 0.65 by
   default; configurable under `routing:`).
3. **Rank the models of that tier or above** by expected cost (70%) and measured latency (30%),
   counting only models with a healthy deployment and room in their context window.
4. **Escalate** to the next-best stronger model if the chosen one fails, subject to the key's budget.

The decision is explained in `x-llmrouter-*` response headers and the logs. A tenant can pin every
`auto` request to one model or cap the tier.

### Resilience
- **Retries and fallbacks:** exponential backoff with jitter, preferring a deployment not yet tried,
  then the model's fallback chain.
- **Circuit breakers:** one per deployment and one per provider endpoint, opening on consecutive
  failures or a failure *rate* over a window, recovering through half-open trial calls. A vendor
  outage opens one breaker for all its models. State is shared across replicas through Redis.
- **Throttling (429):** an upstream 429 does not trip a breaker. The deployment rests for the vendor's
  `Retry-After`, and traffic goes elsewhere meanwhile.
- **Overload:** when the gateway runs out of upstream connections it answers `503 gateway_overloaded`;
  that is never counted against a provider.

### Observability
JSON logs carry a request id. Every request that reaches a model is recorded (who, which model, status,
tokens, cost, latency, routing outcome; never the prompt or the reply) and queryable through the usage
API. `/metrics` exposes request, token, cost, routing, upstream and breaker metrics with bounded labels
(nothing is labelled by tenant or key).

## Evaluation

The evaluation answers the question routing exists for: *how much does it save, and what does it cost
in quality?* It runs a fixed benchmark through the real gateway twice: once always asking the frontier
model, once as `model: "auto"`. The comparison is paired, prompt by prompt.

- **Benchmark:** 90 prompts in six categories (chat, extraction, question answering, maths, code,
  reasoning), 42 easy, 23 medium and 25 hard by the author's judgement. The mix of easy and hard
  prompts drives the headline saving, so always quote it.
- **Quality:** 48 prompts are checked mechanically (exact answer, number, required terms); 42 open-ended
  ones are graded by a judge model against per-criterion rubrics, blind to which system answered. A
  *negative control* gives the judge a non-answer to show how generous it is.
- **Report:** cost saved, quality retained, latency, breakdowns, bootstrap 95% confidence intervals, and
  a check of each figure against a target that says whether the data *establishes* it or only has a
  point estimate above it. Warnings flag failures, unscored answers, policy-shaped routing and a judge
  that also wrote answers.
- **Threshold sweep:** replays 190 routing-threshold settings from stored answers, chooses one on half
  the prompts and measures it on the other half.

```bash
make eval-baseline                 # always-frontier run             (needs a running gateway + keys)
make eval-router                   # model=auto run, same prompts
make eval-score JUDGE=<model>      # grade both; pick a judge that is not a model being compared
make eval-report                   # cost saved, quality retained, latency, with intervals

python -m app.evaluation evaluate --require established     # exits 1 unless both targets are met
make eval-models && make eval-sweep                         # threshold sweep
```

### Results

**There are no measured results in this repository yet.** The evaluation needs real provider keys, real
prices in `config.yaml` and a judge model, and has not been run. Fill in this table from `make eval-report`
and quote the intervals, not just the point estimates:

| | Measured | 95% interval |
|---|---|---|
| Cost saved vs always-frontier | not yet run | |
| Quality retained | not yet run | |
| Prompt mix | 47% easy, 26% medium, 28% hard | |

One early signal to expect: with the default thresholds, the routing *decisions* alone (before any
model is called) send 75 of the 90 benchmark prompts to the small tier, 15 to the mid tier and none to
the frontier tier, including 17 of the 25 prompts labelled hard. Whether the small models answer those
well enough is exactly what the quality measurement is for, and it may mean the default thresholds need
tuning, which the sweep exists to do without tuning on the test prompts.

## Load testing

`make load-test` drives the real gateway against a stand-in model and reports throughput, latency,
behaviour under overload and during an upstream outage, and what each feature costs. On one laptop a
single process sustained roughly 530 to 580 requests per second (CPU-bound) with the stub answering in
50 ms. The full methodology, results, the bugs it found and the things it could not explain are in
[docs/load-test.md](docs/load-test.md).

## Development

```bash
make check          # ruff, mypy (strict) and the test suite
make test-redis     # also run the Redis tests against a real server
make test-postgres  # run the whole suite against a real Postgres (see the Makefile)
make migration m="describe the change"
```

The suite has over two thousand tests, including ones that start real processes (the load-test
harness) and ones that run against real Redis and Postgres. CI runs lint, types and tests on every
push and pull request, and a second job runs the suite against Redis and Postgres service containers.

Project layout:

| Path | |
|---|---|
| `app/api/` | HTTP endpoints and error mapping |
| `app/auth/` | keys, scopes, tenants |
| `app/quota/` | atomic limits and budgets |
| `app/routing/`, `app/router/` | complexity scoring and model choice; deployments, retries, fallbacks |
| `app/resilience/` | circuit breakers, throttling, shared state |
| `app/providers/` | provider adapters |
| `app/usage/`, `app/metrics/` | request log and usage queries; Prometheus |
| `app/evaluation/` | benchmark, runner, judge, report, sweep |
| `app/loadtest/` | stand-in model, load generator, scenarios |
| `migrations/` | Alembic migrations |

## Résumé claims, checked against this repository

The project was rebuilt from scratch after the original was lost. This table says which statements
about it the repository itself supports.

| Claim | Evidence | Status |
|---|---|---|
| OpenAI-compatible gateway in FastAPI | `app/api/chat.py`; tests drive it with the official OpenAI Python SDK | Supported by tests |
| Multi-provider, async adapters | `app/providers/` (OpenAI-compatible and Anthropic), all `httpx` async | Supported by tests |
| Virtual API keys | hashed storage, expiry, revoke, rotate (`app/auth/`) | Supported by tests |
| Scoped RBAC | five scopes, privilege-escalation checks (`app/auth/scopes.py`) | Supported by tests |
| Atomic Redis quotas | Lua token buckets; `tests/test_concurrent_quotas.py` (several replicas, real Redis); the load test sent 300 simultaneous requests at a 60-per-minute key and exactly 60 were allowed | Supported by tests and a measurement |
| Secure multi-tenant isolation | tenant-scoped keys, usage and logs, tested for cross-tenant access; **not independently security-audited** | Tested, not audited |
| Complexity-aware routing between SLMs and frontier models | `app/routing/` | Supported by tests |
| Token-cost and latency scoring | cost and measured-latency ranking in `app/routing/selector.py` | Supported by tests |
| Cross-provider circuit breakers | `app/resilience/`; the load test's outage run (upstream saw 12 of 500 requests once the breaker opened) | Supported by tests and a measurement |
| **70% lower inference cost** | `make eval-report` | **Not reproduced.** Needs the evaluation run above. |
| **93%+ quality retention** | `make eval-report` | **Not reproduced.** Needs the evaluation run above. |

The two numbers came from the original, lost version of the project and **cannot be verified from this
repository until the evaluation is run.** Run it, and put figures on a résumé only if
`python -m app.evaluation evaluate --require established` passes; otherwise quote what the report
actually shows (with its interval and prompt mix), or tune the thresholds with the sweep and re-measure.

## Limitations

- **Routing quality is heuristic.** Complexity comes from surface features of the prompt, which cannot
  see that a short question may need careful reasoning. Escalation happens on failures, not on poor answers.
- **The benchmark is small** (90 prompts), so confidence intervals are wide, and its difficulty labels are
  one author's judgement.
- **Quality depends on a judge model** for 42 of the 90 prompts.
- **One process handles roughly 550 requests per second** here and does not shed load beyond that; run
  more replicas (state is shared through Redis and Postgres) or put a limiter in front.
- **Request-log writes are synchronous**, the largest single cost measured; enforcing limits costs
  about a fifth of capacity (several Redis round trips per request).
- No admin UI, and the code has not been audited by a security professional.

## License

MIT, see [LICENSE](LICENSE).
