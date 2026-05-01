# LLMRouter Roadmap

One row = one commit. Each commit should be small enough to review in about 10 minutes.
Tick the box when the "Done when" check passes. Write tests in the same commit as the feature, never "later".

**Stack (change if you disagree, but decide in commit 1):** Python 3.12, FastAPI, Pydantic v2, httpx, SQLAlchemy + Alembic (SQLite for dev, Postgres for prod), pytest, ruff, mypy, Docker, GitHub Actions, UI in React + Vite + TypeScript.

---

## Phase 0: Repo hygiene and tooling

| ☐ | # | Commit message | What you build | Done when |
|---|---|---|---|---|
| ☐ | 1 | `chore: pyproject and dependency management` | `pyproject.toml` (or uv/poetry), pinned runtime and dev deps, Python version pinned | Fresh clone installs with one command |
| ☐ | 2 | `chore: project layout` | `app/` (api, core, providers, router, db, schemas), `tests/`, `__init__` files, move `main.py` into `app/main.py` | `uvicorn app.main:app` still serves `/health` |
| ☐ | 3 | `chore: ruff lint and format config` | Ruff rules and line length in `pyproject.toml` | `ruff check .` and `ruff format --check .` pass |
| ☐ | 4 | `chore: mypy strict config` | mypy config, fix the existing type errors | `mypy app` passes |
| ☐ | 5 | `test: pytest setup and health smoke test` | pytest config, `conftest.py`, FastAPI `TestClient` fixture | `pytest` is green with 1 test |
| ☐ | 6 | `chore: pre-commit hooks` | `.pre-commit-config.yaml` (ruff, mypy, trailing whitespace, secrets scan) | A bad commit is blocked locally |
| ☐ | 7 | `feat: settings from environment` | `pydantic-settings` class, `.env.example`, `.env` ignored | App reads config only through the settings object |
| ☐ | 8 | `feat: structured logging` | JSON logger, log level from settings | Logs are one JSON object per line |
| ☐ | 9 | `feat: request ID middleware` | Generate or accept `X-Request-ID`, add to logs and response headers | Every log line for a request shares an ID |
| ☐ | 10 | `docs: README skeleton` | Problem statement, goals, non-goals, status badge placeholder | A stranger understands what this is in 1 minute |
| ☐ | 11 | `ci: lint and test workflow` | GitHub Actions: checkout, setup Python, install, ruff, mypy, pytest | Green check on a push |
| ☐ | 12 | `chore: Makefile or task runner` | `make dev / test / lint / fmt` | One short command per task |

## Phase 1: API contract (OpenAI-compatible)

| ☐ | # | Commit message | What you build | Done when |
|---|---|---|---|---|
| ☐ | 13 | `feat(schemas): message model` | `Message` with role (enum or Literal) and content | Invalid role is rejected |
| ☐ | 14 | `feat(schemas): chat completion request` | `model`, `messages` (list, min 1), `temperature`, `max_tokens`, `top_p`, `stop`, `stream`, `n`, `user` | Real sample payload parses, bad payload gives 422 |
| ☐ | 15 | `feat(schemas): tool/function fields` | `tools`, `tool_choice`, tool messages, `tool_calls` on the assistant message | A tool-calling request parses |
| ☐ | 16 | `feat(schemas): multimodal content` | `content` as string or list of parts (text, image_url) | Both content shapes parse |
| ☐ | 17 | `feat(schemas): chat completion response` | `id`, `object`, `created`, `model`, `choices[]`, `finish_reason` | A recorded real response round-trips |
| ☐ | 18 | `feat(schemas): usage` | `prompt_tokens`, `completion_tokens`, `total_tokens` | Totals validate |
| ☐ | 19 | `feat(schemas): OpenAI-style error body` | `{error:{message,type,param,code}}` model | Matches the real error format |
| ☐ | 20 | `feat(api): global exception handlers` | Validation and unhandled errors return the OpenAI error shape | A bad request returns an error the OpenAI SDK understands |
| ☐ | 21 | `feat(api): chat completions stub endpoint` | `POST /v1/chat/completions` returns a hardcoded valid response | curl works |
| ☐ | 22 | `feat(api): models list endpoint` | `GET /v1/models` (static for now) | Response matches the OpenAI list shape |
| ☐ | 23 | `test: official OpenAI SDK against the stub` | Integration test using `openai` with `base_url` pointed at the app | SDK call succeeds |
| ☐ | 24 | `feat(api): liveness and readiness probes` | `/health` (alive), `/ready` (config loaded, DB reachable later) | Both documented |

## Phase 2: Providers

| ☐ | # | Commit message | What you build | Done when |
|---|---|---|---|---|
| ☐ | 25 | `feat(providers): adapter interface` | Abstract `Provider` with `chat()` and `stream()` signatures | Interface is documented, no implementation |
| ☐ | 26 | `feat(providers): domain error types` | `RateLimited`, `AuthFailed`, `Timeout`, `BadRequest`, `ProviderDown`, with a `retryable` flag | Each error has a clear meaning |
| ☐ | 27 | `feat(providers): fake provider for tests` | Deterministic provider with configurable latency, failure rate and token counts | Used by later router tests |
| ☐ | 28 | `feat(core): shared httpx client lifecycle` | Create the client in the FastAPI lifespan, close on shutdown | No connection leaks |
| ☐ | 29 | `feat(providers): OpenAI-compatible adapter (non-stream)` | Request mapping, call, response mapping | Real call returns an answer (use a free-tier compatible API) |
| ☐ | 30 | `feat(providers): timeouts and connect limits` | Separate connect and read timeouts from settings | Slow server gives `Timeout` |
| ☐ | 31 | `feat(providers): map OpenAI-adapter HTTP errors` | 401, 429, 4xx, 5xx, network errors become domain errors | A table-driven test covers each status |
| ☐ | 32 | `feat(providers): Anthropic request translation` | system prompt hoisting, `max_tokens` required, role rules | Unit-tested mapping |
| ☐ | 33 | `feat(providers): Anthropic response translation` | content blocks to `choices`, `stop_reason` to `finish_reason`, usage mapping | Unit-tested mapping |
| ☐ | 34 | `feat(providers): Anthropic error mapping` | Same domain errors as step 31 | Same test table passes |
| ☐ | 35 | `feat(providers): Gemini adapter (optional third)` | Translation both ways | Optional, skip if it drags |
| ☐ | 36 | `test: provider contract test suite` | One shared suite that every adapter must pass (with mocked HTTP) | Adding an adapter means passing this suite |
| ☐ | 37 | `feat(providers): provider factory` | Name to adapter class mapping | Unknown provider gives a clear error |

## Phase 3: Router core

| ☐ | # | Commit message | What you build | Done when |
|---|---|---|---|---|
| ☐ | 38 | `feat(config): model list schema` | Pydantic models for `model_name`, `provider`, `litellm_params`-style fields (model, api_base, key ref, rpm, tpm, weight) | Schema documented |
| ☐ | 39 | `feat(config): YAML loader and validation` | Load `config.yaml`, fail fast with readable errors | Bad config prevents startup with a clear message |
| ☐ | 40 | `feat(config): secrets by env reference` | `os.environ/KEY_NAME` style references, never plaintext in YAML | Keys never appear in logs |
| ☐ | 41 | `feat(router): deployment model and stable ID` | `Deployment` object with deterministic ID (hash of its params) | Same config gives the same ID |
| ☐ | 42 | `feat(router): registry grouped by alias` | Alias to list of deployments | Lookup test passes, unknown alias gives 404 |
| ☐ | 43 | `feat(router): single-deployment passthrough` | `Router.complete()` calls the one deployment | Endpoint now serves real or fake providers |
| ☐ | 44 | `feat(api): wire endpoint to router` | Replace the stub with the router call | Stub removed, tests updated |
| ☐ | 45 | `feat(router): simple shuffle` | Random pick among healthy deployments | Distribution test |
| ☐ | 46 | `feat(router): weighted selection` | Honor `weight` | Statistical test within tolerance |
| ☐ | 47 | `feat(router): retry classification` | Decide retryable and non-retryable from the domain error | Table-driven test |
| ☐ | 48 | `feat(router): exponential backoff with jitter` | Pure function plus an injectable sleep | Unit tests with a fake clock |
| ☐ | 49 | `feat(router): retries in the request path` | Retry on a different deployment where possible | Fake provider that fails twice then succeeds gives a 200 |
| ☐ | 50 | `feat(router): fallback config` | `fallbacks: {model_a: [model_b]}` in config | Parsed and validated |
| ☐ | 51 | `feat(router): fallback execution` | Move to the next alias after retries are exhausted | Test with a dead primary |
| ☐ | 52 | `feat(router): context-window fallback` | Estimate prompt tokens, skip models that can't fit | Long prompt goes to the bigger model |
| ☐ | 53 | `feat(router): per-deployment health state` | Rolling failure counter | Counter resets over time |
| ☐ | 54 | `feat(router): cooldown on repeated failure` | Remove a deployment from the pool for N seconds | Deployment rejoins automatically |
| ☐ | 55 | `feat(router): request deadline` | Total time budget across retries and fallbacks | Never exceeds the budget |
| ☐ | 56 | `feat(router): per-deployment concurrency cap` | In-flight counter, reject or skip when full | Load test shows the cap holds |
| ☐ | 57 | `test: router chaos tests` | Randomised failures across many requests | Overall success rate stays high with 1 of 3 deployments down |

## Phase 4: Persistence, cost and observability

| ☐ | # | Commit message | What you build | Done when |
|---|---|---|---|---|
| ☐ | 58 | `feat(db): SQLAlchemy base and session` | Async engine, session dependency, SQLite default | App starts with DB |
| ☐ | 59 | `feat(db): Alembic migrations setup` | Alembic config, empty initial migration | `alembic upgrade head` works |
| ☐ | 60 | `feat(db): request_logs table` | Columns: id, ts, alias, deployment, status, latency, tokens in/out, cost, error | Migration applied |
| ☐ | 61 | `feat(logging): write request log` | After each request, write a row | Row exists after a request |
| ☐ | 62 | `perf(logging): async non-blocking writes` | Queue plus background writer so logging never slows requests | Latency unchanged under load |
| ☐ | 63 | `feat(cost): pricing table` | `pricing.yaml` with per-1M-token input and output prices, with a source and date | Loaded and validated |
| ☐ | 64 | `feat(cost): cost calculator` | Pure function from usage and model to dollars | Unit-tested with hand-computed cases |
| ☐ | 65 | `feat(cost): unknown model fallback` | Explicit default plus warning log | No silent zero cost |
| ☐ | 66 | `feat(cost): store cost per request` | Cost column populated | Matches a manual calculation |
| ☐ | 67 | `feat(metrics): Prometheus endpoint` | `/metrics` with request count, latency histogram, error count, cost counter | Scrapeable |
| ☐ | 68 | `feat(obs): response headers` | `x-router-deployment`, `x-router-latency-ms`, `x-router-cost-usd` | Visible in curl -i |
| ☐ | 69 | `feat(stats): aggregation queries` | Cost, latency percentiles and error rate by model and by time bucket | Tested on seeded data |
| ☐ | 70 | `feat(api): admin stats endpoints` | `/admin/stats/*` JSON endpoints for the UI | Documented in OpenAPI |
| ☐ | 71 | `chore(db): retention job` | Delete or aggregate rows older than N days | Configurable |

## Phase 5: Smart routing

| ☐ | # | Commit message | What you build | Done when |
|---|---|---|---|---|
| ☐ | 72 | `refactor(router): strategy interface` | `Strategy.pick(candidates, ctx)`, with shuffle moved behind it | All existing tests unchanged |
| ☐ | 73 | `feat(router): latency tracker` | Rolling window or EWMA per deployment | Unit-tested |
| ☐ | 74 | `feat(router): latency-based strategy` | Pick the lowest recent latency | Slow deployment loses traffic |
| ☐ | 75 | `feat(router): latency buffer` | Anything within X% of the fastest qualifies, then shuffle | No single-endpoint stampede |
| ☐ | 76 | `feat(router): cost-based strategy` | Pick the cheapest healthy deployment | Cheapest wins in a test |
| ☐ | 77 | `feat(router): per-alias strategy config` | Choose the strategy per alias in YAML | Two aliases use two strategies |
| ☐ | 78 | `feat(router): prompt feature extraction` | Token estimate, message count, code detection, language, tool use | Pure function with tests |
| ☐ | 79 | `feat(router): rule-based task classifier` | Classes such as simple, complex, code, long-context | Labelled examples file passes |
| ☐ | 80 | `feat(router): model tiers` | Config groups models into cheap, balanced and strong tiers | Validated config |
| ☐ | 81 | `feat(router): task-aware strategy` | Special alias `auto` maps class to tier, then applies cost or latency within the tier | Simple prompts go cheap, hard ones go strong |
| ☐ | 82 | `feat(router): routing explanation` | Return `x-router-reason` and log the decision | You can see why a model was chosen |
| ☐ | 83 | `feat(router): user override` | Explicit `model` always bypasses auto | Documented precedence |
| ☐ | 84 | `feat(eval): replay harness` | Replay logged or sample prompts against strategies offline, compare cost and latency | A report shows savings versus always-strong |
| ☐ | 85 | `feat(router): escalation on weak answer (optional)` | Retry on a stronger tier on empty or refused output | Behind a flag |

## Phase 6: Streaming

| ☐ | # | Commit message | What you build | Done when |
|---|---|---|---|---|
| ☐ | 86 | `feat(schemas): stream chunk models` | `chat.completion.chunk` with deltas | Matches the real chunk format |
| ☐ | 87 | `feat(providers): stream in the OpenAI adapter` | Async iterator of chunks | Chunks arrive incrementally |
| ☐ | 88 | `feat(providers): stream in the Anthropic adapter` | Translate SSE events to OpenAI chunks | OpenAI SDK reads it unchanged |
| ☐ | 89 | `feat(api): SSE response` | `StreamingResponse`, `data: ...` lines, final `[DONE]` | Streaming works through the SDK |
| ☐ | 90 | `feat(router): failover before first token` | Retry or fall back only until the first chunk is sent | Test with a failure before the first chunk |
| ☐ | 91 | `feat(router): mid-stream failure handling` | Defined behaviour once chunks are already sent | Documented and tested |
| ☐ | 92 | `feat(obs): time-to-first-token metric` | TTFT in logs and Prometheus | Visible in the dashboard data |
| ☐ | 93 | `feat(cost): usage capture for streams` | Final usage chunk or token estimate | Streamed requests have a cost |
| ☐ | 94 | `feat(api): client disconnect handling` | Cancel the upstream call when the client leaves | No orphaned upstream requests |

## Phase 7: Gateway: keys, limits, budgets

| ☐ | # | Commit message | What you build | Done when |
|---|---|---|---|---|
| ☐ | 95 | `feat(db): api_keys table` | Hashed key, prefix, name, created, revoked, allowed models | Plaintext key never stored |
| ☐ | 96 | `feat(auth): key generation` | Show the plaintext once at creation | Prefix stored for lookup |
| ☐ | 97 | `feat(auth): bearer auth middleware` | Validate `Authorization: Bearer`, attach the key to the request | Missing or bad key gives 401 in OpenAI format |
| ☐ | 98 | `feat(auth): admin auth` | Separate master key for `/admin/*` | Client keys can't reach admin |
| ☐ | 99 | `feat(admin): key CRUD endpoints` | Create, list, revoke, update | Tested end to end |
| ☐ | 100 | `feat(auth): per-key model allowlist` | Key restricted to certain aliases | Forbidden model gives 403 |
| ☐ | 101 | `feat(limits): token-bucket rate limiter` | Per-key requests per minute, in memory | Over the limit gives 429 with `Retry-After` |
| ☐ | 102 | `feat(limits): TPM limit` | Tokens per minute per key | Tested |
| ☐ | 103 | `feat(budget): spend per key` | Aggregate cost per key per period | Matches the request logs |
| ☐ | 104 | `feat(budget): enforce budget` | Block when the budget is exceeded | Clear error body |
| ☐ | 105 | `feat(budget): budget reset schedule` | Daily or monthly reset | Time-travel test |
| ☐ | 106 | `feat(auth): key attribution in logs` | `key_id` column in `request_logs` | Per-key stats work |

## Phase 8: Extras

| ☐ | # | Commit message | What you build | Done when |
|---|---|---|---|---|
| ☐ | 107 | `feat(cache): exact-match response cache` | Hash of the normalised request, with TTL, in memory | The second identical call is instant and costs $0 |
| ☐ | 108 | `feat(cache): Redis backend` | Pluggable cache and shared limiter state | Works across 2 instances |
| ☐ | 109 | `feat(router): Redis-shared cooldown and usage` | Shared health state | Two instances agree on cooldowns |
| ☐ | 110 | `feat(config): hot reload` | Reload `config.yaml` without restart | Atomic swap, bad config is rejected |
| ☐ | 111 | `feat(api): embeddings endpoint (optional)` | `/v1/embeddings` through the same router | Optional |

## Phase 9: UI (React + Vite + TypeScript)

| ☐ | # | Commit message | What you build | Done when |
|---|---|---|---|---|
| ☐ | 112 | `feat(ui): scaffold` | `ui/` Vite React TS app, eslint, prettier | `npm run dev` shows a page |
| ☐ | 113 | `feat(ui): API client and types` | Typed fetch wrapper, env-based API URL | Types match the backend schemas |
| ☐ | 114 | `feat(ui): admin login` | Enter the master key, store it in memory or session | Wrong key shows an error |
| ☐ | 115 | `feat(ui): app shell and routing` | Sidebar, routes, layout | Navigation works |
| ☐ | 116 | `feat(ui): overview page` | Cards for total cost, requests, p50/p95 latency, error rate | Reads real data |
| ☐ | 117 | `feat(ui): time-series charts` | Cost and latency over time, with range picker | Smooth with 10k rows |
| ☐ | 118 | `feat(ui): requests table` | Paginated, sortable, filters (model, status, key) | Filters hit the API |
| ☐ | 119 | `feat(ui): request detail drawer` | Routing reason, retries, fallback chain, cost breakdown | Explains one request fully |
| ☐ | 120 | `feat(ui): models and health page` | Deployments with status, cooldown timers, latency, success rate | Updates on poll |
| ☐ | 121 | `feat(ui): API keys page` | Create, copy once, revoke, set budget and limits | Full key lifecycle |
| ☐ | 122 | `feat(ui): routing config viewer` | Read-only view of aliases, strategies, fallbacks | Matches the YAML |
| ☐ | 123 | `feat(ui): playground` | Send a chat to any alias, show the answer and routing metadata, streaming supported | Streams token by token |
| ☐ | 124 | `feat(ui): cost savings view` | Actual cost versus a "always strongest model" baseline | Matches the replay harness |
| ☐ | 125 | `feat(ui): empty, loading and error states` | Skeletons and error banners everywhere | No blank screens |
| ☐ | 126 | `feat(ui): dark mode and responsive` | Theme toggle, works at phone width | No horizontal scroll |
| ☐ | 127 | `test(ui): component and API tests` | Vitest and Testing Library for key pages | CI runs them |
| ☐ | 128 | `feat(api): serve UI static build (optional)` | Mount the built UI from the backend | Single container option |

## Phase 10: Docker

| ☐ | # | Commit message | What you build | Done when |
|---|---|---|---|---|
| ☐ | 129 | `build: backend Dockerfile` | Multi-stage, non-root user, slim base | Image under a sensible size |
| ☐ | 130 | `build: .dockerignore` | Exclude venv, git, tests, caches | Faster builds |
| ☐ | 131 | `build: container healthcheck` | `HEALTHCHECK` against `/health` | Docker shows healthy |
| ☐ | 132 | `build: UI Dockerfile` | Build stage, then nginx serves the static files | UI served from a container |
| ☐ | 133 | `build: nginx config` | SPA fallback, `/api` proxy, gzip | Page refresh works on any route |
| ☐ | 134 | `build: docker-compose (api + ui + postgres)` | Volumes, env file, depends_on healthy | `docker compose up` runs everything |
| ☐ | 135 | `feat(db): Postgres support` | Driver, URL via settings, migration run on start | Same tests pass on Postgres |
| ☐ | 136 | `build: add Redis to compose` | Redis service, optional profile | Cache and limiter use it |
| ☐ | 137 | `build: Prometheus and Grafana profile` | Compose profile with prebuilt dashboards | Dashboards show live metrics |
| ☐ | 138 | `docs: running with Docker` | Commands, env vars, ports, volumes | Copy-paste works |

## Phase 11: CI/CD

| ☐ | # | Commit message | What you build | Done when |
|---|---|---|---|---|
| ☐ | 139 | `ci: Python version matrix` | Test on 3.11 and 3.12 | Both green |
| ☐ | 140 | `ci: dependency caching` | Cache pip or uv | Faster runs |
| ☐ | 141 | `ci: coverage report and threshold` | pytest-cov, fail under N% | Coverage shown in the PR |
| ☐ | 142 | `ci: UI lint, test and build job` | Node setup, npm ci, lint, test, build | Runs in parallel with backend |
| ☐ | 143 | `ci: Postgres service integration tests` | Service container in the workflow | DB tests run in CI |
| ☐ | 144 | `ci: Docker build check` | Build both images on every PR | Broken Dockerfile fails the PR |
| ☐ | 145 | `ci: security scanning` | pip-audit, npm audit, CodeQL, secret scanning | Findings reported |
| ☐ | 146 | `ci: Dependabot config` | Weekly updates for pip, npm, Actions, Docker | PRs appear |
| ☐ | 147 | `ci: publish images to GHCR on tag` | Build, tag by version and `latest`, push | Image pullable |
| ☐ | 148 | `ci: release workflow` | On tag, create a GitHub Release with notes | Changelog attached |
| ☐ | 149 | `chore: PR template and issue templates` | Checklist: tests, docs, changelog | Used on the next PR |
| ☐ | 150 | `docs: branch protection and workflow` | Document required checks, trunk-based flow | Settings applied on GitHub |

## Phase 12: Quality, docs and release

| ☐ | # | Commit message | What you build | Done when |
|---|---|---|---|---|
| ☐ | 151 | `test: end-to-end suite with fake providers` | Full path: auth, route, retry, log, cost | Runs in CI |
| ☐ | 152 | `perf: load test script` | Locust or k6 against fake providers | Baseline recorded |
| ☐ | 153 | `perf: router overhead benchmark` | Measure added latency versus a direct call | Number goes in the README |
| ☐ | 154 | `perf: profile and fix the hottest path` | Based on real profiling, not guesses | Before and after numbers |
| ☐ | 155 | `docs: architecture diagram` | Request flow, components, state | Matches the code |
| ☐ | 156 | `docs: config reference` | Every YAML field, with examples | Verified against the schema |
| ☐ | 157 | `docs: routing strategies guide` | How each strategy picks, with tradeoffs | A reader can choose one |
| ☐ | 158 | `docs: ADRs` | Short decision records (why SQLite and Postgres, why these strategies) | 4 or 5 files |
| ☐ | 159 | `docs: full README` | Quickstart, features, screenshots, benchmarks, comparison to LiteLLM, roadmap | Quickstart works on a clean machine |
| ☐ | 160 | `docs: CONTRIBUTING and SECURITY` | Setup, style, how to report vulnerabilities | Linked from the README |
| ☐ | 161 | `chore: CHANGELOG` | Keep a Changelog format | Matches the tags |
| ☐ | 162 | `chore: release v0.1.0` | Version bump, tag, release | Image and release published |

---

## Rules of thumb

1. **Order is deliberate.** Cost and logging (Phase 4) come before smart routing (Phase 5), because you can't route on data you don't have. The admin stats API (step 70) comes before the UI that uses it.
2. **Keep CI green from commit 11.** Never merge a red build.
3. **If a commit takes more than about 2 hours, split it.**
4. **Phases 8 and 9 can run in parallel** with later router work once Phase 4 is done.
5. **MVP cut-off:** steps 1 to 57 plus 61 to 66 is a working, honest LLM router. Everything after that is polish and differentiation.
