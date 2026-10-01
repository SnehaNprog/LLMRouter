import re
from collections.abc import Callable, Iterable, Iterator
from urllib.parse import urlsplit

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from app.resilience.breaker import CircuitState
from app.router.deployment import Deployment
from app.router.router import Router
from app.routing.policy import AUTO_MODEL
from app.usage.record import LogEntry

NAMESPACE = "llmrouter"
UNKNOWN = "unknown"
UNMATCHED = "unmatched"

# Seconds. A completion can run long, so the buckets reach past a minute.
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)

# What an attempt on a deployment can end in; anything else is "other", so a surprising exception
# class can never mint a new label value.
ATTEMPT_OUTCOMES = frozenset(
    {
        "success",
        "refused",
        "ProviderDown",
        "Timeout",
        "RateLimited",
        "AuthFailed",
        "BadRequest",
        "GatewayOverloaded",
        "ContextLengthExceededError",
        "CancelledError",
    }
)
_STATE_VALUE = {CircuitState.CLOSED.value: 0, CircuitState.HALF_OPEN.value: 1}


def snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def safe_name(name: str) -> str:
    """A breaker's name with anything that could be a credential stripped from a URL in it."""

    def clean(match: re.Match[str]) -> str:
        parts = urlsplit(match.group(0))
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        return f"{parts.scheme}://{host}{port}{parts.path}"

    return re.sub(r"https?://\S+", clean, name)


class Metrics:
    """The gateway's Prometheus metrics, in a registry of their own.

    Labels are chosen so their values form a small, fixed set: model names the caller makes up
    are folded into "unknown", routes are the route templates rather than the paths, and nothing
    is labelled by tenant or key (use the usage API for per-customer figures). A label with
    unbounded values would make a metrics server fall over, and a client could cause it.
    """

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.known_models: set[str] = {AUTO_MODEL}
        self._live: LiveState | None = None

        def counter(name: str, doc: str, labels: Iterable[str]) -> Counter:
            return Counter(name, doc, list(labels), namespace=NAMESPACE, registry=self.registry)

        def histogram(name: str, doc: str, labels: Iterable[str]) -> Histogram:
            return Histogram(
                name,
                doc,
                list(labels),
                namespace=NAMESPACE,
                registry=self.registry,
                buckets=LATENCY_BUCKETS,
            )

        self.http_requests = counter(
            "http_requests", "HTTP requests handled.", ["method", "route", "status"]
        )
        self.http_duration = histogram(
            "http_request_duration_seconds", "HTTP request duration.", ["method", "route"]
        )
        self.http_in_progress = Gauge(
            "http_requests_in_progress",
            "HTTP requests being handled right now.",
            namespace=NAMESPACE,
            registry=self.registry,
        )
        self.chat_requests = counter(
            "chat_requests",
            "Chat completion requests that reached the model stage, by how they ended.",
            ["requested_model", "served_model", "tier", "outcome"],
        )
        self.chat_duration = histogram(
            "chat_request_duration_seconds",
            "Chat completion duration, end to end (a stream counts until it ends).",
            ["served_model", "stream"],
        )
        self.tokens = counter("tokens", "Tokens processed.", ["model", "kind"])
        self.cost = counter("cost_usd", "Money spent on upstream models, in USD.", ["model"])
        self.auto_requests = counter(
            "auto_requests",
            "Requests that asked for model=auto, by the tier that answered.",
            ["tier", "escalated"],
        )
        self.attempts = counter(
            "upstream_attempts",
            "Attempts to call an upstream deployment, including ones refused before a call.",
            ["model", "outcome"],
        )
        self.attempt_duration = histogram(
            "upstream_attempt_duration_seconds",
            "Duration of successful upstream calls.",
            ["model"],
        )
        self.transitions = counter(
            "circuit_breaker_transitions",
            "Circuit breaker state changes.",
            ["kind", "to_state"],
        )
        self.throttles = counter(
            "upstream_throttles", "Times an upstream told us to slow down (HTTP 429).", ["model"]
        )

    # ---- what the gateway can name ----

    def set_models(self, aliases: Iterable[str]) -> None:
        self.known_models = {AUTO_MODEL, *aliases}

    def model(self, name: str | None) -> str:
        """A model name that is safe to use as a label: a configured one, else "unknown"."""
        if name is None:
            return "none"
        return name if name in self.known_models else UNKNOWN

    # ---- observers ----

    def observe_chat(self, entry: LogEntry) -> None:
        deployment = entry.deployment
        served = self.model(deployment.alias) if deployment is not None else "none"
        tier = (
            deployment.config.tier.value
            if deployment is not None and deployment.config.tier is not None
            else "none"
        )
        outcome = entry.error_code or "success"
        self.chat_requests.labels(self.model(entry.requested_model), served, tier, outcome).inc()
        self.chat_duration.labels(served, str(entry.stream).lower()).observe(entry.elapsed())
        if deployment is not None:
            self.tokens.labels(served, "prompt").inc(entry.prompt_tokens)
            self.tokens.labels(served, "completion").inc(entry.completion_tokens)
            self.cost.labels(served).inc(entry.price_usd())
            if entry.choice is not None:
                escalated = deployment.alias != entry.choice.alias
                self.auto_requests.labels(tier, str(escalated).lower()).inc()

    def observe_attempt(self, deployment: Deployment, outcome: str, seconds: float) -> None:
        model = self.model(deployment.alias)
        label = outcome if outcome in ATTEMPT_OUTCOMES else "other"
        self.attempts.labels(model, snake(label)).inc()
        if outcome == "success":
            self.attempt_duration.labels(model).observe(seconds)

    def observe_transition(self, name: str, old: CircuitState, new: CircuitState) -> None:
        kind = "provider" if name.startswith("provider:") else "deployment"
        self.transitions.labels(kind, new.value).inc()

    def observe_throttle(self, deployment: Deployment, seconds: float) -> None:
        self.throttles.labels(self.model(deployment.alias)).inc()

    def observe_http(self, method: str, route: str | None, status: int, seconds: float) -> None:
        label = route if route else UNMATCHED
        self.http_requests.labels(method, label, str(status)).inc()
        self.http_duration.labels(method, label).observe(seconds)

    # ---- live state, read when Prometheus scrapes ----

    def bind(self, router: Router) -> None:
        """Expose what only the running gateway knows: breaker states, rests and latencies."""
        if self._live is not None:
            self.registry.unregister(self._live)
        self._live = LiveState(router, self.model)
        self.registry.register(self._live)


class LiveState(Collector):
    """Gauges worked out at scrape time from the router, so they can never go stale."""

    def __init__(self, router: Router, model: Callable[[str | None], str]) -> None:
        self._router = router
        self._model = model

    def collect(self) -> Iterator[Metric]:
        states = GaugeMetricFamily(
            f"{NAMESPACE}_circuit_breaker_state",
            "Circuit breaker state: 0 closed, 1 half-open, 2 open.",
            labels=["kind", "breaker"],
        )
        for name, state in sorted(self._router.breakers.states().items()):
            kind, _, rest = name.partition(":")
            states.add_metric([kind, safe_name(rest)], _STATE_VALUE.get(state, 2))
        yield states

        deployments = {
            d.id: d
            for alias in self._router.registry.aliases()
            for d in self._router.registry.get(alias)
        }
        resting = GaugeMetricFamily(
            f"{NAMESPACE}_deployment_throttled_seconds",
            "Seconds a deployment must still be left alone after an upstream 429.",
            labels=["model", "deployment"],
        )
        for deployment_id, seconds in sorted(self._router.breakers.throttled().items()):
            found = deployments.get(deployment_id)
            if found is not None:
                resting.add_metric([self._model(found.alias), deployment_id], seconds)
        yield resting

        latency = GaugeMetricFamily(
            f"{NAMESPACE}_deployment_latency_seconds",
            "Smoothed latency of a deployment, as the router sees it.",
            labels=["model", "deployment"],
        )
        for deployment_id, deployment in sorted(deployments.items()):
            ewma = self._router.latency.ewma(deployment_id)
            if ewma is not None:
                latency.add_metric([self._model(deployment.alias), deployment_id], ewma)
        yield latency
