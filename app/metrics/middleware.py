import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.metrics.registry import Metrics

METRICS_PATH = "/metrics"


class MetricsMiddleware:
    """Counts and times every HTTP request.

    A plain ASGI middleware rather than a ``BaseHTTPMiddleware``, so that a streamed completion
    is timed until its last byte has been sent, not just until its first. Requests are labelled
    by the *route template* (``/admin/tenants/{tenant_id}``), never the path, and a path that
    matches no route is "unmatched": otherwise anyone could mint endless label values by asking
    for made-up URLs.
    """

    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        status = 500  # what a request that dies before answering will have been
        started = time.perf_counter()

        async def observing_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        self.metrics.http_in_progress.inc()
        try:
            await self.app(scope, receive, observing_send)
        finally:
            self.metrics.http_in_progress.dec()
            route = getattr(scope.get("route"), "path", None)
            if route != METRICS_PATH:  # scraping is not traffic
                self.metrics.observe_http(
                    scope["method"], route, status, time.perf_counter() - started
                )
