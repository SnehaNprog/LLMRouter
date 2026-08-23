import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import request_id_var

REQUEST_ID_HEADER = "X-Request-ID"

logger = logging.getLogger("app.access")


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            fields: dict[str, object] = {
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms,
            }
            auth = getattr(request.state, "auth", None)
            if auth is not None:
                fields["tenant_id"] = auth.tenant_id
                fields["key_id"] = auth.key_id
            logger.info("request completed", extra=fields)
            request_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
