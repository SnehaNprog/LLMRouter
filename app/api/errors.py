import logging
import math

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.auth.errors import AuthError
from app.auth.service import (
    KeyInactiveError,
    KeyNotFoundError,
    TenantInactiveError,
    TenantNotFoundError,
)
from app.auth.tenants import AdminConflictError
from app.providers.errors import (
    AuthFailed,
    BadRequest,
    ProviderError,
    RateLimited,
    Timeout,
)
from app.router.context import ContextLengthExceededError
from app.router.registry import UnknownModelError
from app.schemas.error import ErrorBody, ErrorResponse

logger = logging.getLogger(__name__)

_TYPE_BY_STATUS = {
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
}


def error_type_for(status_code: int) -> str:
    if status_code in _TYPE_BY_STATUS:
        return _TYPE_BY_STATUS[status_code]
    return "api_error" if status_code >= 500 else "invalid_request_error"


def error_response(
    status_code: int,
    message: str,
    type: str | None = None,
    param: str | None = None,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse.of(message, type or error_type_for(status_code), param, code)
    return JSONResponse(status_code=status_code, content=body.model_dump(), headers=headers)


def describe_provider_error(exc: ProviderError) -> tuple[int, ErrorBody]:
    """Gateway status and OpenAI-style body for a failure reported by an upstream provider."""
    if isinstance(exc, RateLimited):
        return 429, ErrorBody(
            message=exc.message, type="rate_limit_error", code="upstream_rate_limited"
        )
    if isinstance(exc, AuthFailed):
        # The caller's key is fine; our credentials for the vendor are not. Do not blame the caller.
        return 502, ErrorBody(
            message="The upstream provider rejected the gateway's credentials",
            type="api_error",
            code="upstream_auth_failed",
        )
    if isinstance(exc, Timeout):
        return 504, ErrorBody(
            message="The upstream provider timed out", type="api_error", code="upstream_timeout"
        )
    if isinstance(exc, BadRequest):
        return 400, ErrorBody(
            message=exc.message, type="invalid_request_error", code="upstream_bad_request"
        )
    return 502, ErrorBody(message=exc.message, type="api_error", code="upstream_unavailable")


async def _provider_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ProviderError)
    status, body = describe_provider_error(exc)
    headers = None
    if isinstance(exc, RateLimited) and exc.retry_after is not None:
        headers = {"Retry-After": str(math.ceil(exc.retry_after))}
    logger.warning(
        "upstream provider error",
        extra={"provider": exc.provider, "upstream_status": exc.upstream_status},
    )
    return JSONResponse(
        status_code=status, content=ErrorResponse(error=body).model_dump(), headers=headers
    )


async def _unknown_model_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, UnknownModelError)
    return error_response(
        404, str(exc), type="invalid_request_error", param="model", code="model_not_found"
    )


async def _auth_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AuthError)
    return error_response(
        exc.status_code,
        exc.message,
        type=exc.type,
        param=exc.param,
        code=exc.code,
        headers=dict(exc.headers),
    )


async def _not_found_handler(_: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, TenantNotFoundError):
        return error_response(404, str(exc), param="tenant_id", code="tenant_not_found")
    assert isinstance(exc, KeyNotFoundError)
    return error_response(404, str(exc), param="key_id", code="key_not_found")


async def _conflict_handler(_: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, AdminConflictError):
        return error_response(409, exc.message, type="invalid_request_error", code=exc.code)
    if isinstance(exc, TenantInactiveError):
        return error_response(409, str(exc), type="invalid_request_error", code="tenant_inactive")
    assert isinstance(exc, KeyInactiveError)
    return error_response(409, str(exc), type="invalid_request_error", code="key_inactive")


async def _context_length_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ContextLengthExceededError)
    return error_response(
        400,
        str(exc),
        type="invalid_request_error",
        param="messages",
        code="context_length_exceeded",
    )


async def _http_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    return error_response(exc.status_code, str(exc.detail), headers=dict(exc.headers or {}))


async def _validation_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    first = exc.errors()[0]
    location = [str(part) for part in first["loc"] if part != "body"]
    param = ".".join(location) or None
    message = f"{param}: {first['msg']}" if param else str(first["msg"])
    return error_response(400, message, type="invalid_request_error", param=param)


async def _unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error", exc_info=exc)
    return error_response(500, "Internal server error", type="api_error")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(ProviderError, _provider_error_handler)
    app.add_exception_handler(AuthError, _auth_error_handler)
    for not_found in (TenantNotFoundError, KeyNotFoundError):
        app.add_exception_handler(not_found, _not_found_handler)
    for conflict in (AdminConflictError, TenantInactiveError, KeyInactiveError):
        app.add_exception_handler(conflict, _conflict_handler)
    app.add_exception_handler(UnknownModelError, _unknown_model_handler)
    app.add_exception_handler(ContextLengthExceededError, _context_length_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
