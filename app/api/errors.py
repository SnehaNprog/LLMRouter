import asyncio
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
    GatewayOverloaded,
    ProviderError,
    RateLimited,
    Timeout,
)
from app.quota.budget_limiter import BudgetExceededError
from app.quota.guard import QuotaUnavailableError
from app.quota.headers import rate_headers_of
from app.quota.limiter import RateLimitExceededError, TokenLimitExceededError
from app.resilience.board import DeploymentUnavailable
from app.router.context import ContextLengthExceededError
from app.router.registry import UnknownModelError
from app.routing.resolve import TenantPolicyError
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
    if isinstance(exc, DeploymentUnavailable):
        # Nothing was tried: every deployment that could serve the request is behind an open
        # circuit breaker. The client should come back when the soonest of them may be tried.
        return 503, ErrorBody(
            message="No upstream provider is currently available for this request; try again "
            "shortly",
            type="api_error",
            code="providers_unavailable",
        )
    if isinstance(exc, GatewayOverloaded):
        return 503, ErrorBody(
            message="The gateway is at its limit of concurrent requests to the model providers; "
            "try again shortly",
            type="api_error",
            code="gateway_overloaded",
        )
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


def outcome_of(exc: BaseException) -> tuple[int, str | None]:
    """The HTTP status and error code the client gets for an exception raised while serving.

    Each case mirrors the handler registered for it below; a test sends every one of these
    through the real API and checks that the two agree, so they cannot drift apart unnoticed.
    A client that hangs up mid-request is recorded as 499, the status nginx made conventional.
    """
    if isinstance(exc, ProviderError):
        status, body = describe_provider_error(exc)
        return status, body.code
    if isinstance(exc, AuthError):
        return exc.status_code, exc.code
    if isinstance(exc, UnknownModelError):
        return 404, "model_not_found"
    if isinstance(exc, ContextLengthExceededError):
        return 400, "context_length_exceeded"
    if isinstance(exc, RateLimitExceededError):
        return 429, "rate_limit_exceeded"
    if isinstance(exc, TokenLimitExceededError):
        return 429, "request_too_large" if exc.too_large else "rate_limit_exceeded"
    if isinstance(exc, BudgetExceededError):
        return 429, "budget_exceeded"
    if isinstance(exc, QuotaUnavailableError):
        return 503, "quota_unavailable"
    if isinstance(exc, TenantPolicyError):
        if exc.kind == "no_model_under_cap":
            return 403, "routing_policy_no_model"
        return 503, "tenant_policy_invalid"
    if isinstance(exc, StarletteHTTPException):
        return exc.status_code, None
    if isinstance(exc, asyncio.CancelledError):
        return 499, "client_disconnected"
    return 500, "internal_error"


async def _provider_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ProviderError)
    status, body = describe_provider_error(exc)
    headers = None
    if isinstance(exc, RateLimited) and exc.retry_after is not None:
        headers = {"Retry-After": str(math.ceil(exc.retry_after))}
    elif isinstance(exc, DeploymentUnavailable):
        headers = {"Retry-After": str(max(1, math.ceil(exc.retry_after)))}
    elif isinstance(exc, GatewayOverloaded):
        headers = {"Retry-After": "1"}
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


async def _rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RateLimitExceededError)
    state = rate_headers_of(request)
    state.requests = exc.window
    headers = state.as_headers()
    if exc.retry_after is not None:
        headers["Retry-After"] = str(max(1, math.ceil(exc.retry_after)))
    return error_response(
        429, str(exc), type="rate_limit_error", code="rate_limit_exceeded", headers=headers
    )


async def _token_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, TokenLimitExceededError)
    state = rate_headers_of(request)  # the request allowance was already counted
    state.tokens = exc.window
    headers = state.as_headers()
    if exc.retry_after is not None:
        headers["Retry-After"] = str(max(1, math.ceil(exc.retry_after)))
    code = "request_too_large" if exc.too_large else "rate_limit_exceeded"
    return error_response(429, str(exc), type="rate_limit_error", code=code, headers=headers)


async def _budget_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, BudgetExceededError)
    headers = rate_headers_of(request).as_headers()
    headers["Retry-After"] = str(max(1, math.ceil(exc.retry_after)))
    return error_response(
        429, str(exc), type="insufficient_quota", code="budget_exceeded", headers=headers
    )


async def _quota_unavailable_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, QuotaUnavailableError)
    return error_response(
        503,
        str(exc),
        type="api_error",
        code="quota_unavailable",
        headers={"Retry-After": str(max(1, math.ceil(exc.retry_after)))},
    )


async def _tenant_policy_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, TenantPolicyError)
    logger.error("organisation routing policy cannot be applied: %s", exc)
    if exc.kind == "no_model_under_cap":
        return error_response(
            403,
            "This organisation's routing policy leaves no model available for automatic routing.",
            type="permission_error",
            code="routing_policy_no_model",
        )
    return error_response(
        503,
        "This organisation's routing policy refers to a model that is not available.",
        type="api_error",
        code="tenant_policy_invalid",
    )


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
    app.add_exception_handler(RateLimitExceededError, _rate_limit_handler)
    app.add_exception_handler(TokenLimitExceededError, _token_limit_handler)
    app.add_exception_handler(BudgetExceededError, _budget_handler)
    app.add_exception_handler(QuotaUnavailableError, _quota_unavailable_handler)
    app.add_exception_handler(TenantPolicyError, _tenant_policy_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
