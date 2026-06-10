import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.error import ErrorResponse

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
    app.add_exception_handler(Exception, _unhandled_exception_handler)
