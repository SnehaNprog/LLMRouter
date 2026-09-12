import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.api.deps import get_limiter, get_router
from app.api.errors import describe_provider_error
from app.auth.dependencies import AuthContext, require_scope
from app.auth.errors import ModelNotAllowedError
from app.auth.model_access import model_allowed
from app.auth.scopes import CHAT_WRITE
from app.providers.errors import ProviderError
from app.quota.limiter import RateLimiter, Reservation
from app.router.router import Router
from app.router.tokens import estimate_text_tokens
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import ChatCompletionChunk, ChatCompletionResponse
from app.schemas.error import ErrorResponse

logger = logging.getLogger(__name__)

chat_scope = require_scope(CHAT_WRITE)


async def rate_limited_chat(
    auth: Annotated[AuthContext, Depends(chat_scope)],
    limiter: Annotated[RateLimiter, Depends(get_limiter)],
) -> AuthContext:
    """Authenticate, check the scope, then count the request against the key's rate limit."""
    await limiter.check_request(auth)
    return auth


router = APIRouter(prefix="/v1")


def _sse(payload: str) -> str:
    return f"data: {payload}\n\n"


async def _event_stream(
    first: ChatCompletionChunk | None,
    rest: AsyncIterator[ChatCompletionChunk],
    reservation: Reservation,
) -> AsyncGenerator[str]:
    """Relay chunks as server-sent events, then settle the token reservation, whatever happens.

    Settling happens in ``finally`` so it also runs when the client disconnects mid-stream. The
    real usage comes from the provider's final usage chunk; failing that, it is estimated from
    the text that was actually sent.
    """
    usage_total: int | None = None
    sent: list[str] = []

    def observe(chunk: ChatCompletionChunk) -> None:
        nonlocal usage_total
        if chunk.usage is not None:
            usage_total = chunk.usage.total_tokens
        for choice in chunk.choices:
            if choice.delta.content:
                sent.append(choice.delta.content)

    try:
        if first is not None:
            observe(first)
            yield _sse(first.model_dump_json(exclude_none=True))
        async for chunk in rest:
            observe(chunk)
            yield _sse(chunk.model_dump_json(exclude_none=True))
    except ProviderError as exc:
        # Headers are already sent, so the failure is reported as an in-stream error event.
        _, body = describe_provider_error(exc)
        yield _sse(ErrorResponse(error=body).model_dump_json())
        return
    except Exception:
        logger.exception("unhandled error while streaming")
        fallback = ErrorResponse.of("Internal server error", "api_error")
        yield _sse(fallback.model_dump_json())
        return
    else:
        yield _sse("[DONE]")
    finally:
        actual = (
            usage_total
            if usage_total is not None
            else reservation.prompt_tokens + estimate_text_tokens("".join(sent))
        )
        await asyncio.shield(reservation.settle(actual))


def _tokens_used(completion: ChatCompletionResponse, reservation: Reservation) -> int:
    if completion.usage is not None:
        return completion.usage.total_tokens
    text = "".join(choice.message.content or "" for choice in completion.choices)
    return reservation.prompt_tokens + estimate_text_tokens(text)


@router.post(
    "/chat/completions", response_model=ChatCompletionResponse, response_model_exclude_none=True
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    gateway: Annotated[Router, Depends(get_router)],
    limiter: Annotated[RateLimiter, Depends(get_limiter)],
    auth: Annotated[AuthContext, Depends(rate_limited_chat)],
) -> Response:
    # Checked before the model is looked up, so a key learns nothing about models it may not use.
    if not model_allowed(auth.allowed_models, request.model):
        raise ModelNotAllowedError(request.model)

    # Set aside the tokens this request is expected to need; settled against real usage below.
    reservation = await limiter.reserve_tokens(auth, request)
    try:
        if not request.stream:
            completion = await gateway.complete(request)
            await reservation.settle(_tokens_used(completion, reservation))
            return JSONResponse(completion.model_dump(exclude_none=True))

        stream = gateway.stream(request)
        # Pull the first chunk before answering so upstream failures still become proper HTTP
        # errors.
        try:
            first: ChatCompletionChunk | None = await anext(stream)
        except StopAsyncIteration:
            first = None
    except BaseException:
        await asyncio.shield(reservation.release())  # nothing was produced: give it all back
        raise
    return StreamingResponse(
        _event_stream(first, stream, reservation),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
