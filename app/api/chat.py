import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.api.deps import get_budgets, get_limiter, get_router
from app.api.errors import describe_provider_error
from app.auth.dependencies import AuthContext, require_scope
from app.auth.errors import ModelNotAllowedError
from app.auth.model_access import model_allowed
from app.auth.scopes import CHAT_WRITE
from app.providers.errors import ProviderError
from app.quota.budget_limiter import BudgetLimiter
from app.quota.headers import rate_headers_of
from app.quota.limiter import RateLimiter
from app.quota.quota import RequestQuota, reserve_request_quota
from app.router.router import Router, Served
from app.router.tokens import estimate_text_tokens
from app.routing.explain import log_decision, log_outcome, routing_headers
from app.routing.policy import AUTO_MODEL
from app.routing.resolve import resolve_model
from app.routing.selector import RoutingChoice
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import ChatCompletionChunk, ChatCompletionResponse
from app.schemas.error import ErrorResponse

logger = logging.getLogger(__name__)

chat_scope = require_scope(CHAT_WRITE)


async def rate_limited_chat(
    http_request: Request,
    auth: Annotated[AuthContext, Depends(chat_scope)],
    limiter: Annotated[RateLimiter, Depends(get_limiter)],
) -> AuthContext:
    """Authenticate, check the scope, then count the request against the key's rate limit."""
    rate_headers_of(http_request).requests = await limiter.check_request(auth)
    return auth


router = APIRouter(prefix="/v1")


def _sse(payload: str) -> str:
    return f"data: {payload}\n\n"


async def _event_stream(
    first: ChatCompletionChunk | None,
    rest: AsyncIterator[ChatCompletionChunk],
    quota: RequestQuota,
    served: Served,
) -> AsyncGenerator[str]:
    """Relay chunks as server-sent events, then settle the quota, whatever happens.

    Settling happens in ``finally`` so it also runs when the client disconnects mid-stream. The
    real usage comes from the provider's final usage chunk; failing that, it is estimated from
    the text that was actually sent.
    """
    usage: tuple[int, int] | None = None  # (prompt tokens, completion tokens)
    sent: list[str] = []

    def observe(chunk: ChatCompletionChunk) -> None:
        nonlocal usage
        if chunk.usage is not None:
            usage = (chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
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
        prompt, completion = usage or (quota.prompt_tokens, estimate_text_tokens("".join(sent)))
        await asyncio.shield(
            quota.settle(
                prompt_tokens=prompt, completion_tokens=completion, deployment=served.deployment
            )
        )


def _routing_headers(
    http_request: Request, choice: RoutingChoice | None, served: Served, auth: AuthContext
) -> dict[str, str]:
    """Explain a smart-routed answer (and log which model finally served it)."""
    if choice is None:
        return {}
    log_outcome(choice, served.deployment, tenant_id=auth.tenant_id, key_id=auth.key_id)
    if not http_request.app.state.routing_headers:
        return {}
    return routing_headers(choice, served.deployment)


def _usage_of(completion: ChatCompletionResponse, quota: RequestQuota) -> tuple[int, int]:
    if completion.usage is not None:
        return completion.usage.prompt_tokens, completion.usage.completion_tokens
    text = "".join(choice.message.content or "" for choice in completion.choices)
    return quota.prompt_tokens, estimate_text_tokens(text)


@router.post(
    "/chat/completions", response_model=ChatCompletionResponse, response_model_exclude_none=True
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
    gateway: Annotated[Router, Depends(get_router)],
    limiter: Annotated[RateLimiter, Depends(get_limiter)],
    budgets: Annotated[BudgetLimiter, Depends(get_budgets)],
    auth: Annotated[AuthContext, Depends(rate_limited_chat)],
) -> Response:
    # Checked before the model is looked up, so a key learns nothing about models it may not use.
    if not model_allowed(auth.allowed_models, request.model):
        raise ModelNotAllowedError(request.model)

    # "auto" is replaced by the model the smart router picks, before anything is priced or
    # reserved, so quotas, retries and fallbacks all see an ordinary request for that model.
    request, choice = resolve_model(request, gateway.registry, gateway.latency)
    escalation = choice.escalation if choice is not None else []
    if choice is not None:
        log_decision(choice, requested=AUTO_MODEL, tenant_id=auth.tenant_id, key_id=auth.key_id)

    # Set aside the tokens and money this request is expected to need; settled below.
    quota = await reserve_request_quota(limiter, budgets, auth, request, gateway.registry)
    rate = rate_headers_of(http_request)
    rate.tokens = limiter.token_window(auth, quota.tokens)
    served = Served()
    try:
        if not request.stream:
            completion = await gateway.complete(
                request, served, escalation=escalation, admit=quota.admit
            )
            prompt, generated = _usage_of(completion, quota)
            await quota.settle(
                prompt_tokens=prompt, completion_tokens=generated, deployment=served.deployment
            )
            rate.tokens = limiter.token_window(auth, quota.tokens)  # now net of the real usage
            headers = {**rate.as_headers(), **_routing_headers(http_request, choice, served, auth)}
            return JSONResponse(completion.model_dump(exclude_none=True), headers=headers)

        stream = gateway.stream(request, served, escalation=escalation, admit=quota.admit)
        # Pull the first chunk before answering so upstream failures still become proper HTTP
        # errors.
        try:
            first: ChatCompletionChunk | None = await anext(stream)
        except StopAsyncIteration:
            first = None
    except BaseException:
        await asyncio.shield(quota.release())  # nothing was produced: give it all back
        raise
    return StreamingResponse(
        _event_stream(first, stream, quota, served),
        media_type="text/event-stream",
        # The headers go out before the stream does, so the token figures are as of the
        # reservation; the final usage is settled once the stream ends.
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            **rate.as_headers(),
            **_routing_headers(http_request, choice, served, auth),
        },
    )
