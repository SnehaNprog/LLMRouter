import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.api.deps import get_router
from app.api.errors import describe_provider_error
from app.auth.dependencies import authenticate
from app.providers.errors import ProviderError
from app.router.router import Router
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import ChatCompletionChunk, ChatCompletionResponse
from app.schemas.error import ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", dependencies=[Depends(authenticate)])


def _sse(payload: str) -> str:
    return f"data: {payload}\n\n"


async def _event_stream(
    first: ChatCompletionChunk | None, rest: AsyncIterator[ChatCompletionChunk]
) -> AsyncIterator[str]:
    try:
        if first is not None:
            yield _sse(first.model_dump_json(exclude_none=True))
        async for chunk in rest:
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
    yield _sse("[DONE]")


@router.post(
    "/chat/completions", response_model=ChatCompletionResponse, response_model_exclude_none=True
)
async def create_chat_completion(
    request: ChatCompletionRequest, gateway: Annotated[Router, Depends(get_router)]
) -> Response:
    if not request.stream:
        completion = await gateway.complete(request)
        return JSONResponse(completion.model_dump(exclude_none=True))

    stream = gateway.stream(request)
    # Pull the first chunk before answering so upstream failures still become proper HTTP errors.
    try:
        first: ChatCompletionChunk | None = await anext(stream)
    except StopAsyncIteration:
        first = None
    return StreamingResponse(
        _event_stream(first, stream),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
