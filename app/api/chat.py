import time

from fastapi import APIRouter

from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import ChatCompletionResponse, Choice, ResponseMessage, Usage

router = APIRouter(prefix="/v1")


@router.post("/chat/completions", response_model_exclude_none=True)
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-stub",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(content="This is a stub response."),
                finish_reason="stop",
            )
        ],
        usage=Usage.of(prompt_tokens=1, completion_tokens=5),
    )
