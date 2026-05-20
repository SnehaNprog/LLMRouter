from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.middleware import RequestIdMiddleware

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name)
app.add_middleware(RequestIdMiddleware)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class Message:
    role = str
    content = str


class ChatRequest:
    model: str
    messages: Message
