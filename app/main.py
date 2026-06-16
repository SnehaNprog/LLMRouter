from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.errors import register_exception_handlers
from app.api.models import router as models_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.middleware import RequestIdMiddleware

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name)
app.add_middleware(RequestIdMiddleware)
register_exception_handlers(app)
app.include_router(chat_router)
app.include_router(models_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
