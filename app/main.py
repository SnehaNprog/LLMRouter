from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.errors import register_exception_handlers
from app.api.health import default_readiness_checks
from app.api.health import router as health_router
from app.api.models import router as models_router
from app.core.config import get_settings
from app.core.http import lifespan
from app.core.logging import configure_logging
from app.core.middleware import RequestIdMiddleware

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
register_exception_handlers(app)
app.state.readiness_checks = default_readiness_checks()
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(models_router)
