from fastapi import FastAPI

from app.api.admin import keys as admin_keys_router
from app.api.admin import tenants as admin_tenants_router
from app.api.chat import router as chat_router
from app.api.errors import register_exception_handlers
from app.api.health import default_readiness_checks
from app.api.health import router as health_router
from app.api.models import router as models_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.middleware import RequestIdMiddleware
from app.lifespan import RegistryFactory, default_registry_factory, make_lifespan


def create_app(
    settings: Settings | None = None, registry_factory: RegistryFactory | None = None
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    factory = registry_factory or default_registry_factory(settings)

    app = FastAPI(title=settings.app_name, lifespan=make_lifespan(settings, factory))
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)
    app.state.readiness_checks = default_readiness_checks()
    app.state.routing_headers = settings.routing_headers
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(models_router)
    app.include_router(admin_tenants_router)
    app.include_router(admin_keys_router)
    return app


app = create_app()
