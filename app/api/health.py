from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings

router = APIRouter(tags=["ops"])

ReadinessCheck = Callable[[], Awaitable[bool]]


async def config_loaded() -> bool:
    return bool(get_settings().app_name)


def default_readiness_checks() -> dict[str, ReadinessCheck]:
    # Later phases register more here (database, Redis).
    return {"config": config_loaded}


@router.get("/health", summary="Liveness: the process is up")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness: dependencies are usable")
async def ready(request: Request) -> JSONResponse:
    checks: dict[str, ReadinessCheck] = request.app.state.readiness_checks
    # A dependency the gateway can run without reports "degraded" instead of failing readiness.
    optional: set[str] = getattr(request.app.state, "optional_checks", set())
    results: dict[str, str] = {}
    for name, check in checks.items():
        try:
            healthy = await check()
        except Exception:
            healthy = False
        results[name] = "ok" if healthy else ("degraded" if name in optional else "failed")
    is_ready = all(status != "failed" for status in results.values())
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={"status": "ready" if is_ready else "not_ready", "checks": results},
    )
