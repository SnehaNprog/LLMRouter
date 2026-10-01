import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.metrics.middleware import METRICS_PATH
from app.metrics.registry import Metrics

router = APIRouter(tags=["ops"])


def _bearer(request: Request) -> str:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


async def metrics_access(request: Request) -> None:
    """Let a scraper in. With a ``metrics_token`` configured it must present it as a bearer token.

    Virtual API keys are deliberately not accepted here: metrics describe the whole gateway, not
    one tenant, so they are for operators. Without a token the endpoint is open, which is right
    only on a private network.
    """
    expected: str | None = request.app.state.metrics_token
    if expected is None:
        return
    if not secrets.compare_digest(_bearer(request).encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="A valid metrics token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_metrics(request: Request) -> Metrics:
    metrics: Metrics = request.app.state.metrics
    return metrics


@router.get(
    METRICS_PATH,
    summary="Prometheus metrics",
    dependencies=[Depends(metrics_access)],
    include_in_schema=False,
)
def scrape(metrics: Annotated[Metrics, Depends(get_metrics)]) -> Response:
    return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)
