import time
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_router
from app.auth.dependencies import AuthContext, require_scope
from app.auth.model_access import model_allowed
from app.auth.scopes import MODELS_READ
from app.router.router import Router
from app.routing.policy import AUTO_MODEL
from app.schemas.models import ModelInfo, ModelList

router = APIRouter(prefix="/v1")

_STARTED = int(time.time())


@router.get("/models")
async def list_models(
    gateway: Annotated[Router, Depends(get_router)],
    auth: Annotated[AuthContext, Depends(require_scope(MODELS_READ))],
) -> ModelList:
    """The models this key may use: restricted keys see only their allow-list.

    "auto" is listed first whenever there are tiered models to route between.
    """
    aliases = list(gateway.registry.aliases())
    if gateway.registry.tiers():
        aliases.insert(0, AUTO_MODEL)
    return ModelList(
        data=[
            ModelInfo(id=alias, created=_STARTED, owned_by="llmrouter")
            for alias in aliases
            if model_allowed(auth.allowed_models, alias)
        ]
    )
