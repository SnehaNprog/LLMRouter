import time
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_router
from app.router.router import Router
from app.schemas.models import ModelInfo, ModelList

router = APIRouter(prefix="/v1")

_STARTED = int(time.time())


@router.get("/models")
async def list_models(gateway: Annotated[Router, Depends(get_router)]) -> ModelList:
    return ModelList(
        data=[
            ModelInfo(id=alias, created=_STARTED, owned_by="llmrouter")
            for alias in gateway.registry.aliases()
        ]
    )
