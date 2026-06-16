from fastapi import APIRouter

from app.schemas.models import ModelInfo, ModelList

router = APIRouter(prefix="/v1")

# Static placeholder; replaced by the YAML model registry later.
_CREATED = 1_730_000_000
_STATIC_MODELS = [
    ModelInfo(id="auto", created=_CREATED, owned_by="llmrouter"),
    ModelInfo(id="stub-small", created=_CREATED, owned_by="llmrouter"),
    ModelInfo(id="stub-large", created=_CREATED, owned_by="llmrouter"),
]


@router.get("/models")
async def list_models() -> ModelList:
    return ModelList(data=_STATIC_MODELS)
