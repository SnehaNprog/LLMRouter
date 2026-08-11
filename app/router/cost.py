from app.core.model_config import ModelConfig

_PER_MILLION = 1_000_000


def cost_usd(config: ModelConfig, prompt_tokens: int, completion_tokens: int) -> float:
    """Dollar cost of a call, from the per-million-token rates in the model's config."""
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("token counts must not be negative")
    return (
        prompt_tokens * config.input_cost_per_mtok + completion_tokens * config.output_cost_per_mtok
    ) / _PER_MILLION
