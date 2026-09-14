import asyncio
from dataclasses import dataclass

from app.auth.dependencies import AuthContext
from app.quota.budget import cost_nano
from app.quota.budget_limiter import BudgetLimiter
from app.quota.limiter import RateLimiter
from app.quota.reservation import Reservation
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.router.tokens import estimate_prompt_tokens
from app.schemas.chat import ChatCompletionRequest


@dataclass
class RequestQuota:
    """Everything set aside for one request: its token allowance and its share of the budgets."""

    tokens: Reservation
    budget: Reservation
    prompt_tokens: int  # our estimate of the prompt, for settling responses that report no usage

    async def settle(
        self, *, prompt_tokens: int, completion_tokens: int, deployment: Deployment | None
    ) -> None:
        """Replace the estimates with what the request really used."""
        await self.tokens.settle(prompt_tokens + completion_tokens)
        if deployment is not None:
            price = deployment.cost(prompt_tokens, completion_tokens)
            await self.budget.settle(cost_nano(price))
        # Without knowing which deployment answered, the budget keeps the (pessimistic) estimate.

    async def release(self) -> None:
        """The request produced nothing: give everything back."""
        await self.tokens.release()
        await self.budget.release()


async def reserve_request_quota(
    limiter: RateLimiter,
    budgets: BudgetLimiter,
    auth: AuthContext,
    request: ChatCompletionRequest,
    registry: ModelRegistry,
) -> RequestQuota:
    """Reserve tokens, then money; if the budget refuses, the tokens go back."""
    prompt_tokens = estimate_prompt_tokens(request)
    tokens = await limiter.reserve_tokens(auth, request)
    try:
        budget = await budgets.reserve(auth, request, registry, prompt_tokens)
    except BaseException:
        await asyncio.shield(tokens.release())
        raise
    return RequestQuota(tokens, budget, prompt_tokens)
