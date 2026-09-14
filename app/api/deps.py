from fastapi import Request

from app.quota.budget_limiter import BudgetLimiter
from app.quota.limiter import RateLimiter
from app.router.router import Router


def get_router(request: Request) -> Router:
    router: Router = request.app.state.router
    return router


def get_limiter(request: Request) -> RateLimiter:
    limiter: RateLimiter = request.app.state.limiter
    return limiter


def get_budgets(request: Request) -> BudgetLimiter:
    budgets: BudgetLimiter = request.app.state.budgets
    return budgets
