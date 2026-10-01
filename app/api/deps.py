from fastapi import Request

from app.quota.budget_limiter import BudgetLimiter
from app.quota.limiter import RateLimiter
from app.router.router import Router
from app.usage.record import RequestLogger


def get_router(request: Request) -> Router:
    router: Router = request.app.state.router
    return router


def get_limiter(request: Request) -> RateLimiter:
    limiter: RateLimiter = request.app.state.limiter
    return limiter


def get_budgets(request: Request) -> BudgetLimiter:
    budgets: BudgetLimiter = request.app.state.budgets
    return budgets


def get_request_logger(request: Request) -> RequestLogger:
    request_logger: RequestLogger = request.app.state.request_logger
    return request_logger
