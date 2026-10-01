"""Dependencies that hand a request the shared objects built at startup.

They are ``async def`` on purpose, though they await nothing. FastAPI runs a plain ``def``
dependency in a worker thread, so each one would cost every request a thread hop: cheap while
the gateway is busy, but several milliseconds when the pool's threads and the CPU have gone idle
waiting for a model, which is exactly the normal state of a gateway.
"""

from fastapi import Request

from app.quota.budget_limiter import BudgetLimiter
from app.quota.limiter import RateLimiter
from app.router.router import Router
from app.usage.record import RequestLogger


async def get_router(request: Request) -> Router:
    router: Router = request.app.state.router
    return router


async def get_limiter(request: Request) -> RateLimiter:
    limiter: RateLimiter = request.app.state.limiter
    return limiter


async def get_budgets(request: Request) -> BudgetLimiter:
    budgets: BudgetLimiter = request.app.state.budgets
    return budgets


async def get_request_logger(request: Request) -> RequestLogger:
    request_logger: RequestLogger = request.app.state.request_logger
    return request_logger
