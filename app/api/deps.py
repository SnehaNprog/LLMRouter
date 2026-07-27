from fastapi import Request

from app.router.router import Router


def get_router(request: Request) -> Router:
    router: Router = request.app.state.router
    return router
