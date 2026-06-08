from typing import Self

from pydantic import BaseModel


class ErrorBody(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody

    @classmethod
    def of(cls, message: str, type: str, param: str | None = None, code: str | None = None) -> Self:
        return cls(error=ErrorBody(message=message, type=type, param=param, code=code))
