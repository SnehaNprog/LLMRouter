import math
from dataclasses import dataclass

from fastapi import Request

_SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class Window:
    """Where a per-minute allowance stands: what the limit is, what is left, when it is full."""

    limit: int
    remaining: int
    reset_seconds: float  # until the allowance has refilled completely

    @classmethod
    def from_balance(cls, limit: int, balance: int) -> "Window":
        """Describe a token bucket holding ``balance`` (negative while in debt) of ``limit``."""
        missing = max(0, limit - balance)
        return cls(
            limit=limit,
            remaining=max(0, balance),
            reset_seconds=missing * _SECONDS_PER_MINUTE / limit,
        )


def format_duration(seconds: float) -> str:
    """A duration in the style OpenAI uses for reset headers: ``20ms``, ``1s``, ``6m0s``.

    Rounded up, so a client told to wait that long never comes back a moment too early.
    """
    if seconds <= 0:
        return "0s"
    milliseconds = math.ceil(seconds * 1000)
    if milliseconds < 1000:
        return f"{milliseconds}ms"
    whole, fraction = divmod(milliseconds, 1000)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    text = f"{secs}" + (f".{fraction:03d}".rstrip("0") if fraction else "") + "s"
    if hours:
        return f"{hours}h{minutes}m{text}"
    if minutes:
        return f"{minutes}m{text}"
    return text


@dataclass
class RateHeaders:
    """What a response should say about the caller's allowances, collected as a request proceeds."""

    requests: Window | None = None
    tokens: Window | None = None

    def as_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        for kind, window in (("requests", self.requests), ("tokens", self.tokens)):
            if window is None:
                continue
            headers[f"x-ratelimit-limit-{kind}"] = str(window.limit)
            headers[f"x-ratelimit-remaining-{kind}"] = str(window.remaining)
            headers[f"x-ratelimit-reset-{kind}"] = format_duration(window.reset_seconds)
        return headers


def rate_headers_of(request: Request) -> RateHeaders:
    """The request's collected allowance info (created on first use)."""
    existing = getattr(request.state, "rate_headers", None)
    if isinstance(existing, RateHeaders):
        return existing
    created = RateHeaders()
    request.state.rate_headers = created
    return created
