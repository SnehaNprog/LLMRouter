from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class SSEEvent:
    event: str | None
    data: str


async def iter_sse(response: httpx.Response) -> AsyncIterator[SSEEvent]:
    """Parse a Server-Sent Events body into events (comments and unknown fields are skipped)."""
    event: str | None = None
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                yield SSEEvent(event, "\n".join(data_lines))
            event, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield SSEEvent(event, "\n".join(data_lines))
