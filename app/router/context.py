from app.router.deployment import Deployment


class ContextLengthExceededError(Exception):
    """No deployment in the model's fallback chain has a context window large enough."""

    def __init__(self, window: int, required: int) -> None:
        super().__init__(
            f"This model's maximum context length is {window} tokens. However, your request "
            f"needs about {required} tokens (messages plus the requested completion). "
            "Please reduce the length of the messages or the completion."
        )
        self.window = window
        self.required = required


def fits(deployment: Deployment, required_tokens: int) -> bool:
    """A deployment with no configured window is assumed to accept anything."""
    window = deployment.context_window
    return window is None or required_tokens <= window
