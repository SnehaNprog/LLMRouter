class AuthError(Exception):
    """A request could not be authenticated or is not allowed. Rendered as an OpenAI-style error."""

    status_code = 401
    type = "authentication_error"
    code = "invalid_api_key"
    param: str | None = None
    headers: dict[str, str] = {"WWW-Authenticate": "Bearer"}

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class MissingApiKeyError(AuthError):
    code = "missing_api_key"

    def __init__(self) -> None:
        super().__init__(
            "You didn't provide an API key. Send your key in an Authorization header "
            "using Bearer auth (Authorization: Bearer YOUR_KEY)."
        )


class InvalidApiKeyError(AuthError):
    code = "invalid_api_key"

    def __init__(self) -> None:
        # The presented key is never echoed: it may be a real secret for some other service.
        super().__init__("Incorrect API key provided.")


class TenantDeactivatedError(AuthError):
    status_code = 403
    type = "permission_error"
    code = "tenant_deactivated"
    headers: dict[str, str] = {}

    def __init__(self) -> None:
        super().__init__("This organisation has been deactivated.")


class ApiKeyRevokedError(AuthError):
    code = "api_key_revoked"

    def __init__(self) -> None:
        super().__init__("This API key has been revoked.")


class ApiKeyExpiredError(AuthError):
    code = "api_key_expired"

    def __init__(self) -> None:
        super().__init__("This API key has expired.")


class InsufficientScopeError(AuthError):
    status_code = 403
    type = "permission_error"
    code = "insufficient_scope"

    def __init__(self, scope: str) -> None:
        super().__init__(f"This API key does not have the required scope '{scope}'.")
        self.scope = scope
        # RFC 6750: tell the client which scope would have been needed.
        self.headers = {"WWW-Authenticate": f'Bearer error="insufficient_scope", scope="{scope}"'}


class ModelNotAllowedError(AuthError):
    status_code = 403
    type = "permission_error"
    code = "model_not_allowed"
    param = "model"
    headers: dict[str, str] = {}

    def __init__(self, model: str) -> None:
        super().__init__(f"This API key is not allowed to use the model '{model}'.")
        self.model = model


class ScopeEscalationError(AuthError):
    status_code = 403
    type = "permission_error"
    code = "scope_escalation"
    headers: dict[str, str] = {}

    def __init__(self, scope: str) -> None:
        super().__init__(f"Your key cannot grant the scope '{scope}' because it does not hold it.")
        self.scope = scope
