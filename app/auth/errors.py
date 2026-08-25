class AuthError(Exception):
    """A request could not be authenticated or is not allowed. Rendered as an OpenAI-style error."""

    status_code = 401
    type = "authentication_error"
    code = "invalid_api_key"
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
