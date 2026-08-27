from collections.abc import Collection, Iterable

# What a key may do. Scopes are "resource:action"; "resource:*" grants every action on a resource.
CHAT_WRITE = "chat:write"
MODELS_READ = "models:read"
ADMIN_TENANTS = "admin:tenants"
ADMIN_KEYS = "admin:keys"

KNOWN_SCOPES = frozenset({CHAT_WRITE, MODELS_READ, ADMIN_TENANTS, ADMIN_KEYS})
RESOURCES = frozenset(scope.split(":")[0] for scope in KNOWN_SCOPES)

# Least privilege: a new key can call models, but never administer anything.
DEFAULT_SCOPES = (CHAT_WRITE, MODELS_READ)


def _is_valid(scope: str) -> bool:
    if scope in KNOWN_SCOPES:
        return True
    resource, _, action = scope.partition(":")
    return action == "*" and resource in RESOURCES


def normalize_scopes(scopes: Iterable[str]) -> list[str]:
    """Validate scopes and return them de-duplicated and sorted, ready to store."""
    unique = set(scopes)
    if not unique:
        raise ValueError("a key needs at least one scope")
    invalid = sorted(scope for scope in unique if not _is_valid(scope))
    if invalid:
        known = ", ".join(sorted(KNOWN_SCOPES))
        raise ValueError(f"unknown scope(s): {', '.join(invalid)} (known: {known})")
    return sorted(unique)


def scope_allows(granted: Collection[str], required: str) -> bool:
    """Whether a key holding ``granted`` scopes may perform an action needing ``required``."""
    if required not in KNOWN_SCOPES:
        raise ValueError(f"'{required}' is not a known scope")  # a typo in our own code
    resource = required.split(":")[0]
    return required in granted or f"{resource}:*" in granted
