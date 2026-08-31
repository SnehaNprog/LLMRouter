import re
from collections.abc import Iterable, Sequence

_MAX_PATTERN_LENGTH = 200


def normalize_allowed_models(patterns: Iterable[str]) -> list[str]:
    """Validate a key's model allow-list and return it de-duplicated and sorted.

    Entries are model aliases, or patterns where ``*`` matches any run of characters
    (``fast*``, ``*-large``). An empty list is rejected: leave the list out to allow every model.
    """
    cleaned = {pattern.strip() for pattern in patterns}
    if not cleaned:
        raise ValueError("allowed_models must not be empty; omit it to allow every model")
    for pattern in cleaned:
        if not pattern:
            raise ValueError("allowed_models entries must not be blank")
        if len(pattern) > _MAX_PATTERN_LENGTH:
            raise ValueError(
                f"allowed_models entries are limited to {_MAX_PATTERN_LENGTH} characters"
            )
        if any(character.isspace() for character in pattern):
            raise ValueError(f"model names do not contain whitespace: {pattern!r}")
    return sorted(cleaned)


def model_allowed(allowed: Sequence[str] | None, model: str) -> bool:
    """``None`` means the key is unrestricted. Matching is exact and case-sensitive."""
    if allowed is None:
        return True
    return any(
        re.fullmatch(".*".join(re.escape(part) for part in pattern.split("*")), model)
        for pattern in allowed
    )
