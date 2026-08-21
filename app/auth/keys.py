import hashlib
import secrets
from dataclasses import dataclass

KEY_PREFIX = "llmr_"
_RANDOM_BYTES = 32  # 256 bits of entropy
_DISPLAY_LENGTH = (
    12  # "llmr_" plus 7 random characters: enough to recognise a key, useless to guess
)


@dataclass(frozen=True)
class GeneratedKey:
    secret: str  # shown to the user once, never stored
    prefix: str  # safe to display and store, for recognising a key
    hash: str  # what the database keeps


def hash_key(secret: str) -> str:
    """SHA-256 hex digest of a key.

    A fast, unsalted hash is the right tool here. Passwords need slow salted hashes because
    people choose guessable ones; these keys are 256 random bits, so brute force is hopeless
    whatever the hash, and a deterministic digest lets us look a key up by index.
    """
    return hashlib.sha256(secret.encode()).hexdigest()


def generate_key() -> GeneratedKey:
    secret = KEY_PREFIX + secrets.token_urlsafe(_RANDOM_BYTES)
    return GeneratedKey(secret=secret, prefix=secret[:_DISPLAY_LENGTH], hash=hash_key(secret))


def looks_like_key(value: str) -> bool:
    """Cheap shape check, so obviously wrong credentials never reach the database."""
    expected = len(KEY_PREFIX) + len(secrets.token_urlsafe(_RANDOM_BYTES))
    return value.startswith(KEY_PREFIX) and len(value) == expected
