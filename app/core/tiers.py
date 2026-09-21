from enum import StrEnum


class Tier(StrEnum):
    """How capable (and costly) a model is. The smart router starts cheap and escalates.

    Tiers compare by strength, not alphabetically: ``Tier.SLM < Tier.MID < Tier.FRONTIER``.
    """

    SLM = "slm"  # small language model: cheap and fast, enough for easy requests
    MID = "mid"  # a capable general-purpose model
    FRONTIER = "frontier"  # the strongest (and dearest) models, kept for hard requests

    @property
    def rank(self) -> int:
        return _ORDER.index(self)

    # A Tier is also a str, and str brings its own (alphabetical) <, <=, > and >=. All four must
    # be overridden, or "frontier" > "mid" would be decided by spelling rather than strength.
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return self.rank >= other.rank

    def stronger(self) -> "Tier | None":
        """The next tier up, or None for the strongest."""
        return _ORDER[self.rank + 1] if self.rank + 1 < len(_ORDER) else None

    def weaker(self) -> "Tier | None":
        """The next tier down, or None for the weakest."""
        return _ORDER[self.rank - 1] if self.rank > 0 else None


_ORDER = (Tier.SLM, Tier.MID, Tier.FRONTIER)  # weakest first
TIERS = _ORDER
