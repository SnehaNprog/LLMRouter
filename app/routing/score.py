import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from app.routing.features import ComplexityFeatures, extract_features
from app.schemas.chat import ChatCompletionRequest

# How many of a thing counts as "a lot": at this point a signal is at full strength. These are
# fixed; what operators tune is how much each signal matters (the strengths below).
_LAST_USER_TOKENS = 1_500  # a long ask
_CONTEXT_TOKENS = 8_000  # a lot of history and system prompt beyond the ask
_CUES = 4  # reasoning phrases, constraint phrases, maths signals, parts of a request
_TOOLS = 3
_IMAGES = 2
_OUTPUT_TOKENS = 4_000  # a very long requested answer
_TURNS = 10
_NON_ASCII = 0.5
_CODE_LINES = 0.5  # share of lines that are code
_CODE_REQUEST_LEVEL = 0.6  # asking for code is weaker evidence than pasting it

SIGNALS = (
    "last_user_length",
    "context_length",
    "code",
    "math",
    "reasoning",
    "constraints",
    "multi_part",
    "structured_output",
    "tools",
    "images",
    "output_length",
    "turns",
    "non_english",
)

# How strongly each signal, at full level, argues that a request is hard (0: ignore it, 1:
# alone it makes the request maximally hard). Reasoning, maths and code are the best evidence.
DEFAULT_STRENGTHS: Mapping[str, float] = {
    "last_user_length": 0.40,
    "context_length": 0.25,
    "code": 0.45,
    "math": 0.55,
    "reasoning": 0.55,
    "constraints": 0.30,
    "multi_part": 0.35,
    "structured_output": 0.20,
    "tools": 0.30,
    "images": 0.20,
    "output_length": 0.30,
    "turns": 0.10,
    "non_english": 0.15,
}
DEFAULT_SMALL_TALK_FACTOR = 0.3
SMALL_TALK_KEY = "small_talk_factor"


def _level(value: float, full: float) -> float:
    return max(0.0, min(1.0, value / full))


def _log_level(tokens: int, full: int) -> float:
    return max(0.0, min(1.0, math.log1p(max(0, tokens)) / math.log1p(full)))


def signal_levels(f: ComplexityFeatures) -> dict[str, float]:
    """Each signal as a level from 0 (absent) to 1 (a lot), whatever its natural unit."""
    return {
        "last_user_length": _log_level(f.last_user_tokens, _LAST_USER_TOKENS),
        # history and system prompt only: a single-turn request carries no extra context
        "context_length": _log_level(f.prompt_tokens - f.last_user_tokens, _CONTEXT_TOKENS),
        "code": 1.0
        if f.code_blocks > 0
        else max(
            _level(f.code_line_ratio, _CODE_LINES), _CODE_REQUEST_LEVEL if f.code_request else 0.0
        ),
        "math": _level(f.math_signals, _CUES),
        "reasoning": _level(f.reasoning_cues, _CUES),
        "constraints": _level(f.constraint_cues, _CUES),
        "multi_part": _level(f.list_items + max(0, f.question_marks - 1), _CUES),
        "structured_output": 1.0 if f.structured_output else 0.0,
        "tools": _level(f.tool_count / _TOOLS + (0.34 if f.forces_tool else 0.0), 1.0),
        "images": _level(f.image_count, _IMAGES),
        "output_length": _level(f.requested_completion_tokens or 0, _OUTPUT_TOKENS),
        "turns": _level(f.turns - 1, _TURNS - 1),
        "non_english": _level(f.non_ascii_ratio, _NON_ASCII),
    }


@dataclass(frozen=True)
class ComplexityWeights:
    """How much each signal matters. Tunable from the ``routing`` section of the config."""

    strengths: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_STRENGTHS))
    small_talk_factor: float = DEFAULT_SMALL_TALK_FACTOR  # greetings count for this share

    @staticmethod
    def problems(overrides: Mapping[str, float]) -> list[str]:
        """Everything wrong with a set of overrides, one readable line each."""
        problems = []
        for name, value in overrides.items():
            if name != SMALL_TALK_KEY and name not in SIGNALS:
                known = ", ".join((*SIGNALS, SMALL_TALK_KEY))
                problems.append(f"unknown signal '{name}' (known: {known})")
            elif isinstance(value, bool) or not isinstance(value, int | float):
                problems.append(f"{name}: must be a number, not {value!r}")
            elif not 0.0 <= value <= 1.0 or math.isnan(value):
                problems.append(f"{name}: must be between 0 and 1, not {value}")
        return problems

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, float]) -> "ComplexityWeights":
        """The defaults with some strengths replaced. Signals not mentioned keep their default."""
        if problems := cls.problems(overrides):
            raise ValueError("invalid complexity weights: " + "; ".join(problems))
        strengths = dict(DEFAULT_STRENGTHS)
        strengths.update({k: float(v) for k, v in overrides.items() if k != SMALL_TALK_KEY})
        return cls(
            strengths=strengths,
            small_talk_factor=float(overrides.get(SMALL_TALK_KEY, DEFAULT_SMALL_TALK_FACTOR)),
        )


@dataclass(frozen=True)
class ComplexityScore:
    value: float  # 0 (trivial) to 1 (very demanding)
    contributions: Mapping[str, float]  # each signal's share of the evidence, largest first
    small_talk: bool

    def top(self, count: int = 3) -> list[str]:
        """The signals that most raised the score (ignoring those that contributed nothing)."""
        return [name for name, share in list(self.contributions.items())[:count] if share > 0]


DEFAULT_WEIGHTS = ComplexityWeights()


def score_features(
    features: ComplexityFeatures, weights: ComplexityWeights = DEFAULT_WEIGHTS
) -> ComplexityScore:
    """Combine the signals into one number.

    Each signal supplies evidence ``strength x level``, and the pieces are combined as
    independent evidence ("noisy OR"): ``1 - (1-e1)(1-e2)...``. One strong signal is enough to
    raise the score, several reinforce each other, and the result can never leave 0 to 1. A plain
    average would let a clearly hard prompt be diluted by all the signals it does not trigger.
    """
    levels = signal_levels(features)
    evidence = {name: weights.strengths.get(name, 0.0) * level for name, level in levels.items()}
    doubt = 1.0
    for share in evidence.values():
        doubt *= 1.0 - share
    value = 1.0 - doubt
    if features.small_talk:
        value *= weights.small_talk_factor
    ranked = dict(sorted(evidence.items(), key=lambda item: item[1], reverse=True))
    return ComplexityScore(
        value=round(max(0.0, min(1.0, value)), 4),
        contributions={name: round(share, 4) for name, share in ranked.items()},
        small_talk=features.small_talk,
    )


def score_request(
    request: ChatCompletionRequest, weights: ComplexityWeights = DEFAULT_WEIGHTS
) -> ComplexityScore:
    return score_features(extract_features(request), weights)
