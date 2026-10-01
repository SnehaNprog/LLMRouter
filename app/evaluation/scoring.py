import re
from dataclasses import dataclass

from app.evaluation.dataset import Contains, Exact, Number

_NUMBER = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|[-+]?\.\d+")
_WRAPPERS = "\"'`*_ \t\n"


@dataclass(frozen=True)
class Score:
    """How good one answer was, from 0 to 1, and why."""

    value: float
    detail: str


def normalize(text: str, *, case_sensitive: bool = False) -> str:
    """Make two spellings of the same short answer comparable.

    Surrounding quotes, backticks and emphasis marks, a final full stop, runs of white space and
    the space after a comma are all ignored. Nothing else is: an answer padded with extra words
    is not an exact match, because the prompts ask for the answer alone.
    """
    cleaned = text.strip(_WRAPPERS).rstrip(".").strip(_WRAPPERS)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    return cleaned if case_sensitive else cleaned.casefold()


def score_exact(scoring: Exact, reply: str) -> Score:
    want = normalize(scoring.answer, case_sensitive=scoring.case_sensitive)
    got = normalize(reply, case_sensitive=scoring.case_sensitive)
    if got == want:
        return Score(1.0, "matches the expected answer")
    return Score(0.0, f"expected {scoring.answer!r}, got {reply.strip()[:80]!r}")


def extract_number(reply: str) -> float | None:
    """The number an answer states.

    A reply that is just a number is read as one (dollar signs, percent signs and thousands
    separators allowed). Otherwise, if it mentions several different numbers the last is taken,
    because an answer states its conclusion at the end; if it mentions one, that one.
    """
    found: list[float] = []
    for match in _NUMBER.finditer(reply):
        text = match.group(0).replace(",", "")
        try:
            found.append(float(text))
        except ValueError:
            continue
    if not found:
        return None
    return found[-1]


def score_number(scoring: Number, reply: str) -> Score:
    got = extract_number(reply)
    if got is None:
        return Score(0.0, "no number in the reply")
    if abs(got - scoring.answer) <= scoring.tolerance:
        return Score(1.0, f"{got:g} is the expected {scoring.answer:g}")
    return Score(0.0, f"expected {scoring.answer:g}, got {got:g}")


def _mentions(text: str, term: str) -> bool:
    """Whether ``term`` appears as a whole word or phrase, ignoring case.

    Whole words, so that "au" is not found in "Australia" and "7" is not found in "17".
    """
    pattern = r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)"
    return re.search(pattern, text.casefold()) is not None


def score_contains(scoring: Contains, reply: str) -> Score:
    missing = [term for term in scoring.all_of if not _mentions(reply, term)]
    if missing:
        return Score(0.0, f"missing {missing}")
    if scoring.any_of and not any(_mentions(reply, term) for term in scoring.any_of):
        return Score(0.0, f"mentions none of {scoring.any_of}")
    return Score(1.0, "mentions what was expected")
