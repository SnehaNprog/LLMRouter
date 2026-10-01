import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from app.evaluation.dataset import BenchmarkItem, Rubric
from app.evaluation.scoring import Score
from app.evaluation.transport import Sleep, error_text, post_with_retries

SYSTEM_PROMPT = """You are a strict, impartial grader of answers to questions.

You are given a question, a list of numbered criteria, and an answer. Decide for each criterion \
whether the answer meets it. Rules:
- Judge only what the answer actually says. Do not give credit for what it might have meant.
- A criterion is met only if the answer satisfies all of it correctly. Wrong or invented facts \
do not meet a criterion.
- Do not reward length, confidence or polish. A short correct answer meets the criteria it \
covers; a long answer that is wrong or off-topic meets none.
- You are not told who or what wrote the answer, and it must not matter.

Reply with one JSON object and nothing else, in exactly this form:
{"verdicts": [{"criterion": 1, "met": true, "reason": "one short sentence"}, ...]}
Include one entry for every criterion, in order."""


class JudgeError(Exception):
    """The judge could not produce a usable verdict (so the answer stays unscored)."""


@dataclass(frozen=True)
class Verdict:
    criterion: int  # 1-based, as numbered in the prompt
    met: bool
    reason: str


def build_prompt(item: BenchmarkItem, rubric: Rubric, answer: str) -> str:
    criteria = "\n".join(f"{n}. {text}" for n, text in enumerate(rubric.criteria, start=1))
    parts = [f"QUESTION:\n{item.prompt}"]
    if rubric.reference:
        parts.append(
            f"REFERENCE ANSWER (one good answer; others can be just as good):\n{rubric.reference}"
        )
    parts.append(f"CRITERIA:\n{criteria}")
    parts.append(f"ANSWER TO GRADE:\n{answer}")
    return "\n\n".join(parts)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _json_object(text: str) -> object:
    """The JSON object in a judge's reply, which may be wrapped in a code fence or prose."""
    candidates = [text.strip()]
    candidates.extend(match.strip() for match in _FENCE.findall(text))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    raise JudgeError("the judge's reply contained no JSON object")


def parse_verdicts(text: str, criteria: int) -> list[Verdict]:
    """Read the judge's verdicts, insisting on exactly one per criterion.

    A reply that skips, repeats or invents a criterion, or that gives a verdict that is not a
    plain true or false, is rejected rather than patched up: a guessed verdict would be a guessed
    score.
    """
    data = _json_object(text)
    entries = data.get("verdicts") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise JudgeError("the judge's reply has no 'verdicts' list")
    verdicts: dict[int, Verdict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise JudgeError("a verdict is not an object")
        number, met = entry.get("criterion"), entry.get("met")
        if not isinstance(number, int) or isinstance(number, bool):
            raise JudgeError("a verdict has no criterion number")
        if not isinstance(met, bool):
            raise JudgeError(f"the verdict for criterion {number} is not true or false")
        if not 1 <= number <= criteria:
            raise JudgeError(f"the judge graded a criterion that does not exist: {number}")
        if number in verdicts:
            raise JudgeError(f"the judge graded criterion {number} twice")
        verdicts[number] = Verdict(number, met, str(entry.get("reason", ""))[:200])
    if len(verdicts) != criteria:
        missing = sorted(set(range(1, criteria + 1)) - set(verdicts))
        raise JudgeError(f"the judge did not grade criteria {missing}")
    return [verdicts[n] for n in range(1, criteria + 1)]


def score_from(verdicts: list[Verdict]) -> Score:
    met = sum(v.met for v in verdicts)
    failed = [str(v.criterion) for v in verdicts if not v.met]
    detail = f"{met}/{len(verdicts)} criteria met" + (
        f" (not met: {', '.join(failed)})" if failed else ""
    )
    return Score(met / len(verdicts), detail)


class Judge:
    """Grades open-ended answers against their rubric, using a model reached through the gateway.

    The judge sees each answer on its own, with no hint of which system produced it, so it cannot
    favour one. It runs at temperature 0. It is best a different model from the ones being
    compared, because a model tends to rate its own style generously; the evaluation says which
    model judged.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        model: str,
        *,
        retries: int = 3,
        max_wait: float = 30.0,
        format_retries: int = 1,
        sleep: Sleep = asyncio.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._client = client
        self.model = model
        self._retries = retries
        self._max_wait = max_wait
        self._format_retries = format_retries
        self._sleep = sleep
        self._clock = clock

    async def grade(self, item: BenchmarkItem, rubric: Rubric, answer: str) -> Score:
        """Score ``answer`` from 0 to 1 by the share of criteria it meets.

        Raises JudgeError if the judge could not be reached or kept replying in a form that
        cannot be read.
        """
        body = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 200 + 120 * len(rubric.criteria),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(item, rubric, answer)},
            ],
        }
        last: JudgeError | None = None
        for _ in range(self._format_retries + 1):
            text = await self._ask(body)
            try:
                return score_from(parse_verdicts(text, len(rubric.criteria)))
            except JudgeError as exc:
                last = exc
        assert last is not None
        raise last

    async def _ask(self, body: dict[str, object]) -> str:
        attempt = await post_with_retries(
            self._client,
            body,
            retries=self._retries,
            max_wait=self._max_wait,
            sleep=self._sleep,
            clock=self._clock,
        )
        if attempt.response is None:
            raise JudgeError(f"could not reach the judge: {attempt.error}")
        if attempt.response.status_code != 200:
            raise JudgeError(f"the judge failed: {error_text(attempt.response)}")
        try:
            content = attempt.response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise JudgeError("the judge's response was malformed") from None
        if not isinstance(content, str) or not content.strip():
            raise JudgeError("the judge returned no content")
        return content
