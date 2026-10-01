import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.evaluation.dataset import Benchmark, BenchmarkItem, Rubric
from app.evaluation.judge import Judge, JudgeError
from app.evaluation.runner import EvalRecord
from app.evaluation.scoring import Score, score_contains, score_exact, score_number

Method = Literal["exact", "number", "contains", "rubric", "no_answer"]

# What the negative control hands the judge: a reply that is polite and brief but answers nothing.
NON_ANSWER = "I'm sorry, I'm not able to help with that."


class ScoreRecord(BaseModel):
    """The quality of one answer. ``score`` is None only when it could not be worked out."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    run: str
    method: Method
    score: float | None  # 0 to 1
    detail: str = ""
    error: str | None = None  # why there is no score
    judge_model: str | None = None  # set when a judge decided it


class GroupQuality(BaseModel):
    n: int
    mean: float


class QualitySummary(BaseModel):
    run: str
    items: int
    scored: int
    unscored: int  # not counted in any mean: their quality is unknown, not zero
    mean: float | None
    by_method: dict[str, GroupQuality]
    by_category: dict[str, GroupQuality]
    by_difficulty: dict[str, GroupQuality]
    no_answer: int  # items that got no usable answer at all; these count as 0


def load_scores(path: Path) -> list[ScoreRecord]:
    """Read a scores file. If an item appears more than once the last record wins."""
    latest: dict[str, ScoreRecord] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = ScoreRecord.model_validate(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"{path}, line {number}: {exc}") from None
        latest[record.item_id] = record
    return list(latest.values())


def score_deterministic(item: BenchmarkItem, reply: str) -> Score | None:
    """The score for an answer that can be checked mechanically; None for an open-ended one."""
    scoring = item.scoring
    if scoring.kind == "exact":
        return score_exact(scoring, reply)
    if scoring.kind == "number":
        return score_number(scoring, reply)
    if scoring.kind == "contains":
        return score_contains(scoring, reply)
    return None


async def score_one(
    item: BenchmarkItem, record: EvalRecord, judge: Judge | None, run: str | None = None
) -> ScoreRecord:
    """Score one recorded answer: a failed request is a 0, a checkable answer is checked, and an
    open-ended one goes to the judge."""
    label = run or record.run
    if not record.ok or record.reply is None:
        return ScoreRecord(
            item_id=item.id,
            run=label,
            method="no_answer",
            score=0.0,
            detail=record.error or "no answer",
        )
    checked = score_deterministic(item, record.reply)
    if checked is not None:
        return ScoreRecord(
            item_id=item.id,
            run=label,
            method=item.scoring.kind,
            score=checked.value,
            detail=checked.detail,
        )
    assert isinstance(item.scoring, Rubric)
    if judge is None:
        return ScoreRecord(
            item_id=item.id, run=label, method="rubric", score=None, error="no judge configured"
        )
    try:
        graded = await judge.grade(item, item.scoring, record.reply)
    except JudgeError as exc:
        return ScoreRecord(
            item_id=item.id,
            run=label,
            method="rubric",
            score=None,
            error=str(exc),
            judge_model=judge.model,
        )
    return ScoreRecord(
        item_id=item.id,
        run=label,
        method="rubric",
        score=graded.value,
        detail=graded.detail,
        judge_model=judge.model,
    )


async def score_records(
    records: list[EvalRecord],
    benchmark: Benchmark,
    judge: Judge | None,
    *,
    concurrency: int = 4,
    out: Path | None = None,
    resume: bool = False,
    on_score: Callable[[ScoreRecord], None] | None = None,
) -> list[ScoreRecord]:
    """Score every recorded answer, in the order given.

    With ``out`` each score is appended to the file as soon as it is known, and with ``resume``
    the items already scored there are not scored (or paid for) again; unscored ones are retried.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    for record in records:
        try:
            benchmark.get(record.item_id)
        except KeyError:
            raise ValueError(
                f"the results contain '{record.item_id}', which is not in the benchmark"
            ) from None
    done: dict[str, ScoreRecord] = {}
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        if resume and out.exists():
            done = {s.item_id: s for s in load_scores(out) if s.score is not None}
        elif not resume:
            out.write_text("", encoding="utf-8")
    gate = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    results = dict(done)

    async def one(record: EvalRecord) -> None:
        async with gate:
            scored = await score_one(benchmark.get(record.item_id), record, judge)
        results[record.item_id] = scored
        if out is not None:
            async with write_lock:
                with out.open("a", encoding="utf-8") as handle:
                    handle.write(scored.model_dump_json() + "\n")
        if on_score is not None:
            on_score(scored)

    await asyncio.gather(*(one(r) for r in records if r.item_id not in done))
    return [results[r.item_id] for r in records]


async def negative_control(
    benchmark: Benchmark, judge: Judge, *, concurrency: int = 4
) -> list[ScoreRecord]:
    """Ask the judge to grade a non-answer for every open-ended item.

    A sound judge gives it (close to) nothing. The mean score here is the floor that any real
    answer's rubric score should be read against, and any item scoring above zero shows a
    criterion that is satisfied too easily.
    """
    gate = asyncio.Semaphore(concurrency)
    items = [item for item in benchmark if isinstance(item.scoring, Rubric)]

    async def one(item: BenchmarkItem) -> ScoreRecord:
        record = EvalRecord(
            item_id=item.id,
            run="negative-control",
            requested_model="none",
            ok=True,
            reply=NON_ANSWER,
        )
        async with gate:
            return await score_one(item, record, judge)

    return list(await asyncio.gather(*(one(item) for item in items)))


def _group(values: dict[str, list[float]]) -> dict[str, GroupQuality]:
    return {
        key: GroupQuality(n=len(scores), mean=sum(scores) / len(scores))
        for key, scores in sorted(values.items())
    }


def quality_summary(scores: list[ScoreRecord], benchmark: Benchmark) -> QualitySummary:
    scored = [s for s in scores if s.score is not None]
    by_method: dict[str, list[float]] = {}
    by_category: dict[str, list[float]] = {}
    by_difficulty: dict[str, list[float]] = {}
    for s in scored:
        assert s.score is not None
        item = benchmark.get(s.item_id)
        by_method.setdefault(s.method, []).append(s.score)
        by_category.setdefault(item.category, []).append(s.score)
        by_difficulty.setdefault(item.difficulty, []).append(s.score)
    values = [s.score for s in scored if s.score is not None]
    return QualitySummary(
        run=scores[0].run if scores else "",
        items=len(scores),
        scored=len(scored),
        unscored=len(scores) - len(scored),
        mean=sum(values) / len(values) if values else None,
        by_method=_group(by_method),
        by_category=_group(by_category),
        by_difficulty=_group(by_difficulty),
        no_answer=sum(1 for s in scores if s.method == "no_answer"),
    )
