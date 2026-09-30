import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.schemas.chat import ChatCompletionRequest, Message

DEFAULT_DATASET = Path(__file__).parent / "data" / "benchmark_v1.jsonl"

Category = Literal["chat", "extraction", "qa", "math", "code", "reasoning"]
Difficulty = Literal["easy", "medium", "hard"]


class DatasetError(Exception):
    """The benchmark file is missing or invalid. The message lists every problem found."""


class Exact(BaseModel):
    """The whole reply must equal ``answer`` (ignoring surrounding space, case and a final dot)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["exact"]
    answer: str = Field(min_length=1)
    case_sensitive: bool = False


class Number(BaseModel):
    """The reply must be a number within ``tolerance`` of ``answer``."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["number"]
    answer: float
    tolerance: float = Field(default=1e-6, ge=0)


class Contains(BaseModel):
    """The reply must mention every term of ``all_of`` and at least one of ``any_of``."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["contains"]
    all_of: list[str] = Field(default_factory=list)
    any_of: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _needs_a_term(self) -> "Contains":
        terms = [*self.all_of, *self.any_of]
        if not terms:
            raise ValueError("give at least one term in all_of or any_of")
        if any(not term.strip() for term in terms):
            raise ValueError("terms must not be blank")
        return self


class Rubric(BaseModel):
    """Open-ended answers, graded against ``criteria`` (by a judge model, in the evaluation)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["rubric"]
    criteria: list[str] = Field(min_length=1)
    reference: str | None = None  # an example of a good answer, if there is a canonical one


Scoring = Annotated[Exact | Number | Contains | Rubric, Field(discriminator="kind")]


class BenchmarkItem(BaseModel):
    """One prompt of the benchmark and how its answers are judged.

    ``difficulty`` is the author's judgement of how much model it takes. It is used to describe
    the benchmark and to break results down, never to route: the router has to find out for
    itself from the prompt.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z]+-\d{2,}$")
    category: Category
    difficulty: Difficulty
    prompt: str = Field(min_length=1)
    system: str | None = None
    scoring: Scoring
    max_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _id_matches_category(self) -> "BenchmarkItem":
        if not self.id.startswith(self.category + "-"):
            raise ValueError(f"id '{self.id}' should start with its category '{self.category}-'")
        if not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        return self

    def messages(self) -> list[Message]:
        system = [Message(role="system", content=self.system)] if self.system else []
        return [*system, Message(role="user", content=self.prompt)]

    def to_request(self, model: str, *, temperature: float | None = 0.0) -> ChatCompletionRequest:
        """The request that asks ``model`` this question (deterministic by default)."""
        return ChatCompletionRequest(
            model=model,
            messages=self.messages(),
            max_tokens=self.max_tokens,
            temperature=temperature,
        )


class Benchmark:
    """An ordered, validated set of benchmark items."""

    def __init__(self, items: list[BenchmarkItem]) -> None:
        if not items:
            raise DatasetError("the benchmark has no items")
        seen: set[str] = set()
        for item in items:
            if item.id in seen:
                raise DatasetError(f"duplicate item id '{item.id}'")
            seen.add(item.id)
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[BenchmarkItem]:
        return iter(self.items)

    def get(self, item_id: str) -> BenchmarkItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(item_id)

    def subset(
        self,
        *,
        categories: set[Category] | None = None,
        difficulties: set[Difficulty] | None = None,
        limit: int | None = None,
    ) -> "Benchmark":
        """The items in the given categories and difficulties, in order, at most ``limit``."""
        chosen = [
            item
            for item in self.items
            if (categories is None or item.category in categories)
            and (difficulties is None or item.difficulty in difficulties)
        ]
        return Benchmark(chosen[:limit] if limit is not None else chosen)

    def counts(self) -> dict[str, dict[str, int]]:
        """How many items there are per category, per difficulty and per scoring kind."""
        return {
            "category": dict(Counter(item.category for item in self.items)),
            "difficulty": dict(Counter(item.difficulty for item in self.items)),
            "scoring": dict(Counter(item.scoring.kind for item in self.items)),
        }

    def share(self, difficulty: Difficulty) -> float:
        return sum(1 for item in self.items if item.difficulty == difficulty) / len(self.items)


def parse_benchmark(text: str, source: str = "benchmark") -> Benchmark:
    """Read JSON Lines: one item per line, blank lines ignored. Reports every bad line at once."""
    items: list[BenchmarkItem] = []
    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(BenchmarkItem.model_validate(json.loads(line)))
        except json.JSONDecodeError as exc:
            problems.append(f"line {number}: not valid JSON ({exc.msg})")
        except ValidationError as exc:
            for error in exc.errors():
                where = ".".join(str(part) for part in error["loc"])
                problems.append(f"line {number}: {where + ': ' if where else ''}{error['msg']}")
    if problems:
        raise DatasetError(f"invalid benchmark {source}:\n  - " + "\n  - ".join(problems))
    try:
        return Benchmark(items)
    except DatasetError as exc:
        raise DatasetError(f"invalid benchmark {source}: {exc}") from None


def load_benchmark(path: Path = DEFAULT_DATASET) -> Benchmark:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"cannot read benchmark {path}: {exc.strerror or exc}") from None
    return parse_benchmark(text, str(path))
