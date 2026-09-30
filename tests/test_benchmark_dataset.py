import json
import math
from pathlib import Path
from typing import Any

import pytest

from app.evaluation.dataset import (
    DEFAULT_DATASET,
    Benchmark,
    BenchmarkItem,
    DatasetError,
    Number,
    load_benchmark,
    parse_benchmark,
)


def item(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "qa-01",
        "category": "qa",
        "difficulty": "easy",
        "prompt": "What is the capital of France?",
        "scoring": {"kind": "contains", "any_of": ["paris"]},
    }
    return {**base, **overrides}


def jsonl(*items: dict[str, Any]) -> str:
    return "\n".join(json.dumps(i) for i in items) + "\n"


# ---------- the shipped benchmark ----------


@pytest.fixture(scope="module")
def shipped() -> Benchmark:
    return load_benchmark()


def test_the_shipped_benchmark_loads(shipped: Benchmark) -> None:
    assert len(shipped) == 90


def test_every_category_has_the_same_number_of_items(shipped: Benchmark) -> None:
    assert shipped.counts()["category"] == dict.fromkeys(
        ["chat", "extraction", "qa", "math", "code", "reasoning"], 15
    )


def test_the_difficulty_mix_is_what_the_documentation_says(shipped: Benchmark) -> None:
    """If this changes, every cost-saving figure quoted from the evaluation changes with it."""
    assert shipped.counts()["difficulty"] == {"easy": 42, "medium": 23, "hard": 25}
    assert shipped.share("easy") == pytest.approx(42 / 90)


def test_ids_are_unique_and_numbered_in_order_within_each_category(shipped: Benchmark) -> None:
    seen: dict[str, int] = {}
    for entry in shipped:
        seen[entry.category] = seen.get(entry.category, 0) + 1
        assert entry.id == f"{entry.category}-{seen[entry.category]:02d}"


def test_every_item_makes_a_valid_request(shipped: Benchmark) -> None:
    for entry in shipped:
        request = entry.to_request("auto")
        assert request.model == "auto"
        assert request.messages[-1].content == entry.prompt
        assert request.temperature == 0.0
        assert request.max_tokens is not None


def test_open_ended_items_have_enough_criteria_to_judge(shipped: Benchmark) -> None:
    for entry in shipped:
        if entry.scoring.kind == "rubric" and entry.category in {"code", "reasoning", "qa"}:
            assert len(entry.scoring.criteria) >= 2, entry.id


def test_the_checkable_items_cover_every_scoring_kind(shipped: Benchmark) -> None:
    assert set(shipped.counts()["scoring"]) == {"exact", "number", "contains", "rubric"}


def test_contains_terms_are_written_in_lowercase_so_matching_can_ignore_case(
    shipped: Benchmark,
) -> None:
    for entry in shipped:
        if entry.scoring.kind == "contains":
            for term in [*entry.scoring.all_of, *entry.scoring.any_of]:
                assert term == term.lower(), (entry.id, term)


def numeric_answer(entry: BenchmarkItem) -> float:
    assert isinstance(entry.scoring, Number)
    return entry.scoring.answer


def test_numeric_answers_are_right(shipped: Benchmark) -> None:
    """Each answer worked out again here, separately from however the file was produced."""
    expected: dict[str, float] = {
        "extraction-08": len("The quick brown fox jumps".split()),
        "extraction-09": 1249.50,
        "extraction-15": len({"red", "blue", "green"}),
        "math-01": 17 + 26,
        "math-02": 9 * 8,
        "math-03": 144 // 12,
        "math-04": 200 * 15 / 100,
        "math-05": 180 / 2.5,
        "math-06": 40 * 3 / 4,
        "math-07": 1.35 / 3 * 8,
        "math-08": 50 * 51 // 2,
        # width w, length 2w: 2 * (w + 2w) = 36
        "math-09": (36 // 6) * (2 * (36 // 6)),
        "math-10": 10 * 9 * 8 * 7,
        "math-11": 5 / 8 * 4 / 7,
        "math-12": max(r for r in range(-20, 21) if 3 * r * r - 12 * r + 9 == 0),
        "math-13": pow(7, 100, 5),
        "math-14": 10_000 * 1.05**3,
        "math-15": math.factorial(5),
        "reasoning-02": next(b for b in range(1, 40) if 2 * b + b + (2 * b - 5) == 40),
        "reasoning-11": sum(str(n).count("7") for n in range(1, 101)),
        "reasoning-12": 5,  # machine-minutes per widget stay the same: 5
    }
    numeric_ids = {e.id for e in shipped if e.scoring.kind == "number"}
    assert numeric_ids == set(expected)  # every numeric item is checked, and none is missing
    for item_id, value in expected.items():
        entry = shipped.get(item_id)
        scoring = entry.scoring
        assert isinstance(scoring, Number)
        assert abs(scoring.answer - value) <= max(scoring.tolerance, 1e-9), item_id


def test_the_comma_separated_names_answer_follows_the_prompt_order(shipped: Benchmark) -> None:
    entry = shipped.get("extraction-13")
    assert entry.scoring.kind == "exact"
    text = entry.prompt
    names = entry.scoring.answer.split(", ")
    assert [text.index(name) for name in names] == sorted(text.index(name) for name in names)


def test_the_shipped_file_is_valid_utf8_json_lines() -> None:
    for number, line in enumerate(DEFAULT_DATASET.read_text(encoding="utf-8").splitlines(), 1):
        assert json.loads(line), number


# ---------- loading ----------


def test_blank_lines_are_ignored() -> None:
    text = "\n" + jsonl(item()) + "\n\n"
    assert len(parse_benchmark(text)) == 1


def test_items_keep_their_file_order() -> None:
    text = jsonl(item(id="qa-02"), item(id="qa-01"))
    assert [i.id for i in parse_benchmark(text)] == ["qa-02", "qa-01"]


def test_every_problem_is_reported_with_its_line_number() -> None:
    text = "\n".join(
        [
            json.dumps(item()),
            "{not json",
            json.dumps(item(id="qa-02", difficulty="impossible")),
            json.dumps(item(id="qa-03", prompt="")),
        ]
    )
    with pytest.raises(DatasetError) as caught:
        parse_benchmark(text, "mine.jsonl")
    message = str(caught.value)
    assert "mine.jsonl" in message
    assert "line 2: not valid JSON" in message
    assert "line 3: difficulty" in message
    assert "line 4: prompt" in message


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(DatasetError, match="duplicate item id 'qa-01'"):
        parse_benchmark(jsonl(item(), item()))


def test_an_empty_file_is_rejected() -> None:
    with pytest.raises(DatasetError, match="no items"):
        parse_benchmark("\n\n")


def test_a_missing_file_is_a_dataset_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="cannot read benchmark"):
        load_benchmark(tmp_path / "nope.jsonl")


def test_a_file_can_be_loaded_from_a_path(tmp_path: Path) -> None:
    path = tmp_path / "mine.jsonl"
    path.write_text(jsonl(item(prompt="Café ☕ — what is Paris?")), encoding="utf-8")
    assert load_benchmark(path).items[0].prompt == "Café ☕ — what is Paris?"


# ---------- item validation ----------


def test_an_unknown_field_is_rejected() -> None:
    with pytest.raises(DatasetError, match="not permitted"):
        parse_benchmark(jsonl(item(colour="blue")))


def test_the_id_must_start_with_the_category() -> None:
    with pytest.raises(DatasetError, match="should start with its category"):
        parse_benchmark(jsonl(item(id="math-01")))


@pytest.mark.parametrize("bad_id", ["QA-01", "qa_01", "qa-1", "qa-", "01"])
def test_ids_have_a_fixed_shape(bad_id: str) -> None:
    with pytest.raises(DatasetError):
        parse_benchmark(jsonl(item(id=bad_id)))


def test_a_blank_prompt_is_rejected() -> None:
    with pytest.raises(DatasetError):
        parse_benchmark(jsonl(item(prompt="   ")))


@pytest.mark.parametrize("max_tokens", [0, -5])
def test_max_tokens_must_be_positive(max_tokens: int) -> None:
    with pytest.raises(DatasetError):
        parse_benchmark(jsonl(item(max_tokens=max_tokens)))


@pytest.mark.parametrize(
    "scoring",
    [
        {"kind": "contains"},
        {"kind": "contains", "any_of": []},
        {"kind": "contains", "any_of": ["  "]},
        {"kind": "exact", "answer": ""},
        {"kind": "number", "answer": "twelve"},
        {"kind": "number", "answer": 1, "tolerance": -1},
        {"kind": "rubric", "criteria": []},
        {"kind": "vibes"},
        {"answer": "x"},
        {"kind": "exact", "answer": "x", "extra": 1},
    ],
)
def test_bad_scoring_is_rejected(scoring: dict[str, Any]) -> None:
    with pytest.raises(DatasetError):
        parse_benchmark(jsonl(item(scoring=scoring)))


def test_each_scoring_kind_is_accepted() -> None:
    scorings = [
        {"kind": "exact", "answer": "yes"},
        {"kind": "number", "answer": 4.5, "tolerance": 0.1},
        {"kind": "contains", "all_of": ["a"], "any_of": ["b", "c"]},
        {"kind": "rubric", "criteria": ["is correct"], "reference": "an answer"},
    ]
    for index, scoring in enumerate(scorings, start=1):
        parse_benchmark(jsonl(item(id=f"qa-{index:02d}", scoring=scoring)))


# ---------- using a benchmark ----------


def sample() -> Benchmark:
    return parse_benchmark(
        jsonl(
            item(id="qa-01", difficulty="easy"),
            item(id="qa-02", difficulty="hard"),
            item(
                id="math-01",
                category="math",
                difficulty="easy",
                scoring={"kind": "number", "answer": 2},
            ),
            item(
                id="math-02",
                category="math",
                difficulty="medium",
                scoring={"kind": "number", "answer": 3},
            ),
        )
    )


def test_subset_by_category_and_difficulty() -> None:
    assert [i.id for i in sample().subset(categories={"math"})] == ["math-01", "math-02"]
    assert [i.id for i in sample().subset(difficulties={"easy"})] == ["qa-01", "math-01"]
    both = sample().subset(categories={"qa"}, difficulties={"hard"})
    assert [i.id for i in both] == ["qa-02"]


def test_subset_limit_keeps_the_first_items() -> None:
    assert [i.id for i in sample().subset(limit=2)] == ["qa-01", "qa-02"]


def test_an_empty_subset_is_an_error_not_a_silent_nothing() -> None:
    with pytest.raises(DatasetError, match="no items"):
        sample().subset(categories={"code"})


def test_counts_and_shares() -> None:
    counts = sample().counts()
    assert counts["category"] == {"qa": 2, "math": 2}
    assert counts["difficulty"] == {"easy": 2, "hard": 1, "medium": 1}
    assert sample().share("easy") == 0.5


def test_get_finds_an_item_by_id() -> None:
    assert sample().get("math-02").difficulty == "medium"
    with pytest.raises(KeyError):
        sample().get("nope")


def test_a_system_prompt_comes_first() -> None:
    entry = BenchmarkItem.model_validate(item(system="Answer briefly."))
    roles = [m.role for m in entry.messages()]
    assert roles == ["system", "user"]


def test_requests_are_deterministic_unless_asked_otherwise() -> None:
    entry = BenchmarkItem.model_validate(item(max_tokens=50))
    request = entry.to_request("gpt-x")
    assert (request.temperature, request.max_tokens, request.model) == (0.0, 50, "gpt-x")
    assert entry.to_request("gpt-x", temperature=None).temperature is None
