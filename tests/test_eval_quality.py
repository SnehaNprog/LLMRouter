import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.evaluation import cli
from app.evaluation.dataset import (
    BenchmarkItem,
    Contains,
    Exact,
    Number,
    Rubric,
    load_benchmark,
)
from app.evaluation.judge import (
    SYSTEM_PROMPT,
    Judge,
    JudgeError,
    Verdict,
    build_prompt,
    parse_verdicts,
    score_from,
)
from app.evaluation.quality import (
    NON_ANSWER,
    ScoreRecord,
    load_scores,
    negative_control,
    quality_summary,
    score_deterministic,
    score_one,
    score_records,
)
from app.evaluation.runner import EvalRecord
from app.evaluation.scoring import (
    extract_number,
    normalize,
    score_contains,
    score_exact,
    score_number,
)

BENCH = load_benchmark()


# ---------- normalising and exact answers ----------


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("  Positive.  ", "positive"),
        ("**Positive**", "positive"),
        ("`positive`", "positive"),
        ('"Positive"', "positive"),
        ("'positive'.", "positive"),
        ("Alice,Bob ,  Carol", "alice, bob, carol"),
        ("two   spaces\nand a newline", "two spaces and a newline"),
        ("Lyon, France", "lyon, france"),
    ],
)
def test_normalising_makes_spellings_of_one_answer_comparable(raw: str, clean: str) -> None:
    assert normalize(raw) == clean


def test_case_can_be_kept() -> None:
    assert normalize("Release Notes", case_sensitive=True) == "Release Notes"


def exact(answer: str, case_sensitive: bool = False) -> Exact:
    return Exact(kind="exact", answer=answer, case_sensitive=case_sensitive)


@pytest.mark.parametrize("reply", ["positive", "Positive", " Positive.\n", "**positive**"])
def test_exact_accepts_the_answer_however_it_is_dressed(reply: str) -> None:
    assert score_exact(exact("positive"), reply).value == 1.0


@pytest.mark.parametrize(
    "reply", ["negative", "The sentiment is positive.", "positive positive", "", "positively"]
)
def test_exact_wants_the_answer_alone(reply: str) -> None:
    assert score_exact(exact("positive"), reply).value == 0.0


def test_exact_can_be_case_sensitive() -> None:
    scoring = exact("RELEASE NOTES", case_sensitive=True)
    assert score_exact(scoring, "RELEASE NOTES").value == 1.0
    assert score_exact(scoring, "release notes").value == 0.0


def test_a_missed_exact_answer_says_what_was_expected() -> None:
    assert "expected 'positive'" in score_exact(exact("positive"), "no").detail


def test_lists_match_whatever_the_spacing_after_commas() -> None:
    assert score_exact(exact("Alice, Bob, Carol, Dave"), "Alice,Bob,Carol,Dave").value == 1.0
    assert score_exact(exact("Alice, Bob, Carol, Dave"), "Bob, Alice, Carol, Dave").value == 0.0


# ---------- numbers ----------


@pytest.mark.parametrize(
    ("reply", "number"),
    [
        ("72", 72),
        ("$1,249.50", 1249.5),
        ("1,249.50.", 1249.5),
        ("15%", 15),
        ("-3", -3),
        ("0.3571", 0.3571),
        (".5", 0.5),
        ("The answer is 43.", 43),
        ("17 + 26 = 43", 43),  # an answer states its conclusion last
        ("About 3.6 dollars", 3.6),
        ("5.", 5),
    ],
)
def test_the_number_in_a_reply(reply: str, number: float) -> None:
    assert extract_number(reply) == pytest.approx(number)


@pytest.mark.parametrize("reply", ["", "no idea", "many", "..."])
def test_a_reply_without_a_number_has_none(reply: str) -> None:
    assert extract_number(reply) is None


def number(answer: float, tolerance: float = 1e-6) -> Number:
    return Number(kind="number", answer=answer, tolerance=tolerance)


def test_a_number_within_the_tolerance_is_right() -> None:
    assert score_number(number(0.3571, 0.00006), "0.3571").value == 1.0
    assert score_number(number(0.3571, 0.00006), "0.35714").value == 1.0
    assert score_number(number(0.3571, 0.00006), "0.357").value == 0.0
    assert score_number(number(11576.25, 0.005), "$11,576.25").value == 1.0


def test_a_number_exactly_at_the_tolerance_is_right() -> None:
    assert score_number(number(10, 0.5), "10.5").value == 1.0
    assert score_number(number(10, 0.5), "9.5").value == 1.0
    assert score_number(number(10, 0.5), "10.6").value == 0.0


def test_a_wrong_or_missing_number_scores_zero_and_says_why() -> None:
    wrong = score_number(number(72), "73")
    assert wrong.value == 0.0 and "expected 72, got 73" in wrong.detail
    assert "no number" in score_number(number(72), "seventy-two").detail


# ---------- contains ----------


def contains(all_of: list[str] | None = None, any_of: list[str] | None = None) -> Contains:
    return Contains(kind="contains", all_of=all_of or [], any_of=any_of or [])


def test_terms_match_whole_words_only() -> None:
    assert score_contains(contains(any_of=["au"]), "The symbol is Au.").value == 1.0
    assert score_contains(contains(any_of=["au"]), "Australia").value == 0.0
    assert score_contains(contains(any_of=["7"]), "Seven, or 7").value == 1.0
    assert score_contains(contains(any_of=["7"]), "17 and 27").value == 0.0


def test_matching_ignores_case_and_handles_phrases() -> None:
    assert score_contains(contains(any_of=["carbon dioxide"]), "Carbon Dioxide (CO2)").value == 1.0
    assert score_contains(contains(any_of=["co₂"]), "Takes in CO₂ from the air").value == 1.0


def test_every_all_of_term_is_needed_and_one_any_of_term_will_do() -> None:
    scoring = contains(all_of=["red", "green"], any_of=["blue", "violet"])
    assert score_contains(scoring, "Red, green and blue").value == 1.0
    assert score_contains(scoring, "red green violet").value == 1.0
    assert "missing ['green']" in score_contains(scoring, "red and blue").detail
    assert "none of" in score_contains(scoring, "red and green").detail


# ---------- the shipped benchmark's checkable items ----------


def right_reply(item: BenchmarkItem) -> str:
    scoring = item.scoring
    if isinstance(scoring, Exact):
        return scoring.answer
    if isinstance(scoring, Number):
        return (
            f"{scoring.answer:g}" if scoring.answer == int(scoring.answer) else str(scoring.answer)
        )
    assert isinstance(scoring, Contains)
    return " ".join([*scoring.all_of, *scoring.any_of[:1]])


def test_the_right_answer_scores_full_marks_on_every_checkable_item() -> None:
    for item in BENCH:
        if not isinstance(item.scoring, Rubric):
            checked = score_deterministic(item, right_reply(item))
            assert checked is not None and checked.value == 1.0, item.id


def test_a_non_answer_scores_zero_on_every_checkable_item() -> None:
    """A blind control: nothing that says nothing should ever pass a mechanical check."""
    for item in BENCH:
        if not isinstance(item.scoring, Rubric):
            for reply in (NON_ANSWER, "", "I don't know", "Sorry, I can't help with that."):
                checked = score_deterministic(item, reply)
                assert checked is not None and checked.value == 0.0, (item.id, reply)


def test_open_ended_items_are_left_to_the_judge() -> None:
    rubric_item = next(i for i in BENCH if isinstance(i.scoring, Rubric))
    assert score_deterministic(rubric_item, "anything") is None


# ---------- the judge: prompt and verdicts ----------


def rubric_item(criteria: int = 3, reference: str | None = None) -> tuple[BenchmarkItem, Rubric]:
    scoring = Rubric(
        kind="rubric",
        criteria=[f"criterion number {n}" for n in range(1, criteria + 1)],
        reference=reference,
    )
    item = BenchmarkItem(
        id="qa-01", category="qa", difficulty="easy", prompt="Why is the sky blue?", scoring=scoring
    )
    return item, scoring


def test_the_prompt_shows_the_question_the_numbered_criteria_and_the_answer() -> None:
    item, rubric = rubric_item(2)
    prompt = build_prompt(item, rubric, "Because of scattering.")
    assert "Why is the sky blue?" in prompt
    assert "1. criterion number 1\n2. criterion number 2" in prompt
    assert prompt.endswith("ANSWER TO GRADE:\nBecause of scattering.")
    assert "REFERENCE" not in prompt


def test_a_reference_answer_is_included_when_there_is_one() -> None:
    item, rubric = rubric_item(1, reference="Rayleigh scattering.")
    assert "REFERENCE ANSWER" in build_prompt(item, rubric, "x")
    assert "Rayleigh scattering." in build_prompt(item, rubric, "x")


def test_the_instructions_ask_for_strict_blind_grading_in_json() -> None:
    assert "not told who or what wrote the answer" in SYSTEM_PROMPT
    assert "Do not reward length" in SYSTEM_PROMPT
    assert '"verdicts"' in SYSTEM_PROMPT


def verdicts_json(*met: bool) -> str:
    return json.dumps(
        {"verdicts": [{"criterion": n, "met": m, "reason": "r"} for n, m in enumerate(met, 1)]}
    )


def test_verdicts_are_read_from_plain_json() -> None:
    parsed = parse_verdicts(verdicts_json(True, False, True), 3)
    assert [v.met for v in parsed] == [True, False, True]
    assert [v.criterion for v in parsed] == [1, 2, 3]


def test_verdicts_are_read_from_a_fenced_or_chatty_reply() -> None:
    fenced = f"```json\n{verdicts_json(True, False)}\n```"
    chatty = f"Sure! Here is my grading:\n{verdicts_json(True, False)}\nHope that helps."
    for text in (fenced, chatty):
        assert [v.met for v in parse_verdicts(text, 2)] == [True, False]


def test_a_fence_rescues_a_reply_with_stray_braces_around_it() -> None:
    text = (
        f"I used the {{criterion}} format.\n```json\n{verdicts_json(True, False)}\n```\nDone {{}}"
    )
    assert [v.met for v in parse_verdicts(text, 2)] == [True, False]


def test_verdicts_come_back_in_criterion_order_whatever_the_judge_wrote() -> None:
    shuffled = json.dumps(
        {"verdicts": [{"criterion": 2, "met": False}, {"criterion": 1, "met": True}]}
    )
    assert [v.met for v in parse_verdicts(shuffled, 2)] == [True, False]


@pytest.mark.parametrize(
    ("text", "complaint"),
    [
        ("no json here", "no JSON"),
        ("[1, 2]", "no 'verdicts' list"),
        ('{"verdicts": "yes"}', "no 'verdicts' list"),
        ('{"verdicts": [1]}', "not an object"),
        ('{"verdicts": [{"met": true}]}', "no criterion number"),
        ('{"verdicts": [{"criterion": true, "met": true}]}', "no criterion number"),
        ('{"verdicts": [{"criterion": 1, "met": "true"}]}', "not true or false"),
        ('{"verdicts": [{"criterion": 1}]}', "not true or false"),
        ('{"verdicts": [{"criterion": 4, "met": true}]}', "does not exist"),
        ('{"verdicts": [{"criterion": 0, "met": true}]}', "does not exist"),
        (
            '{"verdicts": [{"criterion": 1, "met": true}, {"criterion": 1, "met": false}]}',
            "twice",
        ),
        ('{"verdicts": [{"criterion": 1, "met": true}]}', "did not grade criteria [2, 3]"),
        ('{"verdicts": []}', "did not grade"),
    ],
)
def test_a_verdict_that_cannot_be_trusted_is_refused(text: str, complaint: str) -> None:
    with pytest.raises(JudgeError, match=re.escape(complaint)):
        parse_verdicts(text, 3)


def test_the_score_is_the_share_of_criteria_met() -> None:
    verdicts = [
        Verdict(1, True, ""),
        Verdict(2, False, ""),
        Verdict(3, True, ""),
        Verdict(4, True, ""),
    ]
    score = score_from(verdicts)
    assert score.value == 0.75
    assert score.detail == "3/4 criteria met (not met: 2)"
    assert score_from([Verdict(1, True, "")]).detail == "1/1 criteria met"
    assert score_from([Verdict(1, False, "")]).value == 0.0


# ---------- the judge: talking to the gateway ----------


class Gateway:
    """A scripted judge endpoint."""

    def __init__(
        self, *replies: httpx.Response | Callable[[httpx.Request], httpx.Response]
    ) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply(request) if callable(reply) else reply

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


def says(content: str | None) -> httpx.Response:
    body = {
        "id": "x",
        "object": "chat.completion",
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
    }
    return httpx.Response(200, json=body)


class Sleeps:
    async def sleep(self, seconds: float) -> None:
        pass


def judge_for(gateway: Gateway, **kwargs: Any) -> tuple[Judge, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(gateway), base_url="http://gw")
    return Judge(client, "grader", sleep=Sleeps().sleep, **kwargs), client


async def test_the_judge_asks_its_model_at_temperature_zero_with_the_rubric() -> None:
    gateway = Gateway(says(verdicts_json(True, True)))
    judge, client = judge_for(gateway)
    item, rubric = rubric_item(2)

    score = await judge.grade(item, rubric, "my answer")

    body = gateway.bodies()[0]
    assert body["model"] == "grader" and body["temperature"] == 0.0
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "my answer" in body["messages"][1]["content"]
    assert body["max_tokens"] >= 200
    assert score.value == 1.0
    await client.aclose()


async def test_the_judge_knows_nothing_about_which_system_wrote_the_answer() -> None:
    gateway = Gateway(says(verdicts_json(True)))
    judge, client = judge_for(gateway)
    item, rubric = rubric_item(1)
    await judge.grade(item, rubric, "an answer")
    sent = json.dumps(gateway.bodies()[0]).lower()
    for giveaway in ("baseline", "router", "frontier", "slm", "auto"):
        assert giveaway not in sent
    await client.aclose()


async def test_an_unreadable_reply_is_asked_again_once() -> None:
    gateway = Gateway(says("I think it's fine!"), says(verdicts_json(True, False)))
    judge, client = judge_for(gateway)
    item, rubric = rubric_item(2)

    score = await judge.grade(item, rubric, "x")

    assert score.value == 0.5 and len(gateway.requests) == 2
    await client.aclose()


async def test_a_judge_that_keeps_rambling_is_an_error_not_a_guess() -> None:
    gateway = Gateway(says("no json"))
    judge, client = judge_for(gateway)
    item, rubric = rubric_item(2)
    with pytest.raises(JudgeError, match="no JSON"):
        await judge.grade(item, rubric, "x")
    assert len(gateway.requests) == 2  # the original plus one more try
    await client.aclose()


async def test_no_format_retries_means_one_try() -> None:
    gateway = Gateway(says("no json"))
    judge, client = judge_for(gateway, format_retries=0)
    item, rubric = rubric_item(1)
    with pytest.raises(JudgeError):
        await judge.grade(item, rubric, "x")
    assert len(gateway.requests) == 1
    await client.aclose()


async def test_transient_gateway_errors_are_retried() -> None:
    gateway = Gateway(httpx.Response(503, json={}), says(verdicts_json(True)))
    judge, client = judge_for(gateway)
    item, rubric = rubric_item(1)
    assert (await judge.grade(item, rubric, "x")).value == 1.0
    assert len(gateway.requests) == 2
    await client.aclose()


async def test_a_refusal_by_the_gateway_is_an_error() -> None:
    gateway = Gateway(httpx.Response(403, json={"error": {"message": "model not allowed"}}))
    judge, client = judge_for(gateway)
    item, rubric = rubric_item(1)
    with pytest.raises(JudgeError, match="HTTP 403: model not allowed"):
        await judge.grade(item, rubric, "x")
    await client.aclose()


async def test_an_unreachable_gateway_is_an_error() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    judge, client = judge_for(Gateway(boom), retries=0)
    item, rubric = rubric_item(1)
    with pytest.raises(JudgeError, match="could not reach the judge"):
        await judge.grade(item, rubric, "x")
    await client.aclose()


@pytest.mark.parametrize("content", [None, "", "   "])
async def test_an_empty_judge_reply_is_an_error(content: str | None) -> None:
    judge, client = judge_for(Gateway(says(content)), format_retries=0)
    item, rubric = rubric_item(1)
    with pytest.raises(JudgeError, match="no content"):
        await judge.grade(item, rubric, "x")
    await client.aclose()


async def test_a_malformed_gateway_response_is_an_error() -> None:
    judge, client = judge_for(Gateway(httpx.Response(200, json={"nope": 1})), format_retries=0)
    item, rubric = rubric_item(1)
    with pytest.raises(JudgeError, match="malformed"):
        await judge.grade(item, rubric, "x")
    await client.aclose()


# ---------- scoring recorded answers ----------


def record(item_id: str, reply: str | None = "an answer", **kwargs: Any) -> EvalRecord:
    fields: dict[str, Any] = {
        "item_id": item_id,
        "run": "baseline",
        "requested_model": "smart",
        "ok": reply is not None,
        "reply": reply,
    }
    return EvalRecord(**{**fields, **kwargs})


def all_met_judge(n_met: int | None = None) -> Gateway:
    """A judge that grades every criterion of any prompt as met (or the first ``n_met``)."""

    def reply(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][1]["content"]
        count = len(re.findall(r"^\d+\. ", prompt.split("CRITERIA:")[1], re.MULTILINE))
        met = [True] * count if n_met is None else [i < n_met for i in range(count)]
        return says(verdicts_json(*met))

    return Gateway(reply)


async def test_a_failed_request_scores_zero_without_asking_anyone() -> None:
    item = BENCH.get("math-01")
    failed = record("math-01", None, error="HTTP 503: down")
    scored = await score_one(item, failed, None)
    assert (scored.method, scored.score, scored.detail) == ("no_answer", 0.0, "HTTP 503: down")


async def test_checkable_answers_need_no_judge() -> None:
    right = await score_one(BENCH.get("math-01"), record("math-01", "43"), None)
    wrong = await score_one(BENCH.get("math-01"), record("math-01", "44"), None)
    assert (right.method, right.score, right.judge_model) == ("number", 1.0, None)
    assert wrong.score == 0.0


async def test_an_open_ended_answer_without_a_judge_is_unscored_not_zero() -> None:
    scored = await score_one(BENCH.get("chat-01"), record("chat-01"), None)
    assert scored.score is None and scored.error == "no judge configured"


async def test_a_judge_failure_leaves_the_answer_unscored_not_zero() -> None:
    judge, client = judge_for(Gateway(httpx.Response(403, json={})))
    scored = await score_one(BENCH.get("chat-01"), record("chat-01"), judge)
    assert scored.score is None and "the judge failed" in (scored.error or "")
    assert scored.judge_model == "grader"
    await client.aclose()


async def test_a_judged_answer_carries_the_judge_and_the_detail() -> None:
    judge, client = judge_for(all_met_judge(n_met=1))
    item = BENCH.get("chat-01")  # two criteria
    scored = await score_one(item, record("chat-01"), judge)
    assert scored.method == "rubric" and scored.score == 0.5
    assert scored.judge_model == "grader" and "1/2 criteria met" in scored.detail
    await client.aclose()


async def test_scores_come_back_in_the_order_of_the_records() -> None:
    judge, client = judge_for(all_met_judge())
    records = [record("chat-01"), record("math-01", "43"), record("math-02", "70")]
    scores = await score_records(records, BENCH, judge)
    assert [s.item_id for s in scores] == ["chat-01", "math-01", "math-02"]
    assert [s.score for s in scores] == [1.0, 1.0, 0.0]
    await client.aclose()


async def test_scores_are_written_as_they_are_known(tmp_path: Path) -> None:
    judge, client = judge_for(all_met_judge())
    out = tmp_path / "deep" / "b.scores.jsonl"
    seen: list[str] = []
    await score_records(
        [record("chat-01"), record("math-01", "43")],
        BENCH,
        judge,
        out=out,
        on_score=lambda s: seen.append(s.item_id),
    )
    assert sorted(s.item_id for s in load_scores(out)) == ["chat-01", "math-01"]
    assert sorted(seen) == ["chat-01", "math-01"]
    await client.aclose()


async def test_resume_pays_only_for_what_is_missing(tmp_path: Path) -> None:
    out = tmp_path / "s.jsonl"
    records = [record("chat-01"), record("chat-02"), record("chat-03")]
    first = Gateway(
        all_met_judge().replies[0], httpx.Response(403, json={}), all_met_judge().replies[0]
    )
    judge, client = judge_for(first)
    await score_records(records, BENCH, judge, out=out, concurrency=1)
    await client.aclose()
    assert sum(1 for s in load_scores(out) if s.score is None) == 1

    second = all_met_judge()
    judge, client = judge_for(second)
    scores = await score_records(records, BENCH, judge, out=out, resume=True)
    await client.aclose()

    assert len(second.requests) == 1  # only the one the judge had failed on
    assert all(s.score == 1.0 for s in scores)


async def test_a_fresh_run_replaces_an_old_scores_file(tmp_path: Path) -> None:
    out = tmp_path / "s.jsonl"
    await score_records([record("math-01", "43"), record("math-02", "72")], BENCH, None, out=out)
    await score_records([record("math-01", "43")], BENCH, None, out=out)
    assert len(out.read_text().splitlines()) == 1


async def test_results_for_an_unknown_item_are_refused() -> None:
    with pytest.raises(ValueError, match="not in the benchmark"):
        await score_records([record("qa-99")], BENCH, None)


async def test_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError):
        await score_records([], BENCH, None, concurrency=0)


async def test_no_more_judgements_run_at_once_than_the_concurrency() -> None:
    running = peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return says(verdicts_json(True, True))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://gw")
    judge = Judge(client, "grader")
    await score_records([record(f"chat-{n:02d}") for n in range(1, 9)], BENCH, judge, concurrency=2)
    assert peak == 2
    await client.aclose()


# ---------- the negative control ----------


async def test_the_control_grades_a_non_answer_for_each_open_ended_item() -> None:
    gateway = Gateway(lambda r: says(verdicts_json(*[False] * 8)))  # more verdicts than needed
    gateway.replies = [
        lambda r: says(
            verdicts_json(
                *[False]
                * len(
                    re.findall(
                        r"^\d+\. ",
                        json.loads(r.content)["messages"][1]["content"].split("CRITERIA:")[1],
                        re.MULTILINE,
                    )
                )
            )
        )
    ]
    judge, client = judge_for(gateway)

    scores = await negative_control(BENCH, judge)

    rubric_ids = {i.id for i in BENCH if isinstance(i.scoring, Rubric)}
    assert {s.item_id for s in scores} == rubric_ids
    assert all(s.score == 0.0 and s.run == "negative-control" for s in scores)
    assert all(NON_ANSWER in body["messages"][1]["content"] for body in gateway.bodies())
    await client.aclose()


async def test_a_generous_judge_shows_up_in_the_control() -> None:
    judge, client = judge_for(all_met_judge())
    scores = await negative_control(BENCH.subset(categories={"chat"}), judge)
    assert all(s.score == 1.0 for s in scores)  # nothing was withheld, and it shows
    await client.aclose()


# ---------- summaries and files ----------


def sc(item_id: str, method: str, score: float | None, run: str = "baseline") -> ScoreRecord:
    return ScoreRecord.model_validate(
        {"item_id": item_id, "run": run, "method": method, "score": score}
    )


def test_the_summary_averages_overall_and_by_group() -> None:
    scores = [
        sc("math-01", "number", 1.0),  # math, easy
        sc("math-10", "number", 0.0),  # math, hard
        sc("chat-01", "rubric", 0.5),  # chat, easy
        sc("code-01", "no_answer", 0.0),  # code, easy: counts as 0
        sc("qa-09", "rubric", None),  # unscored: left out of every mean
    ]
    summary = quality_summary(scores, BENCH)

    assert (summary.items, summary.scored, summary.unscored, summary.no_answer) == (5, 4, 1, 1)
    assert summary.mean == pytest.approx(1.5 / 4)
    assert summary.by_category["math"].mean == 0.5 and summary.by_category["math"].n == 2
    assert summary.by_difficulty["easy"].n == 3
    assert summary.by_method["rubric"].mean == 0.5
    assert "qa" not in summary.by_category  # its only item was unscored


def test_an_empty_summary_has_no_mean() -> None:
    summary = quality_summary([], BENCH)
    assert summary.mean is None and summary.items == 0


def test_when_an_item_is_scored_twice_the_last_score_wins(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text(
        sc("chat-01", "rubric", None).model_dump_json()
        + "\n\n"
        + sc("chat-01", "rubric", 1.0).model_dump_json()
        + "\n"
    )
    loaded = load_scores(path)
    assert len(loaded) == 1 and loaded[0].score == 1.0


def test_a_corrupt_scores_file_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text(sc("chat-01", "rubric", 1.0).model_dump_json() + "\n{oops\n")
    with pytest.raises(ValueError, match="line 2"):
        load_scores(path)


# ---------- the command ----------


@pytest.fixture
def mock_gateway(monkeypatch: pytest.MonkeyPatch) -> Callable[[Gateway], None]:
    real = httpx.AsyncClient

    def install(gateway: Gateway) -> None:
        def factory(**kwargs: Any) -> httpx.AsyncClient:
            return real(transport=httpx.MockTransport(gateway), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    return install


def write_results(tmp_path: Path, name: str = "baseline", **overrides: Any) -> Path:
    path = tmp_path / f"{name}.jsonl"
    rows = [
        record("math-01", "43", run=name, **overrides),
        record("math-02", "70", run=name, **overrides),
        record("chat-01", "Hello there!", run=name, **overrides),
    ]
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    return path


def run_score(tmp_path: Path, *extra: str, results: Path | None = None) -> int:
    return cli.main(
        [
            "score",
            str(results or write_results(tmp_path)),
            "--judge-model",
            "grader",
            "--api-key",
            "k",
            *extra,
        ]
    )


def test_the_score_command_grades_a_results_file(
    tmp_path: Path, mock_gateway: Callable[[Gateway], None], capsys: pytest.CaptureFixture[str]
) -> None:
    mock_gateway(all_met_judge())
    results = write_results(tmp_path)

    code = run_score(tmp_path, results=results)

    assert code == 0
    out = capsys.readouterr().out
    assert "quality of 'baseline': mean 0.667 over 3/3 items" in out
    assert "number" in out and "rubric" in out
    scores = load_scores(tmp_path / "baseline.scores.jsonl")
    assert {s.item_id: s.score for s in scores} == {"math-01": 1.0, "math-02": 0.0, "chat-01": 1.0}


def test_unscored_answers_fail_the_command_unless_allowed(
    tmp_path: Path, mock_gateway: Callable[[Gateway], None], capsys: pytest.CaptureFixture[str]
) -> None:
    mock_gateway(Gateway(httpx.Response(403, json={"error": {"message": "no"}})))
    assert run_score(tmp_path) == 1
    assert "quality is unknown" in capsys.readouterr().err
    assert run_score(tmp_path, "--allow-unscored") == 0


def test_the_negative_control_is_reported(
    tmp_path: Path, mock_gateway: Callable[[Gateway], None], capsys: pytest.CaptureFixture[str]
) -> None:
    mock_gateway(all_met_judge())
    run_score(tmp_path, "--negative-control")
    out = capsys.readouterr().out
    assert "negative control: the judge gave a non-answer 1.000 on average" in out
    assert "42 open-ended items" in out and "42 scored above zero" in out


def test_a_judge_that_also_wrote_answers_is_flagged(
    tmp_path: Path, mock_gateway: Callable[[Gateway], None], capsys: pytest.CaptureFixture[str]
) -> None:
    mock_gateway(all_met_judge())
    results = write_results(tmp_path, served_model="grader")
    run_score(tmp_path, results=results)
    assert "also wrote answers" in capsys.readouterr().err


def test_several_files_are_scored_in_turn(
    tmp_path: Path, mock_gateway: Callable[[Gateway], None], capsys: pytest.CaptureFixture[str]
) -> None:
    mock_gateway(all_met_judge())
    first, second = write_results(tmp_path, "baseline"), write_results(tmp_path, "router")
    code = cli.main(["score", str(first), str(second), "--judge-model", "grader", "--api-key", "k"])
    assert code == 0
    out = capsys.readouterr().out
    assert "quality of 'baseline'" in out and "quality of 'router'" in out
    assert (tmp_path / "router.scores.jsonl").exists()


def test_resuming_does_not_ask_the_judge_again(
    tmp_path: Path, mock_gateway: Callable[[Gateway], None]
) -> None:
    mock_gateway(all_met_judge())
    results = write_results(tmp_path)
    run_score(tmp_path, results=results)
    again = all_met_judge()
    mock_gateway(again)
    assert run_score(tmp_path, "--resume", results=results) == 0
    assert again.requests == []


def test_bad_inputs_are_refused_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(["score", str(tmp_path / "nope.jsonl"), "--judge-model", "g", "--api-key", "k"])
        == 2
    )
    assert capsys.readouterr().err.startswith("error:")
    stranger = tmp_path / "r.jsonl"
    stranger.write_text(record("qa-99").model_dump_json() + "\n")
    assert cli.main(["score", str(stranger), "--judge-model", "g", "--api-key", "k"]) == 2
    assert "not in the benchmark" in capsys.readouterr().err
    assert cli.main(["score", str(stranger), "--judge-model", "g"]) == 2


def test_a_judge_model_is_required() -> None:
    with pytest.raises(SystemExit):
        cli.main(["score", "x.jsonl", "--api-key", "k"])
