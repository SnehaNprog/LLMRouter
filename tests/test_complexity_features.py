import time
from typing import Any

import pytest

from app.routing.features import ComplexityFeatures, extract_features
from app.schemas.chat import ChatCompletionRequest


def req(content: Any = "hello", **extra: Any) -> ChatCompletionRequest:
    messages = extra.pop("messages", [{"role": "user", "content": content}])
    return ChatCompletionRequest.model_validate({"model": "auto", "messages": messages, **extra})


def features(content: Any = "hello", **extra: Any) -> ComplexityFeatures:
    return extract_features(req(content, **extra))


# ---------- length and shape ----------


def test_an_empty_ask_is_all_zeros() -> None:
    f = features("")

    assert (f.prompt_tokens, f.last_user_tokens, f.reasoning_cues, f.math_signals) == (0, 0, 0, 0)
    assert not f.has_code and not f.structured_output and not f.small_talk


def test_longer_prompts_measure_longer() -> None:
    short, long = features("hello there"), features("hello there " * 200)

    assert long.prompt_tokens > short.prompt_tokens * 50
    assert long.last_user_tokens == long.prompt_tokens


def test_the_final_user_turn_is_measured_separately_from_the_conversation() -> None:
    f = extract_features(
        req(
            messages=[
                {"role": "system", "content": "You are a careful assistant. " * 20},
                {"role": "user", "content": "first question " * 30},
                {"role": "assistant", "content": "an answer " * 30},
                {"role": "user", "content": "ok"},
            ]
        )
    )

    assert f.last_user_tokens == 1
    assert f.prompt_tokens > 100
    assert f.system_tokens > 100
    assert f.turns == 2


def test_turns_count_user_messages_only() -> None:
    f = extract_features(
        req(
            messages=[
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
                {"role": "user", "content": "e"},
            ]
        )
    )

    assert f.turns == 3


def test_with_no_user_message_the_final_turn_is_empty() -> None:
    f = extract_features(req(messages=[{"role": "system", "content": "be brief"}]))

    assert f.last_user_tokens == 0 and f.turns == 0 and f.system_tokens > 0


def test_multimodal_text_parts_are_read_and_images_counted() -> None:
    f = extract_features(
        req(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in these pictures? Compare them."},
                        {"type": "image_url", "image_url": {"url": "https://x/a.png"}},
                        {"type": "image_url", "image_url": {"url": "https://x/b.png"}},
                    ],
                }
            ]
        )
    )

    assert f.image_count == 2
    assert f.reasoning_cues == 1  # "compare", read from the text part


def test_a_message_with_only_tool_calls_has_no_text_and_does_not_break() -> None:
    f = extract_features(
        req(
            messages=[
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c", "function": {"name": "w", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c", "content": "18C"},
            ]
        )
    )

    assert f.turns == 1 and f.last_user_tokens > 0


# ---------- code ----------


def test_fenced_code_is_detected() -> None:
    f = features("Fix this:\n```python\ndef add(a, b):\n    return a - b\n```\nWhy does it fail?")

    assert f.has_code and f.code_blocks == 1
    assert f.code_line_ratio == 0.5  # two code lines of four non-empty lines


def test_several_blocks_are_counted() -> None:
    f = features("```a```\ntext\n```\nx = 1\n```\n```\ny = 2\n```")

    assert f.code_blocks == 3


def test_an_unclosed_fence_still_counts_as_code() -> None:
    f = features("Look:\n```js\nconst x = 1;")

    assert f.has_code and f.code_blocks == 1


@pytest.mark.parametrize(
    "snippet",
    [
        "def f(x):\n    return x\n\nclass A:\n    pass",
        "function go() {\n  return 1;\n}",
        "SELECT name FROM users WHERE id = 1;",
        '<div class="a">\n  <span>hi</span>\n</div>',
        "const add = (a, b) => a + b;\nconsole.log(add(1, 2));",
    ],
)
def test_unfenced_code_is_recognised_by_its_shape(snippet: str) -> None:
    assert features(snippet).has_code


@pytest.mark.parametrize(
    "prose",
    [
        "Write me a short poem about autumn leaves.",
        "What is the capital of France?",
        "I would like a recipe for banana bread, please. It should be moist; not too sweet.",
        "Summarise the plot of Hamlet in three sentences.",
    ],
)
def test_ordinary_prose_is_not_mistaken_for_code(prose: str) -> None:
    f = features(prose)

    assert not f.has_code
    assert f.code_blocks == 0


def test_code_from_earlier_in_the_conversation_still_counts() -> None:
    f = extract_features(
        req(
            messages=[
                {"role": "user", "content": "```python\ndef f():\n    return 1\n```"},
                {"role": "assistant", "content": "Looks fine."},
                {"role": "user", "content": "now make it faster"},
            ]
        )
    )

    assert f.has_code  # "it" is code the user pasted two turns ago


@pytest.mark.parametrize(
    "text",
    [
        "Write a Python function that checks if a number is prime.",
        "Implement a binary search algorithm.",
        "Can you create a script to rename these files?",
        "Generate a SQL query for monthly revenue.",
        "Build an endpoint that returns the current user.",
        "write unit tests for the parser",
        "Port this to Rust",
        "Convert the loop using python",
    ],
)
def test_requests_for_code_are_recognised_even_when_none_is_pasted(text: str) -> None:
    f = features(text)

    assert f.code_request
    assert not f.has_code  # nothing was pasted: this is the case the flag exists for


@pytest.mark.parametrize(
    "text",
    [
        "Write a poem about autumn.",
        "What is a function in mathematics?",
        "Tell me about the Python snake.",
        "Create a meal plan for the week.",
        "Explain how algorithms are taught.",
        "hello",
    ],
)
def test_ordinary_requests_are_not_code_requests(text: str) -> None:
    assert not features(text).code_request


def test_only_the_final_turn_decides_whether_code_is_being_requested() -> None:
    f = extract_features(
        req(
            messages=[
                {"role": "user", "content": "Write a Python function to sort a list."},
                {"role": "assistant", "content": "Here you go."},
                {"role": "user", "content": "thanks"},
            ]
        )
    )

    assert not f.code_request


# ---------- maths ----------


@pytest.mark.parametrize(
    "text",
    [
        "Solve $x^2 - 5x + 6 = 0$.",
        "Simplify $a$ and then $b+c$.",
        "Evaluate \\int_0^1 x^2 dx using \\frac{1}{3}.",
        "Find the derivative of the polynomial.",
        "Prove the theorem.",
        "What is ∑ i from 1 to n, and √2 ≈ 1.41?",
        "Compute 12 * 7 = ?",
    ],
)
def test_maths_is_detected(text: str) -> None:
    assert features(text).math_signals >= 1


@pytest.mark.parametrize(
    "text",
    [
        "Tell me about the history of Rome.",
        "I paid $5 for lunch and $7 for dinner.",
        "It costs $10 or $12 depending on the size.",
        "Send $ to my account, not $",
        "hello",
    ],
)
def test_everyday_text_has_no_maths_signals(text: str) -> None:
    assert features(text).math_signals == 0


def test_more_maths_means_more_signals() -> None:
    plain = features("Solve $x + 1 = 2$.")
    heavy = features("Prove the theorem: the integral ∫ of the derivative, with $a^2 + b^2 = c^2$.")

    assert heavy.math_signals > plain.math_signals


# ---------- reasoning, constraints, structure ----------


def test_each_kind_of_reasoning_cue_is_counted() -> None:
    f = features("Compare A and B, compare C and D, and analyze the trade-offs step by step.")

    assert f.reasoning_cues == 4  # compare, analyze, trade-offs, step by step


def test_reasoning_cues_are_distinct_not_repeated() -> None:
    assert features("compare compare compare").reasoning_cues == 1
    assert features("compare, analyze, and justify").reasoning_cues == 3


def test_cues_match_case_insensitively_and_on_word_boundaries() -> None:
    assert features("EXPLAIN WHY this happens").reasoning_cues == 1
    assert features("The comparer and the designer").reasoning_cues == 0  # not whole words


def test_only_the_final_user_turn_is_read_for_cues() -> None:
    f = extract_features(
        req(
            messages=[
                {"role": "user", "content": "analyze, compare, prove, and justify everything"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "thanks"},
            ]
        )
    )

    assert f.reasoning_cues == 0


def test_constraints_prefer_the_longer_phrase() -> None:
    f = features("It must handle load and must not lose data. Use exactly three items, only once.")

    assert f.constraint_cues == 4  # must not, must, exactly, only


def test_structured_output_requests_are_detected() -> None:
    assert features("Respond in JSON with keys a and b.").structured_output
    assert features("Return the results as a table.").structured_output
    assert features("Please output a YAML schema for this.").structured_output
    assert not features("Tell me a story about Jason.").structured_output
    assert not features("What is JSON?").structured_output  # asking about it is not asking for it


def test_multi_part_asks_are_counted() -> None:
    f = features("Please:\n1. Do this\n2. Then that\n- also this\n* and this\nthanks")

    assert f.list_items == 4


def test_question_marks_are_counted_in_the_final_turn() -> None:
    assert features("Why? How? When?").question_marks == 3
    assert features("no questions here.").question_marks == 0


# ---------- small talk ----------


@pytest.mark.parametrize(
    "text", ["hi", "Hello!", "thanks", "Thank you so much", "ok", "good morning", "how are you?"]
)
def test_greetings_and_acknowledgements_are_small_talk(text: str) -> None:
    assert features(text).small_talk


@pytest.mark.parametrize(
    "text",
    [
        "hi, can you prove that the square root of 2 is irrational and explain each step in depth?",
        "Write a function that parses dates",
        "What is the capital of France?",
    ],
)
def test_real_requests_are_not_small_talk(text: str) -> None:
    assert not features(text).small_talk


# ---------- tools, output length, language ----------


def test_offered_tools_are_counted() -> None:
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}

    assert features(tools=[tool, tool]).tool_count == 2
    assert features().tool_count == 0


def test_forced_tool_use_is_flagged() -> None:
    tool = {"type": "function", "function": {"name": "f"}}
    named = {"type": "function", "function": {"name": "f"}}

    assert features(tools=[tool], tool_choice="required").forces_tool
    assert features(tools=[tool], tool_choice=named).forces_tool
    assert not features(tools=[tool], tool_choice="auto").forces_tool
    assert not features(tools=[tool], tool_choice="none").forces_tool
    assert not features(tools=[tool]).forces_tool


def test_the_requested_answer_length_is_reported() -> None:
    assert features(max_tokens=4000).requested_completion_tokens == 4000
    assert features().requested_completion_tokens is None


def test_non_ascii_text_is_measured() -> None:
    assert features("hello world").non_ascii_ratio == 0.0
    assert features("你好世界").non_ascii_ratio == 1.0
    assert 0.0 < features("hello 世界").non_ascii_ratio < 0.5


# ---------- robustness ----------


def test_the_same_request_always_gives_the_same_features() -> None:
    request = req("Prove it step by step:\n```py\nx = 1\n```\nRespond in JSON.")

    assert extract_features(request) == extract_features(request)


def test_the_request_is_not_modified() -> None:
    request = req("analyze this")
    before = request.model_dump()

    extract_features(request)

    assert request.model_dump() == before


def test_features_are_plain_data() -> None:
    f = features("compare these")

    assert f.as_dict()["reasoning_cues"] == 1
    assert set(f.as_dict()) >= {"prompt_tokens", "code_blocks", "small_talk"}
    with pytest.raises(AttributeError):
        f.turns = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    "hostile",
    [
        "a" * 1_000_000,
        " " * 1_000_000,
        "$" * 200_000,
        "(" * 200_000,
        "{" * 200_000 + "\n" + "}" * 200_000,
        "\n".join("1. item" for _ in range(100_000)),
        "step by " * 100_000,
        "<" + "a" * 500_000 + ">",
        ("x = (" * 50_000) + ")",
        "```" * 100_000,
    ],
)
def test_hostile_input_is_handled_quickly(hostile: str) -> None:
    started = time.perf_counter()

    extract_features(req(hostile))

    assert (
        time.perf_counter() - started < 10.0
    )  # a pattern blow-up takes minutes; this is a hang guard


def test_a_huge_prompt_costs_no_more_to_scan_than_a_moderate_one() -> None:
    moderate = req("compare this " * 1_500)  # about the scan limit
    huge = req("compare this " * 80_000)

    started = time.perf_counter()
    for _ in range(3):
        extract_features(moderate)
    moderate_time = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(3):
        extract_features(huge)
    huge_time = time.perf_counter() - started

    assert huge_time < moderate_time * 40 + 0.5  # linear token counting only: no pattern blow-up


def test_only_the_most_recent_text_is_scanned_for_patterns() -> None:
    old_code = "```python\nx = 1\n```\n" * 5_000  # far outside the scan window
    request = req(
        messages=[{"role": "user", "content": old_code}, {"role": "user", "content": "hi"}]
    )

    f = extract_features(request)

    assert f.code_blocks < 5_000  # bounded by the window, not by the size of the history
    assert f.prompt_tokens > 10_000  # but the length is still measured in full


# ---------- do the features separate easy from hard? ----------

EASY = [
    "hi",
    "thanks!",
    "What is the capital of France?",
    "Translate 'good morning' to Spanish.",
    "Write a short poem about autumn.",
    "Who wrote Hamlet?",
    "Give me a synonym for happy.",
    "What time zone is Tokyo in?",
]

HARD = [
    "Prove that the sum of two odd numbers is even, then derive a general rule for sums of "
    "k odd numbers and justify each step. Use $n = 2k + 1$.",
    "Design a distributed rate limiter. Compare token bucket vs sliding window, analyze the "
    "trade-offs step by step, and respond in JSON. It must not lose requests.\n1. Data model\n"
    "2. Failure modes\n3. Edge cases",
    "Debug this and explain the root cause:\n```python\ndef f(xs):\n    for i in range(len(xs)):\n"
    "        xs.pop(i)\n    return xs\n```\nThen refactor it and analyze its time complexity.",
    "Evaluate the implications of moving our monolith to microservices. Compare the strategy "
    "options, critique each, and justify a recommendation with the trade-offs.",
]


def difficulty_signals(f: ComplexityFeatures) -> int:
    """A deliberately crude tally, just to check that the features point the right way."""
    return (
        f.reasoning_cues
        + f.constraint_cues
        + f.math_signals
        + f.code_blocks
        + f.list_items
        + int(f.structured_output)
        + int(f.has_code)
    )


def test_hard_prompts_light_up_more_features_than_easy_ones() -> None:
    easy = [difficulty_signals(features(p)) for p in EASY]
    hard = [difficulty_signals(features(p)) for p in HARD]

    assert max(easy) < min(hard)


def test_the_easy_prompts_are_short_and_the_hard_ones_are_longer() -> None:
    easy = [features(p).last_user_tokens for p in EASY]
    hard = [features(p).last_user_tokens for p in HARD]

    assert max(easy) < min(hard)
