import random
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.core.model_config import ConfigError, load_config, parse_config
from app.main import create_app
from app.routing.features import ComplexityFeatures, extract_features
from app.routing.score import (
    DEFAULT_SMALL_TALK_FACTOR,
    DEFAULT_STRENGTHS,
    DEFAULT_WEIGHTS,
    SIGNALS,
    ComplexityWeights,
    score_features,
    score_request,
    signal_levels,
)
from app.schemas.chat import ChatCompletionRequest
from tests.conftest import AuthedClient
from tests.test_complexity_features import EASY, HARD

ZERO: dict[str, Any] = {
    "prompt_tokens": 0,
    "last_user_tokens": 0,
    "system_tokens": 0,
    "turns": 1,
    "code_blocks": 0,
    "code_line_ratio": 0.0,
    "code_request": False,
    "math_signals": 0,
    "reasoning_cues": 0,
    "constraint_cues": 0,
    "question_marks": 0,
    "list_items": 0,
    "structured_output": False,
    "tool_count": 0,
    "forces_tool": False,
    "image_count": 0,
    "requested_completion_tokens": None,
    "non_ascii_ratio": 0.0,
    "small_talk": False,
}


def feats(**overrides: Any) -> ComplexityFeatures:
    return ComplexityFeatures(**{**ZERO, **overrides})


def value(**overrides: Any) -> float:
    return score_features(feats(**overrides)).value


def req(content: str) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "auto", "messages": [{"role": "user", "content": content}]}
    )


# ---------- the shape of the score ----------


def test_a_request_with_no_signals_scores_zero() -> None:
    result = score_features(feats())

    assert result.value == 0.0
    assert all(share == 0 for share in result.contributions.values())
    assert result.top() == []


def test_every_signal_has_a_default_strength_and_nothing_else_does() -> None:
    assert set(DEFAULT_STRENGTHS) == set(SIGNALS)
    assert all(0.0 <= strength <= 1.0 for strength in DEFAULT_STRENGTHS.values())


def test_signal_levels_are_always_between_zero_and_one() -> None:
    rng = random.Random(7)
    for _ in range(500):
        f = feats(
            prompt_tokens=rng.randint(0, 10**8),
            last_user_tokens=rng.randint(0, 10**8),
            turns=rng.randint(0, 10**5),
            code_blocks=rng.randint(0, 10**4),
            code_line_ratio=rng.random(),
            math_signals=rng.randint(0, 10**6),
            reasoning_cues=rng.randint(0, 10**3),
            constraint_cues=rng.randint(0, 10**3),
            question_marks=rng.randint(0, 10**5),
            list_items=rng.randint(0, 10**5),
            tool_count=rng.randint(0, 10**3),
            image_count=rng.randint(0, 10**3),
            requested_completion_tokens=rng.choice([None, rng.randint(0, 10**9)]),
            non_ascii_ratio=rng.random(),
        )

        assert all(0.0 <= level <= 1.0 for level in signal_levels(f).values())


def test_the_score_never_leaves_zero_to_one_whatever_the_input() -> None:
    rng = random.Random(11)
    for _ in range(1_000):
        f = feats(
            prompt_tokens=rng.randint(0, 10**9),
            last_user_tokens=rng.randint(0, 10**9),
            code_blocks=rng.randint(0, 10**5),
            math_signals=rng.randint(0, 10**7),
            reasoning_cues=rng.randint(0, 10**4),
            constraint_cues=rng.randint(0, 10**4),
            list_items=rng.randint(0, 10**6),
            structured_output=rng.random() < 0.5,
            tool_count=rng.randint(0, 100),
            forces_tool=rng.random() < 0.5,
            image_count=rng.randint(0, 100),
            requested_completion_tokens=rng.choice([None, 10**9]),
            small_talk=rng.random() < 0.3,
        )

        assert 0.0 <= score_features(f).value <= 1.0


def test_a_request_that_triggers_everything_is_near_but_below_the_top() -> None:
    result = score_features(
        feats(
            prompt_tokens=50_000,
            last_user_tokens=10_000,
            turns=20,
            code_blocks=3,
            math_signals=20,
            reasoning_cues=10,
            constraint_cues=10,
            list_items=10,
            question_marks=10,
            structured_output=True,
            tool_count=5,
            forces_tool=True,
            image_count=4,
            requested_completion_tokens=8_000,
            non_ascii_ratio=0.9,
        )
    )

    assert 0.99 < result.value <= 1.0


# ---------- the arithmetic ----------


def test_a_single_signal_contributes_its_strength_times_its_level() -> None:
    assert value(reasoning_cues=4) == pytest.approx(DEFAULT_STRENGTHS["reasoning"])
    assert value(reasoning_cues=2) == pytest.approx(DEFAULT_STRENGTHS["reasoning"] * 0.5)
    assert value(math_signals=4) == pytest.approx(DEFAULT_STRENGTHS["math"])
    assert value(code_blocks=1) == pytest.approx(DEFAULT_STRENGTHS["code"])
    assert value(structured_output=True) == pytest.approx(DEFAULT_STRENGTHS["structured_output"])


def test_signals_saturate_rather_than_growing_without_limit() -> None:
    assert value(reasoning_cues=4) == value(reasoning_cues=40) == value(reasoning_cues=4_000)


def test_independent_signals_combine_as_noisy_or() -> None:
    a, b = DEFAULT_STRENGTHS["reasoning"], DEFAULT_STRENGTHS["math"]

    both = value(reasoning_cues=4, math_signals=4)

    assert both == pytest.approx(1 - (1 - a) * (1 - b))
    assert max(a, b) < both < a + b  # reinforcing, but not simply added


def test_one_strong_signal_is_not_diluted_by_the_signals_that_are_absent() -> None:
    assert value(reasoning_cues=4) == pytest.approx(0.55)  # a plain average would give ~0.04


def test_more_evidence_always_means_a_higher_score() -> None:
    steps = [
        value(),
        value(reasoning_cues=2),
        value(reasoning_cues=2, math_signals=2),
        value(reasoning_cues=4, math_signals=2, code_blocks=1),
        value(reasoning_cues=4, math_signals=4, code_blocks=1, list_items=4),
    ]

    assert steps == sorted(steps) and len(set(steps)) == len(steps)


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [
        ("last_user_tokens", 5, 500),
        ("prompt_tokens", 100, 5_000),
        ("turns", 1, 6),
        ("code_blocks", 0, 1),
        ("code_line_ratio", 0.0, 0.4),
        ("math_signals", 0, 3),
        ("reasoning_cues", 0, 3),
        ("constraint_cues", 0, 3),
        ("question_marks", 0, 4),
        ("list_items", 0, 3),
        ("structured_output", False, True),
        ("tool_count", 0, 2),
        ("forces_tool", False, True),
        ("image_count", 0, 2),
        ("requested_completion_tokens", None, 2_000),
        ("non_ascii_ratio", 0.0, 0.4),
        ("code_request", False, True),
    ],
)
def test_raising_any_feature_never_lowers_the_score(name: str, low: Any, high: Any) -> None:
    extra = {"tool_count": 1} if name == "forces_tool" else {}
    base = {"last_user_tokens": 20, "prompt_tokens": 20, "reasoning_cues": 1, **extra}

    assert value(**{**base, name: high}) > value(**{**base, name: low})


def test_longer_asks_score_higher_with_diminishing_returns() -> None:
    scores = [value(last_user_tokens=n, prompt_tokens=n) for n in (1, 10, 100, 1_000, 10_000)]

    assert scores == sorted(scores)
    first_jump, last_jump = scores[1] - scores[0], scores[4] - scores[3]
    assert last_jump < first_jump  # a log scale: the thousandth token matters less than the first


def test_history_adds_context_only_beyond_the_final_ask() -> None:
    single_turn = value(last_user_tokens=50, prompt_tokens=50)
    with_history = value(last_user_tokens=50, prompt_tokens=5_050, turns=4)

    assert with_history > single_turn
    assert signal_levels(feats(last_user_tokens=50, prompt_tokens=50))["context_length"] == 0.0


def test_asking_for_code_counts_for_less_than_pasting_it() -> None:
    asked = value(code_request=True)
    pasted = value(code_blocks=1)

    assert 0 < asked < pasted


def test_forcing_a_tool_adds_to_the_tools_signal() -> None:
    assert value(tool_count=1, forces_tool=True) > value(tool_count=1)


# ---------- small talk ----------


def test_small_talk_is_scaled_down() -> None:
    plain = value(last_user_tokens=8, prompt_tokens=8)
    chatty = value(last_user_tokens=8, prompt_tokens=8, small_talk=True)

    assert chatty == pytest.approx(plain * DEFAULT_SMALL_TALK_FACTOR, abs=1e-3)


def test_the_small_talk_factor_is_tunable() -> None:
    f = feats(last_user_tokens=8, prompt_tokens=8, small_talk=True)

    off = score_features(f, ComplexityWeights.from_overrides({"small_talk_factor": 0.0}))
    unchanged = score_features(f, ComplexityWeights.from_overrides({"small_talk_factor": 1.0}))

    assert off.value == 0.0
    assert unchanged.value == pytest.approx(value(last_user_tokens=8, prompt_tokens=8))


# ---------- tuning ----------


def test_overriding_one_strength_leaves_the_rest_alone() -> None:
    weights = ComplexityWeights.from_overrides({"reasoning": 0.9})

    assert weights.strengths["reasoning"] == 0.9
    assert weights.strengths["math"] == DEFAULT_STRENGTHS["math"]
    assert weights.small_talk_factor == DEFAULT_SMALL_TALK_FACTOR


def test_no_overrides_means_the_defaults() -> None:
    assert dict(ComplexityWeights.from_overrides({}).strengths) == dict(DEFAULT_STRENGTHS)


def test_a_higher_strength_raises_the_score_and_a_lower_one_lowers_it() -> None:
    f = feats(reasoning_cues=4)

    base = score_features(f).value
    higher = score_features(f, ComplexityWeights.from_overrides({"reasoning": 0.9})).value
    lower = score_features(f, ComplexityWeights.from_overrides({"reasoning": 0.1})).value

    assert lower < base < higher
    assert lower == pytest.approx(0.1)
    assert higher == pytest.approx(0.9)


def test_a_strength_of_zero_switches_a_signal_off() -> None:
    off = ComplexityWeights.from_overrides({"reasoning": 0.0})

    assert score_features(feats(reasoning_cues=4), off).value == 0.0


def test_a_strength_of_one_makes_a_saturated_signal_decisive() -> None:
    decisive = ComplexityWeights.from_overrides({"math": 1.0})

    assert score_features(feats(math_signals=4), decisive).value == 1.0


def test_overrides_do_not_leak_into_the_defaults() -> None:
    ComplexityWeights.from_overrides({"reasoning": 0.9, "small_talk_factor": 0.9})

    assert DEFAULT_WEIGHTS.strengths["reasoning"] == DEFAULT_STRENGTHS["reasoning"]
    assert DEFAULT_WEIGHTS.small_talk_factor == DEFAULT_SMALL_TALK_FACTOR


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"reasonin": 0.5}, "unknown signal 'reasonin'"),
        ({"reasoning": 1.5}, "reasoning: must be between 0 and 1"),
        ({"reasoning": -0.1}, "reasoning: must be between 0 and 1"),
        ({"reasoning": float("nan")}, "reasoning: must be between 0 and 1"),
        ({"reasoning": "high"}, "reasoning: must be a number"),
        ({"reasoning": True}, "reasoning: must be a number"),
        ({"reasoning": None}, "reasoning: must be a number"),
        ({"small_talk_factor": 2}, "small_talk_factor: must be between 0 and 1"),
    ],
)
def test_bad_overrides_are_rejected_with_a_clear_message(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ComplexityWeights.from_overrides(overrides)


def test_every_problem_is_listed_not_just_the_first() -> None:
    bad: dict[str, Any] = {"nope": 1, "reasoning": 5, "math": "x"}
    problems = ComplexityWeights.problems(bad)

    assert len(problems) == 3
    assert "reasoning" in "".join(problems) and "known:" in problems[0]


# ---------- explaining a score ----------


def test_contributions_are_listed_largest_first() -> None:
    result = score_features(feats(reasoning_cues=4, code_blocks=1, structured_output=True))

    shares = list(result.contributions.values())
    assert shares == sorted(shares, reverse=True)
    assert list(result.contributions)[:3] == ["reasoning", "code", "structured_output"]


def test_top_names_the_signals_that_mattered_and_skips_the_silent_ones() -> None:
    result = score_features(feats(reasoning_cues=4, math_signals=2))

    assert result.top(2) == ["reasoning", "math"]
    assert result.top(10) == ["reasoning", "math"]  # nothing else contributed
    assert result.top(1) == ["reasoning"]


def test_the_score_is_exactly_what_the_contributions_imply() -> None:
    result = score_features(
        feats(reasoning_cues=3, code_blocks=1, list_items=2, last_user_tokens=120)
    )

    doubt = 1.0
    for share in result.contributions.values():
        doubt *= 1 - share
    assert result.value == pytest.approx(1 - doubt, abs=1e-3)


def test_scoring_is_deterministic_and_does_not_touch_its_input() -> None:
    f = feats(reasoning_cues=3, code_blocks=1)

    assert score_features(f) == score_features(f)
    assert f == feats(reasoning_cues=3, code_blocks=1)


# ---------- real prompts ----------


def score_of(text: str) -> float:
    return score_request(req(text)).value


def test_easy_prompts_score_low_and_hard_prompts_score_high() -> None:
    assert max(score_of(p) for p in EASY) < 0.2
    assert min(score_of(p) for p in HARD) > 0.55


def test_there_is_a_wide_gap_between_the_easy_and_the_hard() -> None:
    gap = min(score_of(p) for p in HARD) - max(score_of(p) for p in EASY)

    assert gap > 0.35  # room to place thresholds between tiers


def test_prompts_are_ranked_in_a_sensible_order() -> None:
    ladder = [
        "hi",
        "What is the capital of France?",
        "Write a Python function that checks if a number is prime.",
        "Fix this:\n```python\ndef add(a, b):\n    return a - b\n```\nWhy does it fail?",
        "Design a distributed rate limiter. Compare token bucket and sliding window, analyze the "
        "trade-offs step by step, and respond in JSON.\n1. Data model\n2. Failure modes",
    ]

    scores = [score_of(p) for p in ladder]

    assert scores == sorted(scores) and len(set(scores)) == len(scores)


def test_asking_for_code_scores_higher_than_a_similar_plain_request() -> None:
    assert score_of("Write a Python function that sorts a list.") > score_of(
        "Write a short story that sorts a list of feelings."
    )


def test_a_greeting_never_reaches_the_expensive_tiers() -> None:
    assert score_of("hello!") < 0.05


def test_scoring_a_request_matches_scoring_its_features() -> None:
    request = req("Compare A and B and justify the trade-offs.")

    assert score_request(request) == score_features(extract_features(request))


# ---------- from the config file ----------


def models(**routing: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "models": [
            {
                "name": "a",
                "provider": "openai_compatible",
                "model": "m",
                "base_url": "https://api.example.com/v1",
            }
        ]
    }
    if routing:
        body["routing"] = routing
    return body


def test_the_routing_section_is_optional() -> None:
    assert parse_config(models(), {}).complexity == {}


def test_overrides_are_read_from_the_routing_section() -> None:
    config = parse_config(models(complexity={"reasoning": 0.7, "small_talk_factor": 0.1}), {})

    assert config.complexity == {"reasoning": 0.7, "small_talk_factor": 0.1}
    assert ComplexityWeights.from_overrides(config.complexity).strengths["reasoning"] == 0.7


def test_whole_numbers_are_accepted_as_strengths() -> None:
    assert parse_config(models(complexity={"math": 1, "code": 0}), {}).complexity == {
        "math": 1.0,
        "code": 0.0,
    }


@pytest.mark.parametrize(
    ("routing", "message"),
    [
        ({"complexity": {"reasonin": 0.5}}, "routing.complexity: unknown signal 'reasonin'"),
        (
            {"complexity": {"reasoning": 3}},
            "routing.complexity: reasoning: must be between 0 and 1",
        ),
        ({"complexity": {"reasoning": "high"}}, r"routing\.complexity\.reasoning"),
        ({"complexity": {"reasoning": True}}, r"routing\.complexity\.reasoning"),
        ({"speed": 1}, r"routing\.speed"),
    ],
)
def test_bad_routing_sections_stop_startup_with_a_clear_path(
    routing: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(models(**routing), {})


def test_routing_problems_are_reported_with_the_other_problems() -> None:
    bad = models(complexity={"nope": 1})
    bad["models"][0]["tier"] = "gigantic"

    with pytest.raises(ConfigError) as excinfo:
        parse_config(bad, {})

    assert "tier" in str(excinfo.value)


def test_the_running_gateway_uses_the_configured_strengths(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "models:\n"
        "  - {name: a, provider: openai_compatible, model: m, base_url: 'https://a.example/v1'}\n"
        "routing:\n"
        "  complexity:\n"
        "    reasoning: 0.7\n"
        "    small_talk_factor: 0.05\n"
    )
    app = create_app(Settings(_env_file=None, config_path=config))

    with AuthedClient(app):
        weights = app.state.registry.complexity_weights

    assert weights.strengths["reasoning"] == 0.7
    assert weights.small_talk_factor == 0.05
    assert weights.strengths["code"] == DEFAULT_STRENGTHS["code"]


def test_without_a_routing_section_the_gateway_uses_the_defaults(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "models:\n"
        "  - {name: a, provider: openai_compatible, model: m, base_url: 'https://a.example/v1'}\n"
    )
    app = create_app(Settings(_env_file=None, config_path=config))

    with AuthedClient(app):
        weights = app.state.registry.complexity_weights

    assert weights == DEFAULT_WEIGHTS


def test_a_bad_routing_section_prevents_startup(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "models:\n"
        "  - {name: a, provider: openai_compatible, model: m, base_url: 'https://a.example/v1'}\n"
        "routing:\n  complexity:\n    reasoning: 9\n"
    )
    app = create_app(Settings(_env_file=None, config_path=config))

    with pytest.raises(ConfigError, match="must be between 0 and 1"), AuthedClient(app):
        pass


def test_the_commented_example_in_the_shipped_config_is_valid_when_enabled(tmp_path: Path) -> None:
    text = (Path(__file__).parent.parent / "config.example.yaml").read_text()
    enabled = []
    in_routing = False
    for line in text.splitlines():
        if line.startswith("# routing:"):
            in_routing = True
        if in_routing and line.startswith("#"):
            enabled.append(line[2:] if line.startswith("# ") else line[1:])
        elif not (in_routing and line.startswith("#")):
            enabled.append(line)
    config = tmp_path / "config.yaml"
    config.write_text("\n".join(enabled))

    loaded = load_config(config, {"GROQ_API_KEY": "a", "ANTHROPIC_API_KEY": "b"})

    assert loaded.complexity["reasoning"] == 0.6
    assert loaded.complexity["small_talk_factor"] == 0.2
