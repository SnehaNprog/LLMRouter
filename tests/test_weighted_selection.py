import random
from collections import Counter

import pytest

from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.router.router import Router
from app.router.selection import weighted_choice
from app.schemas.chat import ChatCompletionRequest, Message
from tests.conftest import TEST_ALIAS, model_config

DRAWS = 10_000


def deployments(*weights: int) -> list[Deployment]:
    return [
        Deployment.create(model_config(model=f"model-{i}", weight=w), FakeProvider(f"p{i}"))
        for i, w in enumerate(weights)
    ]


def shares(chosen: list[Deployment]) -> dict[str, float]:
    counts = Counter(d.upstream_model for d in chosen)
    return {model: count / len(chosen) for model, count in counts.items()}


def test_distribution_follows_weights() -> None:
    items = deployments(1, 3)
    rng = random.Random(7)

    result = shares([weighted_choice(items, rng) for _ in range(DRAWS)])

    assert result["model-0"] == pytest.approx(0.25, abs=0.02)
    assert result["model-1"] == pytest.approx(0.75, abs=0.02)


def test_three_way_split() -> None:
    items = deployments(1, 1, 2)
    rng = random.Random(11)

    result = shares([weighted_choice(items, rng) for _ in range(DRAWS)])

    assert result["model-0"] == pytest.approx(0.25, abs=0.02)
    assert result["model-1"] == pytest.approx(0.25, abs=0.02)
    assert result["model-2"] == pytest.approx(0.50, abs=0.02)


def test_equal_weights_split_evenly() -> None:
    items = deployments(1, 1)
    rng = random.Random(3)

    result = shares([weighted_choice(items, rng) for _ in range(DRAWS)])

    assert result["model-0"] == pytest.approx(0.5, abs=0.02)


def test_single_deployment_is_always_chosen() -> None:
    items = deployments(5)

    assert all(weighted_choice(items, random.Random(i)) is items[0] for i in range(20))


def test_same_seed_gives_same_sequence() -> None:
    items = deployments(1, 2, 3)

    rng_a, rng_b = random.Random(5), random.Random(5)
    sequence_a = [weighted_choice(items, rng_a).id for _ in range(50)]
    sequence_b = [weighted_choice(items, rng_b).id for _ in range(50)]

    assert sequence_a == sequence_b
    assert len(set(sequence_a)) > 1


def test_empty_list_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        weighted_choice([], random.Random(0))


async def test_router_splits_real_traffic_by_weight() -> None:
    light, heavy = FakeProvider("light"), FakeProvider("heavy")
    registry = ModelRegistry(
        [
            Deployment.create(model_config(model="light-model", weight=1), light),
            Deployment.create(model_config(model="heavy-model", weight=4), heavy),
        ]
    )
    router = Router(registry, rng=random.Random(42))
    request = ChatCompletionRequest(model=TEST_ALIAS, messages=[Message(role="user", content="hi")])

    for _ in range(2_000):
        await router.complete(request)

    assert light.calls + heavy.calls == 2_000
    assert heavy.calls / 2_000 == pytest.approx(0.8, abs=0.03)
    assert {r.model for r in light.requests} == {"light-model"}
    assert {r.model for r in heavy.requests} == {"heavy-model"}


async def test_weight_one_deployments_never_starve() -> None:
    a, b = FakeProvider("a"), FakeProvider("b")
    registry = ModelRegistry(
        [
            Deployment.create(model_config(model="a", weight=1), a),
            Deployment.create(model_config(model="b", weight=1), b),
        ]
    )
    router = Router(registry, rng=random.Random(1))
    request = ChatCompletionRequest(model=TEST_ALIAS, messages=[Message(role="user", content="hi")])

    for _ in range(200):
        await router.complete(request)

    assert a.calls > 0
    assert b.calls > 0
