import random
from collections.abc import Sequence

from app.router.deployment import Deployment


def weighted_choice(deployments: Sequence[Deployment], rng: random.Random) -> Deployment:
    """Pick one deployment with probability proportional to its weight."""
    if not deployments:
        raise ValueError("cannot choose from an empty list of deployments")
    if len(deployments) == 1:
        return deployments[0]
    return rng.choices(deployments, weights=[d.weight for d in deployments], k=1)[0]
