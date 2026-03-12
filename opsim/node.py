"""Node attributes, opinion arrays, and initialisation utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from config import SimConfig, ATTR_NAMES, N_ATTRIBUTES


def initialize_attributes(config: SimConfig, rng: np.random.Generator) -> NDArray:
    """Create an (N, A) array of node attributes sampled from Beta distributions.

    Returns
    -------
    attributes : ndarray of shape (n_nodes, N_ATTRIBUTES)
        Each column corresponds to an attribute in ATTR_NAMES order.
    """
    n = config.n_nodes
    attrs = np.empty((n, N_ATTRIBUTES), dtype=np.float64)
    for col, name in enumerate(ATTR_NAMES):
        a, b = config.attribute_distributions[name]
        attrs[:, col] = rng.beta(a, b, size=n)
    return attrs


def initialize_opinions(config: SimConfig, rng: np.random.Generator) -> NDArray:
    """Create an (N, P) array of raw opinion scores (pre-softmax).

    Each entry is drawn from Normal(opinion_mean, opinion_std).
    """
    return rng.normal(
        config.opinion_mean,
        config.opinion_std,
        size=(config.n_nodes, config.n_parties),
    )


def compute_susceptibility(
    attributes: NDArray, config: SimConfig
) -> NDArray:
    """Derive per-node peer susceptibility from attributes.

    Returns an (N,) array in [0, 1].
    """
    weights = config.susceptibility_weights
    result = np.full(attributes.shape[0], weights.get("base", 0.5))
    for col, name in enumerate(ATTR_NAMES):
        w = weights.get(name, 0.0)
        if w != 0.0:
            result += w * attributes[:, col]
    return np.clip(result, 0.0, 1.0)


def compute_message_receptivity(attributes: NDArray) -> NDArray:
    """Baseline message receptivity = credulity.

    Returns an (N,) array in [0, 1].
    """
    from config import ATTR_CREDULITY

    return attributes[:, ATTR_CREDULITY].copy()


def compute_influence_power(attributes: NDArray) -> NDArray:
    """Influence power = charisma (scales outgoing edge weights).

    Returns an (N,) array in [0, 1].
    """
    from config import ATTR_CHARISMA

    return attributes[:, ATTR_CHARISMA].copy()
