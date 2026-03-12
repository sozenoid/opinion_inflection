"""Analysis, prediction, and plotting utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

if TYPE_CHECKING:
    from opsim.simulation import Simulation


# ── Vote prediction ──────────────────────────────────────────────────────


def predict_vote(opinions: NDArray) -> NDArray:
    """Convert raw opinions (N, P) to aggregate vote shares via softmax.

    Returns a (P,) array of party vote fractions summing to 1.
    """
    probs = softmax(opinions, axis=1)  # (N, P)
    return probs.mean(axis=0)


def node_vote_probabilities(opinions: NDArray) -> NDArray:
    """Per-node vote probabilities (N, P) via softmax."""
    return softmax(opinions, axis=1)


# ── Polarisation metrics ─────────────────────────────────────────────────


def compute_polarization(opinions: NDArray) -> float:
    """Polarisation index: mean variance of per-node softmax distributions.

    High value → nodes are concentrated on one party each (polarised).
    Low value → nodes are spread across parties (consensus / indifference).
    """
    probs = softmax(opinions, axis=1)  # (N, P)
    per_node_var = probs.var(axis=1)  # (N,)
    return float(per_node_var.mean())


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_vote_share_evolution(
    sim: "Simulation",
    party_names: list[str] | None = None,
    ax=None,
):
    """Stacked area chart of party vote shares over time."""
    import matplotlib.pyplot as plt

    names = party_names or [f"Party {i}" for i in range(sim.config.n_parties)]
    steps = sim.history_steps
    shares = np.array([predict_vote(op) for op in sim.opinion_history])  # (T, P)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(steps, shares.T, labels=names, alpha=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Vote share")
    ax.set_title("Vote Share Evolution")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    return ax.figure


def plot_opinion_trajectories(
    sim: "Simulation",
    party_index: int = 0,
    sample_n: int = 50,
    ax=None,
):
    """Line plot of raw opinion scores for a sample of nodes (one party)."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    nodes = rng.choice(sim.config.n_nodes, size=min(sample_n, sim.config.n_nodes), replace=False)
    steps = sim.history_steps
    trajectories = np.array(
        [op[nodes, party_index] for op in sim.opinion_history]
    )  # (T, sample_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(trajectories.shape[1]):
        ax.plot(steps, trajectories[:, i], alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel(f"Opinion (party {party_index})")
    ax.set_title(f"Opinion Trajectories — Party {party_index}")
    return ax.figure


def plot_spatial_opinions(
    sim: "Simulation",
    t_index: int = -1,
    ax=None,
):
    """2D scatter plot of node positions coloured by leading party."""
    import matplotlib.pyplot as plt

    opinions = sim.opinion_history[t_index]
    probs = softmax(opinions, axis=1)
    leading_party = probs.argmax(axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        sim.positions[:, 0],
        sim.positions[:, 1],
        c=leading_party,
        cmap="tab10",
        s=1,
        alpha=0.6,
    )
    ax.set_title(f"Spatial Opinion Map (step {sim.history_steps[t_index]})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(scatter, ax=ax, label="Leading party")
    return ax.figure


def plot_opinion_histogram(
    sim: "Simulation",
    party_index: int = 0,
    t_indices: list[int] | None = None,
    ax=None,
):
    """Histogram of opinion scores at selected time snapshots."""
    import matplotlib.pyplot as plt

    if t_indices is None:
        n = len(sim.opinion_history)
        t_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    for idx in t_indices:
        vals = sim.opinion_history[idx][:, party_index]
        ax.hist(
            vals, bins=50, alpha=0.4,
            label=f"step {sim.history_steps[idx]}",
            density=True,
        )
    ax.set_xlabel(f"Opinion (party {party_index})")
    ax.set_ylabel("Density")
    ax.set_title(f"Opinion Distribution — Party {party_index}")
    ax.legend()
    return ax.figure
