#!/usr/bin/env python3
"""Regenerate all basic (3-party, 5-city) gallery plots.

Run after changing opinion_std default from 0.3 → 0.15 in config.py.
With opinion_std=0.3, initial L2 distances (~0.73) far exceeded the
confidence_threshold (0.4), leaving ~85% of peer-influence edges cut from
step 0.  Between events only noise (std=0.01) acted on opinions, giving
near-flat straight-line trajectories.  With opinion_std=0.15 the typical
distance (~0.37) is within epsilon so ~70% of edges stay active and genuine
peer-convergence dynamics are visible.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

from config import SimConfig
from opsim.events import ExternalEvent, EventSchedule
from opsim.simulation import Simulation
from opsim.analysis import (
    predict_vote,
    node_vote_probabilities,
    compute_polarization,
    plot_vote_share_evolution,
    plot_opinion_trajectories,
    plot_spatial_opinions,
    plot_opinion_histogram,
)

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
PARTY_NAMES = ["Conservative", "Progressive", "Green"]
COLORS = {"Conservative": "tab:red", "Progressive": "tab:blue", "Green": "tab:green"}

# ── Shared config & events ────────────────────────────────────────────────

def base_config(**kwargs):
    defaults = dict(
        n_nodes=10_000,
        n_parties=3,
        n_cities=5,
        city_sizes=[0.30, 0.25, 0.20, 0.15, 0.10],
        intra_city_density=0.005,
        inter_city_density=0.0005,
        confidence_threshold=0.4,
        noise_std=0.01,
        n_steps=100,
        history_interval=1,
        random_seed=42,
        # Give each city a distinct partisan lean so spatial clusters are
        # visually identifiable.  Cities 0–2 have strong leans; 3–4 have
        # moderate secondary leans.
        city_opinion_biases=[
            (0, 0.6),   # city 0 (30 %) → Conservative
            (1, 0.6),   # city 1 (25 %) → Progressive
            (2, 0.6),   # city 2 (20 %) → Green
            (0, 0.4),   # city 3 (15 %) → Conservative (secondary)
            (1, 0.4),   # city 4 (10 %) → Progressive  (secondary)
        ],
    )
    defaults.update(kwargs)
    return SimConfig(**defaults)


def base_events():
    return EventSchedule([
        ExternalEvent(
            name="Economic recession", time_step=30, party_index=1,
            strength=0.7, effectiveness=0.6,
            attribute_appeal={"capital": 0.8, "family_status": 0.3},
            attribute_deltas={"capital": -0.15},
        ),
        ExternalEvent(
            name="Pro-family campaign", time_step=50, party_index=0,
            strength=0.8, effectiveness=0.7,
            attribute_appeal={"family_status": 0.9, "health_status": 0.2},
            attribute_deltas={},
        ),
        ExternalEvent(
            name="Healthcare crisis", time_step=70, party_index=2,
            strength=0.6, effectiveness=0.8,
            attribute_appeal={"health_status": 0.9, "age": 0.4},
            attribute_deltas={"health_status": -0.2},
        ),
    ])


EVENT_ANNOTATIONS = [
    (30, "Recession\n(→ Progressive)",      "tab:blue"),
    (50, "Pro-family\ncampaign\n(→ Conservative)", "tab:red"),
    (70, "Health\ncrisis\n(→ Green)",        "tab:green"),
]

# ── 1–4  basic_run.py plots ───────────────────────────────────────────────

def make_basic_run_plots():
    print("Running base simulation …")
    sim = Simulation(base_config(), base_events())
    sim.run()

    # 1. Vote share stacked area
    fig = plot_vote_share_evolution(sim, PARTY_NAMES)
    fig.savefig(os.path.join(ASSETS, "vote_share_evolution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  vote_share_evolution.png")

    # 2. Opinion trajectories — Conservative — WITH event annotations
    rng = np.random.default_rng(0)
    nodes = rng.choice(sim.config.n_nodes, size=80, replace=False)
    steps = sim.history_steps
    traj = np.array([op[nodes, 0] for op in sim.opinion_history])

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(traj.shape[1]):
        ax.plot(steps, traj[:, i], alpha=0.3, linewidth=0.5)
    ymin, ymax = ax.get_ylim()
    for step_t, label, color in EVENT_ANNOTATIONS:
        ax.axvline(step_t, color=color, linestyle="--", alpha=0.7, linewidth=1.2)
        ax.text(step_t + 1, ymax - (ymax - ymin) * 0.04,
                label, fontsize=7, color=color, va="top")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Opinion (party 0 — Conservative)")
    ax.set_title("Opinion Trajectories — Conservative (Party 0)")
    fig.savefig(os.path.join(ASSETS, "opinion_trajectories.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  opinion_trajectories.png")

    # 3. Spatial opinion map (final step)
    fig = plot_spatial_opinions(sim, t_index=-1)
    fig.savefig(os.path.join(ASSETS, "spatial_opinions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  spatial_opinions.png")

    # 4. Opinion histogram — Conservative
    fig = plot_opinion_histogram(sim, party_index=0)
    fig.savefig(os.path.join(ASSETS, "opinion_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  opinion_histogram.png")

    return sim


# ── 5. Vote share annotated ───────────────────────────────────────────────

def make_vote_share_annotated(sim):
    steps = sim.history_steps
    shares = np.array([predict_vote(op) for op in sim.opinion_history])  # (T, 3)

    fig, ax = plt.subplots(figsize=(10, 5))
    for p, (name, color) in enumerate(COLORS.items()):
        ax.plot(steps, shares[:, p], label=name, color=color, linewidth=2)

    ymin, ymax = ax.get_ylim()
    for step_t, label, color in EVENT_ANNOTATIONS:
        ax.axvline(step_t, color=color, linestyle="--", alpha=0.7, linewidth=1.2)
        ax.text(step_t + 1, ymax - (ymax - ymin) * 0.04,
                label, fontsize=8, color=color, va="top")

    ax.set_xlabel("Time step")
    ax.set_ylabel("Vote share")
    ax.set_title("Vote share evolution with annotated events")
    ax.legend()
    fig.savefig(os.path.join(ASSETS, "vote_share_annotated.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  vote_share_annotated.png")


# ── 6. Spatial before / after ─────────────────────────────────────────────

def make_spatial_before_after(sim):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Spatial opinion map — leading party per node")

    for ax, t_idx, label in [(axes[0], 0, "Step 0 — initial"),
                              (axes[1], -1, "Step 100 — final")]:
        opinions = sim.opinion_history[t_idx]
        leading = softmax(opinions, axis=1).argmax(axis=1)
        colors_map = ["tab:red", "tab:blue", "tab:green"]
        for p, (name, c) in enumerate(COLORS.items()):
            mask = leading == p
            ax.scatter(sim.positions[mask, 0], sim.positions[mask, 1],
                       c=c, s=1, alpha=0.5, label=name)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if ax is axes[1]:
            ax.legend(markerscale=6, loc="upper right")

    fig.savefig(os.path.join(ASSETS, "spatial_before_after.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  spatial_before_after.png")


# ── 7. Confidence threshold comparison (spatial) ──────────────────────────

def make_confidence_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Effect of bounded-confidence threshold on opinion clustering")

    for ax, eps, label in [
        (axes[0], 0.40, "Confidence threshold ε = 0.40  (default)"),
        (axes[1], 0.15, "Confidence threshold ε = 0.15  (tight)"),
    ]:
        cfg = base_config(confidence_threshold=eps, n_steps=100)
        sim = Simulation(cfg, base_events())
        sim.run()
        opinions = sim.opinion_history[-1]
        leading = softmax(opinions, axis=1).argmax(axis=1)
        for p, (name, c) in enumerate(COLORS.items()):
            mask = leading == p
            ax.scatter(sim.positions[mask, 0], sim.positions[mask, 1],
                       c=c, s=1, alpha=0.5, label=name)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(markerscale=6, loc="upper right")

    fig.savefig(os.path.join(ASSETS, "confidence_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  confidence_comparison.png")


# ── 8. High-polarisation vote share (tight ε, no events) ─────────────────

def make_high_polarisation_vote_share():
    # Use a smaller opinion_std so initial distances are within ε=0.15 and
    # nodes can actually interact before fragmenting into frozen blocs.
    # With the default std=0.15, typical distances (≈0.37) already exceed ε=0.15
    # so edges are cut from step 0 and nothing moves at all.
    cfg = base_config(confidence_threshold=0.15, n_steps=150, history_interval=1,
                      opinion_std=0.05)
    sim = Simulation(cfg, EventSchedule())
    sim.run()

    fig = plot_vote_share_evolution(sim, PARTY_NAMES)
    fig.axes[0].set_title("High-polarisation run  (ε = 0.15, no events)")
    fig.savefig(os.path.join(ASSETS, "high_polarisation_vote_share.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  high_polarisation_vote_share.png")


# ── 9. Attribute prior distributions (no simulation needed) ───────────────

def make_attribute_distributions():
    from scipy.stats import beta as beta_dist
    from config import ATTR_NAMES

    attr_params = {
        "health_status": (2.0, 2.0),
        "capital":       (2.0, 5.0),
        "family_status": (2.0, 3.0),
        "education":     (2.0, 3.0),
        "age":           (2.0, 2.0),
        "credulity":     (3.0, 3.0),
        "charisma":      (2.0, 5.0),
    }

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle("Node attribute prior distributions")
    axes = axes.flatten()
    x = np.linspace(0, 1, 300)

    for i, (name, (a, b)) in enumerate(attr_params.items()):
        axes[i].fill_between(x, beta_dist.pdf(x, a, b), alpha=0.7)
        axes[i].set_title(f"{name}\nBeta({a:.0f}, {b:.0f})")
        axes[i].set_xlabel("value")
        axes[i].set_ylabel("density")

    axes[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "attribute_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  attribute_distributions.png")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("=== Basic 3-party plots ===")
    sim = make_basic_run_plots()
    make_vote_share_annotated(sim)
    make_spatial_before_after(sim)
    print("=== Confidence comparison (2× simulation) ===")
    make_confidence_comparison()
    print("=== High-polarisation run ===")
    make_high_polarisation_vote_share()
    print("=== Attribute distributions ===")
    make_attribute_distributions()
    print(f"\nAll done in {time.time() - t0:.1f}s")
