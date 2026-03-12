#!/usr/bin/env python3
"""Regenerate the polarisation, city-vote-share, and extended 5-party gallery plots.

All events use the platform-based model: events describe real-world attribute
changes; which party benefits emerges from each party's platform alignment with
those changes.  No party_index is set on any event.
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
from opsim.events import ExternalEvent, EventSchedule, Party
from opsim.simulation import Simulation
from opsim.analysis import compute_polarization, node_vote_probabilities, predict_vote

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# ── 5-party definitions (used by extended run) ────────────────────────────
# Platform weights:  +value → appeals to voters HIGH in that attribute
#                    -value → appeals to voters LOW in that attribute
FIVE_PARTIES = [
    Party("Conservative", platform={
        "family_status": +8.0,   # traditional family voters
        "capital":       +4.0,   # wealthier voters
        "age":           +5.0,   # older voters
    }),
    Party("Progressive", platform={
        "capital":       -7.0,   # lower-income voters
        "education":     +5.0,   # higher-education voters
        "health_status": +2.0,
    }),
    Party("Green", platform={
        "health_status": -6.0,   # health-concerned voters
        "age":           -4.0,   # younger voters
        "family_status": -2.0,
    }),
    Party("Libertarian", platform={
        "capital":       +6.0,   # wealthier, pro-market voters
        "education":     +4.0,   # educated, self-reliant voters
        "age":           +2.0,
    }),
    Party("Populist", platform={
        "credulity":     +7.0,   # high-credulity, susceptible voters
        "family_status": +4.0,   # traditional households
        "capital":       -3.0,   # lower-income resentment
    }),
]


# ── 1. Polarisation over time ─────────────────────────────────────────────

def make_polarisation_plot():
    print("1/3  polarisation_over_time.png …")
    epsilons = [
        (0.40, "ε = 0.40 (default)"),
        (0.25, "ε = 0.25"),
        (0.15, "ε = 0.15 (tight)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for eps, label in epsilons:
        config = SimConfig(
            n_nodes=5_000,
            n_parties=3,
            n_cities=5,
            confidence_threshold=eps,
            noise_std=0.005,
            n_steps=300,
            history_interval=5,
            random_seed=42,
            opinion_std=0.10,
        )
        sim = Simulation(config, EventSchedule())
        sim.run()
        pol = [compute_polarization(op) for op in sim.opinion_history]
        ax.plot(sim.history_steps, pol, label=label)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Polarisation index")
    ax.set_title("Polarisation over time at varying confidence thresholds")
    ax.legend()
    fig.savefig(os.path.join(ASSETS, "polarisation_over_time.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   saved.")


# ── 2. City vote shares — 12 cities, 5 parties ───────────────────────────

def make_city_vote_shares():
    print("2/3  city_vote_shares.png …")

    party_names = [p.name for p in FIVE_PARTIES]

    city_sizes = [
        0.22, 0.18, 0.14,
        0.09, 0.08, 0.07,
        0.06, 0.05, 0.04,
        0.03, 0.02, 0.02,
    ]

    # City partisan leans: one per city, matching city_sizes length
    city_biases = [
        (0, 0.6), (1, 0.6), (2, 0.6), (3, 0.6), (4, 0.6),  # cities 0-4: one party each
        (0, 0.4), (1, 0.4), (2, 0.4), (3, 0.4),              # cities 5-8: secondary
        (0, 0.3), (4, 0.4), (1, 0.3),                         # cities 9-11: mixed
    ]

    config = SimConfig(
        n_nodes=12_000,
        n_parties=5,
        n_cities=12,
        city_sizes=city_sizes,
        intra_city_density=0.005,
        inter_city_density=0.00015,
        confidence_threshold=0.4,
        noise_std=0.01,
        n_steps=150,
        history_interval=10,
        random_seed=7,
        parties=FIVE_PARTIES,
        city_opinion_biases=city_biases,
    )

    # Events: attribute deltas only — party benefit emerges from platform alignment
    events = EventSchedule([
        # capital ↓ → Progressive (capital:-7) + Populist (capital:-3) gain
        ExternalEvent("Recession shock", 10, 0.7, 0.6,
                      {"capital": 0.8}, {"capital": -0.15}),
        # capital ↑ + education ↑ → Libertarian (cap:+6,edu:+4) + Conservative gain
        ExternalEvent("Tax-cut campaign", 20, 0.6, 0.7,
                      {"capital": 0.7, "education": 0.4}, {"capital": +0.10, "education": +0.05}),
        # health ↑ slightly, age ↓ (youth mobilisation) → Green (health:-6, age:-4) benefits
        ExternalEvent("Green climate bill", 35, 0.7, 0.6,
                      {"education": 0.8, "age": 0.3}, {"health_status": +0.05, "age": -0.03}),
        # capital ↓ slightly (highlighting hardship) → Progressive + Populist gain
        ExternalEvent("Progressive housing plan", 50, 0.6, 0.7,
                      {"family_status": 0.9, "capital": 0.4}, {"capital": -0.05}),
        # health ↓ → Green (health:-6) gains strongly
        ExternalEvent("Health system crisis", 60, 0.5, 0.8,
                      {"health_status": 0.9, "age": 0.5}, {"health_status": -0.2}),
        # capital ↑ + education ↑ → Libertarian gains
        ExternalEvent("Libertarian deregulation", 70, 0.7, 0.6,
                      {"capital": 0.7, "education": 0.5}, {"capital": +0.08, "education": +0.04}),
        # credulity ↑ + family ↑ → Populist (credulity:+7, family:+4) gains strongly
        ExternalEvent("Populist anti-elite surge", 80, 0.8, 0.5,
                      {"credulity": 0.8, "family_status": 0.6}, {"credulity": +0.10, "family_status": +0.05}),
        # age ↑ + family ↑ → Conservative (age:+5, family:+8) gains
        ExternalEvent("Conservative security push", 95, 0.6, 0.7,
                      {"age": 0.5, "family_status": 0.4}, {"age": +0.05, "family_status": +0.06}),
        # education ↑ → Progressive (edu:+5) + Libertarian (edu:+4) gain
        ExternalEvent("Education reform", 110, 0.5, 0.8,
                      {"education": 0.9}, {"education": +0.10}),
        # capital ↑ → Libertarian + Conservative gain; Progressive loses
        ExternalEvent("Late economic recovery", 130, 0.6, 0.6,
                      {"capital": 0.8}, {"capital": +0.10}),
    ])

    sim = Simulation(config, events)
    sim.run()

    final_opinions = sim.opinion_history[-1]
    probs = node_vote_probabilities(final_opinions)

    city_shares = np.zeros((config.n_cities, len(party_names)))
    for c in range(config.n_cities):
        mask = sim.city_ids == c
        city_shares[c] = probs[mask].mean(axis=0)

    actual = np.array(city_sizes) / sum(city_sizes)
    city_labels = [f"City {i}\n(~{int(actual[i]*12_000):,})" for i in range(config.n_cities)]

    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    x = np.arange(config.n_cities)
    width = 0.15

    fig, ax = plt.subplots(figsize=(16, 5))
    for p, (name, color) in enumerate(zip(party_names, colors)):
        ax.bar(x + p * width, city_shares[:, p], width, label=name, color=color, alpha=0.8)

    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(city_labels, fontsize=8)
    ax.set_ylim(0, 0.60)
    ax.set_ylabel("Mean vote share")
    ax.set_title("Final vote shares by city — 12-city run (unequal city sizes)")
    ax.legend()
    fig.savefig(os.path.join(ASSETS, "city_vote_shares.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   saved.")

    return sim, config, events  # reuse for extended-run gallery


# ── 3. Extended-run gallery (5 parties · 12 cities · 10 events) ──────────

def make_extended_gallery(sim, config, events):
    print("3/3  extended-run gallery …")

    party_names = [p.name for p in FIVE_PARTIES]
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    event_steps = [e.time_step for e in events.events]

    # ── spatial_12cities_3steps.png ──────────────────────────────────
    t_indices = [0, len(sim.history_steps)//2, -1]
    t_labels = [f"Step {sim.history_steps[i]}" for i in t_indices]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Spatial opinion map — initial, mid, and final (12 cities · 5 parties)")
    for ax, t_idx, label in zip(axes, t_indices, t_labels):
        opinions = sim.opinion_history[t_idx]
        leading = softmax(opinions, axis=1).argmax(axis=1)
        for p, (name, c) in enumerate(zip(party_names, colors)):
            mask = leading == p
            ax.scatter(sim.positions[mask, 0], sim.positions[mask, 1],
                       c=c, s=1, alpha=0.4, label=name)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    axes[-1].legend(markerscale=6, loc="upper right")
    fig.savefig(os.path.join(ASSETS, "spatial_12cities_3steps.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── vote_share_5party_events.png (line) ──────────────────────────
    steps = sim.history_steps
    shares = np.array([predict_vote(op) for op in sim.opinion_history])

    fig, ax = plt.subplots(figsize=(14, 6))
    for p, (name, c) in enumerate(zip(party_names, colors)):
        ax.plot(steps, shares[:, p], label=name, color=c, linewidth=2)
    ymin, ymax = ax.get_ylim()
    for es in event_steps:
        ax.axvline(es, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Vote share")
    ax.set_title("Vote share evolution — 5 parties, 10 events")
    ax.legend()
    fig.savefig(os.path.join(ASSETS, "vote_share_5party_events.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── stacked_5party_events.png ────────────────────────────────────
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    ax.stackplot(steps, shares.T, labels=party_names, colors=colors, alpha=0.8)
    for es in event_steps:
        ax.axvline(es, color="white", linestyle="--", alpha=0.6, linewidth=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Vote share")
    ax.set_title("Stacked vote share — 5 parties, 10 events")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(ASSETS, "stacked_5party_events.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── per_party_spread.png ─────────────────────────────────────────
    from opsim.analysis import node_vote_probabilities
    spreads = []
    for op in sim.opinion_history:
        probs = node_vote_probabilities(op)
        spreads.append(probs.var(axis=0))
    spreads = np.array(spreads)

    fig, ax = plt.subplots(figsize=(14, 5))
    for p, (name, c) in enumerate(zip(party_names, colors)):
        ax.plot(steps, spreads[:, p], label=name, color=c)
    for es in event_steps:
        ax.axvline(es, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Variance of vote probability")
    ax.set_title("Per-party opinion spread over time")
    ax.legend()
    fig.savefig(os.path.join(ASSETS, "per_party_spread.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── trajectories_5parties.png ────────────────────────────────────
    rng = np.random.default_rng(0)
    nodes = rng.choice(config.n_nodes, size=40, replace=False)
    traj = np.array([op[nodes] for op in sim.opinion_history])  # (T, 40, 5)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
    fig.suptitle("Raw opinion trajectories — 40 sampled nodes, all 5 parties")
    for p, (name, c) in enumerate(zip(party_names, colors)):
        ax = axes[p]
        for i in range(traj.shape[1]):
            ax.plot(steps, traj[:, i, p], alpha=0.3, linewidth=0.5, color=c)
        for es in event_steps:
            ax.axvline(es, color="gray", linestyle="--", alpha=0.4, linewidth=0.7)
        ax.set_title(name)
        ax.set_xlabel("Step")
        if p == 0:
            ax.set_ylabel("Raw opinion score")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "trajectories_5parties.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("   saved all 5 extended-run images.")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    make_polarisation_plot()
    sim, config, events = make_city_vote_shares()
    make_extended_gallery(sim, config, events)
    print(f"\nAll plots regenerated in {time.time() - t0:.1f}s")
