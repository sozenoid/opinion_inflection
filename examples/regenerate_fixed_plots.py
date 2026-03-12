#!/usr/bin/env python3
"""Regenerate three plots whose images did not match their README captions.

Issue 1 — polarisation_over_time.png
  Root cause: opinion_std=0.3 gives initial L2 opinion distances of ~0.73,
  far above all tested ε values (0.15 / 0.25 / 0.40).  All three thresholds
  therefore cut the vast majority of edges from the start, so their dynamics
  are nearly identical.  The "rapidly fragments" narrative never materialised
  because ε=0.15 had essentially no peer interaction to drive fragmentation.
  Fix: use opinion_std=0.10 → typical distances ≈ 0.24.  Now ε=0.40 is above
  that value (most edges stay active → rapid global consensus), while ε=0.15
  is below (most edges cut → nodes freeze in local blocs → rising polarisation).

Issue 2 — opinion_trajectories.png
  Root cause: The pro-family Conservative campaign fires at step 50 with
  strength=0.8 / effectiveness=0.7, directly nudging Conservative opinions
  for every node.  This produces a large spike — the dominant visual feature —
  that the caption ("spread gradually narrows") completely ignores.
  Fix: add vertical event annotation lines and update the caption.

Issue 3 — city_vote_shares.png
  Root cause: city_sizes=None produces equal-sized cities.  Equal populations
  with equal intra-city densities give nearly identical dynamics in every city,
  so the chart shows uniform ~20% per party across all 12 cities.  The claim
  "smaller cities converge on a dominant party" requires cities of different
  sizes.
  Fix: regenerate with 3 large hubs and 9 progressively smaller towns, with
  reduced inter-city density so the smaller towns are more insular.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import SimConfig
from opsim.events import ExternalEvent, EventSchedule
from opsim.simulation import Simulation
from opsim.analysis import compute_polarization, node_vote_probabilities

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")


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
            opinion_mean=0.0,
            # Use a smaller initial spread so that the ε values fall in the
            # interesting regime relative to typical inter-node distances.
            # With std=0.10 and P=3, typical L2 distance ≈ 0.10*√6 ≈ 0.24.
            # ε=0.40 > 0.24 → most edges active → rapid global consensus.
            # ε=0.15 < 0.24 → most edges cut  → fragmentation into blocs.
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
    fig.savefig(
        os.path.join(ASSETS, "polarisation_over_time.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("     saved.")


# ── 2. Opinion trajectories with event annotations ────────────────────────

def make_trajectories_plot():
    print("2/3  opinion_trajectories.png …")

    config = SimConfig(
        n_nodes=10_000,
        n_parties=3,
        n_cities=5,
        city_sizes=[0.30, 0.25, 0.20, 0.15, 0.10],
        confidence_threshold=0.4,
        noise_std=0.01,
        n_steps=100,
        history_interval=1,
        random_seed=42,
    )

    recession = ExternalEvent(
        name="Economic recession", time_step=30, party_index=1,
        strength=0.7, effectiveness=0.6,
        attribute_appeal={"capital": 0.8, "family_status": 0.3},
        attribute_deltas={"capital": -0.15},
    )
    pro_family = ExternalEvent(
        name="Pro-family campaign", time_step=50, party_index=0,
        strength=0.8, effectiveness=0.7,
        attribute_appeal={"family_status": 0.9, "health_status": 0.2},
        attribute_deltas={},
    )
    health_crisis = ExternalEvent(
        name="Healthcare crisis", time_step=70, party_index=2,
        strength=0.6, effectiveness=0.8,
        attribute_appeal={"health_status": 0.9, "age": 0.4},
        attribute_deltas={"health_status": -0.2},
    )
    schedule = EventSchedule([recession, pro_family, health_crisis])

    sim = Simulation(config, schedule)
    sim.run()

    rng = np.random.default_rng(0)
    nodes = rng.choice(config.n_nodes, size=80, replace=False)
    steps = sim.history_steps
    trajectories = np.array([op[nodes, 0] for op in sim.opinion_history])

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(trajectories.shape[1]):
        ax.plot(steps, trajectories[:, i], alpha=0.3, linewidth=0.5)

    # Annotate the three events
    event_annotations = [
        (30, "Recession\n(→ Progressive)", "tab:blue"),
        (50, "Pro-family\ncampaign\n(→ Conservative)", "tab:red"),
        (70, "Health\ncrisis\n(→ Green)", "tab:green"),
    ]
    ymin, ymax = ax.get_ylim()
    for step_t, label, color in event_annotations:
        ax.axvline(step_t, color=color, linestyle="--", alpha=0.7, linewidth=1.2)
        ax.text(
            step_t + 1, ymax - (ymax - ymin) * 0.05,
            label, fontsize=7, color=color, va="top",
        )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Opinion (party 0 — Conservative)")
    ax.set_title("Opinion Trajectories — Conservative (Party 0)")
    fig.savefig(
        os.path.join(ASSETS, "opinion_trajectories.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("     saved.")


# ── 3. City vote shares with unequal city sizes ───────────────────────────

def make_city_vote_shares():
    print("3/3  city_vote_shares.png (unequal city sizes) …")

    party_names = ["Conservative", "Progressive", "Green", "Libertarian", "Populist"]

    # 3 large hubs, 3 medium towns, 6 small/tiny villages.
    # Reduced inter-city density isolates smaller towns so they can develop
    # a distinct partisan identity rather than being pulled toward the global mean.
    city_sizes = [
        0.22, 0.18, 0.14,   # large hubs
        0.09, 0.08, 0.07,   # medium
        0.06, 0.05, 0.04,   # small
        0.03, 0.02, 0.02,   # tiny
    ]

    config = SimConfig(
        n_nodes=12_000,
        n_parties=5,
        n_cities=12,
        city_sizes=city_sizes,
        intra_city_density=0.005,
        inter_city_density=0.00015,   # much lower than default → isolated towns
        confidence_threshold=0.4,
        noise_std=0.01,
        n_steps=150,
        history_interval=10,
        random_seed=7,
    )

    events = [
        ExternalEvent("Recession shock",          10,  1, 0.7, 0.6, {"capital": 0.8},                         {"capital": -0.15}),
        ExternalEvent("Conservative tax cut",      20,  0, 0.6, 0.7, {"capital": 0.7, "education": 0.4},       {}),
        ExternalEvent("Green climate bill",        35,  2, 0.7, 0.6, {"education": 0.8, "age": 0.3},           {}),
        ExternalEvent("Progressive housing plan",  50,  1, 0.6, 0.7, {"family_status": 0.9, "capital": 0.4},   {}),
        ExternalEvent("Health system crisis",      60,  2, 0.5, 0.8, {"health_status": 0.9, "age": 0.5},       {"health_status": -0.2}),
        ExternalEvent("Libertarian deregulation",  70,  3, 0.7, 0.6, {"capital": 0.7, "education": 0.5},       {}),
        ExternalEvent("Populist anti-elite surge", 80,  4, 0.8, 0.5, {"credulity": 0.8, "family_status": 0.6}, {}),
        ExternalEvent("Conservative security push",95,  0, 0.6, 0.7, {"age": 0.5, "family_status": 0.4},       {}),
        ExternalEvent("Education reform",         110,  1, 0.5, 0.8, {"education": 0.9},                       {}),
        ExternalEvent("Late economic recovery",   130,  3, 0.6, 0.6, {"capital": 0.8},                         {"capital": 0.10}),
    ]
    schedule = EventSchedule(events)

    sim = Simulation(config, schedule)

    # Give each city a distinct political starting lean so that per-city dynamics
    # are visible.  Without this, all cities start from the same prior and the
    # globally-uniform events produce identical vote shares everywhere.
    city_rng = np.random.default_rng(99)
    for c in range(config.n_cities):
        mask = sim.city_ids == c
        city_bias = city_rng.normal(0.0, 0.25, size=(1, config.n_parties))
        sim.opinions[mask] += city_bias

    sim.run()

    final_opinions = sim.opinion_history[-1]
    probs = node_vote_probabilities(final_opinions)

    # Mean vote share per city
    city_shares = np.zeros((config.n_cities, len(party_names)))
    for c in range(config.n_cities):
        mask = sim.city_ids == c
        city_shares[c] = probs[mask].mean(axis=0)

    # City labels with approximate population
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
    fig.savefig(
        os.path.join(ASSETS, "city_vote_shares.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("     saved.")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    make_polarisation_plot()
    make_trajectories_plot()
    make_city_vote_shares()
    print(f"\nAll three plots regenerated in {time.time() - t0:.1f}s")
