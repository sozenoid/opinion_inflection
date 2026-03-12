#!/usr/bin/env python3
"""End-to-end demo: 3 parties, 10 000 nodes, 5 cities, three external events.

Events describe real-world changes only.  Which party benefits is determined
automatically by each party's platform alignment with the attribute deltas.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import SimConfig
from opsim.events import ExternalEvent, EventSchedule, Party
from opsim.simulation import Simulation
from opsim.analysis import (
    predict_vote,
    compute_polarization,
    plot_vote_share_evolution,
    plot_opinion_trajectories,
    plot_spatial_opinions,
    plot_opinion_histogram,
)


# Party platforms:  positive weight → appeals to voters HIGH in that attribute
#                   negative weight → appeals to voters LOW in that attribute
PARTIES = [
    Party("Conservative", platform={
        "family_status": +8.0,   # family-values voters
        "capital":       +5.0,   # wealthier voters
        "age":           +3.0,   # older voters
    }),
    Party("Progressive", platform={
        "capital":       -7.0,   # lower-income voters
        "education":     +4.0,   # higher-education voters
        "health_status": +2.0,
    }),
    Party("Green", platform={
        "health_status": -6.0,   # voters with health concerns
        "age":           -3.0,   # younger voters
        "family_status": -2.5,
    }),
]


def main():
    # ── Configuration ─────────────────────────────────────────────────
    config = SimConfig(
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
        parties=PARTIES,
        city_opinion_biases=[
            (0, 0.6),   # city 0 → Conservative
            (1, 0.6),   # city 1 → Progressive
            (2, 0.6),   # city 2 → Green
            (0, 0.4),   # city 3 → Conservative (secondary)
            (1, 0.4),   # city 4 → Progressive  (secondary)
        ],
    )

    # ── Events ────────────────────────────────────────────────────────
    # Capital falls → Progressive (capital: -7) gains; Conservative loses.
    recession = ExternalEvent(
        name="Economic recession",
        time_step=30,
        strength=0.7,
        effectiveness=0.6,
        attribute_appeal={"capital": 0.8},
        attribute_deltas={"capital": -0.15},
    )

    # Campaign raises family_status → Conservative (family: +8) gains; Green loses.
    pro_family = ExternalEvent(
        name="Pro-family campaign",
        time_step=50,
        strength=0.8,
        effectiveness=0.7,
        attribute_appeal={"family_status": 0.9},
        attribute_deltas={"family_status": +0.08},
    )

    # Health deteriorates → Green (health: -6) gains; Progressive loses slightly.
    health_crisis = ExternalEvent(
        name="Healthcare crisis",
        time_step=70,
        strength=0.6,
        effectiveness=0.8,
        attribute_appeal={"health_status": 0.9, "age": 0.4},
        attribute_deltas={"health_status": -0.2},
    )

    schedule = EventSchedule([recession, pro_family, health_crisis])

    # ── Run simulation ────────────────────────────────────────────────
    print(f"Building network: {config.n_nodes} nodes, {config.n_cities} cities …")
    sim = Simulation(config, schedule)
    print(f"Network edges: {sim.W.nnz}")

    print(f"Running {config.n_steps} steps …")
    t0 = time.time()
    sim.run()
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # ── Results ───────────────────────────────────────────────────────
    party_names = [p.name for p in PARTIES]
    initial_shares = predict_vote(sim.opinion_history[0])
    final_shares = predict_vote(sim.opinion_history[-1])

    print("\nVote shares:")
    for i, name in enumerate(party_names):
        print(f"  {name:15s}  initial={initial_shares[i]:.3f}  final={final_shares[i]:.3f}")

    print(f"\nPolarisation: {compute_polarization(sim.opinion_history[-1]):.4f}")

    # ── Export table ──────────────────────────────────────────────────
    df = sim.to_dataframe(party_names)
    print("\nVote share time series (first 5 rows):")
    print(df.head().to_string(index=False))

    # ── Plots (save to files) ─────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")

    fig = plot_vote_share_evolution(sim, party_names)
    fig.savefig("vote_share_evolution.png", dpi=150, bbox_inches="tight")
    print("\nSaved vote_share_evolution.png")

    fig = plot_opinion_trajectories(sim, party_index=0, sample_n=80)
    fig.savefig("opinion_trajectories.png", dpi=150, bbox_inches="tight")
    print("Saved opinion_trajectories.png")

    fig = plot_spatial_opinions(sim, t_index=-1)
    fig.savefig("spatial_opinions.png", dpi=150, bbox_inches="tight")
    print("Saved spatial_opinions.png")

    fig = plot_opinion_histogram(sim, party_index=0)
    fig.savefig("opinion_histogram.png", dpi=150, bbox_inches="tight")
    print("Saved opinion_histogram.png")


if __name__ == "__main__":
    main()
