"""Tests for the simulation engine."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from config import SimConfig
from opsim.events import ExternalEvent, EventSchedule
from opsim.simulation import Simulation
from opsim.analysis import predict_vote


def _small_config(**kwargs):
    defaults = dict(
        n_nodes=200,
        n_parties=3,
        n_cities=2,
        n_steps=20,
        history_interval=5,
        random_seed=42,
    )
    defaults.update(kwargs)
    return SimConfig(**defaults)


def test_simulation_runs():
    cfg = _small_config()
    sim = Simulation(cfg)
    sim.run()
    # Should have snapshots at 0, 5, 10, 15, 20 (final)
    assert len(sim.opinion_history) == 5
    assert sim.history_steps == [0, 5, 10, 15, 20]


def test_vote_shares_sum_to_one():
    cfg = _small_config()
    sim = Simulation(cfg)
    sim.run()
    shares = predict_vote(sim.opinion_history[-1])
    np.testing.assert_almost_equal(shares.sum(), 1.0)


def test_event_fires_at_correct_step():
    """Verify that an event modifies attributes at the right step."""
    cfg = _small_config(n_steps=10, history_interval=1)
    event = ExternalEvent(
        name="shock",
        time_step=5,
        attribute_deltas={"capital": -0.5},
    )
    sim = Simulation(cfg, EventSchedule([event]))
    # Record capital before running
    capital_before = sim.attributes[:, 1].mean()
    sim.run()
    # Capital should have dropped after step 5
    # History index 5 is snapshot at step 5 (before step 5 fires)
    # History index 6 is snapshot at step 6 (after step 5 fired)
    capital_at_6 = sim.attribute_history[6][:, 1].mean()
    assert capital_at_6 < capital_before


def test_to_dataframe():
    cfg = _small_config()
    sim = Simulation(cfg)
    sim.run()
    df = sim.to_dataframe(["A", "B", "C"])
    assert list(df.columns) == ["step", "A", "B", "C"]
    assert len(df) == len(sim.history_steps)
    # Each row should sum to ~1
    for _, row in df.iterrows():
        np.testing.assert_almost_equal(row["A"] + row["B"] + row["C"], 1.0)


def test_reproducibility():
    """Same seed should produce identical results."""
    cfg = _small_config()
    sim1 = Simulation(cfg)
    sim1.run()
    sim2 = Simulation(cfg)
    sim2.run()
    np.testing.assert_array_equal(
        sim1.opinion_history[-1], sim2.opinion_history[-1]
    )
