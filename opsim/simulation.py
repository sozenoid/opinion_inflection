"""Simulation engine — orchestrates network, nodes, events, and dynamics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from config import SimConfig
from opsim.dynamics import step
from opsim.events import EventSchedule
from opsim.network import build_network
from opsim.node import initialize_attributes, initialize_opinions


class Simulation:
    """Run and record an opinion-dynamics simulation.

    Parameters
    ----------
    config : SimConfig
    event_schedule : EventSchedule
    """

    def __init__(self, config: SimConfig, event_schedule: EventSchedule | None = None):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        # Build network
        self.W: sparse.csr_matrix
        self.positions: NDArray
        self.city_ids: NDArray
        self.W, self.positions, self.city_ids = build_network(config, self.rng)

        # Initialise node state
        self.attributes: NDArray = initialize_attributes(config, self.rng)
        self.opinions: NDArray = initialize_opinions(config, self.rng)

        # Events
        self.event_schedule = event_schedule or EventSchedule()

        # History (recorded at intervals)
        self.opinion_history: list[NDArray] = []  # each entry is (N, P)
        self.attribute_history: list[NDArray] = []  # each entry is (N, A)
        self.history_steps: list[int] = []  # which time steps are recorded

    def run(self, n_steps: int | None = None) -> None:
        """Execute the simulation for *n_steps* (default: config.n_steps)."""
        steps = n_steps if n_steps is not None else self.config.n_steps
        interval = self.config.history_interval

        for t in range(steps):
            # Record snapshot
            if t % interval == 0:
                self.opinion_history.append(self.opinions.copy())
                self.attribute_history.append(self.attributes.copy())
                self.history_steps.append(t)

            # Get events for this step
            events_now = self.event_schedule.get_events_at(t)

            # Advance one step
            self.attributes, self.opinions = step(
                self.W,
                self.attributes,
                self.opinions,
                events_now,
                self.config,
                self.rng,
            )

        # Final snapshot
        self.opinion_history.append(self.opinions.copy())
        self.attribute_history.append(self.attributes.copy())
        self.history_steps.append(steps)

    def get_opinion_at(self, t_index: int) -> NDArray:
        """Return the (N, P) opinion snapshot at history index *t_index*."""
        return self.opinion_history[t_index]

    def to_dataframe(self, party_names: list[str] | None = None) -> pd.DataFrame:
        """Export vote-share time series as a DataFrame.

        Columns: step, party_0, party_1, …  (or custom names).
        Each row is the mean softmax vote share at that recorded step.
        """
        from opsim.analysis import predict_vote

        rows = []
        names = party_names or [f"party_{i}" for i in range(self.config.n_parties)]
        for idx, t in enumerate(self.history_steps):
            shares = predict_vote(self.opinion_history[idx])
            row = {"step": t}
            for i, name in enumerate(names):
                row[name] = shares[i]
            rows.append(row)
        return pd.DataFrame(rows)
