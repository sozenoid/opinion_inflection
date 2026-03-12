"""External events and party messaging model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from config import ATTR_NAMES


@dataclass
class ExternalEvent:
    """An event or party message that fires at a specific time step.

    Parameters
    ----------
    name : str
        Human-readable label, e.g. "Party A pro-family campaign".
    time_step : int
        The simulation step at which this event fires.
    party_index : int | None
        Which party's opinion column receives the nudge.
        None means a non-partisan shock (only attribute deltas apply).
    strength : float
        Message strength / budget in [0, 1].
    effectiveness : float
        How well-crafted the message is in [0, 1].
    attribute_appeal : dict[str, float]
        Maps attribute names to appeal weights.
        The dot product of this vector with a node's attributes gives
        *relevance* — how much the message resonates with that node.
    attribute_deltas : dict[str, float]
        Direct changes to node attributes (e.g. {"capital": -0.1}
        for an economic shock).  Applied to all targeted nodes.
    target_filter : callable or None
        Optional predicate ``(attributes_row) -> bool`` to restrict which
        nodes are affected.  ``None`` means all nodes.
    """

    name: str
    time_step: int
    party_index: int | None = None
    strength: float = 1.0
    effectiveness: float = 1.0
    attribute_appeal: dict[str, float] = field(default_factory=dict)
    attribute_deltas: dict[str, float] = field(default_factory=dict)
    target_filter: Callable[[NDArray], bool] | None = None

    # --- helpers used by dynamics.step() ---

    def appeal_vector(self) -> NDArray:
        """Return a (N_ATTRIBUTES,) array of appeal weights."""
        vec = np.zeros(len(ATTR_NAMES), dtype=np.float64)
        for name, w in self.attribute_appeal.items():
            idx = ATTR_NAMES.index(name)
            vec[idx] = w
        return vec

    def delta_vector(self) -> NDArray:
        """Return a (N_ATTRIBUTES,) array of attribute deltas."""
        vec = np.zeros(len(ATTR_NAMES), dtype=np.float64)
        for name, d in self.attribute_deltas.items():
            idx = ATTR_NAMES.index(name)
            vec[idx] = d
        return vec

    def compute_target_mask(self, attributes: NDArray) -> NDArray:
        """Return a boolean (N,) mask of which nodes are affected."""
        n = attributes.shape[0]
        if self.target_filter is None:
            return np.ones(n, dtype=bool)
        mask = np.array(
            [self.target_filter(attributes[i]) for i in range(n)], dtype=bool
        )
        return mask


@dataclass
class EventSchedule:
    """A collection of events indexed by time step."""

    events: list[ExternalEvent] = field(default_factory=list)

    def get_events_at(self, t: int) -> list[ExternalEvent]:
        """Return all events scheduled for step *t*."""
        return [e for e in self.events if e.time_step == t]
