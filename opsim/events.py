"""External events and party platform model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from config import ATTR_NAMES


@dataclass
class Party:
    """A political party defined by its platform — which voter attributes it appeals to.

    Platform weights follow a linear convention:
    - positive weight on attribute X → party appeals to voters *high* in X
    - negative weight on attribute X → party appeals to voters *low* in X

    When a world event shifts attribute X by delta, the opinion toward this party
    changes (for each node) by::

        Δopinion[p] += (platform · event_delta) × (event_appeal · attrs) × receptivity

    so parties never need to know which events benefit them — the alignment
    emerges automatically from their platform and the event's attribute changes.
    """

    name: str
    platform: dict[str, float] = field(default_factory=dict)

    def platform_vector(self) -> NDArray:
        """Return a (N_ATTRIBUTES,) array of platform weights."""
        vec = np.zeros(len(ATTR_NAMES), dtype=np.float64)
        for attr, w in self.platform.items():
            vec[ATTR_NAMES.index(attr)] = w
        return vec


@dataclass
class ExternalEvent:
    """A world event or contextual shock that fires at a specific time step.

    Events describe changes in the world (e.g. a recession, a health crisis,
    a campaign that shifts issue salience).  They do NOT specify which party
    benefits — that emerges from each party's platform alignment with the
    attribute changes.

    Parameters
    ----------
    name : str
        Human-readable label, e.g. "Economic recession".
    time_step : int
        The simulation step at which this event fires.
    strength : float
        Overall event intensity in [0, 1].
    effectiveness : float
        How salient / well-communicated the event is in [0, 1].
    attribute_appeal : dict[str, float]
        Maps attribute names to weights that control which nodes are most
        *susceptible* to this event.  ``event_appeal · attrs[i]`` gives the
        per-node resonance — high-resonance nodes experience a larger nudge.
    attribute_deltas : dict[str, float]
        Persistent changes applied to node attributes (e.g. ``{"capital": -0.15}``
        for an economic shock).  Determines which parties gain/lose support via
        their platform alignment.
    target_filter : callable or None
        Optional predicate ``(attributes_row) -> bool`` to restrict affected nodes.
    """

    name: str
    time_step: int
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
        return np.array(
            [self.target_filter(attributes[i]) for i in range(n)], dtype=bool
        )


@dataclass
class EventSchedule:
    """A collection of events indexed by time step."""

    events: list[ExternalEvent] = field(default_factory=list)

    def get_events_at(self, t: int) -> list[ExternalEvent]:
        """Return all events scheduled for step *t*."""
        return [e for e in self.events if e.time_step == t]
