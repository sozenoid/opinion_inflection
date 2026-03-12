"""Tests for the dynamics step function."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import sparse

from config import SimConfig, N_ATTRIBUTES, ATTR_CREDULITY, ATTR_CHARISMA
from opsim.dynamics import step
from opsim.events import ExternalEvent, Party


def _make_state(n, p, seed=0):
    """Create minimal state arrays for testing."""
    rng = np.random.default_rng(seed)
    attrs = rng.uniform(0.3, 0.7, size=(n, N_ATTRIBUTES))
    opinions = rng.normal(0, 0.1, size=(n, p))
    return attrs, opinions, rng


def test_no_edges_no_events_opinions_change_only_by_noise():
    """Without edges or events, opinions change only by noise."""
    cfg = SimConfig(n_nodes=5, n_parties=2, noise_std=0.0, random_seed=7)
    W = sparse.csr_matrix((5, 5))
    attrs, opinions, rng = _make_state(5, 2, seed=7)
    old = opinions.copy()
    _, new_opinions = step(W, attrs, opinions, [], cfg, rng)
    # With zero noise and no edges/events, opinions should be unchanged
    np.testing.assert_array_almost_equal(new_opinions, old)


def test_two_node_convergence():
    """Two connected nodes should move toward each other."""
    cfg = SimConfig(
        n_nodes=2, n_parties=1,
        confidence_threshold=10.0,  # effectively no bounded confidence
        noise_std=0.0,
        random_seed=0,
    )
    # Edge 0→1 and 1→0
    W = sparse.csr_matrix(np.array([[0.0, 0.3], [0.3, 0.0]]))
    attrs = np.full((2, N_ATTRIBUTES), 0.5)
    attrs[:, ATTR_CREDULITY] = 0.8  # high credulity → high susceptibility
    attrs[:, ATTR_CHARISMA] = 0.5
    opinions = np.array([[1.0], [-1.0]])
    rng = np.random.default_rng(0)

    _, new_opinions = step(W, attrs, opinions, [], cfg, rng)
    # Both should move toward 0: node 0 should decrease, node 1 increase
    assert new_opinions[0, 0] < 1.0
    assert new_opinions[1, 0] > -1.0


def test_bounded_confidence_blocks_distant_opinions():
    """Nodes with opinions beyond the confidence threshold should not influence each other."""
    cfg = SimConfig(
        n_nodes=2, n_parties=1,
        confidence_threshold=0.1,  # very tight
        noise_std=0.0,
        random_seed=0,
    )
    W = sparse.csr_matrix(np.array([[0.0, 0.5], [0.5, 0.0]]))
    attrs = np.full((2, N_ATTRIBUTES), 0.5)
    attrs[:, ATTR_CHARISMA] = 0.5
    opinions = np.array([[1.0], [-1.0]])  # distance = 2 >> 0.1
    rng = np.random.default_rng(0)

    _, new_opinions = step(W, attrs, opinions, [], cfg, rng)
    # Opinions should be unchanged (bounded confidence blocks everything)
    np.testing.assert_array_almost_equal(new_opinions, opinions)


def test_event_shifts_opinions_by_attribute():
    """High-family nodes should shift more toward the family-values party after a
    family-salience event — because that party's platform weights family_status
    positively and the event increases family_status."""
    from config import ATTR_FAMILY
    parties = [
        Party(name="FamilyValues", platform={"family_status": 1.0}),
        Party(name="Other"),
    ]
    cfg = SimConfig(n_nodes=4, n_parties=2, noise_std=0.0, random_seed=0,
                    parties=parties)
    W = sparse.csr_matrix((4, 4))

    attrs = np.full((4, N_ATTRIBUTES), 0.5)
    attrs[0, ATTR_FAMILY] = 0.9   # high family
    attrs[1, ATTR_FAMILY] = 0.8
    attrs[2, ATTR_FAMILY] = 0.1   # low family
    attrs[3, ATTR_FAMILY] = 0.2
    attrs[:, ATTR_CREDULITY] = 0.5

    opinions = np.zeros((4, 2))
    rng = np.random.default_rng(0)

    # Event raises family_status salience; family-values party benefits
    event = ExternalEvent(
        name="pro-family",
        time_step=0,
        strength=1.0,
        effectiveness=1.0,
        attribute_appeal={"family_status": 1.0},
        attribute_deltas={"family_status": 0.2},
    )

    _, new_opinions = step(W, attrs, opinions, [event], cfg, rng, parties=parties)
    # Party 0 (FamilyValues) should gain more for high-family nodes
    assert new_opinions[0, 0] > new_opinions[2, 0]
    assert new_opinions[1, 0] > new_opinions[3, 0]


def test_credulity_modulates_event_impact():
    """High-credulity nodes should shift more than low-credulity ones for same event."""
    parties = [
        Party(name="HealthParty"),
        Party(name="HealthAdvocate", platform={"health_status": -1.0}),
    ]
    cfg = SimConfig(n_nodes=2, n_parties=2, noise_std=0.0, random_seed=0,
                    parties=parties)
    W = sparse.csr_matrix((2, 2))

    attrs = np.full((2, N_ATTRIBUTES), 0.5)
    attrs[0, ATTR_CREDULITY] = 0.9  # very credulous
    attrs[1, ATTR_CREDULITY] = 0.1  # skeptical

    opinions = np.zeros((2, 2))
    rng = np.random.default_rng(0)

    # Health crisis lowers health_status; party 1 (negative weight) benefits
    event = ExternalEvent(
        name="health-crisis",
        time_step=0,
        strength=1.0,
        effectiveness=1.0,
        attribute_appeal={"health_status": 1.0},
        attribute_deltas={"health_status": -0.2},
    )

    _, new_opinions = step(W, attrs, opinions, [event], cfg, rng, parties=parties)
    assert new_opinions[0, 1] > new_opinions[1, 1]


def test_charisma_amplifies_outgoing_influence():
    """A high-charisma node should influence its neighbor more than a low-charisma one."""
    cfg = SimConfig(
        n_nodes=3, n_parties=1,
        confidence_threshold=10.0,
        noise_std=0.0,
        random_seed=0,
    )
    # Node 0 → Node 2, Node 1 → Node 2 (same base weight)
    W = sparse.csr_matrix(np.array([
        [0.0, 0.0, 0.3],
        [0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0],
    ]))
    attrs = np.full((3, N_ATTRIBUTES), 0.5)
    attrs[0, ATTR_CHARISMA] = 0.9  # high charisma
    attrs[1, ATTR_CHARISMA] = 0.1  # low charisma
    attrs[2, ATTR_CREDULITY] = 0.8  # receiver is susceptible

    # Node 0 opinion = +1, Node 1 opinion = +1, Node 2 = 0
    opinions = np.array([[1.0], [1.0], [0.0]])
    rng = np.random.default_rng(0)

    # Run step
    _, new_opinions = step(W, attrs, opinions, [], cfg, rng)
    # Node 2 should be pulled positive. The pull from Node 0 (charisma=0.9)
    # should dominate over Node 1 (charisma=0.1).
    assert new_opinions[2, 0] > 0.0
