"""Tests for node initialisation and derived quantities."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from config import SimConfig, N_ATTRIBUTES
from opsim.node import (
    initialize_attributes,
    initialize_opinions,
    compute_susceptibility,
    compute_message_receptivity,
    compute_influence_power,
)


def _small_config(**kwargs):
    defaults = dict(n_nodes=100, n_parties=3, random_seed=1)
    defaults.update(kwargs)
    return SimConfig(**defaults)


def test_attributes_shape_and_range():
    cfg = _small_config()
    rng = np.random.default_rng(cfg.random_seed)
    attrs = initialize_attributes(cfg, rng)
    assert attrs.shape == (100, N_ATTRIBUTES)
    assert np.all(attrs >= 0.0) and np.all(attrs <= 1.0)


def test_opinions_shape():
    cfg = _small_config()
    rng = np.random.default_rng(cfg.random_seed)
    ops = initialize_opinions(cfg, rng)
    assert ops.shape == (100, 3)


def test_susceptibility_range():
    cfg = _small_config()
    rng = np.random.default_rng(cfg.random_seed)
    attrs = initialize_attributes(cfg, rng)
    sus = compute_susceptibility(attrs, cfg)
    assert sus.shape == (100,)
    assert np.all(sus >= 0.0) and np.all(sus <= 1.0)


def test_receptivity_equals_credulity():
    cfg = _small_config()
    rng = np.random.default_rng(cfg.random_seed)
    attrs = initialize_attributes(cfg, rng)
    rec = compute_message_receptivity(attrs)
    from config import ATTR_CREDULITY
    np.testing.assert_array_equal(rec, attrs[:, ATTR_CREDULITY])


def test_influence_power_equals_charisma():
    cfg = _small_config()
    rng = np.random.default_rng(cfg.random_seed)
    attrs = initialize_attributes(cfg, rng)
    ip = compute_influence_power(attrs)
    from config import ATTR_CHARISMA
    np.testing.assert_array_equal(ip, attrs[:, ATTR_CHARISMA])
