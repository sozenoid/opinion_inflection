"""Core opinion-update step — vectorised with sparse matrix operations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from config import SimConfig, ATTR_CREDULITY, ATTR_CHARISMA
from opsim.events import ExternalEvent
from opsim.node import compute_susceptibility


def _apply_attribute_deltas(
    attributes: NDArray,
    event: ExternalEvent,
    mask: NDArray,
) -> None:
    """Modify attributes in-place for nodes selected by *mask*."""
    delta_vec = event.delta_vector()
    nonzero = np.nonzero(delta_vec)[0]
    for col in nonzero:
        attributes[mask, col] = np.clip(
            attributes[mask, col] + delta_vec[col], 0.0, 1.0
        )


def _apply_message_nudge(
    opinions: NDArray,
    attributes: NDArray,
    event: ExternalEvent,
    mask: NDArray,
) -> None:
    """Nudge opinions toward *event.party_index* for targeted nodes."""
    if event.party_index is None:
        return
    appeal = event.appeal_vector()  # (A,)
    # relevance = attributes @ appeal → (N,)
    relevance = attributes @ appeal
    credulity = attributes[:, ATTR_CREDULITY]
    receptivity = credulity * event.strength * event.effectiveness
    delta = relevance * receptivity  # (N,)
    # Apply only to masked nodes
    opinions[mask, event.party_index] += delta[mask]


def _peer_influence(
    W_base: sparse.csr_matrix,
    opinions: NDArray,
    charisma: NDArray,
    susceptibility: NDArray,
    confidence_threshold: float,
) -> NDArray:
    """Compute bounded-confidence peer influence delta for all nodes.

    Returns an (N, P) array of opinion deltas (before susceptibility scaling).
    """
    n, p = opinions.shape
    # Build effective weight matrix: W_eff[i,j] = W_base[i,j] * charisma[i]
    # For CSR, scale each row by charisma[row]
    W_eff = W_base.copy()
    # Multiply each stored value by the charisma of its row
    # CSR stores rows contiguously; use indptr to scale
    for i in range(n):
        start, end = W_eff.indptr[i], W_eff.indptr[i + 1]
        W_eff.data[start:end] *= charisma[i]

    # Bounded confidence: zero out edges where opinion distance > ε
    # Work on COO for easy element-wise access
    W_coo = W_eff.tocoo()
    if W_coo.nnz > 0:
        src_opinions = opinions[W_coo.row]  # (nnz, P)
        dst_opinions = opinions[W_coo.col]  # (nnz, P)
        diffs = src_opinions - dst_opinions  # (nnz, P)
        distances = np.linalg.norm(diffs, axis=1)  # (nnz,)
        beyond = distances >= confidence_threshold
        W_coo.data[beyond] = 0.0
        W_masked = W_coo.tocsr()
        W_masked.eliminate_zeros()
    else:
        W_masked = W_eff

    # peer_delta[j] = Σ_i w_ij * (opinions[i] - opinions[j])
    #               = (W^T @ opinions)[j] - (sum_of_incoming_weights[j]) * opinions[j]
    incoming_sum = np.array(W_masked.sum(axis=0)).ravel()  # (N,)
    weighted_opinions = W_masked.T @ opinions  # (N, P)
    peer_delta = weighted_opinions - incoming_sum[:, None] * opinions

    # Scale by receiver susceptibility
    return susceptibility[:, None] * peer_delta


def step(
    W_base: sparse.csr_matrix,
    attributes: NDArray,
    opinions: NDArray,
    events_now: list[ExternalEvent],
    config: SimConfig,
    rng: np.random.Generator,
) -> tuple[NDArray, NDArray]:
    """Execute one synchronous time step.

    Parameters
    ----------
    W_base     : (N, N) sparse base weight matrix
    attributes : (N, A) mutable attribute array
    opinions   : (N, P) current opinion scores
    events_now : events firing this step
    config     : simulation configuration
    rng        : random generator

    Returns
    -------
    new_attributes : (N, A) — updated (modified in-place AND returned)
    new_opinions   : (N, P) — new opinion array (fresh copy)
    """
    # Phase 1: apply event attribute deltas
    for event in events_now:
        mask = event.compute_target_mask(attributes)
        _apply_attribute_deltas(attributes, event, mask)

    # Recompute derived quantities after attribute changes
    susceptibility = compute_susceptibility(attributes, config)
    charisma = attributes[:, ATTR_CHARISMA]

    # Phase 2: message nudges (accumulate into a copy)
    new_opinions = opinions.copy()
    for event in events_now:
        mask = event.compute_target_mask(attributes)
        _apply_message_nudge(new_opinions, attributes, event, mask)

    # Phase 3: peer influence (reads from *opinions*, writes to *new_opinions*)
    peer_delta = _peer_influence(
        W_base, opinions, charisma, susceptibility, config.confidence_threshold
    )
    new_opinions += peer_delta

    # Phase 4: noise
    new_opinions += rng.normal(0.0, config.noise_std, size=new_opinions.shape)

    return attributes, new_opinions
