"""Spatial 2D clustered network construction with sparse adjacency."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from config import SimConfig


def _place_cities(
    config: SimConfig, rng: np.random.Generator
) -> tuple[NDArray, NDArray]:
    """Return city center positions and per-city node counts.

    Returns
    -------
    centers : (K, 2) array of city center positions in [0, 1]^2
    counts  : (K,) int array of how many nodes each city has
    """
    k = config.n_cities
    centers = rng.uniform(0.0, 1.0, size=(k, 2))

    if config.city_sizes is not None:
        weights = np.asarray(config.city_sizes, dtype=np.float64)
        weights /= weights.sum()
    else:
        weights = np.ones(k) / k

    # Distribute nodes proportionally, ensuring total == n_nodes
    counts = np.round(weights * config.n_nodes).astype(int)
    diff = config.n_nodes - counts.sum()
    # Adjust the largest city to absorb rounding error
    counts[np.argmax(counts)] += diff
    return centers, counts


def build_network(
    config: SimConfig, rng: np.random.Generator
) -> tuple[sparse.csr_matrix, NDArray, NDArray]:
    """Build a sparse directed weighted graph on a spatial 2D cluster layout.

    Returns
    -------
    W          : csr_matrix (N, N) — base edge weights (before charisma scaling)
    positions  : (N, 2) — 2D coordinates for visualisation
    city_ids   : (N,) int — which city each node belongs to
    """
    centers, counts = _place_cities(config, rng)
    n = config.n_nodes
    k = config.n_cities

    # --- Assign positions ---
    positions = np.empty((n, 2), dtype=np.float64)
    city_ids = np.empty(n, dtype=np.int32)
    offset = 0
    for c in range(k):
        nc = counts[c]
        positions[offset : offset + nc] = rng.normal(
            loc=centers[c], scale=config.city_radius, size=(nc, 2)
        )
        city_ids[offset : offset + nc] = c
        offset += nc

    # --- Build sparse adjacency (COO then convert to CSR) ---
    rows: list[int] = []
    cols: list[int] = []

    # Pre-compute city node ranges for efficient access
    city_ranges: list[tuple[int, int]] = []
    offset = 0
    for c in range(k):
        city_ranges.append((offset, offset + counts[c]))
        offset += counts[c]

    # Intra-city edges
    for c in range(k):
        lo, hi = city_ranges[c]
        nc = hi - lo
        if nc < 2:
            continue
        n_possible = nc * (nc - 1)  # directed: exclude self-loops
        n_edges = int(round(config.intra_city_density * n_possible))
        if n_edges == 0:
            continue
        # Sample random directed edges within the city
        src = rng.integers(lo, hi, size=n_edges * 2)
        dst = rng.integers(lo, hi, size=n_edges * 2)
        mask = src != dst
        src = src[mask][:n_edges]
        dst = dst[mask][:n_edges]
        rows.extend(src.tolist())
        cols.extend(dst.tolist())

    # Inter-city edges
    for c1 in range(k):
        for c2 in range(k):
            if c1 == c2:
                continue
            lo1, hi1 = city_ranges[c1]
            lo2, hi2 = city_ranges[c2]
            n_possible = counts[c1] * counts[c2]
            n_edges = int(round(config.inter_city_density * n_possible))
            if n_edges == 0:
                continue
            src = rng.integers(lo1, hi1, size=n_edges)
            dst = rng.integers(lo2, hi2, size=n_edges)
            rows.extend(src.tolist())
            cols.extend(dst.tolist())

    # Remove duplicate edges (keep first occurrence)
    if rows:
        row_arr = np.array(rows, dtype=np.int32)
        col_arr = np.array(cols, dtype=np.int32)
        # De-duplicate using a set-based approach via linear indexing
        linear = row_arr.astype(np.int64) * n + col_arr.astype(np.int64)
        _, unique_idx = np.unique(linear, return_index=True)
        row_arr = row_arr[unique_idx]
        col_arr = col_arr[unique_idx]
        # Assign random asymmetric weights
        n_edges = len(row_arr)
        weights = rng.uniform(
            config.influence_weight_min,
            config.influence_weight_max,
            size=n_edges,
        )
        W = sparse.csr_matrix((weights, (row_arr, col_arr)), shape=(n, n))
    else:
        W = sparse.csr_matrix((n, n))

    return W, positions, city_ids
