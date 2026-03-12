"""Simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


# Attribute index constants
ATTR_HEALTH = 0
ATTR_CAPITAL = 1
ATTR_FAMILY = 2
ATTR_EDUCATION = 3
ATTR_AGE = 4
ATTR_CREDULITY = 5
ATTR_CHARISMA = 6

ATTR_NAMES = [
    "health_status",
    "capital",
    "family_status",
    "education",
    "age",
    "credulity",
    "charisma",
]

N_ATTRIBUTES = len(ATTR_NAMES)


@dataclass
class SimConfig:
    """All parameters for an opinion dynamics simulation."""

    # --- Network (spatial clusters) ---
    n_nodes: int = 10_000
    n_parties: int = 3
    n_cities: int = 5
    city_sizes: list[float] | None = None  # relative sizes; None → equal
    city_radius: float = 0.1  # Gaussian spread around city center
    intra_city_density: float = 0.005  # directed-edge probability within a city
    inter_city_density: float = 0.0005  # directed-edge probability between cities

    # --- Node attribute distributions (Beta α, β for each) ---
    attribute_distributions: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "health_status": (2.0, 2.0),
            "capital": (2.0, 5.0),
            "family_status": (2.0, 3.0),
            "education": (2.0, 3.0),
            "age": (2.0, 2.0),
            "credulity": (3.0, 3.0),
            "charisma": (2.0, 5.0),
        }
    )

    # --- Opinion initialisation ---
    opinion_mean: float = 0.0
    opinion_std: float = 0.15  # 0.3 put initial L2 distances (~0.73) far above the
    # Per-city bias: list of (party_index, offset) tuples, one per city.
    # Adds `offset` to every node in that city's opinion score for `party_index`,
    # giving each city a distinct initial lean.  None → uniform initialisation.
    city_opinion_biases: list[tuple[int, float]] | None = None
    # default confidence_threshold (0.4), cutting ~85% of edges immediately and
    # making peer influence negligible between events.  0.15 gives distances ~0.37,
    # keeping ~70% of edges active so convergence dynamics are clearly visible.

    # --- Dynamics ---
    confidence_threshold: float = 0.4  # bounded-confidence ε
    noise_std: float = 0.01

    # Weights for deriving peer_susceptibility from attributes:
    #   susceptibility = clip(base + Σ w_a * attr_a, 0, 1)
    susceptibility_weights: dict[str, float] = field(
        default_factory=lambda: {
            "base": 0.5,
            "health_status": -0.05,
            "capital": -0.15,
            "family_status": 0.10,
            "education": -0.10,
            "age": -0.05,
            "credulity": 0.30,
        }
    )

    # --- Edge weights ---
    influence_weight_min: float = 0.1
    influence_weight_max: float = 0.5

    # --- Parties ---
    # list[Party] defining each party's platform.  None → n_parties anonymous
    # parties with empty platforms (events have no opinion effect).
    parties: list | None = None  # elements are opsim.events.Party

    # --- Simulation ---
    n_steps: int = 100
    history_interval: int = 1  # record every N steps
    random_seed: int | None = 42
