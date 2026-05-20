"""Distance and similarity functions for sequence comparison."""
"""Distance functions for sequence and subspace comparison."""

from src.distances.canonical_angles import canonical_angles, canonical_singular_values
from src.distances.subspace_distances import (
    DISTANCE_NAMES,
    distance_from_singular_values,
    pairwise_subspace_distances,
    subspace_distance,
)

__all__ = [
    "DISTANCE_NAMES",
    "canonical_angles",
    "canonical_singular_values",
    "distance_from_singular_values",
    "pairwise_subspace_distances",
    "subspace_distance",
]
