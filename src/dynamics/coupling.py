from __future__ import annotations

from typing import Sequence, Set

import numpy as np

from src.manifold.geometry import Face
from src.manifold.mesh import edges_from_faces


def diffusive_coupling(state: np.ndarray, adjacency: Sequence[Set[int]]) -> np.ndarray:
    diffusive = np.zeros_like(state)
    for index, neighbors in enumerate(adjacency):
        if not neighbors:
            continue
        neighbor_indices = np.fromiter(neighbors, dtype=int)
        diffusive[index] = state[neighbor_indices].mean(axis=0) - state[index]
    return diffusive


def field_mismatch_energy(state: np.ndarray, faces: Sequence[Face]) -> float:
    edges = edges_from_faces(faces)
    energy = 0.0
    for a, b in edges:
        energy += float(np.sum((state[a] - state[b]) ** 2))
    return energy / max(1, len(edges))