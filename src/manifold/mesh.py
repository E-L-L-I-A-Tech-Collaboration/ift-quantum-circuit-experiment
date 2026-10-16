from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .geometry import Edge, Face, canonical_edge, orient_ccw


def build_rectangular_triangulation(
    nx: int = 8,
    ny: int = 8,
    jitter: float = 0.035,
    seed: int = 7,
) -> Tuple[np.ndarray, List[Face]]:
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    points = np.column_stack([xs.ravel(), ys.ravel()])

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            idx = j * nx + i
            points[idx] += rng.normal(0.0, jitter, size=2)

    faces: List[Face] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = j * nx + (i + 1)
            v01 = (j + 1) * nx + i
            v11 = (j + 1) * nx + (i + 1)

            if (i + j) % 2 == 0:
                cells = [(v00, v10, v11), (v00, v11, v01)]
            else:
                cells = [(v00, v10, v01), (v10, v11, v01)]
            faces.extend(orient_ccw(points, face) for face in cells)

    return points, faces


def edge_to_faces(faces: Sequence[Face]) -> Dict[Edge, List[int]]:
    mapping: Dict[Edge, List[int]] = defaultdict(list)
    for face_index, (a, b, c) in enumerate(faces):
        for edge in (canonical_edge(a, b), canonical_edge(b, c), canonical_edge(c, a)):
            mapping[edge].append(face_index)
    return mapping


def edges_from_faces(faces: Sequence[Face]) -> Set[Edge]:
    return set(edge_to_faces(faces))


def adjacency_from_faces(n_vertices: int, faces: Sequence[Face]) -> List[Set[int]]:
    adjacency = [set() for _ in range(n_vertices)]
    for a, b, c in faces:
        for source, target in ((a, b), (b, c), (c, a)):
            adjacency[source].add(target)
            adjacency[target].add(source)
    return adjacency