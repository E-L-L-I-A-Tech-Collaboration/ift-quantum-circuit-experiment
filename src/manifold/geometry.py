from __future__ import annotations

from typing import Sequence, Tuple

import math
import numpy as np


Edge = Tuple[int, int]
Face = Tuple[int, int, int]


def canonical_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def oriented_area(points: np.ndarray, face: Face) -> float:
    a, b, c = points[list(face)]
    ab = b - a
    ac = c - a
    return 0.5 * float(ab[0] * ac[1] - ab[1] * ac[0])


def orient_ccw(points: np.ndarray, face: Sequence[int]) -> Face:
    oriented = tuple(int(vertex) for vertex in face)
    if oriented_area(points, oriented) < 0:
        return (oriented[0], oriented[2], oriented[1])
    return oriented


def triangle_quality(points: np.ndarray, face: Face) -> float:
    vertices = points[list(face)]
    lengths2 = [
        float(np.sum((vertices[0] - vertices[1]) ** 2)),
        float(np.sum((vertices[1] - vertices[2]) ** 2)),
        float(np.sum((vertices[2] - vertices[0]) ** 2)),
    ]
    area = abs(oriented_area(points, face))
    denominator = sum(lengths2)
    if denominator <= 1e-15:
        return 0.0
    return 4.0 * math.sqrt(3.0) * area / denominator