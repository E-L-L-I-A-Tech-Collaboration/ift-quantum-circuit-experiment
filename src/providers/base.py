from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Sequence, Set

import numpy as np


@dataclass(frozen=True)
class ProviderContext:
    dt: float
    sigma: float
    rho: float
    beta: float
    coupling: float


class StateProvider(Protocol):
    def initialize(self, vertex_count: int, seed: int, config: Any) -> np.ndarray:
        ...

    def evolve(
        self,
        state: np.ndarray,
        adjacency: Sequence[Set[int]],
        config: Any,
        step: int,
    ) -> np.ndarray:
        ...

    def metadata(self) -> Dict[str, Any]:
        ...
