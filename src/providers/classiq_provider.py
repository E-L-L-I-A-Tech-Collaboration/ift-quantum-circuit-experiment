from __future__ import annotations

from typing import Any, Dict, Sequence, Set

import numpy as np

from src.quantum.classiq_provider import DEFAULT_FP_FRAC, DEFAULT_FP_SIZE, synthesize_lorenz_step

from .base import StateProvider


class ClassiqStateProvider(StateProvider):
    """Optional Classiq provider scaffold.

    This provider currently validates a local reversible arithmetic primitive
    (Lorenz Euler step synthesis) and does not execute the complete adaptive
    manifold feedback loop on quantum hardware.
    """

    def __init__(
        self,
        fp_size: int = DEFAULT_FP_SIZE,
        fp_frac: int = DEFAULT_FP_FRAC,
        enable_synthesis_check: bool = False,
    ):
        self.fp_size = fp_size
        self.fp_frac = fp_frac
        self.enable_synthesis_check = enable_synthesis_check
        self._synthesis_checked = False

    def _ensure_classiq(self) -> None:
        try:
            import classiq  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "ClassiqStateProvider requires optional dependency `classiq`. "
                "Install with: pip install .[quantum]"
            ) from exc

    def initialize(self, vertex_count: int, seed: int, config: Any) -> np.ndarray:
        del config
        self._ensure_classiq()
        rng = np.random.default_rng(seed)
        state = rng.normal(0.0, 0.08, size=(vertex_count, 3))
        state[:, 0] += 0.1
        if self.enable_synthesis_check and not self._synthesis_checked:
            synthesize_lorenz_step(fp_size=self.fp_size, fp_frac=self.fp_frac)
            self._synthesis_checked = True
        return state

    def evolve(
        self,
        state: np.ndarray,
        adjacency: Sequence[Set[int]],
        config: Any,
        step: int,
    ) -> np.ndarray:
        del adjacency, config, step
        self._ensure_classiq()
        raise NotImplementedError(
            "ClassiqStateProvider does not yet provide full manifold-loop state evolution. "
            "It currently supports local arithmetic synthesis validation only."
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": "classiq",
            "mode": "quantum_local_arithmetic_prototype",
            "fp_size": self.fp_size,
            "fp_frac": self.fp_frac,
            "note": "Local arithmetic primitive validation only; not full adaptive-manifold execution.",
        }
