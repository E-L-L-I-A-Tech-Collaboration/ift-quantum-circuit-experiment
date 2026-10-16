from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Set

import numpy as np

from .base import StateProvider


class RecordedStateProvider(StateProvider):
    """Replay provider for externally recorded state trajectories.

    Supported formats:
    - NPZ with key `states` and optional `metadata`
    - JSON containing `states` and optional `metadata`
    - CSV with columns: step, vertex, x, y, z
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._states: np.ndarray | None = None
        self._metadata: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Recorded states path does not exist: {self.path}")

        suffix = self.path.suffix.lower()
        if suffix == ".npz":
            data = np.load(self.path, allow_pickle=True)
            if "states" not in data:
                raise ValueError("NPZ recorded data must contain `states` array.")
            self._states = np.asarray(data["states"], dtype=float)
            if "metadata" in data:
                metadata = data["metadata"]
                if isinstance(metadata, np.ndarray) and metadata.shape == ():
                    metadata = metadata.item()
                if isinstance(metadata, dict):
                    self._metadata = metadata
        elif suffix == ".json":
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._states = np.asarray(payload.get("states"), dtype=float)
            self._metadata = dict(payload.get("metadata", {}))
        elif suffix == ".csv":
            self._states, self._metadata = self._load_csv(self.path)
        else:
            raise ValueError("RecordedStateProvider supports .npz, .json, and .csv files only.")

        if self._states is None:
            raise ValueError("No states were loaded from recorded data file.")
        if self._states.ndim != 3 or self._states.shape[-1] != 3:
            raise ValueError(
                "Recorded states must have shape (steps + 1, vertex_count, 3)."
            )
        if not np.isfinite(self._states).all():
            raise ValueError("Recorded states contain non-finite values.")

    def _load_csv(self, path: Path) -> tuple[np.ndarray, Dict[str, Any]]:
        rows: list[tuple[int, int, float, float, float]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            expected = {"step", "vertex", "x", "y", "z"}
            if not expected.issubset(set(reader.fieldnames or [])):
                raise ValueError("CSV recorded data must include step, vertex, x, y, z columns.")
            for record in reader:
                rows.append(
                    (
                        int(record["step"]),
                        int(record["vertex"]),
                        float(record["x"]),
                        float(record["y"]),
                        float(record["z"]),
                    )
                )
        if not rows:
            raise ValueError("CSV recorded data is empty.")

        max_step = max(row[0] for row in rows)
        max_vertex = max(row[1] for row in rows)
        states = np.full((max_step + 1, max_vertex + 1, 3), np.nan, dtype=float)
        for step, vertex, x, y, z in rows:
            states[step, vertex] = np.asarray([x, y, z], dtype=float)

        if np.isnan(states).any():
            raise ValueError("CSV recorded data has missing step/vertex entries.")
        return states, {"source_format": "csv"}

    def initialize(self, vertex_count: int, seed: int, config: Any) -> np.ndarray:
        del seed
        assert self._states is not None
        if self._states.shape[1] != vertex_count:
            raise ValueError(
                f"Recorded vertex count {self._states.shape[1]} does not match mesh vertex count {vertex_count}."
            )
        if self._states.shape[0] < config.steps + 1:
            raise ValueError(
                f"Recorded states contain {self._states.shape[0]} steps, but config requires {config.steps + 1}."
            )
        return np.array(self._states[0], dtype=float)

    def evolve(
        self,
        state: np.ndarray,
        adjacency: Sequence[Set[int]],
        config: Any,
        step: int,
    ) -> np.ndarray:
        del state, adjacency
        assert self._states is not None
        next_index = step + 1
        if next_index >= self._states.shape[0]:
            raise IndexError(f"Recorded states exhausted at step={step}.")
        return np.array(self._states[next_index], dtype=float)

    def metadata(self) -> Dict[str, Any]:
        claimed_backend = str(self._metadata.get("backend", "unknown")).lower()
        provenance = "recorded_replay"
        if "quantum" in claimed_backend:
            provenance = "recorded_quantum_replay"
        return {
            "provider": "recorded_data",
            "mode": provenance,
            "source": str(self.path),
            "recorded_metadata": self._metadata,
            "note": "Replay mode only; backend claims come from recorded metadata.",
        }
