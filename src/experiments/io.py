from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
import numpy as np
import scipy


def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {key: _to_serializable(value) for key, value in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(key): _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(value) for value in obj]
    return obj


def config_hash(config: Any) -> str:
    payload = _to_serializable(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def build_run_id(config: Any) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{config_hash(config)}"


def _git_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def environment_metadata() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "git_commit_sha": _git_commit_sha(),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_serializable(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_time_series(path: Path, result: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time", "objective", "synchronization"])
        dt = float(result["config"].dt)
        for idx, (time_value, objective_value, sync_value) in enumerate(
            zip(result["history_t"], result["history_obj"], result["history_sync"])
        ):
            writer.writerow([idx, float(time_value if time_value is not None else idx * dt), float(objective_value), float(sync_value)])


def write_topology_history(path: Path, result: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "vertices",
                "edges",
                "faces",
                "euler_characteristic",
                "connected_components",
                "boundary_loops",
                "nonmanifold_edges",
                "orientable",
                "beta0",
                "beta1",
                "beta2",
            ]
        )
        for snap in result["topology_history"]:
            inv = snap.invariants
            betti = snap.betti
            writer.writerow(
                [
                    snap.step,
                    inv.vertices,
                    inv.edges,
                    inv.faces,
                    inv.euler_characteristic,
                    inv.connected_components,
                    inv.boundary_loops,
                    inv.nonmanifold_edges,
                    inv.orientable,
                    betti.beta0,
                    betti.beta1,
                    betti.beta2,
                ]
            )


def write_surgery_log(path: Path, result: dict[str, Any]) -> None:
    records = list(result.get("surgery_log", []))
    if not records:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "step",
                    "candidate_edge",
                    "accepted",
                    "rejection_reason",
                    "objective_before",
                    "candidate_objective",
                    "objective_delta",
                    "triangle_quality_before",
                    "triangle_quality_after",
                ]
            )
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "candidate_edge",
                "accepted",
                "rejection_reason",
                "objective_before",
                "candidate_objective",
                "objective_delta",
                "triangle_quality_before",
                "triangle_quality_after",
            ]
        )
        for row in records:
            edge = row.get("candidate_edge", [None, None])
            edge_repr = f"{edge[0]}-{edge[1]}"
            writer.writerow(
                [
                    row.get("step"),
                    edge_repr,
                    row.get("accepted"),
                    row.get("rejection_reason", ""),
                    row.get("objective_before"),
                    row.get("candidate_objective"),
                    row.get("objective_delta"),
                    row.get("triangle_quality_before"),
                    row.get("triangle_quality_after"),
                ]
            )


def write_snapshots(snapshot_dir: Path, result: dict[str, Any]) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[list[Any]] = []

    for snapshot in result.get("snapshots", []):
        step = int(snapshot["step"])
        np.savez_compressed(
            snapshot_dir / f"snapshot_{step:05d}.npz",
            step=step,
            time=float(snapshot["time"]),
            state=np.asarray(snapshot["state"], dtype=float),
            faces=np.asarray(snapshot["faces"], dtype=int),
            objective=float(snapshot["objective"]),
            synchronization=float(snapshot["synchronization"]),
        )
        write_json(snapshot_dir / f"snapshot_{step:05d}.json", snapshot)
        summary_rows.append(
            [
                step,
                float(snapshot["time"]),
                float(snapshot["objective"]),
                float(snapshot["synchronization"]),
                float(snapshot["mesh_quality"]["mean_triangle_quality"]),
                float(snapshot["mesh_quality"]["min_triangle_quality"]),
                snapshot["graph"]["graph_diameter"],
                snapshot["graph"]["laplacian_spectral_gap"],
            ]
        )

    with (snapshot_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "time",
                "objective",
                "synchronization",
                "mean_triangle_quality",
                "min_triangle_quality",
                "graph_diameter",
                "laplacian_spectral_gap",
            ]
        )
        writer.writerows(summary_rows)


def save_run(
    result: dict[str, Any],
    output_root: str | Path,
    data_provenance: str,
    run_id: str | None = None,
) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = result["config"]
    run_id_value = run_id or build_run_id(config)
    run_dir = output_root / run_id_value
    run_dir.mkdir(parents=True, exist_ok=False)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "config.json", config)
    write_json(
        run_dir / "metadata.json",
        {
            "run_id": run_id_value,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed": True,
            "provider": result.get("provider_metadata", {}).get("provider", config.provider),
            "topology_mode": config.topology_mode,
            "seed": config.seed,
            "data_provenance": data_provenance,
            "configuration_hash": config_hash(config),
            "environment": environment_metadata(),
        },
    )

    write_time_series(run_dir / "time_series.csv", result)
    write_topology_history(run_dir / "topology_history.csv", result)
    write_surgery_log(run_dir / "surgery_log.csv", result)
    write_json(run_dir / "summary.json", result["summary"])

    np.savez_compressed(run_dir / "final_state.npz", state=np.asarray(result["state"], dtype=float))
    np.savez_compressed(
        run_dir / "initial_mesh.npz",
        points=np.asarray(result["points"], dtype=float),
        faces=np.asarray(result["initial_faces"], dtype=int),
    )
    np.savez_compressed(
        run_dir / "final_mesh.npz",
        points=np.asarray(result["points"], dtype=float),
        faces=np.asarray(result["faces"], dtype=int),
    )

    write_snapshots(run_dir / "snapshots", result)
    return run_dir


def reload_saved_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    final_state = np.load(run_dir / "final_state.npz")["state"]
    initial_mesh = np.load(run_dir / "initial_mesh.npz")
    final_mesh = np.load(run_dir / "final_mesh.npz")
    return {
        "config": config,
        "metadata": metadata,
        "summary": summary,
        "final_state": final_state,
        "initial_points": initial_mesh["points"],
        "initial_faces": initial_mesh["faces"],
        "final_points": final_mesh["points"],
        "final_faces": final_mesh["faces"],
    }
