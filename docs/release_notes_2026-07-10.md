# Release Notes - 2026-07-10

## Track 2 Prototype Release (Main)

This release upgrades the repository into a reproducible Track 2 quantum-manifold research prototype with explicit scientific boundaries.

## Highlights

- Standardized state evolution providers:
  - `ClassicalLorenzProvider` (default baseline)
  - `RecordedStateProvider` (replay of NPZ/CSV/JSON trajectories)
  - optional `ClassiqStateProvider` scaffold with explicit non-fallback behavior
- Provider-driven experiment engine while preserving compatibility entrypoints.
- Fixed and adaptive topology modes with matched-seed paired benchmarking.
- Structured surgery proposal logging with acceptance/rejection reasons and objective deltas.
- Expanded objective, synchronization, topology, graph, and mesh-quality metrics.
- Reproducible output serialization schema and run folder layout in `results/runs/...`.
- CLI workflow for smoke, fixed/adaptive, and paired benchmark execution.
- Versioned JSON configurations in `configs/`.
- CI smoke checks and expanded pytest coverage.
- Documentation suite updates and Colab-oriented workflow notebook updates.
- GNU GPL v3 license added at repository root.

## Scientific Boundary (Important)

Current demonstrated results are from a **classical adaptive simplicial-manifold simulation**. The Classiq path in this release validates a **local reversible arithmetic primitive** and does **not** execute the full adaptive-manifold feedback loop on quantum hardware.

No claims are made here regarding:

- quantum advantage,
- fault-tolerant quantum computation,
- topological quantum computation,
- emergent genus selection,
- or full quantum-hardware closed-loop execution.

## Validation Snapshot

- Test suite: `pytest -q` passed (`19 passed`).
- Post-merge smoke run completed via CLI and artifacts generated under `results/runs/`.

## Merge Record

- PR merged: `#1`
- Merge commit on `main`: `2028667`
- Release note commit created on `main` after merge.
