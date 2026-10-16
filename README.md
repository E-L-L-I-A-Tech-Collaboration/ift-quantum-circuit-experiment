# IFT Quantum Manifold Research Prototype

This repository originated from an HRM-inspired dynamical systems exploration and has been evolved into a Track 2 research prototype focused on adaptive simplicial-manifold computation.

Current results demonstrate a classical adaptive simplicial-manifold simulation with a separate quantum arithmetic proof of concept. The complete feedback loop has not yet been executed on quantum hardware.

## Scientific Scope

This repository currently contains three distinct tracks:

1. Track 1 (classical baseline): coupled Lorenz-type network dynamics over a triangulated interaction manifold.
2. Track 2 (current focus): adaptive manifold surgery with topology-preserving edge flips and benchmarkable fixed versus adaptive controls.
3. Classiq local quantum arithmetic prototype: a reversible arithmetic/synthesis proof of concept for a small Lorenz-like step.

What is **not** currently claimed:

- Quantum advantage.
- Topological quantum computation.
- Fault-tolerant quantum computation.
- A completed quantum-output-driven manifold loop.

## Current Experiment Loop

State provider
-> local state evolution
-> topological diagnostics
-> validated surgery
-> updated adjacency
-> next state evolution

The provider abstraction allows future quantum-derived state outputs to be integrated without rewriting the manifold engine.

## Installation

Core (classical):

```bash
pip install -e .
```

Development tools:

```bash
pip install -e .[dev]
```

Optional Classiq extras:

```bash
pip install -e .[quantum]
```

## Codespaces Usage

```bash
pip install -e .[dev]
pytest -q
python -m src.experiments.run --mode adaptive --provider classical --steps 200 --seed 11 --output results/runs
```

## Colab Usage

Use the notebook at `IFT_Quantum_Manifold_Prototype.ipynb`.

The notebook:

1. clones/refreshes the repository,
2. installs dependencies,
3. prints commit SHA,
4. runs smoke validation,
5. runs adaptive and fixed simulations,
6. saves run artifacts,
7. zips outputs.

## CLI Usage

Single adaptive run:

```bash
python -m src.experiments.run --mode adaptive --provider classical --steps 1200 --seed 11 --output results/runs
```

Single fixed run:

```bash
python -m src.experiments.run --mode fixed --provider classical --steps 1200 --seed 11 --output results/runs
```

Paired benchmark:

```bash
python -m src.experiments.run --benchmark paired --provider classical --seeds 0 1 2 3 4 --output results/runs
```

## Configurations

Versioned experiment configs are under `configs/`:

- `colab_smoke.json`
- `adaptive_baseline.json`
- `fixed_control.json`
- `paired_benchmark.json`
- `classiq_local_step.json`

## Output Schema

Each run produces:

- `config.json`
- `metadata.json`
- `time_series.csv`
- `topology_history.csv`
- `surgery_log.csv`
- `final_state.npz`
- `initial_mesh.npz`
- `final_mesh.npz`
- `summary.json`
- `snapshots/`
- `figures/`

## Reproducibility and Results

- Runtime metadata includes Python, OS, package versions, commit SHA, seed, provider, topology mode, and configuration hash.
- Curated reference outputs can be placed in `results/reference/`.
- Generated run data is placed in `results/runs/`.

## Repository Structure

- `src/dynamics`: local Lorenz network dynamics.
- `src/manifold`: mesh geometry and invariants.
- `src/topology`: diagnostics and topology-preserving surgery.
- `src/providers`: pluggable state providers (classical, recorded replay, optional Classiq scaffold).
- `src/experiments`: experiment engine, benchmarking, CLI, reproducible serialization.
- `src/quantum`: Classiq local arithmetic prototype and stubs.
- `src/visualization`: plotting helpers.
- `docs/`: architecture, protocol, provenance, roadmap, interpretation guidance.

## Classiq Setup

Classiq remains optional and is not required for classical runs, tests, or CI smoke benchmarks.

## Scientific Limitations

Current evidence supports:

- bounded classical dynamics in tested ranges,
- topology-preserving adaptive surgery,
- benchmarkable adaptive-versus-fixed behavior.

Current evidence does not support claims of full quantum execution of the adaptive manifold loop.
