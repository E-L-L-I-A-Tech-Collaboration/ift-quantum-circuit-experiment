# Results Directory

- `results/runs/`: generated run artifacts (ignored in git by default).
- `results/reference/`: curated reproducible reference runs that may be committed.
- `results/figures/`: curated aggregate figures.
- `results/summaries/`: curated summary tables.

Each generated run folder should include:

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
