# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is a **scaffolded but unimplemented** project. Every module under `src/calibration_drift/` currently contains a one-line docstring and a stub that raises `NotImplementedError`. The scaffold mirrors the six-step plan in `README.md` — there is no working pipeline yet.

When asked to implement something, expect to be filling in stubs rather than refactoring existing logic. Do not delete the stub signatures without checking — they define the intended public API for each step.

## Architecture

The codebase is organized so that **each subpackage under `src/calibration_drift/` maps 1:1 to a step in the README pipeline**. Reading `README.md` is the fastest way to understand the intended end-to-end flow.

Data flow (left → right):

```
ingestion → features → projection → drift → viz / reporting
```

- `ingestion/` — Load synchronized LiDAR/Camera/Radar samples (NuScenes is the reference dataset) and align timestamps across modalities.
- `features/` — Extract 2D image edges (OpenCV) and 3D depth discontinuities / normals (Open3D).
- `projection/` — Project LiDAR points onto the image plane using known intrinsics + extrinsics. This is the "geometric ground truth" against which drift is measured.
- `drift/` — Compute reprojection error per frame (`error_metrics.py`) and detect drift over a sliding temporal window (`detector.py`). This is the core of the tool.
- `viz/` — Foxglove streaming + image overlay rendering for debugging.
- `reporting/` — MLflow logging of per-run calibration quality metrics.

`scripts/` holds thin `argparse` CLI entrypoints (`run_ingest.py`, `run_drift_check.py`, `generate_report.py`) that wire the subpackages together — they should stay thin; real logic lives in `src/`.

## Configuration

The tool is **config-driven**. Two YAML files in `configs/` control behavior:

- `configs/sensors.yaml` — per-sensor intrinsics + initial extrinsics (the "known-good" calibration that drift is measured against). Populated from dataset calibration files (e.g., NuScenes `calibrated_sensor.json`).
- `configs/drift_thresholds.yaml` — tuning knobs: per-frame warn/fail reprojection-error thresholds, temporal window size + slope threshold, Canny + depth-gradient parameters for edge extraction.

When changing detection behavior, prefer adjusting these YAMLs over hardcoding constants in module code.

## Commands

This is a `pyproject.toml`-based project installed in editable mode. From the repo root (PowerShell):

```powershell
# First-time setup
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Tests
pytest                                    # full suite
pytest tests/test_projection.py           # one file
pytest tests/test_projection.py::test_project_points_identity  # one test
pytest -k "drift"                         # by name pattern

# Lint / type-check
ruff check .
ruff format .
mypy src/

# CLI entrypoints (all currently raise NotImplementedError)
python scripts/run_ingest.py --scene <scene_token>
python scripts/run_drift_check.py --scene <scene_token> --config configs/drift_thresholds.yaml
python scripts/generate_report.py --experiment <mlflow_experiment>
```

`pip install -e .` is required before running anything — without it, the `calibration_drift` package isn't importable from `scripts/` or `tests/`.

## Conventions worth preserving

- **Data is never committed.** `data/` is gitignored; assume datasets live elsewhere and are referenced by path.
- **One subpackage per pipeline step.** When adding new functionality, place it in the subpackage matching its README step rather than creating new top-level modules.
- **Stubs use `raise NotImplementedError`**, not `pass` or fake return values. Keep this convention when adding new stubs — it makes "not yet built" loud and obvious.
- **Tests are currently `pytest.mark.skip`** placeholders. When implementing a module, replace the skip with a real test in the same commit.
