# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is a **scaffolded but unimplemented** project. Every module under `src/calibration_drift/` currently contains a one-line docstring and a stub that raises `NotImplementedError`. The scaffold mirrors the six-step plan in `README.md` — there is no working pipeline yet.

When asked to implement something, expect to be filling in stubs rather than refactoring existing logic. Do not delete the stub signatures without checking — they define the intended public API for each step.

## Architecture

The codebase is organized so that **each subpackage under `src/calibration_drift/` maps 1:1 to a step in the README pipeline**. Reading `README.md` is the fastest way to understand the intended end-to-end flow.

Data flow (left → right):

```text
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

## Dev environment

Code is edited on Windows and built/run inside Docker on an EC2 (Linux) instance. There is no local Docker workflow and no local repo on EC2 — the **Dockerfile clones the repo at build time**, so the image is a self-contained artifact and the only file needed on the EC2 host is the Dockerfile itself.

Inside the running container, the repo lives at `/app` and is installed in editable mode (`pip install -e ".[dev]"`). `git pull` from inside the container picks up code changes immediately — no reinstall needed.

When adding a Python dependency, edit `pyproject.toml`, commit + push from Windows, then rebuild the image on EC2 — do not `pip install` ad-hoc inside a running container.

## Commands

Build args `REPO_URL` (required) and `REPO_REF` (optional, defaults to `main`) control which repo + branch the image clones at build time.

```bash
# On EC2 — repo is private, cloned via SSH. ssh-agent must be running with the
# deploy key loaded; BuildKit forwards the agent socket into the build.
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519        # or whichever key is registered with GitHub

docker build \
    --ssh default \
    --build-arg REPO_URL=git@github.com:<user>/Monitoring-Calibration-Drift.git \
    -t calibration-drift .

# Force re-clone when main has moved (Docker can't see remote HEAD changes)
docker build --no-cache --ssh default \
    --build-arg REPO_URL=git@github.com:<user>/Monitoring-Calibration-Drift.git \
    -t calibration-drift .

# Run the container — mount the host's SSH dir read-only so `git pull` inside
# the container can authenticate against the private repo.
docker run --rm -it -v ~/.ssh:/root/.ssh:ro calibration-drift

# Inside the container (WORKDIR is /app, the cloned repo):
pytest                                    # full suite
pytest tests/test_projection.py           # one file
pytest tests/test_projection.py::test_point_on_optical_axis_projects_to_principal_point  # one test
pytest -k "drift"                         # by name pattern

ruff check .
ruff format .
mypy src/

git pull                                  # update code in-place; editable install picks it up

# CLI entrypoints
python scripts/show_overlay.py --out /overlays/overlay.png       # implemented — renders LiDAR-on-image overlay
python scripts/run_ingest.py --scene <scene_token>               # stub
python scripts/run_drift_check.py --scene <scene_token> --config configs/drift_thresholds.yaml  # stub
python scripts/generate_report.py --experiment <mlflow_experiment>  # stub
```

## Conventions worth preserving

- **Data is never committed.** `data/` is gitignored; assume datasets live elsewhere and are referenced by path.
- **One subpackage per pipeline step.** When adding new functionality, place it in the subpackage matching its README step rather than creating new top-level modules.
- **Stubs use `raise NotImplementedError`**, not `pass` or fake return values. Keep this convention when adding new stubs — it makes "not yet built" loud and obvious.
- **Tests are currently `pytest.mark.skip`** placeholders. When implementing a module, replace the skip with a real test in the same commit.
