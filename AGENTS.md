# Repository Guidelines

## Project Structure & Module Organization

The repository currently centers on `train_da_yolo_comparison.py`, a standard-library Python driver for preparing BDD100K data and orchestrating SSDA-YOLO, MS-DAYOLO, and SF-YOLO. Raw datasets belong under `data/`; do not commit them. The driver creates `external/` for upstream model repositories, `prepared_da_yolo/` for generated manifests/configuration, and `runs/da_yolo_comparison/` for logs, checkpoints, metrics, and comparison reports. Keep shared protocol logic in the driver unless the file becomes demonstrably difficult to maintain.

## Build, Test, and Development Commands

There is no build step or project-level dependency file; the driver uses Python's standard library.

- `python train_da_yolo_comparison.py --help` lists available workflows.
- `python train_da_yolo_comparison.py bootstrap` clones the three official implementations into `external/`.
- `python train_da_yolo_comparison.py prepare --domain lighting` creates a leakage-safe dataset protocol.
- `python train_da_yolo_comparison.py train --model sf-yolo --domain lighting ... --dry-run` validates inputs and prints the upstream command without training.
- `python -m compileall -q train_da_yolo_comparison.py` performs a quick syntax check.

Run each upstream repository in its own compatible environment; this repository does not unify their dependencies.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation, type hints, and `pathlib.Path` for filesystem paths. Use `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for constants, and lowercase hyphenated CLI values such as `ssda-yolo`. Prefer small changes, standard-library solutions, explicit validation, and deterministic seeds. Preserve the current command-adapter boundaries rather than reimplementing upstream methods.

## Testing Guidelines

No automated test suite or coverage threshold exists yet. For every change, run the syntax check and the affected command with `--help` or `--dry-run`. Add focused `test_*.py` tests with `unittest` only when introducing non-trivial parsing or transformation logic. Never use target-test labels during preparation or adaptation training.

## Commit & Pull Request Guidelines

History currently contains only `Initial commit`, so no formal convention is established. Use short, imperative subjects (for example, `Validate target budget bounds`). Pull requests should explain the affected workflow, list verification commands, identify dataset/domain assumptions, and include representative metric or log paths when behavior changes. Do not commit datasets, generated runs, model weights, credentials, or cloned upstream repositories.
