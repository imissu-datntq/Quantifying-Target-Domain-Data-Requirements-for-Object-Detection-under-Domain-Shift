# Repository Guidelines

## Project Structure & Module Organization

The current research protocol is `docs/proposal_dataset_aligned.docx`; summarize public-facing status in `README.md` and keep live execution state in `.agent/`. `train_da_yolo_comparison.py` is an earlier standard-library prototype for three YOLO adaptation repositories, not the implementation of the current Faster R-CNN proposal. Raw datasets belong under `data/`; generated manifests, external repositories, checkpoints, and runs remain untracked.

## Build, Test, and Development Commands

There is no build step or project-level dependency file yet. For the existing prototype:

- `python train_da_yolo_comparison.py --help` lists available workflows.
- `python train_da_yolo_comparison.py bootstrap` clones the three official implementations into `external/`.
- `python train_da_yolo_comparison.py prepare --domain lighting` creates a leakage-safe dataset protocol.
- `python train_da_yolo_comparison.py train --model sf-yolo --domain lighting ... --dry-run` validates inputs and prints the upstream command without training.
- `python -c "import ast, pathlib; ast.parse(pathlib.Path('train_da_yolo_comparison.py').read_text(encoding='utf-8'))"` performs a syntax check without creating bytecode.

Run each upstream repository in its own compatible environment. Do not launch a full training campaign until the implementation matches the proposal's fixed split and approval gates.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation, type hints, and `pathlib.Path` for filesystem paths. Use `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for constants, and lowercase hyphenated CLI values such as `ssda-yolo`. Prefer small changes, standard-library solutions, explicit validation, and deterministic seeds. Preserve the current command-adapter boundaries rather than reimplementing upstream methods.

## Testing Guidelines

No automated suite or coverage threshold exists yet. Run the syntax check and affected `--help` or `--dry-run` path for prototype changes. New dataset logic must assert exactly 55,000 training images, 14,863 held-out test images, 10,000 validation images, matching image/annotation counts, and zero basename overlap. Keep samples nested by seed; never use held-out labels for training, tuning, method selection, or backup selection.

## Commit & Pull Request Guidelines

Use short, imperative subjects such as `Validate target budget bounds`. Pull requests should identify the affected research question, dataset/domain assumptions, verification commands, and representative metric or log paths. Call out any change to split membership, holdout access, sampling seeds, recovery definitions, or training schedules. Do not commit datasets, generated runs, model weights, credentials, or cloned upstream repositories.
