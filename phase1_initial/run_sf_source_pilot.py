"""Train the bounded SF-YOLO source model on exactly 1,000 source images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_command(*, python: Path, upstream: Path, output: Path,
                  weights: Path, data: Path, hyp: Path) -> list[str]:
    return [str(python), str(upstream / 'train_source.py'),
            '--weights', str(weights), '--data', str(data), '--hyp', str(hyp),
            '--epochs', '1', '--batch-size', '1', '--imgsz', '1280',
            '--rect', '--workers', '0', '--device', '0',
            '--project', str(output.parent), '--name', output.name,
            '--noautoanchor', '--noplots', '--technical-no-amp', '--seed', '42']


def write_status(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2)
        stream.write('\n')


def update_status(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_source_counts(data: Path) -> tuple[int, int]:
    preparation = data.parent / 'preparation.json'
    if not preparation.is_file():
        raise FileNotFoundError(preparation)
    report = json.loads(preparation.read_text(encoding='utf-8'))
    train = int(report.get('source_train_images', 0))
    validation = int(report.get('source_validation_images', 0))
    if train != 1000 or validation != 1000:
        raise ValueError('SF source pilot requires exactly 1,000 train and 1,000 validation images')
    if report.get('holdout_access') is not False:
        raise ValueError('Source preparation must explicitly exclude holdout access')
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--hyp', type=Path, required=True)
    parser.add_argument('--weights', type=Path,
                        default=ROOT / 'weights/yolov5l-v7.0.pt')
    parser.add_argument('--timeout', type=int, default=14400)
    args = parser.parse_args()

    output = args.output.resolve()
    data = args.data.resolve()
    hyp = args.hyp.resolve()
    weights = args.weights.resolve()
    sidecar = output.with_name(output.name + '.status.json')
    log_path = output.with_name(output.name + '.log')
    if output.exists() or sidecar.exists() or log_path.exists():
        raise FileExistsError(f'Run identity already exists: {output}')

    upstream = ROOT / 'external/sf-yolo'
    python = ROOT / '.venvs/sf-yolo/bin/python'
    paths = [python, upstream / 'train_source.py', data, hyp, weights]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(', '.join(missing))
    if args.timeout <= 0:
        raise ValueError('Timeout must be positive')
    train_images, validation_images = read_source_counts(data)
    command = build_command(python=python, upstream=upstream, output=output,
                            weights=weights, data=data, hyp=hyp)
    state: dict[str, Any] = {
        'status': 'running', 'candidate': 'sf-yolo', 'phase': 'source',
        'domain': 'source', 'source_train_images': train_images,
        'source_validation_images': validation_images,
        'epochs': 1, 'expected_training_batches': train_images,
        'selection_evidence': False, 'holdout_access': False,
        'geometry': 'native 1280 content; rectangular loader; batch=1',
        'weights_sha256': digest(weights), 'data_sha256': digest(data),
        'hyp_sha256': digest(hyp), 'command': command,
        'timeout_seconds': args.timeout,
    }
    write_status(sidecar, state)
    started = time.monotonic()
    try:
        env = os.environ.copy()
        env['PYTHONNOUSERSITE'] = '1'
        with log_path.open('x', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=upstream, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.timeout,
                                    check=False)
        state['exit_code'] = result.returncode
        state['wall_seconds'] = time.monotonic() - started
        text = log_path.read_text(errors='replace')
        state['nonfinite_log'] = bool(re.search(r'\b(?:nan|inf)\b', text, flags=re.I))
        checkpoints = [output / 'weights/last.pt', output / 'weights/best.pt']
        state['checkpoint_paths'] = [str(path) for path in checkpoints if path.is_file()]
        state['status'] = ('completed' if result.returncode == 0
                           and (output / 'weights/last.pt').is_file()
                           and not state['nonfinite_log'] else 'incomplete')
        if output.is_dir():
            update_status(output / 'status.json', state)
    except BaseException as error:
        state.update(status='failed', error=repr(error),
                     wall_seconds=time.monotonic() - started)
        if output.is_dir():
            update_status(output / 'status.json', state)
        raise
    finally:
        update_status(sidecar, state)
    print(json.dumps(state, indent=2))
    if state['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
