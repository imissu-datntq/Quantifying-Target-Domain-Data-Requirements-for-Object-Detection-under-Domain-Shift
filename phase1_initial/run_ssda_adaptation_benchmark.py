"""Run the pinned SSDA image-only graph through the bounded pilot pool."""
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

from pilot_sampling import read_preparation_counts


ROOT = Path(__file__).resolve().parent
DEFAULT_PILOT_UPDATES = 151


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_command(*, python: Path, upstream: Path, output: Path,
                  source_checkpoint: Path, data: Path, hyp: Path,
                  epochs: int, max_optimizer_updates: int) -> list[str]:
    return [str(python), str(upstream / 'ssda_yolov5_train.py'),
            '--weights', str(source_checkpoint), '--data', str(data),
            '--hyp', str(hyp), '--epochs', str(epochs), '--batch-size', '1',
            '--img-size', '1280', '1280', '--rect', '--unlabeled-target',
            '--target-seed', '42', '--noautoanchor', '--notest', '--workers', '0',
            '--max-optimizer-updates', str(max_optimizer_updates),
            '--device', '0', '--project', str(output.parent), '--name', output.name,
            '--conf_thres', '0.001', '--iou_thres', '0.3', '--max_gt_boxes', '20',
            '--consistency_loss', '--alpha_weight', '2.0']


def write_status(path: Path, state: dict[str, Any]) -> None:
    """Create a status artifact once; retries must use a new run identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2)
        stream.write('\n')


def update_status(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-checkpoint', type=Path,
                        default=ROOT / 'weights/yolov5l-v5.0.pt')
    parser.add_argument('--data', type=Path,
                        default=ROOT / 'runs/ssda-adapt-inputs-001/data.yaml')
    parser.add_argument('--hyp', type=Path,
                        default=ROOT / 'runs/ssda-adapt-inputs-001/hyp.yaml')
    # The pinned upstream warms up accumulation for 1,000 batches and then
    # performs roughly two updates per 128-image epoch.  Thirty-six epochs
    # therefore clear the 120-update gate without changing upstream logic.
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max-optimizer-updates', type=int,
                        default=DEFAULT_PILOT_UPDATES)
    parser.add_argument('--timeout', type=int, default=14400)
    args = parser.parse_args()
    output = args.output.resolve()
    sidecar = output.with_name(output.name + '.status.json')
    log_path = output.with_name(output.name + '.log')
    if output.exists() or sidecar.exists() or log_path.exists():
        raise FileExistsError(f'Run identity already exists: {output}')
    upstream = ROOT / 'external/SSDA-YOLO'
    python = ROOT / '.venvs/ssda-yolo/bin/python'
    paths = [python, upstream / 'ssda_yolov5_train.py', args.source_checkpoint,
             args.data, args.hyp]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(', '.join(missing))
    if args.epochs <= 0 or args.max_optimizer_updates <= 0:
        raise ValueError('Epochs and optimizer-update limit must be positive')
    source_images, target_images = read_preparation_counts(args.data)
    command = build_command(python=python, upstream=upstream, output=output,
                            source_checkpoint=args.source_checkpoint.resolve(),
                            data=args.data.resolve(), hyp=args.hyp.resolve(),
                            epochs=args.epochs,
                            max_optimizer_updates=args.max_optimizer_updates)
    state: dict[str, Any] = {
        'status': 'running', 'candidate': 'ssda-yolo', 'phase': 'adapt',
        'domain': 'lighting', 'labeled_budget': 0,
        'source_train_images': source_images,
        'target_train_images': target_images,
        'selection_evidence': False, 'holdout_access': False,
        'expected_optimizer_updates': args.max_optimizer_updates,
        'epochs': args.epochs,
        'geometry': 'native 1280 content with rectangular stride padding; batch=1',
        'source_checkpoint_sha256': digest(args.source_checkpoint),
        'data_sha256': digest(args.data), 'hyp_sha256': digest(args.hyp),
        'command': command, 'timeout_seconds': args.timeout,
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
        trace = output / 'technical_batches.jsonl'
        state['trace_exists'] = trace.is_file()
        updates = []
        if trace.is_file():
            updates = [json.loads(line) for line in trace.read_text().splitlines()
                       if line.strip()]
        state['batches'] = len(updates)
        state['optimizer_updates'] = max((int(item['ema_updates']) for item in updates), default=0)
        text = log_path.read_text(errors='replace')
        state['nonfinite_log'] = bool(re.search(r'\b(?:nan|inf)\b', text, flags=re.I))
        state['checkpoint_paths'] = [str(path) for path in
                                     (output / 'weights/last_student.pt',
                                      output / 'weights/last_teacher.pt')
                                     if path.is_file()]
        complete = (result.returncode == 0
                    and state['optimizer_updates'] == args.max_optimizer_updates
                    and state['batches'] >= max(source_images, target_images)
                    and len(state['checkpoint_paths']) == 2 and not state['nonfinite_log'])
        state['status'] = 'completed' if complete else 'incomplete'
        if output.is_dir():
            update_status(output / 'status.json', state)
    except BaseException as error:
        state.update(status='failed', error=repr(error), wall_seconds=time.monotonic() - started)
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
