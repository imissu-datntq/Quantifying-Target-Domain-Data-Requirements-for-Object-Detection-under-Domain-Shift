"""Run the pinned SF image-only graph through the bounded pilot pool."""
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
                  decoder: Path, encoder: Path, fc1: Path, fc2: Path,
                  epochs: int, max_optimizer_updates: int) -> list[str]:
    return [str(python), str(upstream / 'train_sf-yolo.py'),
            '--unlabeled-target', '--weights', str(source_checkpoint),
            '--data', str(data), '--hyp', str(hyp), '--epochs', str(epochs),
            '--batch-size', '1', '--imgsz', '1280', '--workers', '0',
            '--device', '0', '--project', str(output.parent), '--name', output.name,
            '--max-optimizer-updates', str(max_optimizer_updates),
            '--noautoanchor', '--noplots', '--noval', '--seed', '42',
            '--conf_thres', '0.001', '--iou_thres', '0.3', '--max_gt_boxes', '20',
            '--decoder_path', str(decoder), '--encoder_path', str(encoder),
            '--fc1', str(fc1), '--fc2', str(fc2), '--style_add_alpha', '0.4',
            '--SSM_alpha', '0.5']


def write_status(path: Path, state: dict[str, Any]) -> None:
    """Create a status artifact once; retries must use a new run directory."""
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
                        default=ROOT / 'runs/sf-pretrained-source-smoke-001/weights/last.pt')
    parser.add_argument('--data', type=Path, default=ROOT / 'runs/sf-adapt-smoke-001/data.yaml')
    parser.add_argument('--hyp', type=Path, default=ROOT / 'external/sf-yolo/data/hyps/hyp.scratch-low.yaml')
    parser.add_argument('--decoder', type=Path,
                        default=ROOT / 'runs/targetaugment-lighting-smoke-002/models/decoder_iter_120.pth')
    parser.add_argument('--encoder', type=Path, default=ROOT / 'weights/vgg16_ori.pth')
    parser.add_argument('--fc1', type=Path,
                        default=ROOT / 'runs/targetaugment-lighting-smoke-002/models/fc1_iter_120.pth')
    parser.add_argument('--fc2', type=Path,
                        default=ROOT / 'runs/targetaugment-lighting-smoke-002/models/fc2_iter_120.pth')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--max-optimizer-updates', type=int,
                        default=DEFAULT_PILOT_UPDATES)
    parser.add_argument('--timeout', type=int, default=14400)
    args = parser.parse_args()
    root = ROOT
    output = args.output.resolve()
    sidecar = output.with_name(output.name + '.status.json')
    log_path = output.with_name(output.name + '.log')
    if output.exists() or sidecar.exists() or log_path.exists():
        raise FileExistsError(f'Run identity already exists: {output}')
    upstream = root / 'external/sf-yolo'
    python = root / '.venvs/sf-yolo/bin/python'
    paths = [python, upstream / 'train_sf-yolo.py', args.source_checkpoint,
             args.data, args.hyp, args.decoder, args.encoder, args.fc1, args.fc2]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(', '.join(missing))
    if args.epochs <= 0 or args.max_optimizer_updates <= 0:
        raise ValueError('Epochs and optimizer-update limit must be positive')
    source_images, target_images = read_preparation_counts(args.data)
    command = build_command(python=python, upstream=upstream, output=output,
                            source_checkpoint=args.source_checkpoint.resolve(),
                            data=args.data.resolve(), hyp=args.hyp.resolve(),
                            decoder=args.decoder.resolve(), encoder=args.encoder.resolve(),
                            fc1=args.fc1.resolve(), fc2=args.fc2.resolve(),
                            epochs=args.epochs,
                            max_optimizer_updates=args.max_optimizer_updates)
    state: dict[str, Any] = {
        'status': 'running', 'candidate': 'sf-yolo', 'phase': 'adapt',
        'domain': 'lighting', 'labeled_budget': 0,
        'source_train_images': source_images,
        'target_train_images': target_images,
        'selection_evidence': False, 'holdout_access': False,
        'expected_optimizer_updates': args.max_optimizer_updates,
        'epochs': args.epochs,
        'geometry': 'native 1280 content; stride-compatible loader padding; batch=1',
        'source_checkpoint_sha256': digest(args.source_checkpoint),
        'data_sha256': digest(args.data), 'hyp_sha256': digest(args.hyp),
        'decoder_sha256': digest(args.decoder), 'encoder_sha256': digest(args.encoder),
        'fc1_sha256': digest(args.fc1), 'fc2_sha256': digest(args.fc2),
        'command': command, 'timeout_seconds': args.timeout,
    }
    write_status(sidecar, state)
    started = time.monotonic()
    try:
        env = os.environ.copy()
        env['PYTHONNOUSERSITE'] = '1'
        with log_path.open('x', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=upstream, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
        state['exit_code'] = result.returncode
        state['wall_seconds'] = time.monotonic() - started
        trace = output / 'technical_batches.jsonl'
        state['trace_exists'] = trace.is_file()
        updates = []
        if trace.is_file():
            for line in trace.read_text().splitlines():
                if line.strip():
                    updates.append(json.loads(line))
        state['batches'] = len(updates)
        state['optimizer_updates'] = max((int(item['ema_updates']) for item in updates), default=0)
        text = log_path.read_text(errors='replace')
        state['nonfinite_log'] = bool(re.search(r'\b(?:nan|inf)\b', text, flags=re.I))
        state['checkpoint_paths'] = [str(path) for path in
                                     (output / 'weights/last_student.pt', output / 'weights/last_teacher.pt')
                                     if path.is_file()]
        complete = (result.returncode == 0
                    and state['optimizer_updates'] == args.max_optimizer_updates
                    and state['batches'] >= target_images
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
