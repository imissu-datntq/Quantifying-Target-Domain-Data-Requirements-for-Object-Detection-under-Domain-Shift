"""Bounded native-geometry MS-DAYOLO adaptation graph smoke, not selection training."""
import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def adapt_config(text: str, steps: int, optimized_memory: int = 0) -> str:
    # Pinned upstream's three eight-class detection projections.
    if text.count('filters=39') != 3 or text.count('classes=8') != 3:
        raise ValueError('Unexpected upstream detector heads')
    text = text.replace('filters=39', 'filters=45').replace('classes=8', 'classes=10')
    values = dict(batch=4, subdivisions=2, width=1280, height=736,
                  learning_rate=.0001, burn_in=0, max_batches=steps,
                  policy='constant', mosaic=0, saturation=1, exposure=1, hue=0)
    for key, value in values.items():
        text, count = re.subn(r'^' + key + r'\s*=.*$', f'{key}={value}', text, flags=re.M)
        if count != 1:
            raise ValueError(f'Expected one net setting: {key}')
    text = text.replace('[net]', f'[net]\nletter_box=1\nflip=0\noptimized_memory={optimized_memory}\nworkspace_size_limit_MB=256\npinned_memory_MB=4096')
    text = text.replace('jitter=.3', 'jitter=0\nresize=1\nrandom=0')
    # One upstream head already declared random=0; avoid duplicate options.
    text = text.replace('truth_thresh = 1\nrandom=0', 'truth_thresh = 1')
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=6, choices=range(6, 121))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--optimized-memory', type=int, choices=[0, 1, 2, 3], default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    inputs = ROOT / 'data/research_inputs_v1'
    split = ROOT / 'data/research_split_v1'
    upstream = ROOT / 'external/MS-DAYOLO'
    config = adapt_config((upstream / 'cfg/ms-dayolo.cfg').read_text(), args.steps, args.optimized_memory)
    command = ['stdbuf', '-oL', '-eL', str(upstream / 'darknet'), 'detector', 'train', str(output / 'bdd.data'),
               str(output / 'smoke.cfg'), '-da', '-dont_show', '-gpus', '0']
    # run_detector removes recognized switches before interpreting positional weights.
    print(json.dumps(dict(command=command, initialization='random diagnostic only',
                          geometry=[1280, 736], source_per_update=2, target_per_update=2), indent=2))
    if args.dry_run:
        return
    output.mkdir(parents=True)
    (output / 'backup').mkdir()
    classes = json.loads((inputs / 'metadata/classes.json').read_text())['classes']
    (output / 'classes.names').write_text('\n'.join(item['name'] for item in classes) + '\n')
    for domain, role in [('source', 'source'), ('lighting', 'target')]:
        relative = (f'manifests/domains/{domain}/train.csv' if role == 'source' else
                    f'manifests/unlabeled/{domain}/full_train_pool.csv')
        with (inputs / relative).open(newline='') as stream:
            rows = sorted(csv.DictReader(stream), key=lambda row: row['name'])[:128]
        if len(rows) != 128:
            raise ValueError('Expected 128 technical training examples')
        image_dir = output / role / 'images'
        label_dir = output / role / 'labels'
        image_dir.mkdir(parents=True)
        if role == 'source':
            label_dir.mkdir()
        paths = []
        for row in rows:
            if row['image'] != 'images/train/' + row['name']:
                raise ValueError('Only train images are admitted')
            image = image_dir / row['name']
            image.symlink_to(split / row['image'])
            if role == 'source':
                (label_dir / (image.stem + '.txt')).symlink_to(inputs / row['yolo_label'])
            paths.append(str(image))
        (output / f'{role}.txt').write_text('\n'.join(paths) + '\n')
    (output / 'smoke.cfg').write_text(config)
    (output / 'bdd.data').write_text(
        f'classes=10\ntrain={output}/source.txt\ntrain_target={output}/target.txt\n'
        f'names={output}/classes.names\nbackup={output}/backup\n')
    state = dict(status='running', purpose='random-init technical graph check',
                 selection_evidence=False, exact_resume=False, steps=args.steps,
                 optimized_memory=args.optimized_memory,
                 random_seed='upstream time-based; campaign seed adapter pending', command=command)
    status = output / 'status.json'
    status.write_text(json.dumps(state, indent=2))
    started = time.monotonic()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
    try:
        with (output / 'train.log').open('w') as log:
            result = subprocess.run(command, cwd=output, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=3600)
        text = (output / 'train.log').read_text(errors='replace')
        checkpoint = output / 'backup/smoke_final.weights'
        state['exit_code'] = result.returncode
        state['checkpoint_exists'] = checkpoint.is_file()
        state['nonfinite_log'] = bool(re.search(r'\b(?:nan|inf)\b', text, flags=re.I))
        state['status'] = ('oom' if 'CUDA Error: out of memory' in text else
                          'graph-completed-unverified' if result.returncode == 0
                           and checkpoint.is_file() and not state['nonfinite_log'] else 'failed')
    except subprocess.TimeoutExpired:
        state['status'] = 'timeout'
    finally:
        state['wall_seconds'] = time.monotonic() - started
        status.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))
    if state['status'] != 'graph-completed-unverified':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
