"""Export pinned MS-DAYOLO diagnostic predictions on a complete validation domain."""
from __future__ import annotations

import argparse
import array
import csv
import json
import math
import os
import shutil
import subprocess
import struct
import sys
import time
from pathlib import Path
from typing import BinaryIO, Iterable

from evaluate_validation import digest
from run_ms_dayolo_smoke import adapt_config


def audit_weights(stream: BinaryIO, expected_values: int, expected_seen: int) -> int:
    """Reject Darknet's otherwise silent partial loads for the pinned architecture."""
    header = stream.read(20)
    if len(header) != 20 or struct.unpack('<iiiQ', header) != (0, 2, 5, expected_seen):
        raise ValueError('Unexpected weights header or training image count')
    count = 0
    while True:
        chunk = stream.read(4 * 262144)
        if not chunk:
            break
        if len(chunk) % 4:
            raise ValueError('Truncated FP32 checkpoint payload')
        values = array.array('f')
        values.frombytes(chunk)
        if sys.byteorder != 'little':
            values.byteswap()
        if any(not math.isfinite(value) for value in values):
            raise ValueError('Non-finite checkpoint payload')
        count += len(values)
    if count != expected_values:
        raise ValueError('Wrong checkpoint payload size; partial loads are forbidden')
    return count


def convert_records(records: Iterable[dict], image_ids: list[int],
                    classes: list[dict]) -> tuple[list[dict], list[int]]:
    """Validate indexed native output, then map to canonical COCO without float IDs."""
    if (not image_ids or any(type(i) is not int for i in image_ids)
            or len(set(image_ids)) != len(image_ids)):
        raise ValueError('Expected unique integer validation IDs')
    mapping = {c['yolo_id']: c['coco_id'] for c in classes}
    if (len(classes) != 10 or set(mapping) != set(range(10))
            or set(mapping.values()) != set(range(1, 11))):
        raise ValueError('Expected bijective ten-class research metadata')
    predictions, covered = [], []
    for record in records:
        index = record.get('index')
        if (type(index) is not int or index != len(covered) or index >= len(image_ids)
                or record.get('width') != 1280 or record.get('height') != 720):
            raise ValueError('Invalid image order, coverage or native geometry')
        detections = record.get('detections')
        if not isinstance(detections, list):
            raise ValueError('Missing detection array')
        image_predictions = []
        for detection in detections:
            if not isinstance(detection, list) or len(detection) != 6:
                raise ValueError('Expected class, score and original xyxy')
            category, score, x1, y1, x2, y2 = detection
            if type(category) is not int or category not in mapping:
                raise ValueError('Invalid head class index')
            if (any(type(v) not in (int, float) or not math.isfinite(v)
                    for v in detection[1:]) or not 0 <= score <= 1):
                raise ValueError('Invalid score or non-finite coordinates')
            x1, x2 = max(0, min(1280, x1)), max(0, min(1280, x2))
            y1, y2 = max(0, min(720, y1)), max(0, min(720, y2))
            if score < .001 or x2 <= x1 or y2 <= y1:
                continue
            image_predictions.append(dict(image_id=image_ids[index],
                category_id=mapping[category], score=score, bbox=[x1, y1, x2-x1, y2-y1]))
        predictions.extend(sorted(image_predictions, key=lambda p: -p['score'])[:100])
        covered.append(image_ids[index])
    if covered != image_ids:
        raise ValueError('Incomplete validation inference coverage')
    return predictions, covered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--domain', choices=['source', 'lighting', 'weather', 'compound'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Maximum Darknet validation seconds (default: 3600)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    run_state = json.loads((run / 'status.json').read_text())
    completed_graph = run_state.get('status') == 'graph-completed-unverified'
    completed_supervised = (run_state.get('status') == 'completed'
                            and run_state.get('adaptation') is False
                            and run_state.get('checkpoint_audit') == 'passed')
    if ((not completed_graph and not completed_supervised)
            or run_state.get('exit_code') != 0 or run_state.get('nonfinite_log') is not False):
        raise ValueError('Requires a successfully completed and audited MS graph run')
    checkpoint, config = run / 'backup/smoke_final.weights', run / 'smoke.cfg'
    if not checkpoint.is_file() or not config.is_file():
        raise ValueError('Missing original final weights/config')
    upstream = root / 'external/MS-DAYOLO'
    binary = upstream / 'darknet'
    patch_path = root / 'patches/ms-dayolo-research-export.patch'
    if not binary.is_file() or not os.access(binary, os.X_OK) or not patch_path.is_file():
        raise ValueError('Missing executable or exporter patch provenance')
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != 'c655ceed3b99e6b499a9c907a3ae9c2deba98c1e':
        raise ValueError('Unexpected upstream revision')
    # Restrict this diagnostic adapter to the exact tested ten-class graph.
    if completed_supervised:
        from run_ms_supervised_smoke import adapt_config as adapt_supervised_config
        expected_config = adapt_supervised_config(
            (upstream / 'cfg/ms-dayolo.cfg').read_text(), run_state['steps'],
            run_state['pinned_memory_mb'], run_state.get('learning_rate', .0001))
        expected_seen = run_state['checkpoint_seen']
    else:
        expected_config = adapt_config((upstream / 'cfg/ms-dayolo.cfg').read_text(),
                                       run_state['steps'], run_state['optimized_memory'])
        expected_seen = run_state['steps'] * 4
    if config.read_text() != expected_config:
        raise ValueError('Config differs from the pinned diagnostic graph')
    with checkpoint.open('rb') as stream:
        weight_values = audit_weights(stream, 64745066, expected_seen)
    inputs = root / 'data/research_inputs_v1'
    manifest = inputs / f'manifests/domains/{args.domain}/val.csv'
    with manifest.open(newline='') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row['name'])
    ids = [int(row['coco_image_id']) for row in rows]
    if (not rows or len(set(ids)) != len(ids)
            or len({r['name'] for r in rows}) != len(rows)
            or any(Path(r['name']).name != r['name']
                   or r['image'] != 'images/val/' + r['name'] for r in rows)):
        raise ValueError('Expected unique canonical validation-only membership')
    images = [root / 'data/research_split_v1' / r['image'] for r in rows]
    if any(not p.is_file() or '\n' in str(p) or '\r' in str(p) for p in images):
        raise ValueError('Missing/invalid validation image path')
    class_path = inputs / 'metadata/classes.json'
    classes = json.loads(class_path.read_text())['classes']
    names = '\n'.join(c['name'] for c in sorted(classes, key=lambda c: c['yolo_id'])) + '\n'
    if (run / 'classes.names').read_text() != names:
        raise ValueError('Training head names do not match canonical class order')
    command = [str(binary), 'detector', 'valid', str(output / 'validation.data'),
               str(output / 'inference.cfg'), str(checkpoint), '-out', 'raw', '-dont_show']
    state = dict(status='running', split='val', domain=args.domain,
                 candidate='ms-dayolo', phase=run_state.get('phase', 'adapt'),
                 selection_evidence=False,
                 checkpoint_sha256=digest(checkpoint), config_sha256=digest(config),
                 manifest_sha256=digest(manifest), classes_sha256=digest(class_path),
                 binary_sha256=digest(binary), upstream_revision=revision,
                 finite_checkpoint_values=weight_values,
                 exporter_patch_sha256=digest(patch_path),
                 confidence_threshold=.001, nms_iou=.6, max_detections=100,
                 geometry='native 1280x720; upstream letterbox to 1280x736; original-image boxes',
                 timeout_seconds=args.timeout, command=command, evaluated_image_ids=[])
    output.mkdir(parents=True)
    status = output / 'status.json'
    status.write_text(json.dumps(state, indent=2))
    started = time.monotonic()
    try:
        shutil.copyfile(config, output / 'inference.cfg')
        (output / 'classes.names').write_text(names)
        (output / 'valid.txt').write_text(''.join(str(p) + '\n' for p in images))
        (output / 'validation.data').write_text(
            f'classes=10\nvalid={output}/valid.txt\nnames={output}/classes.names\n'
            f'results={output}\neval=research\n')
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(root / '.venvs/ms-dayolo/lib') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
        with (output / 'inference.log').open('w') as log:
            result = subprocess.run(command, cwd=output, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
        state['exit_code'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'Darknet failed with exit {result.returncode}; see inference.log')
        raw = output / 'raw.jsonl'
        with raw.open() as stream:
            predictions, covered = convert_records((json.loads(line) for line in stream), ids, classes)
        path = output / 'predictions.json'
        path.write_text(json.dumps(predictions, allow_nan=False))
        state.update(status='completed', evaluated_image_ids=covered,
                     prediction_sha256=digest(path), raw_sha256=digest(raw),
                     predictions=len(predictions), wall_seconds=time.monotonic()-started)
        (output / 'inference_manifest.json').write_text(json.dumps(state, indent=2))
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['wall_seconds'] = time.monotonic()-started
        status.write_text(json.dumps(state, indent=2))
        print(json.dumps({k: v for k, v in state.items() if k != 'evaluated_image_ids'}, indent=2))


if __name__ == '__main__':
    main()
