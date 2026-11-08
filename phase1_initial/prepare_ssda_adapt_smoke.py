"""Prepare train-only, provenance-checked paired SSDA diagnostic inputs."""
import argparse
import csv
import json
import re
from pathlib import Path

from evaluate_validation import digest
from pilot_sampling import read_manifest_sample


def verify_pairs(directory, rows, direction, split_root):
    report = json.loads((directory / 'provenance.json').read_text())
    if (report['status'] != 'completed' or report['direction'] != direction
            or report['split'] != 'train' or report['production_asset'] is not False):
        raise ValueError('Expected diagnostic train-only CUT translations')
    records = {item['name']: item for item in report['images']}
    if len(records) != len(report['images']) or set(records) != {r['name'] for r in rows}:
        raise ValueError('CUT and canonical smoke membership differ')
    for row in rows:
        item = records[row['name']]
        if (digest(split_root / row['image']) != item['input_sha256']
                or digest(directory / row['name']) != item['output_sha256']):
            raise ValueError('CUT image provenance mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-pairs', type=Path, required=True)
    parser.add_argument('--target-pairs', type=Path, required=True)
    parser.add_argument('--train-images', type=int, choices=[128, 1000], default=128)
    parser.add_argument('--validation-images', type=int, choices=[64, 1000], default=64)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    inputs = root / 'data/research_inputs_v1'
    split_root = root / 'data/research_split_v1'
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    source = read_manifest_sample(
        inputs / 'manifests/domains/source/train.csv', args.train_images, 'train')
    target = read_manifest_sample(
        inputs / 'manifests/unlabeled/lighting/full_train_pool.csv',
        args.train_images, 'train')
    validation = read_manifest_sample(
        inputs / 'manifests/domains/lighting/val.csv',
        args.validation_images, 'val')
    verify_pairs(args.source_pairs.resolve(), source, 'AtoB', split_root)
    verify_pairs(args.target_pairs.resolve(), target, 'BtoA', split_root)
    output.mkdir(parents=True)
    for rows, split in ((source, 'train'), (validation, 'val')):
        images, labels = output / 'images' / split, output / 'labels' / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for row in rows:
            (images / row['name']).symlink_to(split_root / row['image'])
            (labels / (Path(row['name']).stem + '.txt')).symlink_to(inputs / row['yolo_label'])
    manifest = output / 'target.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'image', 'coco_image_id'])
        writer.writeheader()
        writer.writerows(target)
    classes = json.loads((inputs / 'metadata/classes.json').read_text())['classes']
    config = dict(train_source_real=str(output / 'images/train'),
                  train_source_fake=str(args.source_pairs.resolve()),
                  train_target_real=str(manifest),
                  train_target_fake=str(args.target_pairs.resolve()),
                  test_target_real=str(output / 'images/val'),
                  unlabeled_image_root=str(split_root), nc=10,
                  names=[item['name'] for item in classes])
    (output / 'data.yaml').write_text(json.dumps(config, indent=2))
    hyp = (root / 'external/SSDA-YOLO/data/hyps/hyp.scratch.yaml').read_text()
    for key in ('mosaic', 'mixup', 'degrees', 'translate', 'scale', 'shear',
                'perspective', 'flipud', 'fliplr', 'hsv_h', 'hsv_s', 'hsv_v'):
        hyp = re.sub(r'^' + key + r':.*$', key + ': 0.0', hyp, flags=re.M)
    (output / 'hyp.yaml').write_text(hyp)
    report = dict(selection_evidence=False, holdout_access=False,
                  purpose='paired SSDA full-graph technical smoke',
                  source_train_images=args.train_images,
                  target_train_images=args.train_images,
                  lighting_validation_images=args.validation_images,
                  source_pair_provenance_sha256=digest(args.source_pairs / 'provenance.json'),
                  target_pair_provenance_sha256=digest(args.target_pairs / 'provenance.json'))
    (output / 'preparation.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
