"""Prepare explicit source-train and validation-only SF technical smoke views."""
import argparse
import csv
import json
import re
from pathlib import Path

from pilot_sampling import read_manifest_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-images', type=int, choices=[128, 1000], default=128)
    parser.add_argument('--validation-images', type=int, choices=[64, 1000], default=64)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    inputs = root / 'data/research_inputs_v1'
    output.mkdir(parents=True)
    for split, count in [('train', args.train_images),
                         ('val', args.validation_images)]:
        rows = read_manifest_sample(
            inputs / f'manifests/domains/source/{split}.csv', count, split)
        images = output / 'data/images' / split
        labels = output / 'data/labels' / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for row in rows:
            if row['image'] != f'images/{split}/' + row['name']:
                raise ValueError('Unexpected split image path')
            (images / row['name']).symlink_to(root / 'data/research_split_v1' / row['image'])
            (labels / (Path(row['name']).stem + '.txt')).symlink_to(inputs / row['yolo_label'])
    classes = json.loads((inputs / 'metadata/classes.json').read_text())['classes']
    (output / 'data.yaml').write_text(json.dumps(dict(
        path=str(output / 'data'), train='images/train', val='images/val', nc=10,
        names=[item['name'] for item in classes]), indent=2))
    hyp = (root / 'external/sf-yolo/data/hyps/hyp.scratch-low.yaml').read_text()
    for key in ('mosaic', 'mixup', 'copy_paste', 'degrees', 'translate', 'scale',
                'shear', 'perspective', 'flipud', 'fliplr', 'hsv_h', 'hsv_s', 'hsv_v'):
        hyp = re.sub(r'^' + key + r':.*$', key + ': 0.0', hyp, flags=re.M)
    (output / 'hyp.yaml').write_text(hyp)
    (output / 'preparation.json').write_text(json.dumps(dict(
        purpose='random-initialized source technical smoke only', selection_evidence=False,
        source_train_images=args.train_images,
        source_validation_images=args.validation_images,
        holdout_access=False), indent=2))


if __name__ == '__main__':
    main()
