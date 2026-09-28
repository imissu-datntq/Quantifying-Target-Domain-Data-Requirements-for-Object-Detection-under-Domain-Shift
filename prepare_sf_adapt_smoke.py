"""Prepare bounded image-only SF adaptation inputs; never a campaign registry."""
import argparse
import csv
import json
from pathlib import Path

from pilot_sampling import read_manifest_sample


def preparation_report(*, train_images: int, validation_images: int,
                       source_smoke: Path) -> dict:
    return dict(
        selection_evidence=False, holdout_access=False,
        purpose='bounded SF graph pilot',
        source_train_images=train_images, target_train_images=train_images,
        source_validation_images=validation_images,
        source_smoke=str(source_smoke.resolve()),
        auxiliary_weights='targetaugment-lighting-smoke-002; diagnostic only')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-smoke', type=Path, required=True)
    parser.add_argument('--train-images', type=int, choices=[128, 1000], default=128)
    parser.add_argument('--validation-images', type=int, choices=[64, 1000], default=64)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    inputs = root / 'data/research_inputs_v1'
    rows = read_manifest_sample(
        inputs / 'manifests/unlabeled/lighting/full_train_pool.csv',
        args.train_images, 'train')
    source_preparation = json.loads(
        (args.source_smoke.resolve() / 'preparation.json').read_text())
    if (int(source_preparation.get('source_train_images', 0)) != args.train_images
            or int(source_preparation.get('source_validation_images', 0))
            != args.validation_images):
        raise ValueError('Source and adaptation pilot sizes must match')
    output.mkdir(parents=True)
    manifest = output / 'target.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'image', 'coco_image_id'])
        writer.writeheader()
        writer.writerows(rows)
    classes = json.loads((inputs / 'metadata/classes.json').read_text())['classes']
    config = dict(path=str(root), train=str(manifest),
                  unlabeled_image_root=str(root / 'data/research_split_v1'),
                  val=str(args.source_smoke.resolve() / 'data/images/val'),
                  nc=10, names=[item['name'] for item in classes])
    (output / 'data.yaml').write_text(json.dumps(config, indent=2))
    (output / 'preparation.json').write_text(json.dumps(preparation_report(
        train_images=args.train_images,
        validation_images=args.validation_images,
        source_smoke=args.source_smoke), indent=2))


if __name__ == '__main__':
    main()
