"""Create image-only auxiliary training views from locked train-domain membership."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED = dict(source=9804, lighting=18015, weather=1985, compound=1738)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def validate_domain(rows: list, canonical: dict, domain: str, expected: int) -> list:
    names = [row['name'] for row in rows]
    if len(names) != expected or len(set(names)) != expected:
        raise ValueError('Unexpected domain size or duplicate names')
    expected_names = {name for name, row in canonical.items()
                      if domain in row['domains'].split('|')}
    if set(names) != expected_names:
        raise ValueError('Domain membership differs from locked train split')
    for row in rows:
        if Path(row['name']).name != row['name'] or row['image'] != 'images/train/' + row['name']:
            raise ValueError('Auxiliary images must be flat canonical train paths')
    return sorted(rows, key=lambda row: row['name'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('data/research_inputs_v1'))
    parser.add_argument('--split', type=Path, default=Path('data/research_split_v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    train_manifest = args.split / 'manifests/train.csv'
    train_rows = read_rows(train_manifest)
    canonical = {row['name']: row for row in train_rows}
    if len(train_rows) != 55000 or len(canonical) != 55000:
        raise ValueError('Expected the locked 55,000-image training split')
    records, hashes = {}, {}
    for domain, count in EXPECTED.items():
        manifest = args.inputs / f'manifests/domains/{domain}/train.csv'
        records[domain] = validate_domain(read_rows(manifest), canonical, domain, count)
        hashes[domain] = sha256(manifest)
        if any(not (args.split / row['image']).is_file() for row in records[domain]):
            raise FileNotFoundError(f'Missing {domain} training image')
    report = dict(counts=EXPECTED, train_manifest_sha256=sha256(train_manifest),
                  domain_manifest_sha256=hashes, split='train', annotation_access=False,
                  purpose='auxiliary image inputs only; no trained translation/TargetAugment assets',
                  sf_style_reference='unresolved; no foreign meanfoggy image substituted')
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return
    output.mkdir(parents=True)
    for domain, rows in records.items():
        images = output / 'images' / domain
        images.mkdir(parents=True)
        for row in rows:
            (images / row['name']).symlink_to((args.split / row['image']).absolute())
    for domain in ('lighting', 'weather', 'compound'):
        cut = output / 'cut' / domain
        cut.mkdir(parents=True)
        (cut / 'trainA').symlink_to(output / 'images/source', target_is_directory=True)
        (cut / 'trainB').symlink_to(output / 'images' / domain, target_is_directory=True)
        sf = output / 'sf' / domain
        sf.mkdir(parents=True)
        (sf / 'content').symlink_to(output / 'images' / domain, target_is_directory=True)
    # Written last: absence indicates incomplete preparation; never reuse the directory.
    (output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
