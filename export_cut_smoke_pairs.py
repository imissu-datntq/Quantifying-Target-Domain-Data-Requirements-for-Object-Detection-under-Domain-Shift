"""Generate native-size train-only diagnostic translations from a CUT smoke."""
import argparse
import json
import sys
import time
from pathlib import Path

from evaluate_validation import digest
from pilot_sampling import read_manifest_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-images', type=int, choices=[128, 1000], default=128)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    state = json.loads((args.run / 'status.json').read_text())
    if state['status'] != 'completed' or state['auxiliary'] != 'CUT':
        raise ValueError('Requires completed CUT technical run')
    domain = {'AtoB': 'source', 'BtoA': 'lighting'}[state['direction']]
    checkpoint = args.run / 'technical/smoke_net_G.pth'
    sys.path.insert(0, str(root / 'external/CUT'))
    import numpy as np
    import torch
    from PIL import Image
    from models.networks import define_G
    torch.set_num_threads(2)
    generator = define_G(3, 3, 64, 'resnet_9blocks', norm='instance',
                         use_dropout=False, init_type='xavier', gpu_ids=[0])
    generator.load_state_dict(torch.load(str(checkpoint), map_location='cpu'), strict=True)
    generator.eval()
    manifest = (root / 'data/research_inputs_v1/manifests/domains/source/train.csv'
                if domain == 'source' else
                root / 'data/research_inputs_v1/manifests/unlabeled/lighting/full_train_pool.csv')
    rows = read_manifest_sample(manifest, args.max_images, 'train')
    inputs = [root / 'data/auxiliary_inputs_v1/images' / domain / row['name']
              for row in rows]
    if any(not path.is_file() for path in inputs):
        raise ValueError('Missing bounded CUT input image')
    output.mkdir(parents=True)
    records = []
    start = time.monotonic()
    with torch.no_grad():
        for index, path in enumerate(inputs):
            pixels = np.asarray(Image.open(path).convert('RGB')).copy()
            if pixels.shape != (720, 1280, 3):
                raise ValueError('Unexpected native geometry')
            tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).float().cuda()[None] / 127.5 - 1
            translated = generator(tensor)
            if translated.shape != tensor.shape or not torch.isfinite(translated).all():
                raise ValueError('Invalid native translation')
            array = ((translated[0].clamp(-1, 1).cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            target = output / path.name
            Image.fromarray(array).save(target, quality=95)
            records.append(dict(name=path.name, input_sha256=digest(path), output_sha256=digest(target)))
            if (index + 1) % 32 == 0:
                print(f'Translated {index + 1}/{args.max_images} {domain} train images', flush=True)
    report = dict(status='completed', selection_evidence=False, production_asset=False,
                  direction=state['direction'], source_domain=domain, split='train',
                  requested_images=args.max_images,
                  checkpoint_sha256=digest(checkpoint), images=records,
                  wall_seconds=time.monotonic() - start,
                  note='Native shape verified; no semantic alignment or translation quality claim')
    (output / 'provenance.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
