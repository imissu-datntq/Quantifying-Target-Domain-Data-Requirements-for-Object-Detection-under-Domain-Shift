"""Bounded upstream CUT auxiliary graph benchmark on train-only image views."""
import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=120, choices=range(6, 121))
    parser.add_argument('--direction', choices=['AtoB', 'BtoA'], default='AtoB')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    upstream = root / 'external/CUT'
    sha = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != 'b3ac297708dfb6f7589d04662277e53c0d579c27':
        raise ValueError('Unexpected CUT revision')
    auxiliary = root / 'data/auxiliary_inputs_v1'
    provenance = json.loads((auxiliary / 'provenance.json').read_text())
    if provenance['split'] != 'train':
        raise ValueError('Only prepared train-only auxiliary inputs are allowed')
    sys.path.insert(0, str(upstream))
    import numpy as np
    import torch
    from options.train_options import TrainOptions
    from data import create_dataset
    from models import create_model
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2)
    output.mkdir(parents=True)
    sys.argv = [str(upstream / 'train.py'), '--dataroot', str(auxiliary / 'cut/lighting'),
                '--name', 'technical', '--checkpoints_dir', str(output), '--model', 'cut',
                '--CUT_mode', 'CUT', '--direction', args.direction, '--batch_size', '1',
                '--num_threads', '0', '--max_dataset_size', '128', '--serial_batches',
                '--display_id', '-1', '--no_html', '--gpu_ids', '0']
    state = dict(status='running', selection_evidence=False, auxiliary='CUT',
                 upstream_sha=sha, steps=args.steps, direction=args.direction,
                 geometry='upstream auxiliary resize/crop 286/256; native generation checked separately',
                 seed=42, command=sys.argv, training_input_provenance=provenance)
    status = output / 'status.json'
    status.write_text(json.dumps(state, indent=2))
    started = time.monotonic()
    try:
        opt = TrainOptions().parse()
        dataset = create_dataset(opt)
        model = create_model(opt)
        iterator = iter(dataset)
        durations = []
        torch.cuda.reset_peak_memory_stats()
        with (output / 'metrics.jsonl').open('w') as metrics:
            for step in range(args.steps):
                if time.monotonic() - started > 3600:
                    raise TimeoutError('Auxiliary smoke exceeded one hour')
                data = next(iterator)
                if step == 0:
                    model.data_dependent_initialize(data)
                    model.setup(opt)
                    model.parallelize()
                model.set_input(data)
                torch.cuda.synchronize()
                before = time.monotonic()
                model.optimize_parameters()
                torch.cuda.synchronize()
                durations.append(time.monotonic() - before)
                losses = model.get_current_losses()
                if not all(math.isfinite(value) for value in losses.values()):
                    raise ValueError('Nonfinite CUT loss')
                metrics.write(json.dumps(dict(step=step, losses=losses, seconds=durations[-1])) + '\n')
                metrics.flush()
        model.save_networks('smoke')
        model.load_networks('smoke')
        state['training_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        state['training_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        # A crop-trained auxiliary generator must still accept native detector geometry.
        from PIL import Image
        source_domain = 'source' if args.direction == 'AtoB' else 'lighting'
        native_path = sorted((auxiliary / 'images' / source_domain).iterdir())[0]
        pixels = np.asarray(Image.open(native_path).convert('RGB')).copy()
        if pixels.shape != (720, 1280, 3):
            raise ValueError('Unexpected native auxiliary image geometry')
        native = torch.from_numpy(pixels.transpose(2, 0, 1)).float().cuda()[None] / 127.5 - 1
        model.netG.eval()
        with torch.no_grad():
            generated = model.netG(native)
        if generated.shape != native.shape or not torch.isfinite(generated).all():
            raise ValueError('Native translation changed shape or returned nonfinite pixels')
        state['native_generation_shape'] = list(generated.shape)
        state['native_generation_input'] = str(native_path)
        state.update(status='completed', network_reload=True, optimizer_resume_verified=False,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     mean_update_seconds=sum(durations[20:]) / len(durations[20:]) if len(durations) > 20 else None)
    except Exception as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['wall_seconds'] = time.monotonic() - started
        status.write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
