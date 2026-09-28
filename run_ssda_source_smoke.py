"""Bounded detector-only source smoke using pinned SSDA upstream model/loss."""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=120, choices=range(6, 121))
    parser.add_argument('--resume-run', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(root / 'external/SSDA-YOLO'))
    import cv2
    import numpy as np
    import torch
    import yaml
    from models.yolo import Model
    from utils.loss import ComputeLoss
    torch.set_num_threads(2)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    output.mkdir(parents=True)
    state = dict(status='running', phase='source', candidate='ssda-yolo',
                 selection_evidence=False, seed=42, steps=args.steps,
                 geometry=[1280, 736], augmentation='stride padding only', batch=1)
    status = output / 'status.json'
    status.write_text(json.dumps(state, indent=2))
    start = time.monotonic()
    try:
        upstream = root / 'external/SSDA-YOLO'
        pretrained = torch.load(str(root / 'weights/yolov5l-v5.0.pt'), map_location='cpu')['model'].float()
        model = Model(str(upstream / 'models/yolov5l.yaml'), nc=10)
        compatible = {key: value for key, value in pretrained.state_dict().items()
                      if key in model.state_dict() and value.shape == model.state_dict()[key].shape}
        model.load_state_dict(compatible, strict=False)
        state['pretrained_tensors_loaded'] = len(compatible)
        del pretrained
        model = model.cuda().train()
        model.hyp = yaml.safe_load((upstream / 'data/hyps/hyp.scratch.yaml').read_text())
        model.gr = 1.
        model.nc = 10
        criterion = ComputeLoss(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.0001, momentum=.9)
        scaler = torch.cuda.amp.GradScaler()
        start_step = 0
        if args.resume_run:
            previous = json.loads((args.resume_run / 'status.json').read_text())
            if previous.get('status') != 'completed' or previous.get('candidate') != 'ssda-yolo':
                raise ValueError('Resume requires a completed SSDA source smoke')
            saved = torch.load(str(args.resume_run / 'source_smoke.pt'), map_location='cpu')
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler'])
            start_step = saved['step'] + 1
            state.update(resumed_from=str(args.resume_run.resolve()), start_step=start_step)
            del saved
        inputs = root / 'data/research_inputs_v1'
        with (inputs / 'manifests/domains/source/train.csv').open(newline='') as stream:
            rows = sorted(csv.DictReader(stream), key=lambda row: row['name'])[:128]
        torch.cuda.reset_peak_memory_stats()
        durations = []
        with (output / 'metrics.jsonl').open('w') as metrics:
            for step in range(start_step, start_step + args.steps):
                if time.monotonic() - start > 3600:
                    raise TimeoutError('Technical smoke ceiling exceeded')
                row = rows[step % len(rows)]
                if row['image'] != 'images/train/' + row['name']:
                    raise ValueError('Source smoke only accepts canonical training paths')
                pixels = cv2.imread(str(root / 'data/research_split_v1' / row['image']))
                if pixels is None or pixels.shape != (720, 1280, 3):
                    raise ValueError('Unexpected source image geometry')
                pixels = cv2.copyMakeBorder(pixels, 8, 8, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                image = torch.from_numpy(pixels[:, :, ::-1].transpose(2, 0, 1).copy()).cuda().float()[None] / 255
                lines = (inputs / row['yolo_label']).read_text().splitlines()
                labels = []
                for line in lines:
                    category, x, y, w, h = map(float, line.split())
                    labels.append([0, category, x, (y * 720 + 8) / 736, w, h * 720 / 736])
                targets = torch.tensor(labels, device='cuda', dtype=torch.float32).reshape(-1, 6)
                torch.cuda.synchronize()
                before = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    loss, components = criterion(model(image), targets)
                if not torch.isfinite(loss).all():
                    raise ValueError('Nonfinite source loss')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                durations.append(time.monotonic() - before)
                metrics.write(json.dumps(dict(step=step, loss=float(loss), seconds=durations[-1])) + '\n')
                metrics.flush()
        checkpoint = output / 'source_smoke.pt'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(), step=start_step + args.steps - 1), str(checkpoint))
        saved = torch.load(str(checkpoint), map_location='cpu')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        state.update(status='completed', checkpoint_reload=True, exact_resume=False,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     mean_update_seconds=sum(durations[20:]) / len(durations[20:]) if len(durations) > 20 else None)
    except Exception as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['wall_seconds'] = time.monotonic() - start
        status.write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
