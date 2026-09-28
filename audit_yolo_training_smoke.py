"""Audit recorded diagnostic YOLO batches and saved tensors.

The optional resume arguments verify continuation of the serialized
model/optimizer/EMA bundle.  They deliberately do not claim exact
reproducibility: the upstream checkpoints do not include dataloader/RNG state.
"""
import argparse
import json
import math
import sys
from pathlib import Path

from evaluate_validation import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=['ssda-yolo', 'sf-yolo'], required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expected-batches', type=int, required=True)
    parser.add_argument('--start-batch', type=int, default=0,
                        help='First batch id in the trace (default: 0)')
    parser.add_argument('--resume-input', type=Path,
                        help='Checkpoint used to start this bounded resume')
    parser.add_argument('--require-positive', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    upstream = 'SSDA-YOLO' if args.candidate == 'ssda-yolo' else 'sf-yolo'
    sys.path.insert(0, str(root / 'external' / upstream))
    import torch
    import yaml
    torch.set_num_threads(2)
    destination = args.run / 'tensor_audit.json'
    if destination.exists():
        raise FileExistsError(destination)
    trace = args.run / 'technical_batches.jsonl'
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    expected_batch_ids = list(range(args.start_batch,
                                    args.start_batch + args.expected_batches))
    if args.expected_batches <= 0 or [r['batch'] for r in rows] != expected_batch_ids:
        raise ValueError('Incomplete or noncontiguous batch trace')
    if not all(math.isfinite(r['loss']) for r in rows):
        raise ValueError('Non-finite loss')
    positive = sum(r['pseudo_boxes'] > 0 for r in rows)
    if args.require_positive and positive != len(rows):
        raise ValueError('Expected positive teacher boxes in every diagnostic batch')
    checkpoints = {}
    resume_input = None
    if args.resume_input:
        resume_input = torch.load(args.resume_input, map_location='cpu')
        previous_updates = resume_input.get('updates')
        if not isinstance(previous_updates, int):
            raise ValueError('Resume input has no integer update counter')
        if args.start_batch <= 0:
            raise ValueError('A resumed trace must start after batch zero')
    for name in ('last_student.pt', 'last_teacher.pt'):
        path = args.run / 'weights' / name
        saved = torch.load(path, map_location='cpu')
        components = []
        for key in ('model', 'ema', 'teacher_model'):
            model = saved.get(key)
            if model is not None:
                if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
                    raise ValueError('Non-finite saved model tensor')
                components.append(key)
        if 'model' not in components:
            raise ValueError('Missing saved detector')
        states = (saved.get('optimizer') or {}).get('state', {})
        if not all(torch.isfinite(v).all() for state in states.values()
                   for v in state.values() if torch.is_tensor(v)):
            raise ValueError('Non-finite optimizer state')
        if name == 'last_student.pt' and (not states or saved.get('updates') != rows[-1]['ema_updates']):
            raise ValueError('Student optimizer/update trace mismatch')
        if name == 'last_student.pt' and resume_input is not None:
            if saved.get('updates', 0) <= resume_input.get('updates', 0):
                raise ValueError('Resume did not advance the optimizer/EMA counter')
            if saved.get('epoch') != resume_input.get('epoch', -1) + 1:
                raise ValueError('Resume did not advance exactly one epoch')
        checkpoints[name] = dict(sha256=digest(path), finite_components=components,
                                 optimizer_states=len(states), epoch=saved['epoch'])
    config = yaml.safe_load((args.run / 'opt.yaml').read_text())
    result = dict(audit='passed', selection_evidence=False, resume_verified=False,
                  candidate=args.candidate, batches=len(rows), positive_batches=positive,
                  batch_start=args.start_batch,
                  ema_updates=rows[-1]['ema_updates'],
                  amp_scales=sorted({r['amp_scale'] for r in rows}),
                  trace_sha256=digest(trace), checkpoints=checkpoints,
                  options_sha256=digest(args.run / 'opt.yaml'),
                  diagnostic_confidence=config['conf_thres'],
                  note='Finite saved state and trace audit only; no quality or exact-resume claim')
    if resume_input is not None:
        result['checkpoint_resume_verified'] = True
        result['exact_rng_sampler_resume_verified'] = False
        result['resume_input_sha256'] = digest(args.resume_input)
        result['note'] = ('Serialized model/optimizer/EMA continuation and finite state passed; '
                          'upstream RNG/sampler state is absent, so exact resume is not claimed')
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
