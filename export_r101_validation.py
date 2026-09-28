"""Export R101-C4 diagnostic predictions for the shared validation evaluator."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

from evaluate_validation import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--component', choices=['detector', 'teacher', 'student'], required=True)
    parser.add_argument('--domain', choices=['source', 'lighting', 'weather', 'compound'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-images', type=int, default=64,
                        help='Deterministic validation prefix size for the bounded pilot')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if args.max_images <= 0:
        parser.error('--max-images must be positive')
    if output.exists():
        raise FileExistsError(output)
    run_state = json.loads((args.run / 'smoke_status.json').read_text())
    if run_state['status'] != 'completed':
        raise ValueError('Requires a completed technical run')
    if (run_state['phase'] == 'adapt') != (args.component != 'detector'):
        raise ValueError('Checkpoint phase and requested component disagree')
    sys.path.insert(0, str(root / 'external/adaptive_teacher'))
    import train_net  # Register pinned upstream model architectures.
    import numpy as np
    from PIL import Image
    import torch
    from adapteacher import add_ateacher_config
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model

    torch.set_num_threads(2)
    cfg = get_cfg()
    add_ateacher_config(cfg)
    config = args.run / 'config.yaml'
    cfg.merge_from_file(str(config))
    if cfg.INPUT.FORMAT != 'BGR' or cfg.MODEL.ROI_HEADS.NUM_CLASSES != 10:
        raise ValueError('Expected ten-class BGR research model')
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .001
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = .6
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    model = build_model(cfg)
    checkpoint = args.run / 'model_final.pth'
    saved = torch.load(str(checkpoint), map_location='cpu')['model']
    if args.component != 'detector':
        prefix = {'teacher': 'modelTeacher.', 'student': 'modelStudent.'}[args.component]
        saved = {key[len(prefix):]: value for key, value in saved.items() if key.startswith(prefix)}
    if not saved or not all(torch.isfinite(t).all() for t in saved.values()):
        raise ValueError('Missing or non-finite model tensors')
    model.load_state_dict(saved, strict=True)
    del saved
    model.eval()
    inputs = root / 'data/research_inputs_v1'
    manifest = inputs / f'manifests/domains/{args.domain}/val.csv'
    with manifest.open(newline='') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: r['name'])[:args.max_images]
    ids = [int(row['coco_image_id']) for row in rows]
    if not rows or len(set(ids)) != len(ids) or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Empty/duplicate validation membership')
    if any(r['image'] != 'images/val/' + r['name'] for r in rows):
        raise ValueError('Only canonical validation images are admitted')
    classes = json.loads((inputs / 'metadata/classes.json').read_text())['classes']
    mapping = [item['coco_id'] for item in sorted(classes, key=lambda c: c['coco_id'])]
    output.mkdir(parents=True)
    state = dict(status='running', split='val', domain=args.domain,
                 candidate='adaptive-teacher-r101-c4', selection_evidence=False,
                 component=args.component, checkpoint_sha256=digest(checkpoint),
                 config_sha256=digest(config), manifest_sha256=digest(manifest),
                 validation_subset='lexicographic-name-prefix', max_images=args.max_images,
                 confidence_threshold=.001, nms_iou=.6, max_detections=100,
                 geometry='native 1280x720; internal stride padding; original-image postprocessing',
                 evaluated_image_ids=[])
    status = output / 'status.json'
    status.write_text(json.dumps(state, indent=2))
    started = time.monotonic()
    predictions = []
    try:
        with torch.no_grad():
            for index, row in enumerate(rows):
                with Image.open(root / 'data/research_split_v1' / row['image']) as original:
                    pixels = np.asarray(original.convert('RGB'))[:, :, ::-1].copy()
                if pixels.shape != (720, 1280, 3):
                    raise ValueError('Unexpected native image geometry')
                image = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).float()
                instances = model([dict(image=image, height=720, width=1280)])[0]['instances'].to('cpu')
                if instances.image_size != (720, 1280):
                    raise ValueError('Postprocessed image geometry mismatch')
                instances.pred_boxes.clip((720, 1280))
                for box, score, category in zip(instances.pred_boxes.tensor.tolist(),
                                               instances.scores.tolist(), instances.pred_classes.tolist()):
                    x1, y1, x2, y2 = box
                    if x2 > x1 and y2 > y1:
                        predictions.append(dict(image_id=int(row['coco_image_id']),
                            category_id=mapping[category], bbox=[x1, y1, x2-x1, y2-y1], score=score))
                state['evaluated_image_ids'].append(int(row['coco_image_id']))
                if (index + 1) % 64 == 0:
                    print(f'Exported {index + 1}/{len(rows)} validation images', flush=True)
        path = output / 'predictions.json'
        path.write_text(json.dumps(predictions, allow_nan=False))
        state.update(status='completed', prediction_sha256=digest(path), predictions=len(predictions))
        (output / 'inference_manifest.json').write_text(json.dumps(state, indent=2))
    except Exception as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['wall_seconds'] = time.monotonic() - started
        status.write_text(json.dumps(state, indent=2))
        print(json.dumps({k: v for k, v in state.items() if k != 'evaluated_image_ids'}, indent=2))


if __name__ == '__main__':
    main()
