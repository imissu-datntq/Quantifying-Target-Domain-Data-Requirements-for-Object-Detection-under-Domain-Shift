"""Export native-coordinate YOLO validation predictions for technical integration."""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

from evaluate_validation import digest


def unpad_box(box):
    """Undo only the explicit 8px vertical stride padding; return COCO xywh."""
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError('Expected finite xyxy box')
    x1, y1, x2, y2 = box
    x1, x2 = max(0., min(1280., x1)), max(0., min(1280., x2))
    y1, y2 = max(0., min(720., y1 - 8)), max(0., min(720., y2 - 8))
    return [x1, y1, x2 - x1, y2 - y1] if x2 > x1 and y2 > y1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=['ssda-yolo', 'sf-yolo'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model-key', choices=['model', 'ema'], default='model',
                        help='Explicit checkpoint component; recorded in provenance')
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
    upstream = root / 'external' / ('SSDA-YOLO' if args.candidate == 'ssda-yolo' else 'sf-yolo')
    sys.path.insert(0, str(upstream))
    import cv2
    import torch
    from models.yolo import Model
    from utils.general import non_max_suppression
    torch.set_num_threads(2)
    inputs = root / 'data/research_inputs_v1'
    manifest = inputs / f'manifests/domains/{args.domain}/val.csv'
    with manifest.open(newline='') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row['name'])[:args.max_images]
    if not rows or len({row['name'] for row in rows}) != len(rows):
        raise ValueError('Empty or duplicated validation membership')
    for row in rows:
        if row['image'] != 'images/val/' + row['name']:
            raise ValueError('Only canonical validation paths are available')
    mapping = {item['yolo_id']: item['coco_id'] for item in
               json.loads((inputs / 'metadata/classes.json').read_text())['classes']}
    saved = torch.load(str(args.checkpoint), map_location='cpu')
    component = saved.get(args.model_key)
    if component is None:
        raise ValueError('Requested checkpoint component is absent')
    if isinstance(component, torch.nn.Module):
        model = component.float()
        checkpoint_format = 'serialized-model'
    elif args.candidate == 'ssda-yolo' and isinstance(component, dict):
        model = Model(str(upstream / 'models/yolov5l.yaml'), nc=10)
        model.load_state_dict(component, strict=True)
        checkpoint_format = 'state-dict'
    else:
        raise ValueError('Unsupported candidate checkpoint format')
    if model.model[-1].nc != 10:
        raise ValueError('Expected ten-class research detector')
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Non-finite checkpoint tensor')
    del saved
    model = model.float().cuda().eval()
    output.mkdir(parents=True)
    state = dict(status='running', split='val', domain=args.domain,
                 candidate=args.candidate, selection_evidence=False,
                 model_key=args.model_key, checkpoint_format=checkpoint_format,
                 checkpoint_sha256=digest(args.checkpoint), manifest_sha256=digest(manifest),
                 validation_subset='lexicographic-name-prefix', max_images=args.max_images,
                 confidence_threshold=.001, nms_iou=.6, max_detections=100,
                 geometry='original 1280x720 padded 8 pixels above/below; FP32',
                 evaluated_image_ids=[])
    status = output / 'status.json'
    status.write_text(json.dumps(state, indent=2))
    started = time.monotonic()
    predictions = []
    try:
        with torch.no_grad():
            for index, row in enumerate(rows):
                pixels = cv2.imread(str(root / 'data/research_split_v1' / row['image']))
                if pixels is None or pixels.shape != (720, 1280, 3):
                    raise ValueError('Unexpected validation geometry')
                padded = cv2.copyMakeBorder(pixels, 8, 8, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                tensor = torch.from_numpy(padded[:, :, ::-1].transpose(2, 0, 1).copy()).float().cuda()[None] / 255
                raw = model(tensor)[0]
                detections = non_max_suppression(raw, conf_thres=.001, iou_thres=.6,
                                                multi_label=True, max_det=100)[0].cpu().tolist()
                for x1, y1, x2, y2, score, category in detections:
                    box = unpad_box([x1, y1, x2, y2])
                    if box is not None:
                        predictions.append(dict(image_id=int(row['coco_image_id']),
                            category_id=mapping[int(category)], bbox=box, score=score))
                state['evaluated_image_ids'].append(int(row['coco_image_id']))
                if (index + 1) % 64 == 0:
                    print(f'Exported {index + 1}/{len(rows)} validation images', flush=True)
        path = output / 'predictions.json'
        path.write_text(json.dumps(predictions, allow_nan=False))
        state.update(status='completed', prediction_sha256=digest(path),
                     predictions=len(predictions))
        (output / 'inference_manifest.json').write_text(json.dumps(state, indent=2))
    except Exception as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['wall_seconds'] = time.monotonic() - started
        status.write_text(json.dumps(state, indent=2))
        print(json.dumps({key: value for key, value in state.items() if key != 'evaluated_image_ids'}, indent=2))


if __name__ == '__main__':
    main()
