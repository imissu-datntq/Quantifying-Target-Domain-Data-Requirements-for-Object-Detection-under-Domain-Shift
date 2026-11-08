"""Check validation prediction/evaluation plumbing using a technical checkpoint."""
import argparse
import json
import math
import sys
from pathlib import Path

from adaptive_teacher_bdd import register


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    sys.path.insert(0, str(project / 'external/adaptive_teacher'))
    import train_net
    from adapteacher import add_ateacher_config
    from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer
    from detectron2.config import get_cfg
    from detectron2.data import build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.utils.logger import setup_logger
    import torch

    state = json.loads((args.run / 'smoke_status.json').read_text())
    if state['status'] != 'completed':
        parser.error('Requires a completed technical smoke')
    output = args.run / 'validation_plumbing'
    output.mkdir(exist_ok=False)
    setup_logger(output=str(output))
    domain = state.get('domain', 'lighting')
    register(project / 'data/research_inputs_v1', smoke=True, domain=domain)
    cfg = get_cfg()
    add_ateacher_config(cfg)
    cfg.merge_from_file(str(args.run / 'config.yaml'))
    trainer_cls = ATeacherTrainer if state['phase'] == 'adapt' else BaselineTrainer
    model = trainer_cls.build_model(cfg)
    saved = torch.load(str(args.run / 'model_final.pth'), map_location='cpu')['model']
    if state['phase'] == 'adapt':
        prefix = 'modelTeacher.'
        saved = {key[len(prefix):]: value for key, value in saved.items() if key.startswith(prefix)}
    model.load_state_dict(saved, strict=True)
    del saved
    results = dict(purpose='evaluation-plumbing-only', selection_evidence=False,
                   checkpoint='model_final.pth', model='teacher' if state['phase'] == 'adapt' else 'detector',
                   images_per_domain=64, domains={})
    for name in ('bdd_source_val', 'bdd_target_val'):
        evaluator = COCOEvaluator(name, ('bbox',), False,
                                  output_dir=str(output / name), use_fast_impl=False)
        loader = build_detection_test_loader(cfg, name)
        results['domains'][name] = inference_on_dataset(model, loader, evaluator)
    results['absent_class_ap'] = 'null; COCO excludes categories without ground truth from aggregate AP'
    (output / 'results.json').write_text(json.dumps(json_safe(results), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
