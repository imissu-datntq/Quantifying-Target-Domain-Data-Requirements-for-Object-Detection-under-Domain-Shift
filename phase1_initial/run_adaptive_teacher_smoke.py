"""Run a bounded Adaptive Teacher R101-C4 technical smoke.

Run inside .venvs/adaptive-teacher. Run directories must be new. The upstream
trainer owns checkpointing; this entry point does not implement a new method.
"""
import argparse
import json
import math
import logging
import os
import random
import statistics
import subprocess
import signal
import sys
import time
from pathlib import Path

from adaptive_teacher_bdd import register


def save_rng_sampler_state(path, torch, sampler_cursor):
    """Persist process RNG streams and the single-worker loader cursor.

    The upstream Detectron2 checkpoint omits these values. The smoke adapter
    uses one worker and one image per batch, so replaying ``sampler_cursor``
    batches reconstructs the sampler position before restoring augmentation
    RNG streams. This is intentionally a sidecar rather than a modified
    upstream checkpoint format.
    """
    import numpy as np

    payload = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'sampler_cursor': int(sampler_cursor),
        'num_workers': 0,
        'batch_size': 1,
    }
    torch.save(payload, str(path))


def restore_rng_sampler_state(path, trainer, torch, sampler_cursor):
    """Replay the loader cursor, then restore all process RNG streams."""
    import numpy as np

    payload = torch.load(str(path), map_location='cpu')
    if payload.get('num_workers') != 0 or payload.get('batch_size') != 1:
        raise RuntimeError('RNG sidecar was created with incompatible loader settings')
    if int(payload.get('sampler_cursor', -1)) != int(sampler_cursor):
        raise RuntimeError('RNG sidecar sampler cursor does not match checkpoint iteration')
    iterator_owner = getattr(trainer, '_trainer', trainer)
    for _ in range(int(sampler_cursor)):
        next(iterator_owner._data_loader_iter)
    random.setstate(payload['python'])
    np.random.set_state(payload['numpy'])
    torch.set_rng_state(payload['torch_cpu'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload['torch_cuda'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['source', 'finetune', 'adapt'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--train-images', type=int, choices=[128, 1000], default=128)
    parser.add_argument('--validation-images', type=int, choices=[64, 1000], default=64)
    parser.add_argument('--resume-from', type=Path, help='Completed smoke run; restore model/optimizer/scheduler')
    parser.add_argument('--source-run', type=Path,
                        help='Completed source checkpoint for fine-tuning or adaptation')
    parser.add_argument('--domain', choices=['lighting', 'weather', 'compound'], default='lighting')
    parser.add_argument('--budget', type=int, choices=[100, 1000], default=100)
    parser.add_argument('--deterministic', action='store_true',
                        help='Request deterministic CUDA algorithms for restart validation')
    parser.add_argument('--seed', type=int, choices=[42, 43], default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 6 <= args.steps <= 1000:
        parser.error('Bounded pilot must use 6..1000 updates (upstream timer warmup)')
    if args.steps < args.train_images and args.train_images > 128:
        parser.error('The 1,000-image pilot requires at least 1,000 updates')
    if args.output.exists():
        parser.error('Output already exists; select a fresh run directory')
    if args.phase == 'finetune' and not args.source_run:
        parser.error('--source-run is required for fine-tuning')
    if args.phase == 'source' and args.source_run:
        parser.error('--source-run is only valid for fine-tuning or adaptation')
    project = Path(__file__).resolve().parent
    upstream = project / 'external/adaptive_teacher'
    sha = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != '5256463ad9ec90fd5ba84ebb8d53bed56bd369df':
        raise RuntimeError('Unexpected upstream revision')
    if args.dry_run:
        print(json.dumps(dict(
            status='planned', candidate='adaptive-teacher-r101-c4',
            phase=args.phase, domain='source' if args.phase == 'source' else args.domain,
            labeled_budget=0 if args.phase == 'source' else args.budget,
            seed=args.seed, steps=args.steps,
            source_run=str(args.source_run.resolve()) if args.source_run else None,
            output=str(args.output.resolve()), upstream_sha=sha,
            max_training_images_per_pool=args.train_images,
            validation_images=args.validation_images, resume_allowed=True,
            holdout_access=False, selection_evidence=False), indent=2))
        return
    # cuBLAS may select a different reduction algorithm in a fresh process.
    # Set its deterministic workspace contract before importing torch/CUDA.
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(upstream))
    import torch
    import train_net
    from detectron2.config import get_cfg
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.env import seed_all_rng
    from detectron2.engine import HookBase
    from adapteacher import add_ateacher_config
    from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer

    start_iter = 0
    if args.resume_from:
        previous = json.loads((args.resume_from / 'smoke_status.json').read_text())
        if previous['status'] != 'completed' or previous['phase'] != args.phase or previous['upstream_sha'] != sha:
            parser.error('Resume requires a completed matching smoke phase/revision')
        saved = torch.load(str(args.resume_from / 'model_final.pth'), map_location='cpu')
        start_iter = saved['iteration'] + 1
        del saved

    if args.source_run:
        parent = json.loads((args.source_run / 'smoke_status.json').read_text())
        if parent['status'] != 'completed' or parent['phase'] != 'source' or parent['upstream_sha'] != sha:
            parser.error('Requires a completed matching source run')
        source_config = (args.source_run / 'config.yaml').read_text().splitlines()
        if 'META_ARCHITECTURE: TwoStagePseudoLabGeneralizedRCNN' not in [line.strip() for line in source_config]:
            parser.error('Missing source detector configuration')
    register(project / 'data/research_inputs_v1', smoke=True,
             domain=args.domain, budget=args.budget, seed=args.seed,
             train_limit=args.train_images,
             validation_limit=args.validation_images)
    # Upstream baseline requires an explicit supervision-index file even at 100%.
    args.output.mkdir(parents=True, exist_ok=False)
    seed_file = args.output.absolute() / 'supervision_indices.json'
    count = min(args.budget, args.train_images) if args.phase == 'finetune' else args.train_images
    seed_file.write_text(json.dumps({'100.0': {'0': list(range(count))}}))
    cfg = get_cfg()
    add_ateacher_config(cfg)
    cfg.merge_from_file(str(upstream / 'configs/faster_rcnn_R101_cross_clipart.yaml'))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    if args.phase != 'adapt':
        # Use upstream detector-only meta-architecture: the DA class adds a
        # source discriminator loss even in its supervised branch.
        cfg.MODEL.META_ARCHITECTURE = 'TwoStagePseudoLabGeneralizedRCNN'
    cfg.DATASETS.TRAIN = ('bdd_source_train',)
    if args.phase == 'finetune':
        cfg.DATASETS.TRAIN = ('bdd_finetune_train',)
        cfg.MODEL.WEIGHTS = str(args.source_run.absolute() / 'model_final.pth')
    elif args.source_run and args.phase == 'adapt':
        # A detector-only BaselineTrainer checkpoint has unprefixed keys while
        # Adaptive Teacher owns separate student and teacher modules.  Build a
        # bounded-run initialization artifact with the detector copied into
        # both modules; domain-discriminator weights intentionally remain at
        # their fresh initialization.
        source_checkpoint = torch.load(
            str(args.source_run.absolute() / 'model_final.pth'), map_location='cpu')
        detector_state = source_checkpoint.get('model', {})
        if not detector_state or not all(torch.isfinite(value).all()
                                         for value in detector_state.values()):
            parser.error('Source checkpoint has missing or non-finite detector weights')
        adapted_state = {}
        for key, value in detector_state.items():
            adapted_state['modelStudent.' + key] = value
            adapted_state['modelTeacher.' + key] = value.clone()
        initialization = args.output / 'adapt_init.pth'
        torch.save({'model': adapted_state}, str(initialization))
        cfg.MODEL.WEIGHTS = str(initialization.absolute())
    cfg.DATASETS.TRAIN_LABEL = ('bdd_source_train',)
    cfg.DATASETS.TRAIN_UNLABEL = ('bdd_target_unlabeled',)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.SUP_PERCENT = 100.0
    cfg.DATALOADER.RANDOM_DATA_SEED = 0
    cfg.DATALOADER.RANDOM_DATA_SEED_PATH = str(seed_file)
    cfg.INPUT.MIN_SIZE_TRAIN = (720,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1280
    cfg.INPUT.MIN_SIZE_TEST = 720
    cfg.INPUT.MAX_SIZE_TEST = 1280
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.IMG_PER_BATCH_LABEL = 1
    cfg.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    cfg.SOLVER.MAX_ITER = start_iter + args.steps
    cfg.SOLVER.STEPS = ()
    cfg.SOLVER.FACTOR_LIST = (1,)
    cfg.SOLVER.BASE_LR = 0.0001
    if args.phase != 'adapt':
        cfg.SOLVER.LR_SCHEDULER_NAME = 'WarmupMultiStepLR'
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.CHECKPOINT_PERIOD = args.steps
    cfg.TEST.EVAL_PERIOD = 0
    cfg.SEMISUPNET.BURN_UP_STEP = 1
    cfg.SEMISUPNET.Trainer = 'ateacher' if args.phase == 'adapt' else 'baseline'
    cfg.SEED = args.seed
    cfg.OUTPUT_DIR = str(args.output.absolute())
    cfg.freeze()
    state = dict(status='running', phase=args.phase, upstream_sha=sha,
                 steps=args.steps,
                 purpose='technical-smoke',
                 selection_evidence=False, holdout_access=False,
                 seed=args.seed, max_training_images_per_pool=args.train_images,
                 validation_images=args.validation_images,
                 resume_allowed=True)
    state.update(start_iteration=start_iter, resume_from=str(args.resume_from) if args.resume_from else None,
                 exact_rng_sampler_resume=False, rng_sampler_resume_restored=False,
                 rng_sampler_state_saved=False)
    state.update(domain=args.domain, labeled_budget=args.budget if args.phase == 'finetune' else None,
                 source_run=str(args.source_run) if args.source_run else None,
                 deterministic_requested=args.deterministic)
    state_path = args.output / 'smoke_status.json'
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    def timeout_handler(signum, frame):
        raise TimeoutError('Bounded pilot exceeded four hours')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(14400)
    started = time.monotonic()
    try:
        # Detectron2 0.3 collect_env crashes when CUDA_HOME is None, although
        # prebuilt CUDA wheels do not require a local compiler toolkit.
        setup_logger(output=cfg.OUTPUT_DIR, name='detectron2')
        setup_logger(output=cfg.OUTPUT_DIR, name='adapteacher')
        logging.getLogger('detectron2.engine.defaults').setLevel(logging.WARNING)
        (args.output / 'config.yaml').write_text(cfg.dump())
        seed_all_rng(cfg.SEED)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            cuda_matmul = getattr(getattr(torch.backends, 'cuda', None), 'matmul', None)
            if cuda_matmul is not None and hasattr(cuda_matmul, 'allow_tf32'):
                cuda_matmul.allow_tf32 = False
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = False
            deterministic = getattr(torch, 'use_deterministic_algorithms', None)
            if deterministic is not None:
                deterministic(True)
        else:
            torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
        trainer_cls = ATeacherTrainer if args.phase == 'adapt' else BaselineTrainer
        trainer = trainer_cls(cfg)
        if args.resume_from:
            restored = trainer.checkpointer.load(str(args.resume_from / 'model_final.pth'))
            trainer.start_iter = restored['iteration'] + 1
            if trainer.start_iter != start_iter or not trainer.optimizer.state:
                raise RuntimeError('Resume state was not restored')
            rng_path = args.resume_from / 'rng_sampler_state.pth'
            if rng_path.is_file():
                restore_rng_sampler_state(rng_path, trainer, torch, start_iter)
                state['rng_sampler_resume_restored'] = True
        else:
            trainer.resume_or_load(resume=False)
        class BenchmarkHook(HookBase):
            def __init__(self):
                self.elapsed = []

            def before_step(self):
                torch.cuda.synchronize()
                self.started = time.monotonic()

            def after_step(self):
                torch.cuda.synchronize()
                self.elapsed.append(time.monotonic() - self.started)

        benchmark = BenchmarkHook()
        trainer.register_hooks([benchmark])
        torch.cuda.reset_peak_memory_stats()
        trainer.train()
        save_rng_sampler_state(args.output / 'rng_sampler_state.pth', torch,
                               cfg.SOLVER.MAX_ITER)
        state['rng_sampler_state_saved'] = True
        state['sampler_cursor'] = cfg.SOLVER.MAX_ITER
        state['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        state['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        measured = benchmark.elapsed[20:]
        state['benchmark_warmup_updates'] = 20
        state['benchmark_measured_updates'] = len(measured)
        if measured:
            state['seconds_per_update'] = statistics.mean(measured)
        state['update_seconds'] = benchmark.elapsed
        metrics = [json.loads(line) for line in (args.output / 'metrics.json').read_text().splitlines()]
        losses = [row['total_loss'] for row in metrics if 'total_loss' in row]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise RuntimeError('Missing/nonfinite logged losses')
        checkpoint = args.output / 'model_final.pth'
        checkpoint_data = torch.load(str(checkpoint), map_location='cpu')
        if not {'model', 'optimizer', 'scheduler', 'iteration'} <= set(checkpoint_data):
            raise RuntimeError('Checkpoint lacks resumable training state')
        if checkpoint_data['iteration'] != cfg.SOLVER.MAX_ITER - 1:
            raise RuntimeError('Checkpoint iteration mismatch')
        del checkpoint_data
        trainer.checkpointer.load(str(checkpoint))
        state['checkpoint_reload'] = 'passed'
        state['checkpoint_bytes'] = checkpoint.stat().st_size
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        signal.alarm(0)
        state['wall_seconds'] = time.monotonic() - started
        state_path.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
