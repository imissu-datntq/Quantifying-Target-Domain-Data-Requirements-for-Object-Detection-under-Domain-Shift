# Faster R-CNN Python training/evaluation pipeline

Status: complete. Started and completed 2026-10-02 Asia/Saigon.

## Purpose and observable outcome

Deliver Python data, model, training, checkpoint and COCO evaluation modules, CLI
entry points, checks and a Vietnamese implementation report. User authorized
implementation and verification, not a full research training campaign.

## Context and scope

Use only data/bdd100k_revised, proposal 6.1–6.5 and fixed native 1280×720 input.
Support source training, nested target budgets and full-pool ceiling, plus
generic-pretrained controls. Train/val selection remains independent of test.
Implement final test execution behind a frozen explicit checkpoint plan.
No adaptation algorithm reproduction, full training grid or test scores now.

## Progress and discoveries

- Read current instructions, handoff and proposal. Source dataset remains fixed.
- Existing Python has torch, but lacks torchvision/pycocotools. GPU availability
  is being inspected. Existing Git index has user changes; preserve them.
- Installed compatible torchvision 0.26.0 and pycocotools 2.0.11 without replacing
  torch 2.11.0. Confirmed CPU only; no CUDA/AMP verification possible locally.
- Implemented checked domains, nested budgets, native detector contract,
  weighted accumulation, complete atomic resume, validation selection, shared
  COCO metrics and one-shot frozen final evaluation plan.
- Fixed NumPy category-ID JSON serialization and aligned FrozenBN epsilon across
  COCO/random/reload paths before running the final verification pair.
- 2026-10-02 11:20: 14/14 semantic tests pass; 17 Python files pass syntax checks.
  Native CPU forward/backward, standalone val reload, target fine-tune and COCO
  pretrained pilot pass. Checkpoint resume matches continuous training exactly
  across 295 model tensors, optimizer/scaler/RNG/schedule/position (max delta 0).
- CLI guards reject smoke freeze and altered-LR resume. Source/control dry runs
  and affected CLI help paths pass. No test detection annotations accessed.

## Decisions

Provide reusable Python modules and thin CLI scripts. Record the class order,
source/manifest fingerprints, annotation sanitation, normalization, horizontal
flip and COCO evaluation settings. Save complete optimizer/scaler/RNG/position
state atomically and reject incompatible resumes. Small verification must be
labeled pipeline smoke, not experimental mAP evidence.

## Ordered work

1. Install compatible detection/evaluation dependencies and inspect runtime.
2. Implement checked domain data/budgets and fixed-resolution model.
3. Implement common optimization schedule, resume and validation selection.
4. Implement COCO evaluator and frozen-plan final-test boundary.
5. Run semantic tests and native-resolution train/val smoke, verify resume,
   document measured results and update project state.

## Validation criteria

Native geometry, class mapping, sanitation/flip, nested budgets, manifest hashes,
correct optimizer-update counts and weighted accumulation, complete resume,
perfect/empty predictions in COCO metrics and final-plan validation. Execute
real Faster R-CNN forward/backward/checkpoint/val inference at native resolution.

## Recovery/interfaces/dependencies

Use installed torch plus matching torchvision/pycocotools; avoid replacing
existing torch. Generated checkpoints/metrics remain under ignored runs/.
Run directories and final outputs refuse accidental overwrites. Partial training
can resume from last checkpoint; final-evaluation selection is immutable.

## Outcome

Delivered Python supervised training/evaluation modules and CLI, pinned training
dependencies, usage documentation and `docs/pipeline_report_vi.md` generated
from actual verification evidence in ignored `runs/pipeline_verified/`.
README and live state updated; user Git index preserved. No split membership
changed and no full campaign or held-out performance evaluation launched.

CPU smoke establishes functioning data/model/backward/checkpoint/evaluator paths,
not model convergence. CUDA/AMP/multiworker checks and representative GPU policy
pilot precede research training. Adaptation/recovery analysis remains later work.
