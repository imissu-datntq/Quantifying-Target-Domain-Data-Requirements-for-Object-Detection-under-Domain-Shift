# Quantifying Target-Domain Data Requirements for Object Detection under Domain Shift

This repository supports a BDD100K study of how much labeled target-domain
data is needed to recover object-detection performance after domain shift.
The study compares source-only inference, supervised target fine-tuning and a
single established adaptation pipeline while keeping detector, splits and
evaluation fixed.

## Current scope

The bounded 1,000-image pilot is complete. The selected project pipeline is
**Adaptive Teacher + Faster R-CNN R101-C4**. SSDA-YOLO was the numeric leader
on this deliberately short engineering screen, but the pilot does not support
an accuracy-winner claim: initialization, source pretraining and optimization
semantics differ. R101-C4 remains selected because it matches the project
proposal and the intended teacher-student Faster R-CNN study. MS-DAYOLO remains
excluded after repeatable host-memory failures.

Pilot constraints:

- exactly 1,000 source and 1,000 target training images per candidate;
- 1,000 source plus 1,000 adaptation iterations for R101-C4;
- 1,015 traced batches and 151 optimizer/EMA updates for each YOLO adaptation;
- the same deterministic 1,000-image lighting validation prefix for all three;
- validation data only for comparison; the held-out test split stays blocked;
- pilot scores are screening evidence, not final project metrics.

The retired 60-arm campaign and long-run launch code have been removed. The
authoritative pilot result and decision are in
[`docs/small_sample_testing.md`](docs/small_sample_testing.md).

## Research design

The materialized BDD100K split contains:

- 55,000 training images;
- 14,863 held-out final-test images;
- the unchanged official 10,000-image validation split.

Daytime-clear is the source domain. Target domains are:

| Shift | Target condition | Train | Test | Validation |
| --- | --- | ---: | ---: | ---: |
| Lighting | Night, clear | 18,015 | 4,869 | 3,274 |
| Weather | Daytime, rainy | 1,985 | 537 | 396 |
| Compound | Night, rainy | 1,738 | 470 | 286 |

Target training budgets are nested and sampled from training data only. The
main ladder is `10, 25, 50, 100, 250, 500, 1,000, full`; lighting additionally
uses `2,000, 5,000, 10,000` in the later project experiment.

The primary recovery quantity is:

```text
Recovery(N) = [mAP(N) - mAP(0)] / [mAP_c - mAP(0)]
```

Primary reporting uses COCO mAP50:95. Final held-out evaluation begins only
after the project pipeline and protocol are locked.

## Repository status

The protocol is [`docs/proposal_dataset_aligned.docx`](docs/proposal_dataset_aligned.docx).
Live execution state is under [`.agent/`](.agent/README.md).

`bdd100k_research_split.py` materializes and verifies the versioned split at
`data/research_split_v1`. `bdd100k_model_inputs.py` creates canonical COCO,
YOLO and manifest inputs at `data/research_inputs_v1`. Generated datasets,
external repositories, checkpoints and runs remain untracked.

Small diagnostic integrations are available for all three eligible candidates.
They verify native geometry, finite losses/checkpoints and bounded validation
plumbing. They do not establish final accuracy or justify a full campaign.

`train_da_yolo_comparison.py` is an older prototype based on a different split
and is not the implementation of the current proposal.

## Local checks

```bash
python bdd100k_research_split.py verify
python bdd100k_model_inputs.py verify
python pipeline_doctor.py
python -m unittest discover -s tests -v
```

Bounded R101 1,000-image pilot example:

```bash
.venvs/adaptive-teacher/bin/python run_adaptive_teacher_smoke.py \
  --phase source --steps 1000 --train-images 1000 --validation-images 1000 \
  --output runs/r101-source-pilot

.venvs/adaptive-teacher/bin/python run_adaptive_teacher_smoke.py \
  --phase adapt --domain lighting --steps 1000 \
  --train-images 1000 --validation-images 1000 \
  --source-run runs/r101-source-pilot --output runs/r101-lighting-pilot

.venvs/adaptive-teacher/bin/python export_r101_validation.py \
  --run runs/r101-lighting-pilot --component teacher --domain lighting \
  --max-images 1000 --output runs/r101-lighting-validation

.venvs/adaptive-teacher/bin/python evaluate_validation.py \
  --domain lighting --max-images 1000 \
  --predictions runs/r101-lighting-validation/predictions.json \
  --coverage runs/r101-lighting-validation/inference_manifest.json \
  --output runs/r101-lighting-metrics.json
```

Every output directory must be new. No small-sample command accesses the
held-out test split.
