# Small-sample model-selection pilot

## Scope

The pilot screens SSDA-YOLO, SF-YOLO and Adaptive Teacher Faster R-CNN R101-C4.
MS-DAYOLO is excluded by the prior user decision. The old full comparison
campaign is retired and its launch code and generated planning artifacts have
been removed.

Each candidate uses exactly 1,000 source and 1,000 target training images.
R101-C4 runs 1,000 source iterations followed by 1,000 adaptation iterations.
The YOLO adapters run 1,015 traced batches and stop at exactly 151
optimizer/EMA updates, the smallest trace-demonstrated bound that passes the
entire 1,000-image pool. Comparison uses the same deterministic 1,000-image
lighting-validation prefix; the held-out test split is never read.

## Decision rule

Use the same validation domain and image membership for every candidate.
Inspect COCO mAP50:95, then mAP50. Reject a candidate first if its
run is incomplete, its loss/checkpoint is non-finite, its validation coverage
is incomplete, or it violates the label policy. Because the bounded checkpoints
have different upstream initialization and effective schedules, their AP order
is diagnostic rather than a scientifically controlled accuracy ranking. The
project choice must also account for proposal alignment, auxiliary dependencies
and measured execution cost.

The selected pipeline is a project engineering decision. Pilot accuracy must
not be reported as a final scientific result.

## Kept implementation

- `run_adaptive_teacher_smoke.py` and `evaluate_adaptive_teacher_smoke.py`;
- `run_ssda_adaptation_benchmark.py` and bounded SSDA input preparation;
- `run_sf_adaptation_benchmark.py` and bounded SF input preparation;
- `run_cut_smoke.py`, `export_cut_smoke_pairs.py` and diagnostic auxiliary audit;
- shared split/input verification and validation utilities.

All retained training entry points are bounded technical-smoke paths. There is
no production auxiliary launcher or full model-selection campaign runner.

## Result — 2026-09-28, 1,000-image rerun

All three candidates completed common evaluation on the same 1,000
lighting-validation images. Metrics below are COCO AP points, not fractions:

| Candidate | Training coverage | mAP50:95 | mAP50 |
| --- | --- | ---: | ---: |
| SSDA-YOLO | 1,000 source + 1,000 target; 1,015 batches/151 updates | 2.229456 | 6.614537 |
| SF-YOLO | 1,000-image source epoch; 1,000 target; 1,015 batches/151 updates | 1.856307 | 4.760677 |
| Adaptive Teacher R101-C4 | 1,000 source + 1,000 adaptation iterations | 0.686985 | 1.936026 |

SSDA-YOLO is the numeric leader of this short pilot. The result is not a
controlled accuracy comparison: the models have different pretrained weights,
source-training histories, update semantics and auxiliary inputs. It therefore
does not supersede the architecture decision or qualify as a project result.

The project selection remains **Adaptive Teacher Faster R-CNN R101-C4**. Its
corrected source-to-teacher/student conversion completed both 1,000-iteration
phases with finite losses and common validation coverage. This selection is a
proposal-alignment and research-design decision, not a claim that R101-C4 won
the short pilot numerically. Generated checkpoints, predictions and evaluator
JSON remain under ignored `runs/pilot1000-*` paths.

The earlier 128/64-image result is superseded by this rerun for engineering
screening. It remains historical evidence only.
