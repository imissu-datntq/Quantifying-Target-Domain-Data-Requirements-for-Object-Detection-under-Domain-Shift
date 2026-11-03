# Faster R-CNN training and evaluation pipeline

This Python pipeline implements the supervised core of proposal sections 6.1–6.5
using only `data/bdd100k_revised/`. It supports source training, target fine-tuning,
full-pool reference runs, generic-pretrained controls and shared COCO evaluation.
Implementation verification uses bounded train/validation smoke runs. Adaptation
methods, recovery-curve statistics and a full experimental grid are separate work.

## Files

| Python file | Purpose |
| --- | --- |
| `scripts/train_frcnn.py` | Source, fine-tuning and control CLI |
| `scripts/pilot_frcnn.py` | Bounded native-resolution throughput/smoke pilot |
| `scripts/evaluate_frcnn.py` | Validation or explicit frozen final-test execution |
| `scripts/freeze_evaluation_plan.py` | Freeze checkpoints and settings without test-label access |
| `scripts/verify_pipeline_resume.py` | Exact comparison of paired resumed/continuous smoke runs |
| `scripts/report_pipeline.py` | Vietnamese report from recorded verification artifacts |
| `scripts/bdd_pipeline/data.py` | Manifest integrity, domains, nested budgets, targets and COCO GT |
| `scripts/bdd_pipeline/model.py` | Shared Faster R-CNN ResNet-50 FPN contract |
| `scripts/bdd_pipeline/training.py` | Optimization, accumulation, selection, complete resume |
| `scripts/bdd_pipeline/evaluation.py` | COCO AP/AR and per-class/support diagnostics |
| `scripts/bdd_pipeline/artifacts.py` | Atomic checkpoint/JSON writes, environment and RNG |

Run commands from the repository root. Dependencies are in
`requirements-training.txt`; the verified environment uses PyTorch 2.11.0 CPU,
Torchvision 0.26.0 CPU and pycocotools 2.0.11. A GPU environment needs matching
PyTorch/Torchvision CUDA builds from the official PyTorch distribution.

## Data and model contracts

- Check all three manifest hashes, 55,000/14,863/10,000 counts, native dimensions,
  unique basenames and cross-split basename/content overlap before selecting data.
  Training reads train/val annotations only; hashes those annotation files and
  records them. It never opens held-out detection annotations.
- **Protocol amendment 2026-10-19 (evaluation roles).** Physical folders are
  unchanged, but weather (daytime-rainy) and compound (night-rainy) validation
  images belong to the final test set: final test is 2,650 source / 4,869
  lighting / 933 weather / 756 compound images, target pools are unchanged.
  `selection_rows` refuses weather/compound validation; `final_test_rows` merges
  them into the test role and `read_manifests` asserts every role count.
- Checkpoint selection: source training selects on source validation; every
  target run (fine-tune, ceiling, control) keeps the final update of the common
  fixed schedule, so all shifts share one rule. Lighting validation is used only
  for the schedule/hyperparameter pilot.
- Class IDs 1–10 follow dataset metadata: pedestrian, rider, car, truck, bus,
  train, motorcycle, bicycle, traffic light, traffic sign. ID 0 is background.
- Read RGB tensors in [0,1]. Model applies normalization once using ImageNet
  mean/std. Fixed min/max sizes 720/1280 preserve native 1280×720 geometry;
  internal FPN padding to height 736 is padding, not image resize.
- Reject malformed/nonfinite/nonpositive boxes, clip to bounds and drop empty
  clipped boxes. Same function supplies training targets and evaluator GT. Keep
  valid occluded/truncated objects; neither attribute is mapped to COCO crowd.
- Training augmentation is horizontal flip with matching bbox transformation.
  It is deterministic by seed/epoch/name; no weather or color augmentation is
  introduced. Validation/test have no augmentation.
- ResNet-50 FPN, FrozenBatchNorm with fixed epsilon 0, frozen stem/layer1,
  trainable layer2–4/FPN/RPN/ROI, 11-output task head. Standard RPN proposal counts,
  score cutoff 0.001, NMS 0.5 and max 100 detections per image.
- `coco` initializes backbone/FPN/RPN/ROI from official COCO detector weights and
  resets the task-specific head. `imagenet` initializes the backbone only.
  `none` is offline random initialization and is restricted to smoke runs.

Builder/input references: [Torchvision Faster R-CNN](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html).
COCO evaluation reference: [official COCOeval implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

## Budget and optimization rules

Only target train images are sampled. Sort basenames, shuffle using the budget
seed, then take the first N images. N=10 is a prefix of N=25 for the same seed.
`full` is all available target-train images. Validation membership is unchanged.
The initialization control uses the same subset sampler as target fine-tuning.

Training ends after:

```text
updates_per_epoch = ceil(ceil(N / micro_batch_size) / accumulation)
total_updates = max(S, E * updates_per_epoch)
```

This implements the proposal's longer-of-update-floor-or-epochs rule, including
the full-pool point. Each epoch keeps its final partial batch. Each accumulation
window weights microbatch losses by its actual image count, including the final
short window. The configured effective batch is micro_batch×accumulation; actual
short-window sizes are recorded. S/E/B and optimizer policy must be locked after
a representative GPU pilot before any research campaign.

Defaults **are provisional implementation values**: S=2000, E=30, effective
batch 2, SGD LR 0.005/momentum 0.9/weight decay 0.0005, warmup 100 updates then
cosine to 1% base LR, norm clipping 10, support threshold 100. They are not a
validated research training recipe. AMP is optional on CUDA. All these settings
are recorded; use the same policy across target budgets and full-pool ceilings.

The primary metric is mAP50:95 over one **fixed reliable-class roster shared by
every domain**: pedestrian, car, traffic light, traffic sign. It is undefined
(null) rather than silently shrunk when an evaluation set lacks one of them. The
ten-class mean is the sensitivity analysis. Classes absent from GT have null AP
and remain in the per-class roster. COCO AP50/AP75, area AP and AR100 are
diagnostics. Do not select research policy from smoke mAP.

## Commands

Inspect configuration without loading a detector or training:

```powershell
python scripts/train_frcnn.py --output-dir runs/source_candidate --dry-run
```

Run a bounded pilot with generic COCO weights (downloaded on first use):

```powershell
python scripts/pilot_frcnn.py --output-dir runs/pilot_cpu --device cpu --steps 5 --images 10 --validation-images 8
```

The research commands below are examples for use **after policy/pilot decisions**.
Use a new output directory for each domain/budget/seed; they were not executed
as a research campaign during implementation.

```powershell
python scripts/train_frcnn.py --stage source --domain source --budget full --output-dir runs/source_seed0 --seed 0 --device cuda --amp
python scripts/zero_shot_gap.py --checkpoint runs/source_seed0/best.pt --output-dir runs/gap_pilot --device cuda
python scripts/train_frcnn.py --stage fine-tune --domain lighting --budget 100 --source-checkpoint runs/source_seed0/best.pt --output-dir runs/lighting_N100_seed0 --seed 0 --device cuda --amp
python scripts/train_frcnn.py --stage fine-tune --domain compound --budget full --source-checkpoint runs/source_seed0/best.pt --output-dir runs/compound_full_seed0 --seed 0 --device cuda --amp
python scripts/train_frcnn.py --stage control --domain weather --budget 100 --initialization coco --output-dir runs/weather_control_N100_seed0 --seed 0 --device cuda --amp
```

The zero-shot gap pilot comes first: it compares the source checkpoint on
source validation with each target's training pool (never seen by the source
model, no held-out labels) and bootstraps the gap. A shift whose 95% interval
does not clear `--min-gap` (default 0.01 AP) is flagged before any budget ladder
is launched, because Recovery's denominator would be near zero.

For matched-pool ceilings, pass `--budget 1738` in lighting/weather. Repeat each
non-ceiling rung under the agreed sampling seeds and ceiling runs under at least
two optimization seeds as in the proposal. No architecture/adaptation comparisons
are silently introduced by this pipeline.

Resume uses `--resume <original-run>/last.pt` with the original command settings
and output directory. `--stop-after-updates K` stops an invocation after K updates
without changing the configured total, so resume follows the original schedule.
Run/model/data/source fingerprints, runtime package versions and configuration
must match. Changing the source files or development annotations is rejected.

Checkpoints contain model, optimizer, scaler, Python/NumPy/torch/CUDA RNG,
completed update count, epoch/position, schedule state, best validation score and
processed-image counts. Deterministic epoch sampler positions and flip decisions
are reconstructible. Writes are atomic; checkpoint cadence, epoch boundary,
validation and graceful-stop boundaries save `last.pt`. `best.pt` preserves the
selected weights and selected step, with training-completion metadata attached
after the run finishes. Resume from historical best is intentionally rejected.

Validation evaluation, including source-domain retention for target checkpoints:

```powershell
python scripts/evaluate_frcnn.py --checkpoint runs/lighting_N100_seed0/last.pt --domain source --output-dir runs/lighting_N100_seed0/source_retention --device cuda --amp
```

Evaluation saves COCO-format predictions, filename/image-ID mapping and metrics.
The run's dataset/class contract is checked. Only trusted pipeline checkpoints
should be loaded: optimizer/RNG state uses PyTorch's full local checkpoint loader.

## Final test boundary

The evaluator has no freely selectable `--split test` switch. Its default path
is validation. For final test, first fix *all* checkpoint choices, domains, support
roster, precision and evaluation batch without reading test annotations:

```powershell
python scripts/freeze_evaluation_plan.py --checkpoints runs/source_seed0/best.pt runs/lighting_N100_seed0/last.pt --domains source lighting --amp --output runs/final_plan.json
python scripts/evaluate_frcnn.py --final-plan runs/final_plan.json --output-dir runs/final_results --device cuda --amp
```

Plans use schema 2 and record the fixed reliable roster, the holdout-val domains
and per-domain final-test sizes. Recovery curves, including one curve per reliable
class to separate class-prior re-learning from new features, come from the frozen
metrics:

```powershell
python scripts/recovery_curves.py --index runs/recovery_index.csv --output-dir runs/recovery
```

A plan rejects smoke/incomplete/modified checkpoints, wrong manifests and changed
evaluation implementation. A completed run's selected `best.pt` is valid even
when its selected step precedes the final optimization step. A one-shot opened
marker is created adjacent to the plan before any test-annotation access; this
plan cannot be rerun by choosing a different output directory. Checkpoint/domain/
limit overrides are prohibited. A failure after opening preserves partial output
for diagnosis, and requires an explicit recovery decision rather than automatic
holdout re-execution. No final plan or test evaluation has been executed here.

## Verification

```powershell
python -m unittest discover -s tests -v
python scripts/train_frcnn.py --help
python scripts/evaluate_frcnn.py --help
python scripts/pilot_frcnn.py --help
python scripts/freeze_evaluation_plan.py --help
```

The implementation report `docs/pipeline_report_vi.md` records actual CPU smoke,
resume equivalence, evaluator checks and remaining hardware/research limitations.

## Campaign path (added 2026-10-30)

The grid runs as Kaggle sessions. Held-out labels are opened once, by scoring frozen
prediction files after the whole grid is complete.

| File | Purpose |
| --- | --- |
| `scripts/bdd_pipeline/campaign.py` | Locked target policy (`POLICY`), the 87-run grid in launch order, and per-run cost |
| `scripts/campaign_grid.py` | Writes `runs/campaign/grid.csv`. Use `--assume S,E,LR` for planning estimates while the policy is unlocked |
| `scripts/predict_frcnn.py` | Predicts on final-test images through `BDDImages`, which never opens an annotation file. Writes hashed `predictions_<domain>.json` plus `prediction_manifest.json` |
| `scripts/freeze_evaluation_plan.py --prediction-manifests` | Schema-3 plan. Rejects smoke, incomplete or limited runs, files that changed, and prediction sets that don't cover the exact final-test image list |
| `scripts/evaluate_frcnn.py --final-plan` | Schema 3: re-checks every hash, takes the one-shot lock, then scores each domain's labels once. Predictions are mapped to images by file name. Writes `metrics.json` and `bootstrap_class_ap.npy` (1,000 shared test-image resamples, seed 20261030) |
| `scripts/analyze_campaign.py` | Produces the RQ1–RQ4 outputs listed below |
| `kaggle/bdd_campaign.py` | Queue-runner kernel: restores previous sessions, resumes exactly, stops before the 12 h limit, predicts, and drops optimizer checkpoints |

`analyze_campaign.py` produces:
- **RQ1:** the domain drop and the compound-vs-sum interaction, each with a CI.
- **RQ2:** seed-mean recovery curves with a hierarchical bootstrap band. Each band resamples the sampling seeds, the ceiling seeds and the test images.
- **RQ3:** N_BE/N75/N90/N95, each given as a point estimate, a band interval, a ladder bracket, and a power-fit estimate with CI.
- **RQ4:** the same thresholds measured against matched-pool ceilings.
- Forgetting on the source test set, and the generic-pretrained controls.

**Freeze rule.** `training.py` records the SHA256 of every `bdd_pipeline/*.py` in each run's config, and resume rejects any change. Finish all `bdd_pipeline/` edits before the first campaign session. Analysis changes after launch belong in `scripts/analyze_campaign.py`, which is outside the package.
