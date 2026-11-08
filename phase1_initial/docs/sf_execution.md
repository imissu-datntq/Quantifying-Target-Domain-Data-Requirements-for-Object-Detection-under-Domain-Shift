# SF-YOLO technical execution

## One-thousand-image project pilot — 2026-09-28

`runs/pilot1000-sf-source` completed one source epoch over 1,000 images and
validated on 1,000 source images. `runs/pilot1000-sf-adapt-v2` then used that
checkpoint with 1,000 image-only lighting-target images. The 120-update attempt
was rejected at 503 batches; the accepted run completed 1,015 batches and
exactly 151 optimizer/EMA updates with finite loss and both checkpoints.
Common 1,000-image lighting validation produced mAP50:95 1.856307 and mAP50
4.760677 AP points. The result is bounded engineering evidence, not a final
project score.

## Bounded checkpoint resume — 2026-09-25

The saved SF bundle from `runs/sf-adapt-120-001` was resumed for one bounded
epoch as `runs/sf-adapt-resume-001`. The upstream loader restored the detector,
optimizer, scheduler/scaler and teacher/EMA bundle: the trace continued at
batch 7,680 for 128 batches, and the student checkpoint advanced from epoch 59
/ 139 EMA updates to epoch 60 / 141 updates. The independent audit is recorded
in `runs/sf-adapt-resume-001/tensor_audit.json`; it passed finite losses,
positive diagnostic pseudo-boxes, all model/EMA/teacher tensors and 309
optimizer states. The output student checkpoint SHA-256 is
`cb5196f19213face40f2b2ac39ce687dcfb16d8bc9c6b5ba21f67f7d6fe24949`.

This closes the bounded serialized-state continuation check only. The audit
sets `checkpoint_resume_verified=true` but keeps `resume_verified=false` and
`exact_rng_sampler_resume_verified=false`: the upstream checkpoint has no
dataloader/RNG state, so exact reproducibility across a restart is still a
campaign gate. The run uses diagnostic confidence 0.001 and is not selection
evidence.

## 120-update adaptation benchmark — 2026-09-25

`run_sf_adaptation_benchmark.py` now provides an exclusive, provenance-recorded
launcher for the pinned image-only target graph. It uses the verified v7.0
ten-class source checkpoint, 128 lighting-train target images, TargetAugment
smoke-002 weights, native 1280 input, batch 1, no target labels/auto-anchor/
multiscale, confidence .001, style alpha .4 and SSM alpha .5. A sidecar status
file records hashes and failures before the upstream output exists; a completed
run requires at least 120 EMA/optimizer updates, finite trace values and both
student/teacher checkpoints. `runs/sf-adapt-120-001` completed 7,680 batches,
139 EMA/optimizer updates and 3,196.91 seconds wall time. The independent
`tensor_audit.json` found all 7,680 batches positive at the diagnostic .001
threshold, finite student/teacher/EMA tensors and 309 finite student optimizer
states. Resource sampling observed 5,374 MiB GPU memory and 1,663,436 KiB
Darknet-process RSS. This remains technical evidence only; exact resume,
pseudo-label quality and selection evidence are not claimed.

2026-09-25 pretrained-source runtime: `runs/sf-pretrained-source-smoke-001`
completed the upstream source training loop for one epoch / 128 source-train
images, with 64 source-validation images, seed42, native rectangular geometry,
FP32 and disabled geometric augmentation. Unlike the original source smoke,
this initializes from verified official v7.0 weights and trains a ten-class head.
Canonical source image/label identities were independently checked. Epoch
metrics and all saved model tensors were finite. Last checkpoint SHA-256 is
`b54237c44718487cd6e7f840b211e85019afa125e4d6d780ea4160cfaef68d81`.
Upstream strips the final optimizer: this is not resumable-training evidence or
a selected source model. Console reserved memory was ~3.18 GB, not a formal
benchmark. Source CLI also performed its upstream Git status fetch (no checkout
change; pinned HEAD retained); disable that network side effect in the harness.

2026-09-25 pretrained asset: official v7.0 YOLOv5-L download resumed from its
confirmed failed 70 MiB partial transfer and completed. All tensors were finite
and strict reload into this fork's checkpoint-defined architecture passed
(80 classes, 46,563,709 parameters). Promoted to `weights/yolov5l-v7.0.pt`,
93,622,629 bytes; SHA-256 recorded in `environments/pretrained-assets.json`.
This is COCO initialization, not a BDD-trained source model. Source training
and ten-class transfer with this asset remain to be checked.

Two-epoch follow-up `runs/sf-positive-smoke-002` completed 256 batches and
23 optimizer/EMA updates with style alpha .4, SSM alpha .5 (the second epoch
executes teacher-to-student mixing), diagnostic confidence .001 and FP32.
All 256 batches contained teacher-generated boxes. Loss trace and saved student,
EMA, teacher bundle and 309 optimizer states passed `audit_yolo_training_smoke.py`;
hashes are in `tensor_audit.json`. Both epochs and final source-validation
plumbing completed. Console reserved memory was ~4.36 GB, not a formal benchmark.
This closes the initial nonempty/multi-epoch execution gap but does not verify
pseudo quality, 120 optimizer updates, source pretraining or exact resume.

2026-09-24 positive-branch checkpoint: `runs/sf-positive-smoke-001` completed
128 batches/19 updates with teacher-generated boxes in all batches at diagnostic
confidence .001. Student, teacher, EMA and 309 optimizer states were finite;
source-validation plumbing completed. This run used upstream defaults style
alpha 1 and SSM alpha 0, so it does NOT establish multi-epoch student mixing.
The image-only adapter now preserves gradient accumulation (upstream's student
zero_grad inside every forward discarded it), rejects non-finite loss, and logs
per-batch pseudo counts, AMP scale and EMA updates. Earlier smoke 001 predates
this fix. Low-confidence boxes are branch coverage, not quality/threshold tuning.

Common-export follow-up: the smoke teacher passed native-coordinate prediction
export and shared COCO evaluation on all 286 compound-val images in
`runs/sf-common-compound-001`. This verifies exporter plumbing, not training
quality, full benchmarking or resume.

Pinned revision: `c84dfad79a889b5f172d7192ee0379edfd738835`.
Environment/label-free adapter and auxiliary evidence are recorded in
`environments/sf-yolo.freeze.txt`, `patches/sf-yolo-label-free.patch` and
`docs/auxiliary_execution.md`. This is not selection-quality evidence.

`prepare_sf_source_smoke.py` creates 128 source-train and 64 source-validation
symlink/label views. `sf-source-smoke-001` ran upstream `train_source.py` for
one epoch from random initialization, ten-class YOLOv5-L, batch 1, rectangular
1280 image size and disabled augmentation/multiscale/auto-anchor. The optional
`technical-no-amp` patch avoids an automatic AMP-probe weight download and uses
FP32. Seed was upstream default 0. It completed training and source-validation
plumbing, and saved last/best checkpoints. Upstream strips optimizer from these
final source artifacts; they are not resumable research checkpoints.

`prepare_sf_adapt_smoke.py` writes a 128-image lighting-train manifest without
label paths. `sf-adapt-smoke-001` used the diagnostic source checkpoint and
TargetAugment smoke-002 assets, upstream batch-style mode, alpha .4, SSM .5,
teacher EMA .999, seed 42 and FP32. One epoch of 128 target batches completed,
with finite objectness loss and validation plumbing on 64 source-val images.
The 128 batches comprised **19 optimizer/EMA updates**, not the required
120-update benchmark. Console reserved memory was about 4.3 GB; this is not a
formal peak-allocation measurement.

The weak diagnostic teacher produced no pseudo boxes at confidence .4; box and
class losses stayed zero. Empty-label handling ran without fabricated boxes.
Positive pseudo-box regression/classification, meaningful source initialization,
multi-epoch student mixing and exact restart are not established by this run.

Upstream built `ckpt_student` but did not save it. The saved label-free patch
now saves last/best student checkpoints as well as teacher exports, including
student optimizer, scheduler, scaler and teacher model. Inspection of
`last_student.pt` found epoch 0, 19 EMA updates, 309 optimizer states and finite
student/teacher/EMA tensors. Resume is explicitly marked unverified: existing
upstream loading does not yet restore this complete bundle or RNG/sampler state.
Do not treat the presence of these fields as a tested resume implementation.

Next: verify a source initialization compatible with this fork, exercise
positive pseudo boxes and at least 120 optimizer updates, implement/verify
full checkpoint restore, then export original-coordinate predictions to the
common validation evaluator. This fork's own downloader defaults to official
YOLOv5 release v7.0; its YOLOv5-L asset is being fetched separately from SSDA's
v5.0 weights and must pass hash/architecture checks before use.
