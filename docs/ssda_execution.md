# SSDA-YOLO technical execution

## One-thousand-image project pilot — 2026-09-28

`runs/pilot1000-ssda-adapt-v3` completed on 1,000 source and 1,000 image-only
lighting-target images. The 120-update attempt was correctly rejected because
it stopped after 503 batches. Trace evidence established 151 as the smallest
integer update limit that exceeds one full pool: the accepted run stopped at
1,015 batches/151 optimizer and EMA updates, with finite loss and both student
and teacher checkpoints. Common 1,000-image lighting validation produced
mAP50:95 2.229456 and mAP50 6.614537 AP points. This numerical lead belongs
only to the bounded engineering pilot and does not replace the fixed R101-C4
project architecture.

## Native 120-update adaptation checkpoint — 2026-09-25

The immutable launcher completed `runs/ssda-adapt-120-003` with the pinned
image-only paired graph: 4,608 native batches, 122 optimizer/EMA updates,
2,286.94 seconds wall time and exit 0. The training log's maximum reserved-GPU
display was 4.72 GiB (observational, not a sampled peak). The independent audit passed all trace
rows, finite student/EMA/teacher tensors, 327 finite student optimizer states,
constant AMP scale 65,536 and positive diagnostic pseudo-boxes in all 4,608
batches. Checkpoint SHA-256 values and the full trace hash are in
`runs/ssda-adapt-120-003/tensor_audit.json`.

The bounded continuation `runs/ssda-adapt-resume-001` restored the serialized
student/optimizer/EMA/scheduler/scaler bundle for 128 batches (4,608–4,735),
advancing 122 to 124 updates. Its audit sets
`checkpoint_resume_verified=true`, while `resume_verified=false` and
`exact_rng_sampler_resume_verified=false` because upstream does not serialize
dataloader/RNG state. The `ema` checkpoint was exported on all 286 compound
validation images in `runs/ssda-common-compound-002` and passed the shared COCO
evaluator. These metrics remain diagnostic (`selection_evidence=false`), not
model-ranking evidence.

The two shorter pilots (`ssda-adapt-120-001` at 56 updates and
`ssda-adapt-120-002` at 68) remain preserved as incomplete immutable runs; they
were not overwritten or promoted.

## 120-update adaptation benchmark — 2026-09-25

`run_ssda_adaptation_benchmark.py` now provides an exclusive, provenance-
recorded launcher for the pinned image-only paired graph. It fixes the same
native 1280 geometry, batch 1, seed 42, consistency branch, confidence 0.001
and train-only lighting inputs used by the positive smoke. The upstream
schedule warms up accumulation for 1,000 batches, so the first two bounded
pilots are preserved as incomplete rather than relabeled: `runs/ssda-adapt-
120-001` finished 512 batches/56 updates and `runs/ssda-adapt-120-002`
finished 1,152 batches/68 updates, both with finite losses and positive
diagnostic pseudo-boxes. The native 36-epoch run `runs/ssda-adapt-120-003` then
cleared the gate with 4,608 batches/122 updates without changing upstream
accumulation. Its independent trace/tensor audit and bounded serialized resume
pass; exact RNG/sampler restoration remains open. No run is selection evidence.

Positive-pseudo-label follow-up: `runs/ssda-positive-smoke-001` completed 128
batches/36 optimizer-EMA updates with 20 teacher-produced boxes in every batch,
finite loss/model/EMA/teacher/327 optimizer states and unchanged AMP scale 65536.
This uses diagnostic confidence .001 solely to exercise the nonempty branch;
it neither validates pseudo-label quality nor selects the campaign threshold.
Source/target splits and all other inputs match smoke 002. Upstream validation
on 64 lighting-val images completed. Separately, smoke 002 student EMA exported
and passed common COCO evaluation on all 286 compound-val images, through the
new explicit serialized-model/EMA export path. Earlier outstanding-positive-box
and serialized-export notes below are superseded by this evidence.

2026-09-24 full-graph checkpoint: `runs/ssda-adapt-smoke-002` completed one
epoch / 128 native paired batches / 36 optimizer-and-EMA updates with the
consistency branch enabled, COCO v5.0 initialization and diagnostic CUT pairs.
All batch losses were finite, AMP scale stayed 65536, and teacher pseudo boxes
were empty for all batches at the default .5 threshold. Console training
reserved memory was approximately 4.68 GB; this is not a formal peak benchmark.
Upstream validation completed on 64 lighting-val images. Student, teacher and
student EMA checkpoint tensors reloaded and were finite; all 327 student
optimizer states were finite. This historical smoke remains technical graph
evidence, not model selection or production CUT evidence. The later native
120-update run, bounded restart and serialized-model common exporter are
documented above; fine-tuning, exact RNG/sampler restart and production CUT
quality remain open.

The opt-in `--unlabeled-target` patch uses a target CSV plus translated-image
directory, reads no target annotation/cache files, preserves native paired
pixels with stride padding, inherits teacher class weights from labeled source,
and rejects auto-anchor/image-weight/quad/multiscale target options. Empty
teacher output stays empty, replacing upstream's random-box fallback in this
mode only. `--target-seed` controls target shuffling; global campaign seeding
is still to be unified. One paired-loader runtime test passes. Diagnostic
translation provenance is verified by `prepare_ssda_adapt_smoke.py`; production
translation quality and positive-pseudo-label regression remain unverified.

Paired preparation uses 128 source train images, 128 image-only lighting train
images and 64 lighting validation images. Despite the upstream key name
`test_target_real`, it points only to validation, never the research holdout.
Input/output hashes and membership must match both completed CUT exports.

Run `ssda-adapt-smoke-001` was interrupted before training: upstream iterated
characters of a string path and recursively enumerated filesystem paths from
`/`. It had not reached image/label cache construction or optimizer updates.
The patch now iterates normalized path groups, validates matching group counts,
and has a regression check. The opt-in image-only branch also retains gradient
accumulation across microbatches (upstream zeroed it each forward), fails loudly
on backward/non-finite errors, logs batch loss/pseudo-count/AMP scale/EMA updates,
and preserves optimizer state instead of stripping it at exit. Automatic package
installation and technical-mode Git fetch are disabled. Exact resume remains
unimplemented; these adapter deviations are recorded, not hidden.

Common-export follow-up: the source smoke passed native-coordinate prediction
export and shared COCO evaluation on all 286 compound-val images in
`runs/ssda-common-compound-001`. Paired translated-image adaptation and training
quality evidence remain outstanding.

Pinned upstream: `3ab1f506ded2ca67233699a2e9bb8944aea40013`.
Environment: `.venvs/ssda-yolo`, freeze in `environments/ssda-yolo.freeze.txt`.
Official v5.0 pretrained weight provenance: `environments/pretrained-assets.json`.

`run_ssda_source_smoke.py` exercises the upstream YOLOv5-L detector and loss
with ten classes, loading 644 compatible pretrained tensors. It uses 128 source
train images, batch 1, unchanged 1280x720 content padded to 1280x736, SGD
lr=0.0001/momentum=.9, AMP and seed 42. Padding-only augmentation and this short
schedule are diagnostic, not the selection-training recipe.

`runs/ssda-source-smoke-001` completed 120 steps with finite losses, checkpoint
save/reload (model/optimizer/scaler), 1,645,638,144 bytes peak allocated and
1,746,927,616 bytes reserved. After 20 warmup steps, 100 updates averaged
0.1200859 seconds; this excludes image loading and validation. Wall time was
38.97 seconds. Separate inspection found all saved model tensors finite,
327 optimizer states, last step 119 and AMP growth tracker 120 with unchanged
scale 65536 (no overflow-induced skipped steps observed).

The saved checkpoint is technical source-only evidence, not a selected source
model or adaptation result. Source validation export, ten-class prediction
coordinate checks, full paired translated-image SSDA graph and auxiliary CUT
cost remain required. The launcher supports bounded model/optimizer/scaler/
iteration resume to a fresh directory; it does not claim exact RNG/sampler
restoration. Do not substitute this smoke for full-pipeline eligibility.

Bounded resume `runs/ssda-source-resume-001` passed: restored model, optimizer,
scaler and iteration, continued six steps starting at 120, and reloaded the
result. Wall time 6.15 seconds; exact RNG/sampler continuation remains unproven.
