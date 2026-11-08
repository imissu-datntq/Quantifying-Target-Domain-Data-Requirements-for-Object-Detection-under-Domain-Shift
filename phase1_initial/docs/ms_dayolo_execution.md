# MS-DAYOLO technical feasibility checkpoint

## Supervised source/fine-tune smoke — 2026-09-25

`run_ms_supervised_smoke.py` now exercises Darknet's ordinary supervised path
without `-da`. It uses only canonical train manifests and labels, native
1280x736 geometry, batch 2/subdivisions 2 and the pinned MS-DAYOLO revision.
The first two source attempts are preserved as resource failures: the
ordinary full microbatch exceeded the observed 4-GiB pinned allocation even
with an 8-GiB request. The bounded adapter then completed source smoke
`runs/ms-source-smoke-003` (6 updates, 20.55 seconds) with finite losses and a
strict 64,745,066-value checkpoint audit (`seen=12`, SHA-256
`1c2c182818619275d6b0bea7d58784b7d2df8e06138558ee0a074d06be8c2409`).

Using that checkpoint, the same adapter completed a labeled lighting N=100
fine-tune smoke in `runs/ms-finetune-smoke-001` (6 updates, 19.44 seconds;
checkpoint SHA-256
`82089513068f45306f4eb2de96a47b6a44cdaa5ae82738a6250ee5e178a8a2c1`). Both
runs record `holdout_access=false`, `adaptation=false`, source-checkpoint
provenance where applicable, and finite-log checks. The official
`yolov4.conv.137` asset is now present and verified (170,038,676 bytes,
SHA-256 `11c5286cfbffabc9a3059927401ab3480e9483ceafd3265779a0dc26c769bed0`).
The default-rate official source smoke (`ms-source-smoke-004`) stays finite for
six updates but its 120-update continuation reaches NaN loss at update 28 and
fails the strict checkpoint audit; its six-update exporter also rejects
non-finite detections. A lower-rate diagnostic probe (1e-5) makes the six-update
source export finite and complete (`ms-source-common-compound-003`), but the
labeled N=100 fine-tune from that checkpoint still produces non-finite
validation detections (`ms-finetune-common-compound-003`). These results close
the supervised label-path and asset-loading checks only; the learning-rate and
resource schedule, stable source/fine-tune training, exact restart and
selection eligibility remain open.

The follow-up source diagnostic `runs/ms-source-diagnostic-1e-5-120b` completed
120 official-weight updates at `lr=1e-5` with pinned-memory 4096 in 291.43s;
its finite log and strict payload audit pass. This does not establish a stable
campaign rate: the same-rate labeled fine-tune export remains non-finite, and
the lower rate is intentionally not approved for the registry.

The canonical exporter now accepts these audited supervised checkpoints as a
separate diagnostic phase. The random-initialized smokes
`runs/ms-source-common-compound-001` and
`runs/ms-finetune-common-compound-001` each cover all 286 compound-validation
images and pass the official COCO evaluator (`selection_evidence=false`). The
lower-rate official source export `runs/ms-source-common-compound-003` also
passes complete coverage (126 predictions, diagnostic 0.0 AP). All such output
is execution evidence only and must not be used to rank candidates; the
fine-tune export failure remains preserved separately.

### Lower-rate stability follow-up

An official-weight source probe at `lr=1e-6` completed 120 native updates in
`runs/ms-source-diagnostic-1e-6-120` (finite trace, strict 64,745,066-value
payload audit, 289.30s). Its labeled lighting N=100 six-update continuation
`runs/ms-finetune-diagnostic-1e-6-006` also passed the finite audit. The full
compound validation export `runs/ms-finetune-common-compound-1e-6-001` covered
all 286 images in 2,814.68s, produced 28,600 finite predictions and passed the
shared COCO evaluator (`common_metrics.json`). The diagnostic AP50:95 is about
0.001 and is explicitly excluded from selection. This closes the bounded MS
technical runtime tuple, not the approved campaign rate: default and 1e-5
rates remain failed evidence, and 1e-6 still needs a reviewed resource/cost
schedule and longer supervised stability check.

The independent seed-43 repeat `runs/ms-source-diagnostic-1e-6-120-seed43`
failed before the first update while allocating the 4096-MB pinned arena
(`CUDA invalid argument`, no checkpoint). This is preserved as host-resource
reliability evidence, not as a learning-rate or model failure; the successful
seed-42 tuple remains diagnostic and unapproved.

The first infrastructure retry, `runs/ms-source-diagnostic-1e-6-120-seed43-retry1`,
failed at the same pre-update pinned-arena allocation with no checkpoint. Under
the declared retry policy this confirms a repeatable host-resource blocker for
that configuration; no learning-rate conclusion is drawn from either failure.
The second and final permitted retry, `runs/ms-source-diagnostic-1e-6-120-seed43-retry2`,
failed identically before update 1. The configuration therefore has three
pre-update allocation failures (original plus two retries) against one
successful seed-42 run; the MS resource gate remains open.

## Canonical validation export — 2026-09-25

The user approved a bounded validation-only exporter. `export_ms_validation.py`
uses the pinned upstream inference path and an opt-in `eval=research` printer;
it does not read labels or change the detector. The printer emits head indices,
scores and original-image xyxy coordinates for every image, including empty
images. The wrapper enforces the canonical class mapping, full official-val
membership, finite/clipped xywh predictions, maxDets=100, hashes and a strict
checkpoint payload audit. It never emits a completed manifest for a failed or
partial process.

Run `runs/ms-common-compound-001` timed out at 279/286 after 1800 seconds and
is preserved as failure evidence. Retry `runs/ms-common-compound-002` completed
all 286 images in 1,549.98 seconds, and the shared COCO evaluator accepted its
coverage. Its metrics remain diagnostic (`selection_evidence=false`), because
the checkpoint is random/time-seeded smoke evidence rather than a campaign
source or adaptation model.

## Reliability caveat from resource resampling

After successful run008, identical mode2 six-step resource run
`ms-da-memory-smoke-002` failed at layer159 extending pinned block6 with CUDA
invalid argument (6.23s, no update/checkpoint). Thus pinned allocation remains
host-state dependent; prior success is not proof of repeatable launch-readiness.
Its sampled RSS maximum8,928,008KiB is a failed-construction observation, NOT a
training peak. Whole-GPU samples include other applications. Raw samples are in
`sampled_resources.json`. No driver/OS/model changes were made. Initial sampler
attempt001 failed before training due to a ps child-discovery race; no live
process/checkpoint remained and its stale status must not be treated as live.

## Target-truth boundary regression

`tests/test_ms_native_regressions.py` now compiles actual upstream
`da_train_network_waitkey`, `get_next_batch` and `get_next_batch_target` against
the real Darknet types, replacing only the downstream train/update calls with
boundary assertions. With two source and two target examples, it verifies two
microbatches containing source pixels first and target pixels second, only
source truth, one update and one iteration increment. Target truth storage is
NULL. A deliberately mutated copy that reads target truth fails, demonstrating
that the test detects that leak. This is runtime evidence for batch assembly,
not a full GPU gradient-isolation proof or an annotation-filesystem access audit.
Existing source-only head and target-loader guards remain separately tested.

## 120-update checkpoint and reload — 2026-09-25

`runs/ms-da-smoke-008` completed all 120 native DA updates, exit0, 458.05s wall.
All logged losses were finite and no missing-label warnings occurred. Excluding
the first20 updates, the 100 logged update durations averaged 3.7384257s. These
are upstream loop timings, not CUDA-event timings; peak GPU/host memory was not
sampled and remains unmeasured. The random/time-seeded initialization remains
diagnostic only.

Saved weights: header (0,2,5,seen480), 64,745,066 finite FP32 values, SHA-256
`1871fb77f601c6305db72c1bdc682fcbc21140f13fdd140557c70236e462f047`.
Upstream CPU `darknet partial CFG WEIGHTS NEW_OUTPUT 9999` loaded all178 layers
and saved them again; all258,980,264 payload bytes are identical. That command
intentionally resets seen to0; original weights remain unchanged. This proves
weight serialization/reload only, NOT training restart: the saved format lacks
SGD momentum and RNG/sampler state. Exact resume remains a campaign gate.

Exporter audit: upstream BDD output hardcodes an alphabetical class order that
differs from canonical research IDs and closes its file twice; its NMS is .45.
Do not use raw BDD output as common-evaluator input. A canonical exporter must
map actual head indices, preserve original coordinates/64-bit image IDs, use
declared postprocessing and record full coverage including empty detections.

## First successful native DA updates — 2026-09-25

Two compiled-C regressions first failed on the actual upstream logic: generic
source image paths mapped beside images, and a second pinned allocation
overwrote the first. Minimal fixes enable generic images-to-labels mapping and
advance the new pinned block cursor by allocation_size. Both tests now pass;
the base suite has 42 tests, 9 dependency skips. CUDA allocation itself is stubbed
only in the CPU interval test; full GPU behavior is checked separately.

Rebuilt `patches/ms-dayolo-label-free.patch` and native mode2 run
`runs/ms-da-smoke-007` completed six DA updates, with no missing-label warning or
cuDNN error, then saved `backup/smoke_final.weights`. Wall time 30.39 seconds;
logged updates 3.78–3.99 seconds each. All 64,745,066 FP32 payload values were
finite; header (0,2,5,seen=24). SHA-256:
`7b8fac4a0ea5df767076df1244d92871a6cce3c50866cf01f8fd8d3b161c3d5b`.
The two fixes together remove the observed failures; a separate GPU ablation
has not isolated which change eliminated the cuDNN error. No architecture,
resolution, DA-head or system-driver changes were made. This is random-init
technical evidence, NOT checkpoint reload/resume or selection eligibility.

120-update follow-up `runs/ms-da-smoke-008` has been launched with the same
geometry/mode, preserving all source/target branches. Verify its process/status
before further GPU work. Native source/FT adapter smokes now pass; full source
pretraining, label-loss policy audit, exact restart, seeding, coordinate export
and common validation for supervised checkpoints remain outstanding.

## 2026-09-25 revalidation: construction now passes, new runtime failures

Independent CUDA11.8 ctypes probes now allocate/free ten 512 MiB host blocks
with flags0 and flags2, and eight 1 GiB mapped blocks. The historical fifth-block
failure is therefore not a stable 4 GiB host limit. No system/driver setting was
changed by this investigation; its original cause remains unknown. Current
process memlock limit is 8,397,582,336 bytes; cgroup memory.max is unlimited.

`runs/ms-da-smoke-006` reran unchanged mode2/native geometry and constructed the
complete graph, then failed before any update/checkpoint: cuDNN convolution
forward at `convolutional_kernels.cu:532` returned CUDNN_STATUS_NOT_SUPPORTED.
Exit -6, 7.93 seconds. This supersedes construction failure as the latest
reproducible failure, but does not establish native training feasibility.

Two additional code findings must be resolved before trusting another run:

- Source labels were not loaded: upstream `replace_image_to_label` has generic
  `/images/` to `/labels/` replacement commented out. The adapter's
  `source/images/*.jpg` therefore maps to `source/images/*.txt`, not the supplied
  `source/labels/*.txt`. Use an upstream-supported layout or explicit mapping
  and test actual annotation loading; never accept missing-label warnings.
- The pinned allocator's new-block branch sets `pinned_index=0`, returns the
  beginning of the block, but never advances the cursor for that allocation.
  The next small allocation can overlap it. This is a source-level defect
  requiring an allocation-interval regression before a fix. It is NOT yet proven
  to cause the cuDNN status error; investigate these independently.

Next diagnostic: source-path regression, pinned allocation overlap regression,
then instrument convolution input/output pointer/descriptor/algorithm details.
Do not substitute reduced geometry or remove DA heads to manufacture success.

2026-09-23: workspace CUDA 11.8/cuDNN8/OpenCV4.10 build succeeds.
`build_ms_dayolo.py` preserves the pinned upstream revision and compiles the
traceable `patches/ms-dayolo-label-free.patch`. Binary startup and dynamic
library resolution pass; these are not training evidence.

`run_ms_dayolo_smoke.py` prepares 128 deterministic source and lighting-target
train images in separate symlink views, source labels only, three ten-class
YOLO heads (45 projection channels), native-content letterboxing to 1280x736,
and a bounded DA command. Microbatch is one source + one target, accumulated
twice per update. Technical initialization is random and upstream seeding is
time-based; neither this initialization nor these runs may enter selection.
Geometry conversion still needs a pixel-level loader test. No validation or
holdout is used by this command. Dry-run writes nothing.

Evidence retained under ignored runs:

- `ms-da-smoke-005` (2026-09-24): supported memory mode 1 also fails with CUDA
  OOM constructing layer 144, exit -6 after 10.15 seconds, no update/checkpoint.
  Its delta-buffer sharing occurs after network construction and does not solve
  this construction peak. Native geometry and one-source/one-target microbatch
  were preserved. This is additional host-feasibility evidence. The later
  user decision on 2026-09-28 excludes MS-DAYOLO from current registry v9 based on the
  complete repeatable resource record, not accuracy.

- `ms-da-smoke-001`: default memory mode, CUDA OOM constructing layer 143;
  exit -6 after 3.22 seconds, no update/checkpoint. Its initial status format
  says failed; log establishes OOM.
- `ms-da-smoke-002`: upstream memory mode 2, fails preallocating pinned host
  memory; no update/checkpoint.
- `ms-da-smoke-003` / `004`: initial pinned reserve reduced to 4 GiB; line-buffered
  run 004 shows failure expanding beyond four 1 GiB blocks at layer 15.
  Independent ctypes CUDA runtime test also allocated four 1 GiB mapped host
  blocks successfully and returned invalid argument for the fifth, then freed
  all successful allocations. Root OS/driver cause is not established.

The saved patch now makes the initial reserve configurable (`pinned_memory_MB`,
default upstream 8192; smoke uses 4096), and has been rebuilt successfully.
It does not remove the observed total pinned-memory limit. No driver or OS
configuration was changed. Candidate remains feasibility-unresolved on this
host, not scientifically rejected. Do not lower resolution, remove DA heads,
or silently claim a detector-only smoke as DA success. Further supported memory
options/hardware review are needed before admission. Source/FT, full DA
benchmark, deterministic seeding, checkpoint/restart and common validation
export remain unverified.

Reproduction (new output directory required):

```bash
python build_ms_dayolo.py
python run_ms_dayolo_smoke.py --output runs/ms-da-next --steps 6 --optimized-memory 2 --dry-run
```
