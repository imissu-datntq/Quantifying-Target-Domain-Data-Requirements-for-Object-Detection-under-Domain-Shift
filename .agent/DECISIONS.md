# Decisions

2026-09-28 1,000-image pilot supersedes the earlier 128/64-image screen for
engineering evidence. Use exactly 1,000 source and 1,000 target training images
per candidate and the same lexicographic 1,000-image lighting-validation
prefix. R101-C4 completed 1,000 source plus 1,000 adaptation iterations.
SSDA-YOLO and SF-YOLO each completed 1,015 traced batches and exactly 151
optimizer/EMA updates; 120 updates was rejected because it stopped after only
503 batches. Common COCO results in AP points were SSDA 2.229456/6.614537,
SF 1.856307/4.760677 and R101-C4 0.686985/1.936026 for mAP50:95/mAP50.
Preserve the contrary numeric evidence, but keep **Adaptive Teacher Faster
R-CNN R101-C4** as the project model because the pilot schedules are not
accuracy-comparable and R101-C4 is the architecture fixed by the proposal.
No held-out test data was read.

2026-09-28 bounded-pilot confirmation: run all three remaining candidates on
the same deterministic 64-image lighting-validation sample after bounded
training (at most 128 source/target images and approximately 120 updates).
SSDA is the numeric pilot leader (mAP50:95 0.680424), followed by SF (0.053487)
and R101-C4 (0.002170). Record this contrary evidence, but do not call it an
accuracy winner: every score is below one point and upstream initialization and
effective schedules differ. Select Adaptive Teacher Faster R-CNN R101-C4 for
the project because it matches the proposal, completed the corrected
source-to-teacher/student path at 1.444501 seconds/update, and avoids CUT and
TargetAugment production dependencies. `runs/model-selection-decision-v1.json`
and `runs/small-sample-model-selection-v1.json` are authoritative.

2026-09-28 scope cleanup: remove the retired 60-arm registry, Stage-0/Stage-1
planners/executors, production auxiliary and full YOLO launchers, their tests,
and generated campaign artifacts. Keep only bounded smoke/integration code.
The removed generated artifacts could not be moved to the volume trash and
were deleted in place at the user's request.

2026-09-28 earlier direct model choice (now confirmed by bounded feasibility
pilot): select Adaptive Teacher Faster R-CNN R101-C4 for the project. This is a
protocol-alignment and feasibility decision, not an accuracy-winner claim.
MS remains excluded by the earlier user decision.

2026-09-28 selection campaign cancellation: stop the partial TargetAugment
weather job, cancel its compound supervisor, and leave every CUT output absent.
Lighting TargetAugment remains diagnostic non-selection evidence. The campaign
goal and its 90-day CUT projection are no longer active project requirements.

2026-09-28 historical CUT schedule hold: the full upstream six-job default extrapolates to
2,163.7 GPU-hours (90.2 GPU-days) on the single RTX 3060. The user's question
shows that this duration requires a renewed, explicit schedule choice. Continue
the short TargetAugment jobs and allow the bounded YOLO update-limit probe, but
do not launch CUT while `runs/campaign-aux-v1/cut-launch-control.json` has
`cut_training_authorized=false`. Choosing a staged schedule would require a new
immutable plan/policy version; do not silently shorten v4.

2026-09-28 protocol correction, registry v9: the proposal explicitly places an
unsupervised method on the labeled-budget axis as one constant reference. SSDA,
SF and Adaptive Teacher campaign adapters consume the full unlabeled target pool
and no N=100/1000 labeled subset. Preserve v8 as historical but never launch its
18 duplicated UDA rows. V9 has 60 arms: 6 source, 36 supervised fine-tuning and
18 UDA N=0 references. `campaign-stage1-plan-v2.json` covers all 60 training
commands and 132 run-bound validation jobs. The corrected detector forecast is
195.562 GPU-hours. After the first production TargetAugment task, the
end-to-end rate refresh raises the total optimization-only forecast to
2,364.397 GPU-hours (`campaign-cost-v3.json`).

2026-09-28 staged-launch policy (v3 supersedes v2): approve the 2,364.397 GPU-hour
optimization-only forecast after completing the first-production-task rate refresh
and a hard re-review threshold at 3,000 GPU-hours. Because the pinned R101 stack
is non-identical even in a same-segment control, do not claim or use bitwise
resume. Retain any partial attempt as non-selection evidence and restart the
complete arm from the same immutable initialization, seed and inputs in a new
attempt directory; allow at most two retries. Stage 0 contains six CUT training,
three TargetAugment training and six native CUT generation jobs and must pass provenance/semantic/geometry QA
before detector arms unlock.

2026-09-28 Stage-0 execution: TargetAugment lighting completed 160,000 updates
in 6,161.742 seconds. Its decoder/fc1/fc2 tensors are finite and the production
audit records no target-annotation access. Continue serially with weather, then
compound; all CUT tasks and final cross-asset audit remain mandatory.

2026-09-28 MS-DAYOLO exclusion: the user explicitly removed MS-DAYOLO from the
model-selection campaign after its seed-43 official-weight source launch and
both permitted retries failed before update 1 during pinned-memory allocation.
Preserve all MS technical evidence. Registry v8 first removed MS; registry v9
retains only
SSDA-YOLO, SF-YOLO and Adaptive Teacher R101-C4. V8 also fixes v6's unintended
source-domain fine-tune/adaptation rows and corrects duplicated UDA budgets; the
current grid is 60 arms. Do not rewrite or launch superseded v6/v7/v8.

2026-09-28 historical v8 cost forecast: apply 1,200 updates to all 78 registry-v8
rows in `campaign_cost_report.py`. Use measured SSDA/R101/adaptation rates,
SSDA source as its fine-tune proxy, and the slower SF adaptation rate as a
conservative SF source/fine-tune proxy. Detector subtotal is 281.924 GPU-hours;
CUT plus three 160k-step TargetAugment defaults yields 2,448.722 provisional
optimization-only GPU-hours. Registry v9 and cost report v2 supersede these
figures as recorded above.

2026-09-25 campaign schedule: draft `docs/campaign_schedule.md` as the
machine-checkable review boundary for the then-current 136-arm v6 grid. V8
supersedes that grid with 78 corrected eligible arms. The schedule fixes the
1,200-update selection-pilot target, 100-update validation/checkpoint cadence,
20-update throughput warm-up, seed 42/43 usage, two infrastructure retries and
immediate-stop conditions. Candidate-specific native geometry and microbatch
settings remain explicit; the MS 1280x736 setting is now historical exclusion
evidence. This is a review draft only; production auxiliary preparation, exact
restart and reviewer sign-off keep `ready=false`. Do not launch long training
from the superseded v6 schedule.

2026-09-25 MS diagnostic stability: official `yolov4.conv.137` source training
at `lr=1e-5`, native 1280x736, batch2/subdiv2, pinned-memory 4096 and 120
updates completed in `runs/ms-source-diagnostic-1e-5-120b` (291.43s). The
strict 64,745,066-value checkpoint audit passed and all logged losses were
finite. An earlier 8-GiB pinned-memory invocation failed before iteration and
is preserved separately. This lower-rate source result is diagnostic only:
the labeled fine-tune at the same rate still has a non-finite common export,
so no alternate rate is approved for the campaign.

2026-09-25 R101 restart boundary: add `rng_sampler_state.pth` sidecars to the
Adaptive Teacher smoke adapter. They capture Python/NumPy/Torch streams and
replay the single-worker loader cursor before restoring them. A bounded
12-update uninterrupted run versus 6+6 continuation restored the sidecar but
still produced different model/optimizer tensors, so CUDA/ROI nondeterminism
remains unresolved. Keep `exact_rng_sampler_resume=false`; state capture is
useful plumbing, not an exact-restart pass.

2026-09-25 MS lower-rate extension: official source at `lr=1e-6` completed
120 native 1280x736 updates with finite loss and strict payload audit
(`runs/ms-source-diagnostic-1e-6-120`). A labeled lighting N=100 continuation
completed six finite updates and its 286-image compound export plus shared
COCO evaluation passed (`runs/ms-finetune-common-compound-1e-6-001`). The
diagnostic AP is not selection evidence. This closes the MS technical runtime
tuple only; 1e-6 remains unapproved until stable resource/cost review and a
longer supervised schedule are frozen.

2026-09-25 MS resource repeat: an independent seed-43 official-weight
`lr=1e-6`, pinned-4096 source launch failed before update 1 while allocating
the pinned arena (`CUDA invalid argument`) in
`runs/ms-source-diagnostic-1e-6-120-seed43`. Preserve it as host-state
reliability evidence, not a learning-rate failure; the successful seed-42
tuple remains diagnostic and the stable resource envelope is still open.

2026-09-25 MS resource retry: the first permitted retry with the identical
seed-43/`lr=1e-6`/4096-MB configuration also failed before update 1 during
pinned-arena allocation (`runs/ms-source-diagnostic-1e-6-120-seed43-retry1`).
Treat the pair as repeatable host-resource failure evidence, not as a rate
failure; preserve the declared schedule and keep the resource gate open.

2026-09-25 readiness review packet: consolidate verified evidence and the then
three remaining decisions in `docs/campaign_review_packet.md`. The 2026-09-28
MS exclusion resolves the MS decision; production auxiliary assets and
exact-restart policy remain. Keep the packet explicitly non-authorizing.

The second and final permitted retry, `runs/ms-source-diagnostic-1e-6-120-seed43-retry2`,
also failed before update 1 at the pinned arena. Record three identical
pre-update seed-43 allocation failures and one successful seed-42 run; do not
alter the learning rate to hide the host-resource blocker.

2026-09-25 R101 deterministic retry: rerunning the bounded uninterrupted versus
6+6 comparison with `--deterministic` still produced 96 model/optimizer tensor
 mismatches. Keep exact restart false; cuDNN/ROI nondeterminism remains a pinned
 stack limitation even after process RNG and sampler state restoration.

2026-09-25 R101 cuBLAS retry: set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and disabled
TF32 in deterministic mode. The same-segment control already differed in
200/986 model and 103/105 optimizer tensors; the uninterrupted-versus-resumed
comparison differed in 204/986 model and 104/105 optimizer tensors. Preserve
`runs/r101-cublas-restart-audit-001/report.json`; keep exact restart false and
do not weaken the gate to an allclose threshold.

2026-09-25 MS supervised path: add a separate ordinary Darknet adapter with no
`-da` switch for source and labeled target fine-tuning. The native 1280x736
smokes use batch 2/subdivisions 2 because the full supervised microbatch
exceeded the observed 4-GiB pinned allocation; failed 4-GiB/8-GiB source
attempts remain preserved. `ms-source-smoke-003` and its lighting N=100
`ms-finetune-smoke-001` continuation both complete six finite updates and
pass the strict 64,745,066-value checkpoint audit. The source checkpoint is
randomly initialized while the optional official MS pretraining asset is
missing, so this closes only the supervised loader/adapter technical gate;
source quality, repeatable memory feasibility, exact restart and campaign
eligibility remain open. The official `yolov4.conv.137` asset is now verified;
the default 1e-4 source path reaches NaN at update 28 in a 120-update run.
A 1e-5 diagnostic source probe exports finite predictions, but its labeled
N=100 fine-tune still fails the non-finite detection guard. Do not silently
adopt 1e-5 as the campaign schedule; it requires a predeclared stability and
resource review.

2026-09-25 MS supervised export: allow the canonical MS validation exporter to
consume a completed supervised smoke only when its finite payload audit passed
and `adaptation=false`; retain the existing DA path unchanged. Source and
lighting-N=100 fine-tune checkpoints now each cover all 286 compound-val images
and pass official COCO coverage/evaluation with `selection_evidence=false`.
Their zero-prediction/0.0-AP diagnostic output is not a ranking signal.

2026-09-25 SF restart boundary: preserve the upstream checkpoint format and
verify bounded continuation of model, optimizer, scheduler/scaler and
teacher/EMA state before campaign scheduling. The one-epoch continuation from
`runs/sf-adapt-120-001` passed with a trace starting at batch 7,680 and ending
at 141 EMA updates. Mark this as `checkpoint_resume_verified`, not
`resume_verified`: RNG/sampler state is absent, so exact restart remains a
predeclared campaign gate and no smoke metrics enter selection.

2026-09-25 SSDA update-count boundary: keep the pinned upstream warm-up and
gradient-accumulation schedule unchanged. At native batch 1 it needs roughly
36 epochs to exceed 120 optimizer updates; shorter 4- and 9-epoch pilots are
immutable incomplete evidence (56 and 68 updates), not failed algorithm runs
and not permission to alter accumulation. Run-003 is the bounded native
benchmark; only its audited output can close the technical update gate.

2026-09-25 SSDA adaptation checkpoint: run-003's native schedule is the
accepted technical throughput evidence (122 updates, not a shortened pilot).
Use its serialized EMA for common validation integration, but preserve
`selection_evidence=false`. The bounded restart proves model/optimizer/EMA
continuation only; exact RNG/sampler restoration remains a campaign gate.

2026-09-25: User approved the bounded MS canonical exporter design. Keep native
upstream inference; opt-in eval=research writes head indices and original-image
xyxy plus an explicit record for every validation image. Python maps canonical
yolo_id to coco_id, clips boxes, retains top100 after .001 confidence / .6 NMS,
and rejects incomplete coverage. Do not use upstream BDD's hard-coded class
order or its double-close path. No holdout access or architecture change; this
is diagnostic integration, not candidate ranking or approval of long training.

2026-09-25 campaign registry (superseded by v7): freeze a 136-arm planned grid covering all four
pipelines, source baselines, fine-tuning and adaptation across source/lighting/
weather/compound, N=100/1000 and seeds42/43. Derive run IDs from protocol,
candidate, phase, domain, budget, seed and upstream SHA; write exclusively and
block holdout access. Keep missing throughput/auxiliary rates explicit instead
of fabricating a completed cost estimate. Registry creation does not launch
training.

Registry v8 supersedes this decision after the 2026-09-28 MS exclusion and
removes the unintended source-domain fine-tune/adaptation rows.

2026-09-25 MS correctness fixes: enable generic image-to-label path conversion
for separated source label views, and advance the pinned arena cursor after a
new block's first allocation. Both changes have RED-to-GREEN compiled-C tests.
They preserve the detector, native geometry and DA heads. Six native DA updates
now complete, superseding earlier construction/first-forward failures as current
runtime evidence. Do not infer exact resume or selection eligibility from this.

2026-09-24 nonempty-pseudo regression: diagnostic SSDA/SF runs may lower teacher
confidence to .001 to exercise teacher-generated positive boxes without reading
target labels. This is a branch-coverage test, not a tuning result or campaign
threshold choice. Preserve the default-threshold empty-box evidence separately.
SF's image-only adapter also fixes per-microbatch student zero_grad so intended
gradient accumulation is retained; earlier SF smoke 001 is pre-fix evidence.

2026-09-24 SSDA technical adapter: retain all four source-real/source-fake/
target-fake teacher/target-real student branches, including consistency loss.
In opt-in image-only mode, fix upstream per-microbatch student zero_grad that
discarded accumulated gradients; propagate backward failures and reject
non-finite loss. Preserve optimizer checkpoints instead of stripping them,
but do not claim exact resume (scheduler/scaler/RNG remain outstanding).
Disable implicit package installation, and normalize paired source path groups
before iteration; no broad filesystem discovery is intended. Diagnostic CUT
pairs require canonical train membership plus input/output hash verification;
they are not production translations or selection evidence.

## 2026-09-22

- Historical scope at this checkpoint included SSDA-YOLO, MS-DAYOLO, SF-YOLO
  and Faster R-CNN. The 2026-09-28 decision above supersedes it: MS-DAYOLO is
  excluded, while SSDA-YOLO, SF-YOLO and Faster R-CNN R101-C4 remain eligible.
  No research winner is locked.
- The user chose creation of a new split version with seed 20260916 and proposal counts, rather than claiming reconstruction of an unavailable old manifest. Keep raw data intact and official validation separate.
- Select candidates using validation, with explicit label/source-data access, corresponding detector baselines and measured resource/auxiliary costs. Do not select on final holdout scores.
- The earlier planning request did not authorize execution. The subsequent request to implement the current plan authorized the deterministic split and model-input milestones. It does not by itself open held-out labels to evaluation or launch the full research campaign; those remain explicit gates in the active plan.
- Static inspection rejects the prototype's SF-YOLO synthetic-box workaround: upstream reads target label metadata for loader validation, augmentation, auto-anchor/class weights and callbacks. SF remains blocked until a tested label-free target loader exists.
- Adaptive Teacher at pinned revision `5256463a...` does not establish off-the-shelf Faster R-CNN R50-FPN compatibility. Its supplied ResNet adaptation configs are R101-C4 and its driving-weather config is VGG16-C4. Architecture substitution or an R50-FPN port requires user review before environment/pilot commitment.
- Do not cap weather/compound confirmation at two finalists. The user accepts the additional training time: every candidate that passes the predeclared technical, label-policy, reproducibility and resource-feasibility gates proceeds through lighting, weather and compound confirmation before one research pipeline is selected. Cost remains measured and reported, but it is not an automatic early-pruning rule.
- The user selected Faster R-CNN R101-C4 with Adaptive Teacher at revision `5256463ad9ec90fd5ba84ebb8d53bed56bd369df`, superseding the R50-FPN candidate assumption and the architecture-review gate above. Use the same detector backbone and ROI head for source-only, supervised fine-tuning and adaptation comparisons. This selects one candidate branch, not the final research winner. BDD100K registration, train-only target access and native-resolution single-GPU feasibility on RTX 3060 12 GB still require verification.
- Proactively update relevant Markdown files at decisions and checkpoints, keeping README, decisions, handoff and the active plan consistent with verified execution state.

## 2026-09-23

- Source-only/fine-tuning use upstream `TwoStagePseudoLabGeneralizedRCNN` with the same R101-C4 backbone and two-FC ROI head as adaptation. The DA meta-architecture adds a source discriminator loss even on its supervised branch; diagnostic source smoke 002 is excluded as a valid baseline. Accepted source smoke is 003.
- Technical smokes demonstrate execution only. Smoke AP and shortened burn-in must not enter model ranking. Resume currently restores model/optimizer/scheduler/iteration but not RNG/sampler state; exact reproducibility remains a campaign gate.
# SF-YOLO label-free integration checkpoint — 2026-09-23

2026-09-25 auxiliary integrity checkpoint: retain the 128-image-per-direction
CUT exports and 120-step TargetAugment tensors as train-only diagnostic inputs
only. `audit_auxiliary_assets.py` passes hash, native RGB geometry, finite-state,
and provenance checks; it does not promote these assets to production or close
semantic-quality, optimizer/RNG-resume, or reviewer-approval gates.

2026-09-24 style-mode clarification: use SF's supported empty-`style_path`
default for the BDD technical adapter. It samples style from the current target
batch (self-style at batch 1); no fixed foreign meanfoggy image is needed.
TargetAugment content/style training uses only target-train images. Record this
choice and batch dependence in the eventual frozen schedule; native style-only
forward works, but full SF adaptation eligibility is still unproven.

MS-DAYOLO follow-up: use an explicit image-only target loader rather than dummy
target annotations. Disable target box-aware blur/mosaic, preserve source
supervision, and retain upstream source-only detection heads. The patch fixes
the parser's dependence on the previous layer batch when the DA microbatch is
2. Static regression coverage exists; compiled runtime evidence is still required.

The research adapter uses an explicit image-only train manifest, preserves
1280x720 pixels with stride padding, and never discovers target annotation files.
Target-dependent auto-anchor, image weighting, multiscale and plotting options
are rejected in this mode; class weights come from the source checkpoint.
Empty teacher output stays empty rather than receiving a fabricated random box.
This changes target data augmentation relative to the upstream labeled loader
and must be reported in the experiment configuration. Loader and upstream loss
unit checks pass, but full TargetAugment/teacher/student feasibility remains open.
Automatic dependency installation is disabled; dependencies belong in the
candidate-specific environment, never base Python. Patch provenance is retained
in `patches/sf-yolo-label-free.patch`.
