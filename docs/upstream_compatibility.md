# Candidate pipeline compatibility audit

Updated 2026-09-23. This audit pins the exact source revisions inspected for
the model-selection readiness plan. It records feasibility evidence only; it
does not select a pipeline or report held-out performance.

Runtime update: R101-C4 source/adaptation/fine-tuning checks are recorded in
`adaptive_teacher_execution.md`. AT, SSDA and SF isolated environments exist;
SSDA would need translated assets for any later full run. SF image-only
loading, full TargetAugment/student/teacher execution and empty-pseudo-label
loss/backward tests pass with a saved adapter patch; exact restart and full
auxiliary preparation are outside the bounded pilot. MS has an explicit
unlabeled-loader patch and a batch-2 parser fix with static tests. Its CUDA 11.8
build and binary startup/library resolution passed; a 120-update finite DA graph
and strict weight-payload reload passed, while repeatable pinned-memory, exact
resume and target-loss/resource gates remain open. The original audit table
below describes initial upstream gates, not a claim that these runtime checks
have all passed. No candidate has been selected by validation accuracy.

The bounded MS validation exporter is now implemented separately from the old
BDD printer: it keeps upstream inference, maps canonical head indices, records
empty images and rejects incomplete coverage/truncated checkpoints. DA and
supervised diagnostic exports now cover the full compound validation subset;
no MS metric is selection evidence.

| Candidate | Pinned revision | Verified upstream shape | Current gate |
| --- | --- | --- | --- |
| SSDA-YOLO | `3ab1f506ded2ca67233699a2e9bb8944aea40013` | YOLOv5 v5.0, paired real/translated source and target loaders | Requires translated train-only assets, compatible PyTorch environment, weights, and single-GPU smoke |
| MS-DAYOLO | `c655ceed3b99e6b499a9c907a3ae9c2deba98c1e` | Darknet YOLOv4 with a separate target list and `-da` training path | Requires a CUDA build, adapted ten-class cfg/data files, weights, and proof that empty target annotations do not enter detection loss |
| SF-YOLO | `c84dfad79a889b5f172d7192ee0379edfd738835` | YOLOv5 plus TargetAugment and teacher pseudo-label training | Blocked on a real unlabeled-loader integration; synthetic target boxes are invalid |
| Adaptive Teacher | `5256463ad9ec90fd5ba84ebb8d53bed56bd369df` | Detectron2 0.3-era R101-C4, selected by the user | Requires ten-class BDD100K integration, train-only target checks and 12 GB single-GPU smoke |

## Findings that affect the research design

Adaptive Teacher cannot currently establish the proposal's claimed
Faster R-CNN R50-FPN adaptation compatibility. Its checked-in ResNet configs
inherit `Base-RCNN-C4.yaml`, set depth 101, and attach the discriminator to
`res4`; the Cityscapes-to-Foggy configuration uses VGG16 without FPN. The
README reports multi-GPU tested environments (8 V100 or 4 2080 Ti) and does
not establish a 12 GB single-GPU schedule. On 2026-09-22 the user selected
R101-C4 + Adaptive Teacher, resolving the architecture change. The supplied
ResNet configuration uses `StandardROIHeadsPseudoLab` with a two-FC
`FastRCNNConvFCHead`; preserve that detector head across source-only,
fine-tuning and adaptation. Admission as a complete RQ1-RQ5 pipeline still
requires BDD100K integration, train-only target checks and a measured native
1280x720 single-GPU smoke. No final research winner has been selected.

SF-YOLO's target loader is not label-agnostic. Before adaptation starts,
`train_sf-yolo.py` concatenates dataset labels, checks the maximum class,
runs auto-anchor, computes class weights, optionally computes image weights,
and passes loader targets to callbacks. Its YOLO loader also requires nonempty
labels when augmentation is enabled and uses them in mosaic and geometric
augmentation. Therefore the prototype's one-synthetic-box-per-class workaround
can influence training behavior and must not be used. A valid integration must
load target images and shapes without ground truth, disable label-dependent
sampling/augmentation/auto-anchor for that branch, derive class/anchor state
from the fixed source checkpoint or declared source metadata, and test that no
target ground truth is read.

SSDA-YOLO explicitly pairs each real source/target image with a translated
counterpart and derives label paths from the real image path. Its source side
is supervised; its target side still expects readable label-cache structure.
The smoke gate must prove empty target labels remain absent from supervised
losses while translated-image geometry remains aligned.

MS-DAYOLO exposes a separate `train_target` list and a dedicated
`da_train_network_waitkey` path. The upstream example distributes dummy target
annotations. The candidate is not admitted until a ten-class configuration is
built and a bounded instrumented smoke demonstrates that target annotations
are ignored by detection losses rather than merely required by the loader.

Run `python pipeline_doctor.py` for read-only local evidence. Missing isolated
environments, compiled binaries, pretrained weights, translation models, and
TargetAugment assets are deliberately reported rather than downloaded.
