# Common validation evaluation

SF pretrained-source smoke follow-up (2026-09-25):
`runs/sf-pretrained-source-common-001` covers all 286 compound-val images with
28,600 predictions and completed common COCO metrics. Checkpoint identity and
full coverage are recorded; these remain diagnostic, not ranking evidence.

2026-09-25 R101 integration: `export_r101_validation.py` strictly reloads the
chosen detector/teacher/student state, consumes a deterministic prefix of the
selected domain validation manifest and exports original-coordinate predictions with
canonical integer IDs. Native BGR pixels go through upstream internal stride
padding/postprocessing. Confidence .001, NMS .6 and maxDet100 are explicit.
`runs/r101-common-compound-001` (adaptation teacher) and
`runs/r101-source-common-compound-001` (source detector) each exported all 286
compound-val images/28,600 predictions and completed shared official COCO
evaluation. These diagnostic APs are NOT model-ranking evidence. Independent
box-coordinate regression for R101 remains required; MS integration is recorded
below.

`evaluate_validation.py` provides the shared official COCO bbox evaluator.
Run it with the Adaptive Teacher environment (pycocotools installed). It always
loads `coco/instances_val.json` and the selected domain's official validation
manifest; it has no train/test split argument. The default is the first 64 rows
after sorting by image name. This is bounded pilot evaluation, not proof that
any candidate has a final-quality checkpoint.

```bash
.venvs/adaptive-teacher/bin/python evaluate_validation.py \
  --domain lighting --max-images 64 --predictions RUN/predictions.json \
  --coverage RUN/inference_manifest.json --output RUN/common_metrics.json
```

Predictions use canonical COCO image/category IDs, original-image `xywh` boxes
and scores in [0, 1]. Exporters must undo padding/resizing before writing them.
The inference manifest must contain `status: "completed"`, `split: "val"`,
`evaluated_image_ids` for every image in the deterministic sample (including
zero detections), `prediction_sha256` and
`checkpoint_sha256`. Partial/duplicate/foreign IDs and mismatched prediction
hashes are rejected. The caller remains responsible for truthful inference
coverage and checkpoint provenance; adapter integration is still pending.

All ten canonical categories are retained. Report AP50:95 and AP50 on a 0–100
scale, maxDets=100, IoU=.50:.05:.95 and per-class AP. Classes without ground truth
are null and excluded from aggregate AP, following official COCO behavior.
Existing outputs are never overwritten. Synthetic tests cover perfect AP=100,
wrong/empty AP=0, absent classes, foreign IDs and incomplete inference coverage.
The bounded comparison is recorded in
`runs/small-sample-model-selection-v1.json`; it is not a final scientific result.

## YOLO exporter integration checkpoint

`export_yolo_validation.py` now supports the pinned SSDA source-smoke state-dict
format and SSDA/SF serialized detector/teacher checkpoints. `--model-key model`
(default) or `--model-key ema` explicitly selects the checkpoint component;
format/key are recorded, absent components and non-finite tensors are rejected.
Use each candidate's own
environment. It exports the bounded deterministic sample of a chosen official validation domain,
never train/test, with batch-1 FP32 native-content inference, explicit 8px top/
bottom padding removal, canonical class mapping, confidence .001, NMS IoU .6
and maxDets 100. Tests cover padding reversal, clipped/degenerate boxes and
64-bit canonical COCO IDs without rounding. Failed/incomplete inference records
cannot be evaluated; existing output directories/results are rejected.

`runs/ssda-common-compound-001` and `runs/sf-common-compound-001` each exported
286 compound-val images and 28,600 predictions from diagnostic checkpoints.
They are integration tests, explicitly `selection_evidence=false`, not a model
ranking. Their prediction/coverage hashes and common metric artifacts are kept
in those directories as historical integration evidence.

`runs/ssda-adapt-common-compound-001` also completed all 286 compound-val images
and official COCO evaluation using the adaptation smoke 002 student EMA
(28,598 predictions). This closes serialized adaptation-checkpoint plumbing,
not selection quality. It retains `selection_evidence=false`.

## MS-DAYOLO exporter checkpoint — 2026-09-25

`export_ms_validation.py` is now implemented for the pinned MS-DAYOLO binary.
Its opt-in upstream `eval=research` path writes one JSONL record per image,
including empty detections, with raw head class indices and post-`get_network_boxes`
original-image coordinates. The wrapper maps the ten head indices through
`metadata/classes.json`, clips/filter boxes, keeps top-100 after confidence .001
and NMS IoU .6, and rejects duplicate/partial/foreign coverage and non-finite
values. Integer COCO IDs remain integers. A strict binary audit rejects truncated
or non-finite Darknet checkpoints before inference because the upstream loader
otherwise tolerates short reads.

The first full compound attempt reached 279/286 records before its 1800-second
wrapper timeout and is retained as a failed run (`selection_evidence=false`). A
second run with a 3600-second timeout completed all 286 IDs in
`runs/ms-common-compound-002` (28,028 predictions, 1,549.98 seconds). The
official shared evaluator accepted its coverage and produced
`common_metrics.json` with `selection_evidence=false`; the tiny diagnostic AP
is not a model-selection result.
