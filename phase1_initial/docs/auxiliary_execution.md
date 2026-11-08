# Auxiliary preparation: evidence and unresolved cost

2026-09-28 bounded-pilot export: the verified lighting CUT generators exported
1,000 deterministic source A-to-B images and 1,000 lighting B-to-A images in
`runs/pilot1000-cut-source` and `runs/pilot1000-cut-target`. Native 1280x720
geometry, exact manifest membership and per-image hashes passed. These assets
were used only for the 1,000-image SSDA engineering pilot; they do not establish
semantic preservation or production translation quality.

2026-09-24 paired-export follow-up: reverse CUT smoke
`runs/cut-lighting-reverse-smoke-001` completed 120 updates, mean 0.2584403
seconds after warmup, finite native-size inference and strict network reload.
`export_cut_smoke_pairs.py` exported 128 source-train A-to-B and 128
lighting-train B-to-A images in `runs/cut-pairs-source-smoke-001` and
`runs/cut-pairs-target-smoke-001`. Per-image provenance records generator,
original-image and translated-image SHA-256. Native geometry is 1280x720;
semantic preservation and translation quality are NOT established. These are
diagnostic-only assets, not replacements for reviewed full CUT preparation.

Updated 2026-09-24. Train-only views are at `data/auxiliary_inputs_v1`, built
by `prepare_auxiliary_inputs.py` and validated against canonical train-domain
membership. No validation/holdout image is included. These views are inputs,
not completed translated-image or TargetAugment assets.

## CUT for SSDA

Pinned official CUT revision: `b3ac297708dfb6f7589d04662277e53c0d579c27`.
Isolated `.venvs/cut` cloned the compatible SSDA stack; additional upstream
requirements were installed there only. `environments/cut.freeze.txt`, pip
check and upstream `train.py --help` capture the environment checkpoint.

`run_cut_smoke.py` calls upstream CUT initialization/optimization/checkpoint
functions directly. `runs/cut-lighting-smoke-001` completed 120 A-to-B updates
on 128 source/lighting training examples, batch 1, seed 42, upstream 286-resize/
256-crop auxiliary training. All reported losses were finite; G/F/D weights
saved and reloaded. Mean optimization time over steps 20–119 was 0.2587897 s,
excluding image IO. Wall time: 47.49 s. Peak training allocated/reserved memory:
5,806,846,464 / 6,520,045,568 bytes. Native 1280x720 inference returned a finite
tensor of identical shape; total reserved memory including inference reached
8,445,231,104 bytes. Shape preservation alone does not prove semantic box
alignment or translation quality. Optimizer/scheduler/RNG resume is unverified.

The observed upstream defaults are **200 constant + 200 decay epochs**. At
batch 1, full pools and the measured A-to-B optimization rate, extrapolating
two directions per domain gives approximately:

| Domain | Training updates, both directions | Optimization-only GPU hours |
| --- | ---: | ---: |
| lighting | 14,412,000 | 1,036.0 |
| weather | 7,843,200 | 563.8 |
| compound | 7,843,200 | 563.8 |

Total ~2,163.7 GPU hours (90.2 GPU days), before data IO, image generation,
retries and detector training. This is a provisional linear extrapolation from
one-direction smoke, not a reviewed schedule, reservation or exclusion rule.
Do not silently replace CUT with FastCUT, shorten the schedule, or prune SSDA
because this default is expensive. A justified schedule and measured reverse
direction/inference cost must be reviewed before long preparation.

## TargetAugment for SF

VGG now downloaded and verified against exact upstream Git blob
`e52b7276b75c9b1b0e4332291c51f9e2fd25b353`; strict upstream VGG load passes.
SHA-256 and URL are in `environments/pretrained-assets.json`.

The upstream trainer completed six iterations with `--scenario_name bdd100k`,
batch 1, workers 0, content and style sampled from lighting-train image views,
and saved decoder/fc1/fc2 under `runs/targetaugment-lighting-smoke-001/models`.
All three state dicts loaded strictly and all tensors were finite. This first
execution lacks a finite-loss trace, resource benchmark and optimizer resume;
it is not the required complete auxiliary benchmark. It used upstream random
sampling, not a locked campaign seed. The experimental use of the target pool
as style inputs does not finalize the single style-reference policy used during
detector adaptation. No foreign meanfoggy image was substituted. No six-step
weight may be presented as a production-trained TargetAugment model.

Remaining: lock BDD style-reference derivation, instrument auxiliary metrics,
measure native detector+TargetAugment graph, validate target-label isolation,
and integrate trained artifact identities into the reviewed campaign registry.

### Instrumented follow-up, 2026-09-24

`patches/sf-targetaugment-instrumentation.patch` adds opt-in exclusive JSONL
loss/timing logs and a reproducible local NumPy sampler seed. The default
upstream behavior is retained when the new arguments are omitted. Three SF
tests pass, including repeated sampler cycles without changing global RNG.
`targetaugment-lighting-smoke-002` completed 120 iterations with seed 42 and
finite logged content/constraint/style/content-style losses. Mean two-optimizer
update time over the final 100 iterations: 0.0232331 seconds, excluding image
IO. Peak allocated/reserved: 187,203,072 / 207,618,048 bytes. Decoder/fc1/fc2
strictly reload and all saved tensors are finite. Optimizer resume remains open.

The earlier assumed need for an external fixed style image is superseded by
code inspection: `train_sf-yolo.py` enables `random_style` when `style_path` is
empty; `enhance_base.load_style_img` then selects from the current target batch.
Use this supported default for the BDD technical integration, recording that
batch 1 means self-style. Auxiliary content/style training still uses independent
samples from the same train-only target pool. Do not introduce a foreign meanfoggy
reference or claim a controlled comparison with the README's fixed-style example.

A direct native-style probe using smoke-002 weights and the first lighting-train
image returned a finite `[1,3,736,1280]` tensor: 0.3988 seconds for one cold call,
1,017,990,144 bytes allocated / 1,291,845,632 bytes reserved. This is NOT a latency
benchmark or the complete detector+teacher+TargetAugment memory footprint.
No labels or external style image were read. Shape equality does not establish
semantic translation quality, and 120-step auxiliary weights remain diagnostic.

### Integrity audit, 2026-09-25

`audit_auxiliary_assets.py` verified both diagnostic pair directories: 128/128
outputs per direction, SHA-256 provenance matches, RGB native geometry 1280x720,
and train-only/`production_asset=false` metadata. It also loaded all three
instrumented TargetAugment components and verified finite tensors plus the 120-step
finite-loss summary. The report is `runs/auxiliary-asset-audit-001/report.json`.
This closes only an integrity sub-gate; it does not establish semantic box
alignment, translation quality, optimizer/RNG resume, or production approval.

### Retired full-run path — 2026-09-28

The production auxiliary campaign was cancelled in favor of the bounded pilot.
Its launch, export and audit code and generated planning artifacts were removed.
Only the 128-image CUT and 120-iteration TargetAugment diagnostics above remain
in scope; none is a production asset.
