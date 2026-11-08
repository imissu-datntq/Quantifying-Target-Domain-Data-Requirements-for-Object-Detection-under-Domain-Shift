# Adaptive Teacher R101-C4 execution

## One-thousand-image project pilot — 2026-09-28

`runs/pilot1000-r101-source` completed 1,000 source iterations on 1,000
daytime-clear images. Its detector checkpoint initialized both teacher and
student for `runs/pilot1000-r101-adapt`, which completed 1,000 adaptation
iterations on 1,000 source and 1,000 image-only lighting-target images. The
adaptation run took 1,426.23 seconds including hooks and used 5,445 MiB peak
GPU memory in the training log. The teacher export covered the same 1,000
lighting-validation images used by the YOLO candidates. COCO mAP50:95 was
0.686985 and mAP50 was 1.936026 AP points. This is bounded engineering
evidence, not a final project score.

The candidate uses upstream revision `5256463ad9ec90fd5ba84ebb8d53bed56bd369df`.
The isolated environment is `.venvs/adaptive-teacher` (Python 3.8).
Initial compatibility stack: PyTorch 1.7.1+cu110, torchvision 0.8.2+cu110,
Detectron2 0.3+cu110, NumPy 1.23.5 and Pillow 9.5.0.
Installation is not evidence that GPU operators or training work.

`adaptive_teacher_bdd.py` registers source training, lighting unlabeled training
and source validation. Image paths resolve against `research_split_v1`;
annotations resolve against `research_inputs_v1`. Unlabeled loading opens no
COCO annotation file, and pilot registration exposes no holdout dataset.

`run_adaptive_teacher_smoke.py` uses the upstream R101-C4 configuration with
ten classes, native 1280x720 geometry, one labeled and one unlabeled image per
batch, seed 42 and 128 deterministically selected images per training pool.
The technical learning rate is 0.0001; burn-in is one update so a two-update
adaptation smoke reaches the adaptation graph. The current entry point uses
120 updates by default (20 warmup and 100 measured); fewer than six updates
are rejected because the upstream timing hooks do not handle very short runs
reliably. These settings are exclusively
for technical checks, not a reviewed accuracy-training schedule.

```bash
.venvs/adaptive-teacher/bin/python run_adaptive_teacher_smoke.py --phase source --output runs/r101-source-smoke-001
.venvs/adaptive-teacher/bin/python run_adaptive_teacher_smoke.py --phase adapt --output runs/r101-adapt-smoke-001
```

Existing output directories are rejected. Each run records the resolved config,
upstream revision, completion/failure state, wall time and peak GPU memory when
successful. Resume via `--resume-from RUN --steps 6` restores model, optimizer,
scheduler and iteration into a new run directory. New runs also write an
`rng_sampler_state.pth` sidecar containing Python/NumPy/Torch RNG streams and
the single-worker sampler cursor; a resume replays the cursor before restoring
those streams. This closes state-capture plumbing, not bitwise exactness.

The bounded split-vs-uninterrupted check (`r101-restart-uninterrupted-012`
versus `r101-restart-split-006` + `r101-restart-split-continue-006b`) still
found model/optimizer tensor differences despite sidecar restoration. The
likely remaining source is nondeterministic CUDA/ROI operations; the run is
therefore recorded with `exact_rng_sampler_resume=false`, and exact restart
remains a campaign gate.

The same bounded comparison was repeated with `--deterministic` (cuDNN
deterministic mode and PyTorch deterministic algorithms when available); it
still produced 96 model/optimizer tensor mismatches. This confirms that the
sidecar restores process state but does not establish bitwise continuation on
this pinned CUDA/ROI stack.

A further PyTorch 1.7.1 retry additionally set `CUBLAS_WORKSPACE_CONFIG` and
disabled TF32. The independent same-segment control already differed in 200/986
model tensors (maximum absolute difference `7.46e-6`) and 103/105 optimizer
tensors; the uninterrupted-versus-resumed comparison differed in 204/986 model
and 104/105 optimizer tensors. The structured report is
`runs/r101-cublas-restart-audit-001/report.json`. This makes the limitation
stronger: bitwise exact restart is not attainable on the pinned stack even
before a checkpoint boundary. The gate remains open rather than being silently
weakened to an allclose criterion.

Source-only uses upstream `TwoStagePseudoLabGeneralizedRCNN`, with the same
backbone and ROI head as the adaptation model. Using the DA meta-architecture
for source-only adds an unwanted source discriminator loss; diagnostic run
`r101-source-smoke-002` therefore is not an accepted baseline.

`r101-adapt-smoke-004` completed 120 updates with finite logged losses and
checkpoint reload. Mean update wall time including hooks was 1.4361 s over
100 updates after 20 warmup updates. Peak allocated/reserved memory was
5,709,664,256 / 6,845,104,128 bytes. This sample uses 128 source and 128 target
images; it does not establish the worst-case memory requirement of all images.
Each adaptation update draws one source and one target image and applies
the upstream strong/weak views and multiple teacher/student forwards.

`evaluate_adaptive_teacher_smoke.py --run RUN` strictly reloads the checkpoint
and exports original-coordinate predictions for 64 source-validation and
64 lighting-validation images, evaluated with the official COCO API. This
plumbing check passed for `r101-adapt-smoke-004`; its AP is not selection evidence.
The training run remains distinct from validation output.

Detector-only `r101-source-smoke-003` completed 120 updates and checkpoint
reload at 0.3174 s/update including hooks, with allocated/reserved peaks
2,265,843,200 / 2,382,364,672 bytes. Its validation plumbing also passed.
`r101-adapt-resume-001` continued iterations 120 through 125 successfully.
`r101-finetune-smoke-001` loaded the accepted source checkpoint, used the
canonical lighting N=100 seed-42 manifest, completed six updates and reloaded
its checkpoint. Fine-tuning supports N=100/1000 and the three target domains
for technical checks; N=1000 smoke uses at most 128 images from that manifest.

The retired full selection harness and long-training schedule were removed.
Model screening now uses only bounded smoke runs and common validation samples.
No technical-smoke score is a final scientific result.
