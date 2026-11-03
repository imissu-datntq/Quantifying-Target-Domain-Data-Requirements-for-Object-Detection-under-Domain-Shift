# ExecPlan: RQ5, adaptation vs fine-tuning along the label budget

- **Status:** active. Created 2026-11-03 (Asia/Saigon).
- **Owner:** Claude Code session with the user. Compute: Kaggle `kizhowl` (user, 2026-11-03: "run it on kaggle for now").
- **Depends on:** the completed supervised campaign (`2026-10-30-supervised-campaign.md`), its frozen source model, and its scored fine-tuning (FT) curves in `runs/campaign/scored/`.

## Purpose and observable outcome

RQ5 asks: *when does domain adaptation give a meaningful advantage over plain target fine-tuning as labeled target data increases?*

**Done means:**
- One established adaptation method, run on our Faster R-CNN R50-FPN at 1280×720 from our source model, for each shift:
  - as an unsupervised line (N = 0 labels), and
  - as a semi-supervised curve over labeled budgets.
- Its final-test predictions are scored once from a frozen plan.
- `docs/results_rq5_vi.md` reports, per shift:
  - Δ(N) = mAP_DA(N) − mAP_FT(N) with CI;
  - the crossover budget;
  - the label-equivalent budget of each DA run;
  - an attribution split: unlabeled-target distillation vs. source replay, strong augmentation and EMA.

## Context

**RQ1–RQ4 results** (`docs/results_rq1_rq4_vi.md`):
- FT with the locked schedule (S=500, E=2, LR 1e-3, B 2) is worse than zero-shot at N ≤ 50.
- Break-even is at about 100–250 images.
- N75: lighting 5,000, compound 1,000, weather the full pool.
- The weather ceiling gain is only about 0.013 AP.

The proposal (§6.5) says:
- The method is chosen after the FT curves exist.
- It must have a public implementation compatible with the detector.
- An unsupervised method appears as a constant line. It may use only unlabeled target-train images, and the held-out images are never exposed.
- A semi-supervised method gives a second curve over the same budgets.

**Why this method.** ALDI / ALDI++ (Kay et al., *Align and Distill*, arXiv 2403.12029, CVPR 2024; github.com/justinkay/aldi):
- It is a unified DAOD benchmark whose default detector is **Faster R-CNN R50-FPN**, the same as ours.
- Its framework re-implements Adaptive Teacher, MIC, PT, UMT and SADA on that detector.
- ALDI++ is its strongest configuration, and the paper shows that most earlier methods do not beat a fair source-only baseline.

Adaptive Teacher and 2PCNet were the alternatives:
- Adaptive Teacher's public configs use R101 or VGG16, not R50-FPN.
- 2PCNet is night-specific, so it doesn't fit weather. It is also built on Adaptive Teacher.

**Licensing.** The ALDI repo has no license file, so its code is *not* copied. The method is re-implemented in torchvision from the paper and its published configs (`ALDI-Best-Cityscapes.yaml`, `aldi/config.py`), which were read for exact semantics. This is stated in the report.

**ALDI++ as configured** (`ALDI-Best-Cityscapes.yaml`; our port keeps these):
- EMA teacher, α 0.9996 by default.
- Each step uses 1:1 labeled and unlabeled images.
- Labeled images get strong augmentation plus random erasing.
- Unlabeled images: the student gets strong augmentation plus MIC masking (ratio 0.5, 32-px blocks); the teacher gets the weak view.
- Soft distillation, all at weight 1 and temperature 1:
  - ROI classification: CE against the teacher's softmax.
  - ROI box regression: smooth-L1, β 0, on proposals the teacher calls foreground.
  - RPN objectness: BCE against the teacher's sigmoid, on sampled anchors.
  - RPN regression: smooth-L1 on foreground anchors.
- Hard pseudo-labels at 0.8 are used only to sample anchors and proposals. They are not used as loss targets.
- No feature alignment.
- Evaluation uses the EMA teacher.

**Deviations, all stated in the report:**
- **No multiscale input:** our contract is fixed 1280×720.
- **No ALDI++ "robust burn-in":** we start from *our* source model, the same initialization as FT, so the arms differ only in the adaptation stage.
- **Small batch:** 2 labeled + 2 unlabeled per step, vs. 48 in the paper, with LR 1e-3 to match FT. The number of updates is set by the pilot below.
- **Semi-supervised use is our extension.** ALDI ignores target labels; here, labeled target images join the labeled stream.

## Scope and non-goals

**In scope:**
- The method port, its tests, the lighting-val pilot and the lock.
- A 46-run grid on Kaggle.
- One-shot scoring, the RQ5 analysis and the report.

**Not in scope:**
- A second DA method.
- Retuning FT.
- Proportional-update FT reruns (still optional, separate).
- Any use of weather or compound val.
- Any change to `scripts/bdd_pipeline/`. It stays byte-identical, so FT provenance hashes hold. The new code lives in `scripts/bdd_adapt/`.

## Design

**Arms, all initialized from the source model** (`source_seed0/best.pt`, source-val 0.352):

| arm | labeled stream (2/step) | unlabeled stream (2/step) | distillation |
|---|---|---|---|
| `adapt` N=0 (UDA line) | 2 source | 2 from the target train pool | ALDI++ |
| `adapt` N>0 (semi-supervised) | 1 source + 1 from the N labeled target images | 2 from the target train pool | ALDI++ |
| `replay` (attribution control) | as in `adapt` | none | none (supervised only, same augmentation and EMA) |

**Shared settings:**
- **Unlabeled stream:** the whole target train pool, read through `BDDImages`, which never opens an annotation file. The images are the same as FT's pool; the labels are hidden.
- **Updates:** T_run = max(T, 2N), where T comes from the pilot. Each labeled target image is then seen at least twice, matching FT's E=2.
- **Optimizer:** SGD with momentum 0.9 and weight decay 5e-4, LR 1e-3, warmup 50, clip 10, AMP. These match FT.
  - The LR is constant after warmup, as in ALDI, whose only LR step is at the last iteration. Intermediate teacher checkpoints of a long pilot run therefore stand in for shorter runs.
- **Kept model:** the final EMA teacher. There is no checkpoint selection, which is the same rule as FT.

**Grid (46 runs):**
- `adapt` N=0: 3 shifts × seeds {0, 1} = 6 runs.
- `adapt` semi-supervised, with seeds {0, 1, 2} and the same nested subsets as FT (`nested_rows`), so Δ is paired by sampling seed:
  - compound and weather at N {10, 100, 1000}: 18 runs;
  - lighting at N {10, 100, 1000, 5000}: 12 runs.
- `replay`, seed 0: N=0 (source only, predicted on all domains) as 1 run, plus 3 shifts × N {10, 100, 1000} as 9 runs.

**Cost** (estimate; replaced by the GPU smoke measurement):
- About 0.75 s per `adapt` step and 0.35 s per `replay` step on a T4, plus 5–10 min of prediction per run.
- If T = 3000, the grid is about 30 T4-h. If T = 6000, about 50 T4-h. The pilot adds about 7 h.
- At 30 h/week on one account, that is 1–2 weeks. Lighting N=5000 (3 runs × 10,000 steps) goes last.

## Pre-registered rules (copied to DECISIONS.md before the pilot runs)

**Pilot setup:**
- Lighting val only, on a fixed 1,000-image subset: lighting val rows shuffled with seed 0, first 1,000.
- Runs: `adapt` N=0 and `adapt` N=100 (seed 0), each with EMA α ∈ {0.999, 0.9996}. That is 4 runs of 6,000 updates.
- The teacher's reliable mAP50:95 is evaluated at step 0 and every 500 updates.

**Choosing α:**
- Pick the α with the higher mean peak over its two runs.
- If the two are within δ = 0.005, use 0.9996, the published default.

**Choosing T:**
- T is the smallest evaluated step at which both runs under the chosen α are within δ of their own peak, and stay within δ at every later step.
- If T = 6,000, it is locked at 6,000 and reported as a limitation.
- There is no second pilot round.

**Correctness gate** (a check of the implementation, not a selection):
- If the N=0 run's teacher at T is below its own step-0 value (the source model) by more than δ, the port is treated as suspect.
- The grid does not launch until the cause is found. A bug fix does not reopen the α/T choice unless the fix changes the training maths. In that case the pilot is rerun under the same rule, and the rerun is recorded.

**RQ5 analysis** (primary metric: reliable mAP50:95 on final test):
- **Δ(N)** = mean over seeds of [mAP_DA(N, s) − mAP_FT(N, s)], paired by sampling seed.
  - Its CI comes from the hierarchical bootstrap: the same test resamples (seed 20261030) as the FT scoring, with the seeds resampled as pairs.
  - It is also reported in recovery units, against FT's own mAP(0) and ceiling.
- **Meaningful advantage at N:** Δ(N) ≥ δ = 0.005, and the lower 95% bound is above 0.
- **Crossover N×:** the smallest tested N from which DA has no meaningful advantage at every larger tested N. It is reported as the bracket between rungs, or "beyond the largest tested N".
- **Label-equivalent N_eq(N):** the FT budget whose seed-mean mAP equals mAP_DA(N), interpolated linearly in log N between FT rungs, with a bootstrap CI.
  - Above the FT ceiling, it is reported as "> full pool".
  - The UDA line gets N_eq(0).
- **Attribution at N ∈ {0, 10, 100, 1000}, seed 0:** adapt − replay (the value of unlabeled-target distillation), and replay − FT (source replay plus strong augmentation and EMA).
- **Secondary:** per-class Δ for the four reliable classes, and forgetting on source test.
- **Weather:** always given in raw AP with CI. Recovery units are reported only where the denominator is at least 0.01.

## Ordered work

1. **Plan, decisions, memory.** Done: this file, and the DECISIONS entry with the rules above.
2. **Port** into `scripts/bdd_adapt/`:
   - `augment.py`: strong photometric augmentation, random erasing and MIC, on tensors in [0,1], with seeded randomness.
   - `distill.py`: teacher-student losses on torchvision `FasterRCNN` internals.
   - `train.py`: the trainer, with step-indexed deterministic data streams, exact resume, and a `last.pt` whose `model` is the teacher, so `predict_frcnn.py` and `freeze_evaluation_plan.py` work unchanged.
   - `grid.py`: the policy, rows and arguments.
   - CLI: `scripts/train_adapt.py`.
3. **Tests** in `tests/test_adapt.py`:
   - augmentation keeps the geometry, and MIC masks about the configured ratio;
   - distillation losses go to zero for identical teacher and student on identical inputs (KL form), and to zero gradient on the teacher;
   - the EMA update is correct;
   - T_run = max(T, 2N);
   - the unlabeled stream never reads labels;
   - stream batches are step-deterministic across a simulated resume;
   - grid rows and ids are correct.
   - Then run the full `pytest`.
4. **Local CPU smoke:** `--smoke` with tiny limits. Check the end-to-end losses are finite, resume is exact, and predict plus freeze accept the checkpoint.
5. **Kaggle pilot** with kernel `kaggle/bdd_adapt_pilot.py`:
   - Push a new protocol-dataset version carrying `scripts/bdd_adapt/`.
   - The kernel attaches the source-model output.
   - The first 30 min serve as the GPU smoke: throughput, memory, `amp_step_skipped`.
6. **Lock α and T** by the rule. Record them in DECISIONS.md and `bdd_adapt/grid.py`, then generate `runs/rq5/grid.csv` with the cost estimate.
7. **Grid on Kaggle** with kernel `kaggle/bdd_adapt.py`, a queue runner adapted from `bdd_campaign.py` that resumes across sessions. Order:
   1. compound;
   2. weather;
   3. replay;
   4. lighting ≤ 1000;
   5. lighting N=5000.
8. **Freeze and score once:**
   - `freeze_evaluation_plan.py --prediction-manifests` over all 46 runs.
   - `evaluate_frcnn.py` on CPU locally, with `--resume-interrupted` if needed.
9. **Analysis:** `scripts/analyze_rq5.py` reads `runs/rq5/scored` and `runs/campaign/scored` and writes `runs/rq5/analysis/`.
10. **Report:** `docs/results_rq5_vi.md`, then the lecturer draft, HANDOFF and TODO. Then move both campaign plans to `plans/archive/` once the user has reviewed.

## Validation criteria

- `pytest` passes, with the old 29 tests and the new ones.
- The smoke resume run is bit-identical to an uninterrupted run, on CPU with a tiny limit.
- Every grid run's `config.json` records `target_labels_accessed: false` for the unlabeled stream. Every prediction manifest has `test_labels_accessed: False`.
- Pilot step-0 teacher mAP equals the source model on the same subset. This is the EMA-copy sanity check.
- Scoring: the FT bootstrap replicates are reproduced exactly for one FT entry from `runs/campaign/scored` under the RQ5 plan's settings. This proves the resamples are paired.
- `replay` N=0 at about the source model on source test, give or take forgetting, is a sanity reference, not a gate.

## Recovery, interfaces, dependencies

- **Resume:** `train_adapt.py --resume <run>/last.pt` with identical arguments. The checkpoint holds the student, teacher, optimizer, scaler, step and RNG. The data streams are a pure function of (seed, stream, step).
- **Kaggle:**
  - Inputs: `kizhowl/bdd100k-revised-protocol` (code + manifests), the `solesensei/solesensei_bdd100k` mirror, the `kizhowl/bdd100k-source-train` output, and the kernel's own previous version.
  - Use the `kizhowl` token only, via `KAGGLE_API_TOKEN`.
  - Never `ngdminh31`.
- **Ceiling on effort:** if the GPU smoke shows more than 1.2 s per `adapt` step, re-estimate. Before running more than 60 T4-h, ask the user (for example, drop lighting N=5000 or replay seeds).

## Progress

- 2026-11-03: plan written; rules pre-registered in DECISIONS.md.
- 2026-11-03: port written in `scripts/bdd_adapt/` (augment, distill, train, grid), with `scripts/train_adapt.py`, `kaggle/bdd_adapt_pilot.py` and `kaggle/bdd_adapt.py`.
- 2026-11-04: pilot complete (6.6 GPU-h). Rule gives α 0.9996, T 2000; gate passed. POLICY locked; both protocol datasets updated; grid launched on buihuynhgiahuy (32 runs) and kizhowl (14 lighting runs, 8 h session until the quota reset).
  - `pytest`: 43 passed (29 old, 14 new).
  - CPU smoke on real data with the real source model (`runs/rq5_smoke2/`): losses are finite, with 20–40 pseudo-boxes per image pair.
  - Interrupt and resume is bit-identical (teacher, student, momentum).
  - `replay` runs without an unlabeled stream.
  - `predict_frcnn.py` accepts the checkpoint unchanged.
  - CPU speed is about 23 s per update.
- 2026-11-03: protocol dataset `kizhowl/bdd100k-revised-protocol` has a new version with the RQ5 code; `bdd_pipeline/` is byte-identical to the previous version.
  - Pilot kernel `kizhowl/bdd100k-rq5-pilot` v1 pushed and RUNNING.
  - Expect about 6–7 h if an update takes about 0.75 s on the T4. If the session limit cuts it, push v2 with `kizhowl/bdd100k-rq5-pilot` added to its `kernel_sources`; it resumes.

- 2026-11-03: user approved running part of the grid on `buihuynhgiahuy` (default `~/.kaggle` login).
  - Private datasets aren't shared between accounts, so two private copies were created: `buihuynhgiahuy/bdd100k-revised-protocol` (same files) and `buihuynhgiahuy/bdd100k-source-model` (`best.pt`, sha `2c62adf9…`).
  - `kaggle/bdd_adapt.py` finds the source model under either account's layout and refuses any other hash.
  - Quota at push time: buihuynhgiahuy 30.0 h, kizhowl 15.1 h; both reset 2026-11-07.
  - **Split:** buihuynhgiahuy (`buihuynhgiahuy/bdd100k-rq5`) runs compound, weather and replay (32 runs). kizhowl (`kizhowl/bdd100k-rq5`) runs lighting (14 runs).
  - Both use the same T4 shape and docker image. Kernel staging is in the scratchpad (`kpush/rq5_kizhowl`, `kpush/rq5_bhgh`).
- 2026-11-03: `scripts/rq5_pilot_lock.py` applies the pre-registered α/T rule and gate mechanically; tested on synthetic curves. 44 tests pass.

## Discoveries

- ALDI repo has no license file, so the method is re-implemented, not vendored.
- Creating a `DataLoader` iterator draws from the global torch RNG. This broke exact resume, because torchvision's anchor and proposal samplers use that RNG. Fixed by giving each loader its own generator; a regression test covers it.
- Local source checkpoint for smoke tests: `runs/source_model/runs/source_seed0/best.pt` (sha256 `2c62adf9…e349`).

## Decisions

- Method: ALDI++ (see Context). Semi-supervised by adding labeled target images to the labeled stream (1:1 with source).
- `scripts/bdd_pipeline/` untouched; new package `scripts/bdd_adapt/`.
