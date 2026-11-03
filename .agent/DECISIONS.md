# Decisions

## 2026-10-01: Revised dataset version and EDA scope

- Materialize a new 55,000/14,863/10,000 dataset from original source files
  verified against the user's Kaggle archive, with independent image copies.
- Freeze membership using manifests, hashes and algorithm, not seed alone.
- Stratify hierarchically across timeofday/weather and scene with Hamilton
  apportionment and seed 20260916; keep exact-content groups in one split using
  same-stratum singleton swaps so group quotas remain intact.
- Preserve official validation membership; retain and disclose within-split
  exact duplicates (train one extra image, val two), while rejecting any
  cross-split content leakage. Do not imply near-duplicate/group independence.
- EDA opens only new train/val detection labels. Held-out label access is
  limited to construction/structural validation, with metadata-only census.
- Keep original bbox geometry in the new detection JSON; report invalid geometry
  and clip only within EDA diagnostics. Fix training sanitation consistently later.
- Use budget seeds 0/1/2 for reproducible exploratory nested samples; do not
  interpret class-coverage curves as model performance or recovery results.

## 2026-10-02: Python detector pipeline

- Implement supervised core in scripts/bdd_pipeline and Python CLI entry points;
  preserve the earlier YOLO prototype and the user's existing staged Git changes.
- Native 1280×720 RGB inputs, one model normalization, deterministic horizontal
  flip, canonical 1–10 detection IDs plus background 0, shared bbox sanitation
  for training targets and COCO GT. FPN divisibility padding is not resize.
- Use the same FrozenBN epsilon and trainable layers for generic initialization,
  source-checkpoint loading and resume; replace the generic detection head.
- Common max(update floor, epochs) rule with actual weighted accumulation
  windows. Defaults remain provisional until a representative GPU pilot.
- Save complete local checkpoint state atomically; compatible exact resume uses
  last.pt, not historical best. Preserve selected best weights/step and attach
  training-completion metadata when its run completes.
- Use COCO AP/AR and support-aware diagnostics, with fixed final-test plan,
  checkpoint hashes and one-shot opening marker. Reject smoke/incomplete models
  and selection overrides before accessing test annotations.
- CPU smoke only is authorized in this implementation task. No full experiment
  campaign or real test evaluation has been executed. CUDA/AMP remains unverified.

## 2026-10-19: Review amendments (evaluation roles, metric roster, pilot gate)

- Weather (daytime-rainy) and compound (night-rainy) held-out sets were below the
  600-image target. Their preserved-val images join the final test set
  (933 / 756); source and lighting keep 2,650 / 4,869. Physical folders,
  manifests and 55,000/14,863/10,000 counts are unchanged; roles are derived in
  `bdd_pipeline.data` and asserted on every manifest read.
- Weather/compound have no selection validation. All target runs (fine-tune,
  ceiling, control) keep the final update of the common fixed schedule; only
  source training selects on source val. Lighting val serves the schedule pilot.
- Primary metric: mAP50:95 over the fixed roster pedestrian, car, traffic light,
  traffic sign for every domain; undefined if any is missing. Ten-class mean is
  a sensitivity analysis. Support-threshold rosters are retired.
- Pilot gate first: zero-shot gap of the source model, source val vs target train
  pool (no holdout labels), bootstrap CI; a shift whose CI does not clear 0.01 AP
  is flagged before its ladder. Recovery uses a minimum denominator (default 0.01).
- Recovery is reported per reliable class to separate class-prior re-learning
  from new visual features.
- Labels are the 2018 BDD100K release via Kaggle v2 (officially deprecated by
  det_20). Official hosts were unreachable on 2026-10-19; a seeded 400-record
  sample is frozen for comparison with an official download.
- Disclosure: before this amendment the EDA read weather/compound val labels
  descriptively, and one CPU smoke run validated on one weather val image; no
  modelling or selection decision used them.

## 2026-10-20: GPU execution on Kaggle

- GPU work runs as private kernels on the user's Kaggle account `kizhowl`. The
  frozen protocol (code, annotations, manifests, hashes) is the private dataset
  `kizhowl/bdd100k-revised-protocol`; images are symlinked from the public
  `solesensei/solesensei_bdd100k` mirror. A CPU kernel rehashed all 79,863 images
  against the manifests (pass); GPU sessions rehash a seeded 2,000-image sample.
- AMP: nonfinite gradients from an overflowing loss scale are skipped by
  GradScaler and logged (`amp_step_skipped`); without AMP they remain an error.
  Found by the first T4 pilot.
- Weekly GPU quota (30 h) is shared with the account's other projects; long
  runs stop by update count and resume from the previous version's output.

## 2026-10-30: Labels, compute, campaign plan

- Stay on the 2018 BDD100K labels; no det_20 migration. The local copy (`data/labels_raw/`, moved from
  `D:/BDD100K-weather-classification/data`) is byte-identical to the Kaggle mirror (SHA256).
  That does not prove it matches Berkeley's original files.
- GPU work runs on Kaggle. Ask the user before using the local RTX 5060.
- Campaign plan: `.agent/plans/active/2026-10-30-supervised-campaign.md`. The schedule is locked on
  lighting val before any target run, and test is opened once, after the full grid.

## 2026-10-30: Schedule pilot round 3, rule written before results

Rounds 1 and 2 (teammate, 2026-10-20, kernels `bdd100k-gate-schedule` and `bdd100k-schedule-pilot`) were all run on lighting val, 3,274 images, seed 0, batch 2. Zero-shot N0 = 0.249.

- At LR 5e-3, N100 lands below zero-shot.
- At LR 1e-3, N100 with 500 updates scores 0.254, and N1000 at E5 scores 0.278, the same as E10 (0.277).
- Every result improved with fewer updates, so round 3 brackets downward.

Round 3 kernel `bdd100k-schedule-pilot3` (`kaggle/bdd_schedule_pilot.py`). Same settings as round 2: batch 2, AMP, warmup 50, scored on lighting val reliable mAP50:95.

| run | updates |
|---|---|
| N100 S250 E5, LR 1e-3, seeds 0/1/2 | 250 |
| N100 S125 E1, LR 1e-3 | 125 |
| N10 S250 E5, LR 1e-3 | 250 |
| N10 S500 E5, LR 1e-3 | 500 |
| N1000 E2, LR 1e-3 | 1,000 |
| N1000 E3, LR 1e-3 | 1,500 |
| N1000 E5, LR 5e-4 | 2,500 |

**Rule (fixed before round 3 is read).** Seed-0 runs from rounds 2 and 3 are comparable; all use batch 2 and warmup 50.

δ = max(0.005, sample SD of the three N100 S250 seeds).

- **LR:** 1e-3, unless 5e-4 beats it by more than δ at N1000 E5. It tied at N100 in round 2.
- **S:** among N100 runs at S ∈ {125, 250, 500}, take the best score. If several are within δ of it, choose the largest S, so that small budgets are not starved of updates (proposal §6.4).
- **E:** among N1000 runs at E ∈ {2, 3, 5, 10}, take the best score. If several are within δ of it, choose the smallest E, since the curve has already plateaued and fewer epochs costs less.
- **Fixed:** B_eff = 2 (batch 2, accumulation 1), AMP, warmup 50, cosine to 1%, clip 10.
- **Seeds:** 3 sampling seeds (proposal §10).
- **N10 runs:** reported only, not used to choose. If both N10 runs score below N0 by more than δ, flag it in the write-up; it is a finding, not a reason to retune.
- If the best S or E is the smallest value tested, lock it anyway. Do not run a fourth round. Disclose that the optimum may lie lower still.

## 2026-10-30: Target fine-tuning schedule locked (S=500, E=2, LR 1e-3)

Round 3 (`kizhowl/bdd100k-schedule-pilot3`, outputs in `runs/schedule_pilot/round3/`) was scored by the rule above, unchanged. Lighting val reliable mAP50:95:

| run | updates | score |
|---|---|---|
| N100 S125, LR 1e-3, seed 0 | 125 | 0.2523 |
| N100 S250, LR 1e-3, seeds 0/1/2 | 250 | 0.2565 / 0.2552 / 0.2537 |
| N100 S500, LR 1e-3, seed 0 (round 2) | 500 | 0.2543 |
| N1000 E2, LR 1e-3 | 1,000 | 0.2749 |
| N1000 E3, LR 1e-3 | 1,500 | 0.2771 |
| N1000 E5, LR 1e-3 (round 2) | 2,500 | 0.2781 |
| N1000 E10, LR 1e-3 (round 2) | 5,000 | 0.2770 |
| N1000 E5, LR 5e-4 | 2,500 | 0.2770 |
| N10 S250, LR 1e-3 | 250 | 0.2001 |
| N10 S500, LR 1e-3 | 500 | 0.1862 |

- δ = max(0.005, SD 0.0014) = **0.005**.
- **LR = 1e-3:** 5e-4 does not beat it at N1000 E5 (0.2770 vs 0.2781).
- **S = 500:** the best is S250 at 0.2565. S125 and S500 are both within δ, so the largest S wins.
- **E = 2:** the best is E5 at 0.2781. E2, E3 and E10 are all within δ, so the smallest E wins.
  - E2 is the lowest value tested. Per the rule there is no fourth round. The write-up must disclose that the optimum may lie lower still, and that E2 sits 0.003 below the best E.
- **Flag for the write-up:** both N10 runs land well below zero-shot (N0 = 0.249, a gap of 0.049 and 0.063). At the locked S=500, the N10 rung is expected to show negative transfer. This is reported as a finding, not retuned.
- **Locked policy** (`bdd_pipeline/campaign.POLICY`): S=500, E=2, LR 1e-3, batch 2, accumulation 1, AMP, warmup 50, cosine to 1%, clip 10. Sampling seeds 0/1/2, ceiling seeds 0/1, control seed 0.
- **Grid:** 87 runs, about 277k image-passes, about 12.6 T4-h of training plus prediction on the test images.

## 2026-11-02: One-shot scoring can finish an interrupted pass

The schema-3 scoring pass covers 87 prediction sets across 4 test domains and takes about 7 CPU-hours. Before this change, any crash or reboot after the `.opened` lock left the test set impossible to score again.

`evaluate_frcnn.py --final-plan P --output-dir O --resume-interrupted` continues only that same pass. It requires:
- the lock records the hash of plan P and the path of output O;
- O holds an identical `frozen_plan.json`;
- O has no `results.json`.

Prediction sets that already have a `metrics.json` are skipped. The bootstrap `.npy` is written atomically before `metrics.json`, so a `metrics.json` only exists for a complete entry.

This adds no selection freedom: predictions are frozen and hash-checked, and scoring is deterministic (fixed bootstrap seed). A plain rerun, a different plan, or a rerun after completion are all still refused.

## 2026-11-03: RQ5 method and pre-registered rules

**Method:** ALDI++ (Kay et al., *Align and Distill*, CVPR 2024). It is re-implemented on our torchvision Faster R-CNN R50-FPN (`scripts/bdd_adapt/`) and started from the same source model as FT.

**Why ALDI++:**
- Its benchmark's default detector is R50-FPN.
- It reimplements Adaptive Teacher, MIC, PT, UMT and SADA on that detector.
- It is the strongest method reported there.

The repo has no license file, so its code is not copied. Deviations:
- fixed 1280×720 input, no multiscale;
- no robust burn-in (the source model is reused);
- 2 labeled + 2 unlabeled images per step, LR 1e-3;
- semi-supervised use adds labeled target images to the labeled stream (1 source + 1 target), which is our extension.

Full design: `.agent/plans/active/2026-11-03-rq5-adaptation.md`.

**Pilot rule (lighting val only), written before any pilot result:**
- **Runs:** `adapt` at N=0 and N=100 (seed 0), each with EMA α ∈ {0.999, 0.9996}. 6,000 updates each at constant LR after warmup. The EMA teacher is evaluated at step 0 and every 500 updates on a fixed 1,000-image subset of lighting val (shuffled with seed 0, first 1,000).
- **α:** the α with the higher mean peak over its two runs. If the two are within δ = 0.005, use 0.9996.
- **T:** the smallest evaluated step at which both runs under the chosen α are within δ of their own peak and stay within δ afterwards. If T = 6,000, it is locked there and disclosed. There is no second round.
- **Correctness gate:** if the N=0 teacher at T is more than δ below its own step-0 value, debug the port before launching the grid.
- **Run length:** T_run = max(T, 2N).

**Grid (46 runs):**
- `adapt` N=0: seeds 0/1 for each shift.
- `adapt` semi-supervised: N 10/100/1000 for each shift, plus 5000 for lighting. Seeds 0/1/2, using FT's nested subsets.
- `replay` control: no unlabeled stream, otherwise identical. Seed 0, at N 0, 10, 100 and 1000.

**RQ5 analysis:**
- **Δ(N)** = DA − FT, paired by sampling seed. Its CI comes from the hierarchical bootstrap on FT's test resamples (seed 20261030).
- **Meaningful advantage:** Δ ≥ 0.005, with the lower 95% bound above 0.
- **Crossover N×:** the smallest tested N from which there is no meaningful advantage at every larger tested N.
- **Label-equivalent N_eq:** the FT budget matching mAP_DA, interpolated linearly in log N.
- **Attribution:** adapt − replay, and replay − FT.

## 2026-11-04: RQ5 pilot locked (α = 0.9996, T = 2,000)

The pre-registered rule (2026-11-03) was applied mechanically by `scripts/rq5_pilot_lock.py` to the 4 pilot curves (`runs/rq5/pilot/`, output `runs/rq5/pilot_lock.json`, Kaggle kernel `kizhowl/bdd100k-rq5-pilot` v1, 6.6 GPU-h).

| run (lighting val, 1,000 img) | step 0 | peak | at 2,000 | at 6,000 |
|---|---|---|---|---|
| N=0, α 0.9996 | 0.2533 | 0.2596 | 0.2595 | 0.2587 |
| N=100, α 0.9996 | 0.2533 | 0.2725 | 0.2684 | 0.2718 |
| N=0, α 0.999 | 0.2533 | 0.2563 | 0.2499 | 0.2477 |
| N=100, α 0.999 | 0.2533 | 0.2665 | 0.2640 | 0.2651 |

- **α = 0.9996:** mean peak 0.2661 vs 0.2614 (difference 0.0047 ≤ δ, so the tie rule gives the published default; it is also the higher one).
- **T = 2,000:** N=0 is sustained from 500, N=100 from 2,000. Not the pilot maximum.
- **Gate passed:** N=0 at T is 0.2595, above step 0 (0.2533). Under α = 0.999 the N=0 teacher drifts below step 0 after ~1,500 updates; this is the faster-EMA arm and is not used.
- **Grid:** `POLICY = {floor_updates: 2000, ema_alpha: 0.9996}`; run length max(2000, 2N). Measured 0.789 s/update on a T4 (AMP, 2+2 images). Estimated ~35 T4-h for the 46 runs, under the 60 h limit.
- **Disclosure:** the unsupervised gain on lighting val is small (+0.006 AP); N=100 keeps rising slightly after T (0.268 → 0.272). Both are reported, not tuned around.
