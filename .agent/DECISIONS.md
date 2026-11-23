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

## 2026-11-05: Full fresh rerun on the local RTX 3060, no Kaggle

The user decided to stop using Kaggle and rerun the whole study from scratch on the local Linux machine (RTX 3060 12 GB, driver CUDA 12.8). "From scratch" includes the selection pilots, not just the grids.

- **Scope:** rebuild split → source model → zero-shot gate → schedule pilot (re-run rounds 2+3 configs, apply the 2026-10-30 rule unchanged) → 87-run campaign → RQ5 pilot (apply the 2026-11-03 rule unchanged) → 46-run RQ5 grid → one frozen plan, scored once.
- **Rules stay, numbers are re-derived.** The pre-registered rules (schedule 2026-10-30, RQ5 2026-11-03) are reused as written. Their Kaggle outcomes (S=500/E=2/LR 1e-3, α 0.9996/T 2000) are *not* carried over; the local pilots re-lock them.
- **Second opening of held-out labels.** The Kaggle campaign was scored once on 2026-11-03. The local rerun's final scoring is a second opening, reported as an independent replication, never mixed with or selected against the Kaggle numbers.
- **Source settings** are the Kaggle source kernel's (`kaggle/bdd_source_train.py`): 12 epochs, batch 4, AMP, LR 0.005, warmup 500, seed 0, selection on source val only.
- **Split:** rebuilt from the local copy (all 80,002 files pass the mirror's size/CRC32), seed 20260916. Manifests matched the frozen SHA256 directly; annotations and `dataset_metadata.json` differed only in line endings (Windows CRLF vs Linux LF) and were converted to CRLF, after which all 7 hashes match `data/split_reference/expected_sha256.json`. Membership and content are identical.
- **Environment:** `~/venvs/cv-final`, torch 2.11.0+cu128, torchvision 0.26.0, pycocotools 2.0.11. 44/44 tests pass. Measured 0.46 s/update at batch 4 (≈8.7 img/s).

## 2026-11-05: Local zero-shot gate and schedule lock (S=500, E=3, LR 1e-3)

Local source model `runs/local/source/source_seed0/best.pt` (sha256 `8b1b5b4d…5d8faf`): source-val reliable mAP50:95 0.3517, selected at the final step 29,412.

**Zero-shot gate** (`runs/local/zero_shot_gap/summary.json`, source val vs target train pools, 1,000 bootstrap resamples, no held-out label read). All three shifts pass:

| shift | gap (AP) | 95% CI | Kaggle 2026-10-20 |
|---|---|---|---|
| weather | 0.032 | 0.024–0.041 | 0.031 (0.023–0.040) |
| lighting | 0.100 | 0.091–0.110 | 0.099 |
| compound | 0.129 | 0.121–0.139 | 0.129 |

**Schedule pilot** (`runs/local/schedule_pilot/summary.json`, lighting val 3,274 images, batch 2, AMP, warmup 50; round-3 configs plus the round-2 seed-0 configs the rule compares). The 2026-10-30 rule was applied unchanged:

| run | updates | lighting val reliable |
|---|---|---|
| N0 (source model) | 0 | 0.2468 |
| N100 S125 | 125 | 0.2503 |
| N100 S250 seeds 0/1/2 | 250 | 0.2554 / 0.2547 / 0.2539 |
| N100 S500 | 500 | 0.2514 |
| N1000 E2 | 1,000 | 0.2746 |
| N1000 E3 | 1,500 | 0.2780 |
| N1000 E5, LR 1e-3 | 2,500 | 0.2797 |
| N1000 E10 | 5,000 | 0.2772 |
| N1000 E5, LR 5e-4 | 2,500 | 0.2775 |
| N10 S250 | 250 | 0.1969 |
| N10 S500 | 500 | 0.1834 |

- δ = max(0.005, SD 0.0007) = **0.005**.
- **LR = 1e-3:** 5e-4 is 0.0022 *below* it at N1000 E5.
- **S = 500:** best is S250 (0.2554); S500 is within δ (0.0040), S125 is not (0.00503); largest S within δ wins.
- **E = 3:** best is E5 (0.2797); E3 (0.0017) and E10 (0.0026) are within δ, **E2 is not (0.00511, outside by 0.0001)**; smallest E within δ wins.
  - This differs from the Kaggle lock (E=2), where E2 was 0.003 below the best. The margin is razor-thin and is disclosed as such; the rule is applied as written, not re-argued.
- **N10 flag:** both N10 runs land below N0 by more than δ (0.050, 0.063). Expected negative transfer at the N10 rung; reported as a finding.
- **Locked policy** (`bdd_pipeline/campaign.POLICY`): S=500, **E=3**, LR 1e-3, batch 2, accumulation 1, AMP, warmup 50. Seeds unchanged. 44/44 tests pass after the change.
- **Grid:** 87 runs, 391,368 image-passes, about 12.9 h of training at the measured 8.45 img/s (batch 2), plus test predictions (~0.083 s/image). `runs/local/campaign/grid.csv`.

## 2026-11-06: Local campaign scored once; local RQ5 pilot locked (α = 0.9996, T = 2,000)

- **Campaign:** 87/87 runs finished 2026-11-06 04:54 under S=500/E=3/LR 1e-3, no errors. One plan frozen over all 87 prediction files before any test label was read: `runs/local/campaign/final_plan.json`, schema 3, bootstrap 1,000 resamples seed 20261030, sha256 `a1c5eb9fa457188d8094d97f2a8e8984e7b0b34d633a5b37c9b8175fc9436a44`. Scored once on CPU (176/176 entries, `runs/local/campaign/scored/`), finished before 10:02. This is the second opening of the held-out labels (the first was the Kaggle campaign, 2026-11-03); it is reported as the local replication.
- **RQ5 pilot** (`runs/local/rq5/pilot/`, lighting val 1,000-image subset, 4 × 6,000 updates, ~0.62 s/update). The 2026-11-03 rule applied by `scripts/rq5_pilot_lock.py` → `runs/local/rq5/pilot_lock.json`:

| step | N0 α .9996 | N100 α .9996 | N0 α .999 | N100 α .999 |
|---|---|---|---|---|
| 0 | 0.2497 | 0.2497 | 0.2497 | 0.2497 |
| 1,000 | 0.2558 | 0.2615 | 0.2535 | 0.2629 |
| 2,000 | 0.2588 | 0.2679 | 0.2508 | 0.2656 |
| 4,000 | 0.2593 | 0.2712 | 0.2474 | 0.2653 |
| 6,000 | 0.2584 | 0.2715 | 0.2474 | 0.2654 |

  - **α = 0.9996:** mean peak 0.2656 vs 0.2603 (difference 0.0053 > δ, so chosen as the higher one, not by the tie rule).
  - **T = 2,000:** N=0 sustained from 1,000, N=100 from 2,000. Not the pilot maximum.
  - **Gate passed:** N=0 at T 0.2588 > step 0 0.2497. Under α 0.999 the N=0 teacher again drifts below step 0 after ~2,000 updates.
  - Same lock as Kaggle (2026-11-04), so `bdd_adapt/grid.POLICY` is unchanged. N=100 keeps rising after T (0.268 → 0.272); disclosed, not tuned around.
- **RQ5 grid:** 46 runs started 2026-11-06 10:03 via `scripts/local_rq5_grid.py`; ~18 h of training at the measured rate plus predictions.

## 2026-11-07: Local RQ5 grid scored once; RQ5 results

- **Grid:** 46/46 runs finished 2026-11-07 07:11 (`scripts/local_rq5_grid.py`, ~0.6 s/update). One plan frozen over the 46 prediction files before any RQ5 test label was read: `runs/local/rq5/final_plan.json`, schema 3, bootstrap 1,000 resamples seed 20261030 (shared with the local FT scoring), sha256 `3df3d788a57ebf2eb9cf0d4e996a954857740eabf6f20c8c9ca7b58bbf91b4b7`. Scored once on CPU (94/94, `runs/local/rq5/scored/`, done 09:27). The Kaggle RQ5 grid (launched 2026-11-04) was abandoned and never scored.
- **Analysis** by `scripts/analyze_rq5.py` (rules of 2026-11-03 applied as written) → `runs/local/rq5/analysis/`, report `docs/results_rq5_vi.md`.
  - Δ(N) = DA − FT, reliable: lighting +0.009 / +0.070 / +0.015 / −0.011 / −0.010 (N 0/10/100/1000/5000); weather +0.006 / +0.057 / +0.014 / −0.003; compound +0.011 / +0.061 / +0.016 / −0.012. Meaningful at N ≤ 100 everywhere; at N ≥ 1000 DA is significantly below FT.
  - **Crossover N× ∈ (100, 1000] for all three shifts.** N_eq(0) = 160 / 349 / 153 FT images (lighting/weather/compound).
  - Attribution (seed 0): adapt − replay 0.003–0.010 at N ≤ 10, ≤ 0.004 at N=100, ≈ 0 at N=1000; replay − FT carries the N=10 gap (0.046–0.064, FT overfits) and the N=1000 deficit.
  - N_eq uses the sustained (last upward) crossing of the FT curve, since FT dips below mAP(0) at small N; this detail was not spelled out in the 2026-11-03 rule and is disclosed.
- **Disclosed confound:** T_run = max(T, 2N) was pre-registered as matching FT's E=2; the local FT lock is E=3, and the rule was not changed. At N=1000 FT sees ~3,000 labeled-target image passes vs ~2,000 for adapt/replay (1 labeled target image per step). The N ≥ 1000 deficit is therefore confounded with labeled-training volume.

## 2026-11-21: Lecturer review of the results report; revision rules written before any new run

Source: `docs/review_2026-11-21_lecturer_vi.md`. Plan: `.agent/plans/active/2026-11-21-lecturer-review-revision.md`. Every rule below was written before any revision run existed and before any new test label was read. The analysis rules were written **after** the existing curves had been seen, so they are descriptive, not confirmatory.

**Framing (lecturer).**
- The headline is the control arm: *most of what domain adaptation gives at small labeled budgets is not adaptation.* It is tied to ALDI's thesis (Kay et al., CVPR 2024) that reported DAOD gains depend on implementation and baseline choices.
- The data-efficiency curves are the context that makes the claim measurable. N_BE/N75/N90 are secondary, practical results.
- Weather is the third replication of the decomposition, not a third threshold measurement.
- This supersedes "the contribution is the curve" in `CLAUDE.md`, which is rewritten in the plan's Phase 4.

**Kaggle (user).** Kaggle is dropped as a results source. The report, README and result notes carry local numbers only. The lecturer had asked for both runs to be reported equally, so the user informs the lecturer.
- Pending user confirmation: keep one disclosure sentence that held-out labels were first opened on 2026-11-03 by a Kaggle campaign under a different schedule (E=2). The comparison can go; the count of test-set openings cannot.
- Pending user confirmation: delete `kaggle/`, `scripts/kaggle_*.py`, `docs/kaggle_training.md` and `docs/results_rq1_rq4_vi.md` from `main`. Mentions of the Kaggle *mirror* as the data source stay; that is provenance.

**Provenance constraint.** Every run records the SHA256 of all `scripts/bdd_pipeline/*.py` (and `scripts/bdd_adapt/*.py`). Those folders stay byte-identical. New analysis code lives in `scripts/analyze_campaign.py` and `scripts/analyze_rq5.py`.

**Analysis rules (existing scored outputs only).**
- **Thresholds.** N95 is retired from every reported table (`recovery.LEVELS` keeps it only because the file is frozen). A threshold whose first passing rung is the ceiling rung is reported as "not determined within the pool (> last failing rung)": that rung is 1 by construction.
- **Fit.** The power fit is retired from the report. Replacement: Recovery = α + β·ln N by ordinary least squares.
  - Region: ladder rungs from the first rung whose 95% band lower bound is above 0 and stays above 0 at every larger rung, up to the largest rung below the ceiling. The ceiling rung is excluded, since it is 1 by construction.
  - At least four rungs, otherwise no fit and the ladder bracket is the result.
  - R² and RMSE are computed on the region only.
  - Held-out check: fit the three lowest rungs of the region, predict the remaining rungs and the ceiling rung, report the errors.
  - A fit-based threshold is given only by interpolation inside the fitted range.
  - Sensitivity: the same fit with the region starting at the point-estimate N_BE.
- **Recovery denominator.** Ceiling − zero-shot is tabulated per shift and class with its bootstrap CI and its ratio to the 0.01 floor.
- **Interaction detection limit.** For drop(compound) − drop(lighting) − drop(weather): the 95% CI half-width, and (z₀.₉₇₅ + z₀.₈₀)·SE with SE the bootstrap SD. Both are given in AP and as a share of the weather drop. The claim is "an interaction smaller than this cannot be detected", not "no evidence of an interaction".
- **Replay share.** share(N) = (replay − FT) / (adapt − FT) on reliable mAP, over the sampling seeds common to all three arms, paired by seed. CI: the shared test resamples, seeds redrawn as triples. Undefined where |adapt − FT| < δ = 0.005, for the point value and for each replicate. Reported at N ∈ {10, 100, 1000}; N = 0 is tabulated separately (FT is then the source model).
- **Against no fine-tuning.** adapt(N) − mAP(0) and replay(N) − mAP(0) are tabulated with CI, because FT at N ≤ 50 is below zero-shot.

**Claim rules, fixed before the revision runs are scored.**
- **Headline.** "Most of the gain is not adaptation" is supported at budget N when share(N) > 0.5 with the lower 95% bound above 0.5 in all three shifts, with three replay seeds. It is stated only for labeled budgets; at N = 0 the seed-0 shares are 0.17–0.55 and the claim is not made.
- **Matched passes.** With adapt − FT at N = 1,000 under matched labeled passes: if the upper 95% bound is below 0 and the estimate is ≤ −δ, "FT is ahead from N = 1,000" stands. Otherwise the crossover claim is withdrawn and the report says there is no meaningful difference at N = 1,000.

**Rules for the new runs (Phase 2 is not yet authorized; these apply once it is).**
- **Ceiling.** mAP_c := seed mean of the full-pool fine-tune at **E = 12** (the source model's epoch count), every other setting as `campaign.POLICY`, final update kept, seeds 0/1. It replaces the E = 3 ceiling as the recovery denominator **whatever the result**. Recovery stays unclipped. The E = 3 full-pool runs become the top rung of the ladder. Matched-pool (1,738) ceilings for lighting and weather follow the same rule with seeds 0/1.
- **Materially higher** := paired difference (E12 − E3) ≥ δ = 0.005 AP with the lower 95% bound above 0. Both threshold tables are published either way.
- **Bracket reading.** Ladder rungs stay at E = 3. Thresholds against the E3 ceiling are lower bounds; against the E12 ceiling they are upper bounds wherever large rungs are themselves undertrained. The pilot shows N = 1,000 saturated at E3 (0.2780, vs 0.2797 at E5 and 0.2772 at E10 on lighting val).
- **RQ5 matched passes.** `adapt` and `replay` at N = 1,000 with 3,000 updates (`--floor-updates 3000`), so labeled target image-passes equal FT's 3,000. Seeds 0/1/2 for both arms, same nested subsets. The original runs stay as the pre-registered result; the matched runs are the confound control.
- **Replay seeds.** `replay` at N = 10 and 100 gains seeds 1 and 2 at the original run length.
- **Scoring.** One new frozen plan over all new prediction sets, scored once, bootstrap 1,000 resamples seed 20261030 so they pair with the existing scored outputs. Existing sets are not rescored. This is the third opening of held-out labels (2026-11-03 Kaggle, 2026-11-06/07 local).

## 2026-11-21 (addendum): second seed for the E3 matched-pool ceilings

Written while the revision queue was running, before any revision run was scored. User approved 2026-11-21.

- **Added runs:** `fine-tune_lighting_N1738_s1` and `fine-tune_weather_N1738_s1` under the locked E = 3 policy (2,607 updates each), block `matched_pool_e3` of `scripts/local_revision_runs.py`. The revision plan therefore freezes 42 prediction sets, not 40.
- **Why:** the E12 matched-pool ceilings have seeds 0/1, the E3 ones had seed 0 only. At the 1,738 pool the seed also decides which images are drawn, so the paired E12 − E3 difference of the "materially higher" rule could be formed on one seed only, and the E3 matched-pool table (RQ4) rested on one seed. The gap was found by the independent review of the driver.
- **Unchanged:** every rule of the 2026-11-21 entry. Replay at N = 1,000 at the original 2,000-update length stays seed 0 only, so share(1,000) over three seeds exists only under matched passes; this is disclosed, not filled.

## 2026-11-21 (addendum 2): reporting rules made explicit after the analysis review

Written while the revision queue was running, before any revision run was scored. Six independent reviewers checked the new analysis code; these points were left to a rule decision. Each one narrows what may be claimed.

- **Headline rule, direction and budgets.** "Most of the gain is not adaptation" is evaluated at N = 10 and N = 100 only, and additionally requires adapt − FT > 0 in all three shifts. A share above 0.5 of a *deficit* (N = 1,000 today) is not evidence for the headline; it is reported in the matched-pass section.
- **Baseline of the share.** The share is defined against FT (2026-11-21). FT at N ≤ 50 is below the source model, so at N = 10 the same numbers read differently against the source model: there the replay arm gains about 0 and the whole of ALDI++'s small gain is distillation. Both readings are reported side by side (`rq5_replay_share.csv`, `rq5_vs_zero_shot.csv`), and the headline is worded as a statement about the advantage **over fine-tuning**.
- **Share with undefined replicates.** A share is quoted with its interval only when at most 5% of its bootstrap replicates are undefined (|gain| < δ). Otherwise it is reported as not determined, with the fraction.
- **Fit thresholds.** A fit-based N75/N90 is quoted only when it lies inside its ladder bracket (`fit_inside_ladder_bracket`). A CI bound flagged as censored is written "> largest region rung" (or "< smallest"), not as a number. Per-class fits that contradict a measured rung are not quoted.
- **Held-out check wording.** The log-linear fit on the three lowest region rungs over-predicts the larger lighting rungs by +0.003, +0.008, +0.013 recovery and the ceiling rung by +0.020. The plan's earlier "predicts to three digits" came from the anchored power form and is withdrawn.

## 2026-11-23: Kaggle run code removed; crash cause left alone

- **Removed from `main` (user, 2026-11-23, "if it affects nothing"):** `kaggle/` (9 kernels), `scripts/kaggle_materialize.py`, `docs/kaggle_training.md`, `docs/results_rq1_rq4_vi.md`. Nothing imported them; 151 tests pass and every entry point imports after the removal. They remain in the git history.
- **Kept:** `scripts/kaggle_source.py`. Despite its name it is not run code: it verifies the local raw files against the public Kaggle mirror (README setup step 2), which is data provenance. `docs/review_response_2026-10-20_kaggle_vi.md` and `docs/lecturer_update_2026-10-30_vi.md` are kept as records of correspondence; their pointers to the removed files are historical.
- **Still pending:** whether the report keeps one sentence disclosing that held-out labels were first opened by the Kaggle campaign on 2026-11-03. The notes keep it until the user decides.
- **Runaway `nvidia-smi dmon`** (likely cause of the 2026-11-21 crash): the user chose to leave it alone. The queue finished on 2026-11-22 20:44 without another crash.

## 2026-11-23: Revision plan frozen (42 sets), before any revision label was read

- **Runs:** all 42 finished 2026-11-22 20:44 (`runs/local/revision.log`). One machine crash on 2026-11-21 interrupted run 20; it resumed from `last.pt` at step 2,500 of 3,000 and the 19 finished runs were re-verified by hash.
- **Pre-freeze checklist passed for all 42 runs:** completed; update count equal to the pre-registered count (2,000 / 3,000 / 10,428 / 11,916 / 108,096 / 2,607); `test_labels_accessed` false; training and prediction implementation hashes equal to the original grids'; prediction file hashes equal to the manifests; 42 distinct checkpoints; the 12 fine-tune runs have loadable `weights.pt`; no leftover `last.pt`; no git change in the frozen or hashed files.
- **Plan:** `runs/local/revision/final_plan.json`, schema 3, 42 prediction sets, 84 scoring entries, bootstrap 1,000 resamples seed 20261030, sha256 `de135bccadbb11494901192af0950f45ae22d9f2b6ee173338666dec10b10282`. Bootstrap, split hashes, model contract and evaluation code are identical to `runs/local/campaign/final_plan.json`, so the new sets pair with the existing scored outputs.
- **Scoring:** once, into `runs/local/revision/scored/`. This is the third opening of held-out labels (2026-11-03 Kaggle, 2026-11-06/07 local), approved by the user on 2026-11-21 together with the runs.
- **Scored 2026-11-23, 08:53 to 10:28:** 84 of 84 entries, no error (`runs/local/revision/scoring.log`); every entry has a defined reliable metric and a finite bootstrap of 1,000 resamples with seed 20261030; `scored/frozen_plan.json` is identical to the frozen plan. The plan must not be scored again.

## 2026-11-23 (later): no mention of the Kaggle runs in public documents; provenance script renamed

- **User decision:** drop the Kaggle runs completely, without a disclosure sentence. The user informs the lecturer directly. The assistant had recommended keeping one sentence, because the same held-out labels were opened by that campaign on 2026-11-03; the user decided otherwise.
- **What it covers:** the report (`docs/bao_cao_ket_qua_vi.docx`, at its Phase 4 rewrite), `docs/results_local_rq1_rq4_vi.md`, `docs/results_rq5_vi.md` and `README.md`. All comparisons with, and references to, the Kaggle runs were removed from the two notes and the README on 2026-11-23.
- **Wording constraint that follows:** those documents must not claim that the held-out labels were opened only once in total. They say what is true of each plan: each frozen plan was scored exactly once (campaign, RQ5 grid, revision).
- **Not covered:** this log, `HANDOFF.md`, the plans and the two dated correspondence files (`docs/review_response_2026-10-20_kaggle_vi.md`, `docs/lecturer_update_2026-10-30_vi.md`) keep their history. The user confirmed on 2026-11-23 that the two correspondence files are not to be deleted. The Kaggle *mirror* as the source of the dataset is provenance and stays in the README and the EDA document.
- **Renamed (user request):** `scripts/kaggle_source.py` → `scripts/verify_source_mirror.py`. Nothing imported it. Its outputs keep their names (`data/raw/kaggle_verified/`, `kaggle_zip_directory.json`), because the built dataset and the README setup steps refer to those paths.

## 2026-11-23: Revision results, and reporting rules decided after they were seen

**Analysis.** `scripts/analyze_campaign.py` and `scripts/analyze_rq5.py` take `--revision-scored`. Outputs: `runs/local/campaign/analysis_rev2/`, `runs/local/rq5/analysis_rev2/`. 239 tests pass. Without the option both scripts reproduce `analysis_rev1/` byte for byte; with it, the 2026-11-03 tables and the 3-epoch full-pool rows are unchanged. Runs are told apart by `run.epochs` and `run.total_updates`, never by label. The code was written by two agents, reviewed by four and fixed by two; the coordinator re-ran the regression.

**Results under the rules of 2026-11-21, applied as written.**
- **Ceilings (reliable, paired E12 − E3, 95% CI):** lighting full pool +0.0029 (+0.0014 to +0.0041); compound +0.0005 (−0.0021 to +0.0037); weather full pool −0.0045 (−0.0067 to −0.0025); pool 1,738: lighting −0.0021 (−0.0031 to −0.0007), weather −0.0044 (−0.0067 to −0.0022). **None is materially higher** (δ = 0.005). Per class, only lighting traffic light (+0.0077) and traffic sign (+0.0053) at the full pool are.
- **mAP_c is the E12 mean by rule:** 0.2989 / 0.3310 / 0.2578 (lighting / weather / compound); denominators 0.0551 / 0.0109 / 0.0384.
- **Thresholds, reliable, full pool, E3 ceiling → E12 ceiling:** lighting N_BE 100 → 100, N75 5,000 → 5,000, N90 10,000 → 18,015 (the E3 full-pool rung, recovery 0.948). Compound N_BE 100 → 100, N75 1,000 → 1,000, N90 not determined within the pool → 1,738 (the E3 full-pool rung, recovery 0.986). Weather: see the reporting rule below.
- **Headline:** supported at N = 10 and N = 100 with three replay seeds. Shares against FT: N = 10: 0.875 (0.850–0.902) / 0.939 (0.910–0.962) / 0.835 (0.805–0.863); N = 100: 0.861 (0.782–0.905) / 0.898 (0.808–1.037) / 0.759 (0.619–0.853). Against the source model the N = 10 shares are 0.046, undefined, 0.071, and the N = 100 shares 0.867 / 0.804 / 0.803.
- **Matched passes at N = 1,000 (adapt − FT, three seeds):** lighting −0.0076 (−0.0091 to −0.0058) and compound −0.0074 (−0.0097 to −0.0047): "FT ahead from N = 1,000" stands. Weather −0.0021 (−0.0038 to +0.0002): withdrawn, no meaningful difference. adapt − replay under matched passes is −0.0012 / −0.0005 / −0.0016.

**Reporting rules decided after the results were seen (descriptive, not pre-registered).**
- **Recovery with undefined replicates.** Mirroring addendum 2: when more than 5% of the bootstrap replicates of a recovery curve are undefined (denominator at or below the 0.01 floor), its recovery values, band, thresholds and fit are reported as not determined, with the fraction. This applies to weather against the E12 ceiling (339 of 1,000 at the full pool; the 1,738 pool is undefined outright at 0.0099). Weather is then given in raw AP against both ceilings.
- **Two ceilings, no choice between them.** The 2026-11-21 rule made E12 the denominator "whatever the result" and required both tables. The write-up presents both side by side and says plainly that E12 is not higher on the small pools and is lower on weather, where the 3-epoch full-pool run exceeds the E12 "ceiling". No ceiling is picked as better after the fact.
- **RQ5 reference line.** The RQ5 tables and figures keep the 3-epoch full-pool mean as their reference, labelled "FT full pool, 3 epochs" when the revision runs are included, not "ceiling".
- **`quotable` stays as recorded** (the bootstrap draw of the default seed). The exact undefined fraction is reported next to it. The compound matched-pass share against FT sits at 4.2% against the 5% limit and is not part of the headline.
- **Seeds are strict:** each E12 ceiling needs exactly seeds 0 and 1, and lighting and weather need seed 0 (campaign) and seed 1 (revision) of the 3-epoch run at 1,738.
- **Rounding fix in the note:** three interval bounds of the recovery table in `docs/results_local_rq1_rq4_vi.md` were mis-rounded in the 2026-11-06 version (lighting N = 1,000 low 0.52, N = 5,000 high 0.82, N = 10,000 low 0.89). The table is now generated from `recovery_points.csv`. `docs/bao_cao_ket_qua_vi.docx` still carries the old bounds.
