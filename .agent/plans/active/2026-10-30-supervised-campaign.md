# ExecPlan: supervised campaign (RQ1–RQ4)

Status: active. Created 2026-10-30 (Asia/Saigon). The user approved this plan the same day.

## Progress

- 2026-10-30, Phase 0:
  - Memory notes saved.
  - `CLAUDE.md` rewritten to match the current protocol.
  - Local labels in `data/labels_raw/` match the Kaggle-mirror SHA256 exactly. Train is `c1998792e385fd79213187576ca07d5249563d0ab83ae976688ad60668baca9e`; val is `c835be652f0c002897be51e74deedd16a1dcee0b7230f19c0456fba772f694af`.
  - `pytest tests`: 18/18 pass, after installing `Markdown` from `requirements-eda.txt`.
  - The lecturer update is drafted in `docs/lecturer_update_2026-10-30_vi.md`.
- **Unblocked, 2026-10-30.** The user added the `kizhowl` key; select it with `KAGGLE_API_TOKEN` (see memory `kaggle-accounts`). `ngdminh31` is not part of this project.
  - The protocol dataset was downloaded, and all 7 protocol files pass their SHA256 check.
  - Its code copy matches this repo apart from line endings and the out-of-scope work that was removed.
  - `data/bdd100k_revised` was materialized locally with hardlinks to `data/bdd100k/bdd100k/images/100k`. All 79,863 images were rehashed against the manifests and pass.
  - Pools 9,804 / 18,015 / 1,985 / 1,738, final test 2,650 / 4,869 / 933 / 756, and selection is refused for weather/compound.
- **Found: the teammate already ran a two-round schedule pilot.** Kernels `bdd100k-gate-schedule` and `bdd100k-schedule-pilot`, 2026-10-20. It is missing from the response docs. The outputs are in the scratchpad under `kout/`.
  - Setup: lighting val (all 3,274 images), seed 0, batch 2, AMP.
  - Metric: reliable mAP50:95. Zero-shot N0 = 0.249.

| run | LR | updates | val reliable |
|---|---|---|---|
| N100 S1000 | 5e-3 | 1,500 | 0.211 |
| N100 S2000 | 5e-3 | 2,000 | 0.203 |
| N100 S4000 | 5e-3 | 4,000 | 0.190 |
| N1000 E10 | 5e-3 | 5,000 | 0.272 |
| N1000 E20 | 5e-3 | 10,000 | 0.250 |
| N100 S500 | 1e-3 | 500 | 0.254 |
| N100 S500 | 5e-4 | 500 | 0.255 |
| N100 S1000 | 1e-3 | 1,000 | 0.239 |
| N1000 E5 (S500) | 1e-3 | 2,500 | 0.278 |
| N1000 E10 (S500) | 1e-3 | 5,000 | 0.277 |

  - At LR 5e-3, N100 lands *below* zero-shot, which would make Recovery negative.
  - Every comparison favours fewer updates, and none of the tested settings reaches a peak: the best setting is always the shortest one tried.

- 2026-10-30, Phase 1: round-3 pilot pushed as kernel `kizhowl/bdd100k-schedule-pilot3`. The decision rule was written to DECISIONS.md before the push.
- 2026-10-30, Phase 2 code done, 28/28 tests pass:
  - `BDDImages` and `predict_frcnn.py` (label-free);
  - schema-3 freeze and scoring, plus `fast_bootstrap_class_ap`, which matches the exact bootstrap and is about 3× faster;
  - `recovery.py`: hierarchical bootstrap band, threshold intervals, ladder bracket, power fit, paired differences;
  - `bdd_pipeline/campaign.py`, `campaign_grid.py` (87 runs), `analyze_campaign.py`, `kaggle/bdd_campaign.py`.
  - CPU end-to-end smoke on real data: prediction opened no annotation file, and freezing rejects smoke/limited predictions.
  - Grid estimate: about 28 T4-h at S500/E5, versus 166 h at the old defaults.
- 2026-10-30, Phase 1 done: round 3 was scored by the pre-registered rule, which locked S=500, E=2, LR 1e-3 (`DECISIONS.md`).
  - E2 is the lowest value tested and sits within δ = 0.005 of the best; this must be disclosed.
  - N10 lands below zero-shot; flag it in the write-up.
  - The grid estimate is now about 12.6 T4-h of training.
- 2026-10-30, Phase 3 started:
  - The protocol dataset has a new version with the campaign code.
  - `scripts/zero_shot_gap.py` and `benchmark_detectors.py` were synced from the Kaggle copy. The teammate's snowy probe was missing from this repo.
  - `kizhowl/bdd100k-campaign` v1 was pushed with 54 runs: zero-shot, compound, weather, controls.
- 2026-11-02:
  - Grid complete: 87/87 runs.
  - Plan frozen (`runs/campaign/final_plan.json`).
  - Scoring made safe to resume after an interruption (`--resume-interrupted`, DECISIONS 2026-11-02).
- 2026-11-03: Phase 4 done.
  - Scored once, 176 entries.
  - Analysis is in `runs/campaign/analysis/` and the report is `docs/results_rq1_rq4_vi.md`.
  - Source-test sanity check: 0.343. Recovery at the full pool is 1.
  - This plan can move to `completed/` once the user has reviewed the report.
- **Rule:** all `bdd_pipeline/` code is frozen from the first campaign push, because resume checks implementation hashes.

## Corrections to the plan text below

- The aligned proposal's primary thresholds are **N_BE, N75 and N90**, with N95 as secondary. Wherever "N90/N95" appears below, read it that way. `recovery.py` already implements N_BE/N75/N90/N95.
- The aligned proposal (§10) fixes 3 sampling seeds for the non-ceiling rungs. The pilot only checks that 3 seeds are enough. It does not choose between 3 and 5 without a new decision.
- ~~B_eff = 4~~. The existing pilot used batch 2, which matches the proposal's B_eff = 2. Keep 2.

# Original plan

## Context

On 16/10 the lecturer (Nguyễn Mạnh Toàn) raised 6 points and closed with *"Các em sửa lại như bên trên rồi tiến hành run"*: fix these, then start the runs. The teammate fixed all 6 (`docs/review_response_2026-10-19_vi.md`). On Kaggle they then:
- benchmarked the T4: 6.4 img/s at batch 4 with AMP;
- trained the source model: 12 epochs, source-val reliable mAP50:95 = 0.352;
- passed the official zero-shot gate (`docs/review_response_2026-10-20_kaggle_vi.md`): weather gap 0.031 (CI 0.023–0.040), lighting 0.099, compound 0.129.

Neither response has been sent to the lecturer yet. What's left is the run campaign itself, and the training schedule still has to be locked before it starts.

**What this copy of the repo is missing.** `data/` is empty, `runs/` doesn't exist, and `D:/CV` no longer exists.

**`CLAUDE.md` is out of date.** It says "no code yet", points at `D:/CV`, and describes an older design: rainy+snowy pooled, and val used as the test set. The protocol actually in force is `docs/proposal_dataset_aligned.docx` plus `.agent/DECISIONS.md`:
- 55,000 / 14,863 / 10,000 split;
- final test sets: source 2,650, lighting 4,869, weather 933, compound 756;
- train pools: source 9,804, lighting 18,015, weather 1,985, compound 1,738;
- reliable roster: pedestrian, car, traffic light, traffic sign;
- target runs keep their last update, with no selection.

**User decisions (2026-10-30):**
- Data and labels come from the local BDD100K copy in `D:/BDD100K-weather-classification/data`, which the user is moving into this folder. This means staying on the 2018 labels, with no det_20 migration.
- Compute is Kaggle. **Ask before running anything on the local RTX 5060.**
- The lecturer update is a step in this plan.

## Phase 0: Housekeeping (no GPU)

1. **Save memory notes** in this project's memory folder, then add their lines to `MEMORY.md`:
   - a feedback note: Kaggle is the default compute; ask before using the 5060;
   - a project note: the dataset source is the local BDD100K copy, moving into `data/`, on the 2018 labels.
2. **Rewrite `CLAUDE.md`** to match the current protocol. Keep it short and point to `AGENTS.md`, `.agent/` and the docx; don't duplicate them. Fix the stale paths (`D:/CV` becomes `data/`).
3. **Bring in the protocol files.**
   - Download the private Kaggle dataset `kizhowl/bdd100k-revised-protocol` (annotations, manifests, `expected_sha256.json`) into `data/bdd100k_revised/`. Get the source `best.pt` and the gate summary JSON from the `bdd100k-source-train` and `bdd100k-gate-schedule` kernel outputs.
   - Once the user has moved the data, run SHA256 on the local `labels_raw/bdd100k_labels_images_{train,val}.json` and compare with the hashes recorded in point 5 of the 10-19 response (`c1998792…a9e` / `c835be65…4af`).
     - If they match, the local copy and the mirror are the same 2018 release.
     - Then run `python scripts/label_provenance.py compare --official <local JSONs>` on the frozen 400-record sample.
     - Report honestly: this proves the copies are identical. It proves the labels are Berkeley's official release only if the local copy came from the official portal.
   - Run `python -m pytest tests` (18 tests) and `train_frcnn.py --dry-run`.
4. **Draft the Zalo message to the lecturer** (Vietnamese, short) for the user to send. It covers:
   - the 6 points, each with a one-line status;
   - the weather gate result, staying in the study with CI and minimum denominator 0.01;
   - smoke vs mock test answered: the smoke test only checks the pipeline. Faster R-CNN was chosen because of RQ5, not because of the smoke test;
   - the run plan below, with its estimated time.

## Phase 1: Lighting val pilot to lock the schedule (Kaggle, about 10 GPU-h)

This uses lighting val only (3,274 images). It never touches test, or weather/compound val.

**Why it comes first.** E (the number of epochs) sets the cost of the whole grid:

| E | image-passes | T4 GPU-h | weeks at 30 h |
|---|---|---|---|
| 30 | about 3.8M | about 165 | 6 or more |
| 12 | about 1.8M | about 80 | about 3 |

The lighting ceiling alone at E=30 is 540k passes, or 23 h.

**Fix these to match the source run:** batch 4, accumulation 1, AMP, warmup 500 then cosine. The defaults in `training.py` (batch 1 × acc 2) must not be used.

**Pilot runs (seed 0), with `--eval-every-updates` on a fixed 1,000-image subset of lighting val:**

| N | length | LR |
|---|---|---|
| 100 | 4,000 updates | 0.005 and 0.0025 |
| 2,000 | 40 epochs | 0.005 and 0.0025 |
| 10 and 100 | final update only | best LR from above; 2 more seeds |

The last row measures seed variance, which decides the number of seeds for the main run (3 or 5).

**Decision rules, written into `DECISIONS.md` before the pilot results are read:**
- **S:** the smallest number of updates at which N=100 val reaches 95% of its peak.
- **E:** the smallest number of epochs at which N=2,000 val is within 0.005 AP of its peak.
- **LR:** the one with the higher val at both N.

**Lock S/E/LR/seeds into `DECISIONS.md` before any target run.** The pilot is done in a new kernel, `kaggle/bdd_schedule_pilot.py`, modelled on `kaggle/bdd_source_train.py`, which already resumes across sessions.

## Phase 2: Campaign infrastructure (CPU, in parallel with Phase 1)

1. **Grid manifest.** Add `scripts/campaign_grid.py`. It generates one row per run (shift, N, seed, stage, init) from the locked policy, and writes `runs/campaign/grid.csv` with a status column. The grid:
   - ladder: 10, 25, 50, 100, 250, 500, 1,000, then `full`;
   - lighting adds 2,000, 5,000 and 10,000;
   - ceilings use 2 seeds;
   - matched-pool ceilings run lighting and weather at `--budget 1738`;
   - the control arm uses `--stage control --initialization coco` at N=100 and N=1,000 for all 3 shifts.
2. **Queue-runner kernel.** Add `kaggle/bdd_campaign.py`. It takes a slice of the grid, runs each row with `train_frcnn.py`, resumes when Kaggle's 12 h session limit cuts a run, and puts results in the output.
3. **Inference without label access.** At the end of each run, write predictions for the target final-test images and the source final-test images (test images only, never test labels). Record the prediction-file hashes in the run metadata, then strip the checkpoint down to its weights to keep `/kaggle/working` under 20 GB.
   - This needs a predict-only mode in `scripts/bdd_pipeline/evaluation.py:predict` / `evaluate_frcnn.py` that doesn't build COCO ground truth. Reuse the existing `predict()`.
   - `freeze_evaluation_plan.py` and `evaluate_final` then freeze and score **prediction files** (hash-checked, with the one-shot marker kept) instead of reloading every checkpoint. Opening test becomes a single, cheap CPU pass.
4. **Fill the analysis gaps against proposal §7.3** in `scripts/bdd_pipeline/recovery.py`. It already has `recovery`, `sustained_threshold`, `recovery_table` and `curve_summary` (seed mean ± SD). It needs:
   - image-level bootstrap of test mAP. Reuse `evaluation.py:bootstrap_class_ap`; the same image indices must be used across all runs of a shift;
   - propagation of the spread across the 2 ceiling seeds into the denominator;
   - N90/N95 intervals from where the CI band crosses the threshold, plus the bracket between ladder rungs;
   - a fit of `1 − aN^(−b)` passing through 1 at the full pool, reported with a bootstrap CI and goodness of fit;
   - matched-pool thresholds (RQ4);
   - drop tables (RQ1), forgetting on the source test set, and compound vs the sum of lighting + weather.
   - Each gets a unit test on synthetic records in `tests/`.

## Phase 3: Main run (Kaggle, about 3 to 6 weeks depending on E)

Run in cost order, so a complete curve exists early:
1. **compound:** full ladder × seeds, plus 2 ceilings. The smallest pool and the largest gap.
2. **weather:** full ladder plus ceilings, plus 2 matched-pool runs.
3. **control arm.**
4. **lighting:** ladder up to 1,000 first, then 2,000, 5,000 and 10,000, then 2 ceilings. The longest runs, split across sessions.

Check `kaggle quota` before each push, since the quota is shared. Update `grid.csv` and `.agent/HANDOFF.md` after each batch. If the queue falls badly behind, ask the user about adding the 5060 instead of turning it on unilaterally.

## Phase 4: Open test once, analyse, write up RQ1–RQ4

Do this after every run in the grid is complete:
1. Freeze one plan covering all prediction files, then score once. The marker is one-shot.
2. Run the Phase 2 analysis. Produce: recovery curves (reliable mean plus 4 per-class curves), N90/N95 with intervals, matched-pool comparison, forgetting, and the 10-class sensitivity.
3. Write a Vietnamese results report, `docs/results_rq1_rq4_vi.md`, then a message to the lecturer.

## Phase 5: RQ5, after the curves exist

Pick one public DA method that runs on torchvision Faster R-CNN R50-FPN. Unsupervised gives a horizontal line on the plot; semi-supervised gives a second curve. Plan it in a separate ExecPlan.

## Conventions while working

- The teammate's ExecPlan goes in `.agent/plans/active/2026-10-30-supervised-campaign.md`, per `.agent/PLANS.md`. Durable decisions go in `.agent/DECISIONS.md`.
- The project hasn't been through the ICM restructure. Keep the current layout and suggest the restructure line once the campaign has settled.
- Never commit `data/`, `runs/` or weights. Never use weather/compound val for selection.

## Verification

- **Phase 0:** pytest passes 18 tests or more. The dry-run reports pools of 9,804 / 18,015 / 1,985 / 1,738 and test sets of 2,650 / 4,869 / 933 / 756. The local label SHA256 result is recorded.
- **Phase 1:** the pilot's val curves are saved under `runs/schedule_pilot/`, and S/E/LR are written into `DECISIONS.md` with a link to the evidence.
- **Phase 2:** the new tests pass. On synthetic data, Recovery equals 1 at `full`, thresholds are sustained, and the bootstrap is reproducible by seed. A small grid slice (`--smoke`) runs end to end in the bootstrap-check kernel (CPU) without reading test labels.
- **Phase 3:** every run has completion metadata, `amp_step_skipped` stays low, and the prediction-file hashes are in the metadata.
- **Phase 4:** the source model on the source test set lands near 0.35 (sanity check). Recovery at `full` is exactly 1 in every shift.
