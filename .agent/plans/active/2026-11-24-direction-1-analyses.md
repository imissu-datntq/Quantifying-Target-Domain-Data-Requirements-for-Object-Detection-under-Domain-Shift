# ExecPlan: direction 1 of the lecturer's follow-up (instances vs images, seed spread, source retention)

- **Status:** code written and reviewed on branch `alt-1`, not run on real results. Created 2026-11-24 (Asia/Saigon). The user asked for code first; running any of the three scripts on real results needs their approval ("lúc nào chạy thì báo để mình duyệt").
- **Inputs (retrained local results only):** `runs/local/campaign/scored/`, `runs/local/rq5/scored/`, `runs/local/revision/scored/`, and the train annotations under `data/bdd100k_revised/`. The Kaggle outputs (`runs/campaign/`, `runs/rq5/` on the Windows machine) are dropped and are never read, not even to try the code (user, 2026-11-24).
- **No new training, no new scoring, no held-out label read.** All three analyses read the existing one-shot scoring outputs (per-run `metrics.json` and `bootstrap_class_ap.npy`) and, for direction 1.1, the train annotation file.
- **Status of the rules:** written before any of these three analyses was run, but after the curves, thresholds and RQ5 tables had been seen. They are descriptive, not confirmatory, like the 2026-11-21 analysis rules. They go into `DECISIONS.md` once the user approves the run.

## Purpose and observable outcome

The lecturer's follow-up (2026-11-24) gave two directions. This machine takes direction 1, three analyses:

1. **Instances or images.** Pedestrian density is 13,170 / 9,804 = 1.34 per image in the source pool and 12,265 / 18,015 = 0.68 at night. Pedestrian recovering slowest at night (0.46) may come partly from fewer labeled pedestrians rather than visual difficulty. Regress the per-class ΔAP between consecutive rungs on Δ(instances of that class) and Δ(images). If the gain follows instances, thresholds in images are a BDD100K density artefact and the transferable quantity is instances per class.
2. **Seed spread against N.** Plot the SD over sampling seeds against N. A 1/√N decline would mean random sampling is close to optimal; a plateau would leave room for active selection. This answers "why not active learning" with numbers.
3. **Source retention as an axis.** For every (method, N): (recovery on the target, retention on the source) gives a trade-off surface. ALDI++/replay should dominate at small N; FT should win on the target but lose on the source at large N.

**Done means:** three scripts with unit tests on synthetic data, outputs defined below, and the commands to run them. Nothing is run until approved.

## Scope and non-goals

- New code only in `scripts/analyze_instances.py`, `scripts/analyze_seed_spread.py`, `scripts/analyze_retention.py` and their tests. `scripts/bdd_pipeline/` and `scripts/bdd_adapt/` are not touched (provenance hashes). The existing analysis scripts are imported, not edited.
- Not in scope: active-learning runs, new rungs, rescoring, report or paper text, direction 2.

## Shared conventions

- Load scored directories with `analyze_campaign.load`; tell runs apart by `metrics["run"]` fields (stage, domain, budget, seed, epochs, total_updates), never by folder names.
- Fine-tune ladder = campaign `fine-tune` runs with `run.domain == eval domain`, the locked epoch count (`POLICY["epochs"]`, 3), budgets in `LADDER` + `EXTRA_RUNGS`, plus budget `full` (the E = 3 full-pool runs, seeds 0 and 1). The matched-pool runs at 1,738 of lighting and weather are not rungs (as in `analyze_campaign.py`). N = 0 is the source model scored on the shift.
- Primary metric reliable mAP50:95; per-class AP50:95 for the four reliable classes. The ten-class mean is not used.
- Intervals (sections 1 and 3) follow the project's hierarchical bootstrap: replicate t takes test resample t of every run (all scored sets share the bootstrap seed 20261030 and the resample count; checked), and the sampling seeds are redrawn with replacement. Every interval is a 95% percentile interval over the defined replicates; the count of undefined replicates is reported next to it.
- Section 2 is the exception: its SD, floor and slope intervals are analytic (χ², normal), and the variance model's intervals come from a seeded parametric bootstrap (see there).
- A table without rows writes no file; outputs go to a fresh directory, suggested `runs/local/direction1/<analysis>/`.
- A recovery denominator must **exceed** the minimum 0.01 (strict, as `bdd_pipeline.recovery`); one exactly at the floor is undefined.

## 1. Instances or images (`scripts/analyze_instances.py`)

**Counts.** The train annotation file is streamed once (`stream_records`), boxes pass through `sanitize_labels` (the boxes training saw), and each image gets a count per reliable class. Only train-split names of the four domain pools are kept; no other annotation file is opened. A run's image set is rebuilt with the training code's own call, `nested_rows(domain_rows(manifests, "train", d), budget, seed)`, so it is exactly the set that run trained on.

- `instance_pool_density.csv`: per domain (source and the three targets), images, instances per reliable class and per image. This should reproduce the lecturer's 13,170 / 9,804 and 12,265 / 18,015 up to sanitation; the script prints both.
- `instance_counts.csv`: per shift, seed and rung: images, and per class instances and positive images.

**Observations.** For each shift and sampling seed, the rungs that seed has (ladder, then the full pool for seeds 0 and 1) form a chain; each pair of consecutive rungs is a transition. One observation = (shift, seed, transition, class): ΔAP = AP_c(to) − AP_c(from), ΔN = images added, ΔI = instances of class c added, Δln N = ln(N_to/N_from), Δln I = ln(I_to/I_from). The 0 → first-rung transition is tabulated but left out of every regression: Δln I is undefined from 0, and the N = 10 rung is dominated by the negative transfer under the 500-update floor (DECISIONS 2026-11-05, N10 flag). Observations with I_from = 0 are left out of the log form and counted. The update counts of both rungs are kept as columns, because the 500-update floor changes the schedule along the ladder.

Expected size: per class 20 (compound) + 20 (weather) + 29 (lighting) = 69 transitions, 276 observations; the lecturer estimated about 200.

**Specifications** (`instance_regression.csv`), each fitted by OLS in two forms, **log** (Δln I, Δln N) and **raw** (ΔI, ΔN, the lecturer's literal form):

| spec | model | what it asks |
|---|---|---|
| `images` | ΔAP ~ 1 + x_N | images alone |
| `instances` | ΔAP ~ 1 + x_I | instances alone |
| `both` | ΔAP ~ 1 + x_I + x_N | the lecturer's regression |
| `both+class` | ΔAP ~ class + x_I + x_N | the same, class intercepts |
| `within` | ΔAP ~ cell + x_I, cell = shift × class × transition | at a fixed image step, do seeds that drew more instances gain more? |
| `both` per class | ΔAP ~ 1 + x_I + x_N, one class | per-class version |

Columns: n_obs, coefficients of x_I and x_N with CI, standardized coefficients, R² (within-R² for `within`), the correlation of the two regressors after the specification's intercepts are partialled out (plain Pearson for `both`, within class for `both+class`) and VIF = 1/(1 − r²) for two-regressor specs, and the leave-one-shift-out RMSE (fit on two shifts, predict the third; not for `within`). The bootstrap redraws seed chains within each shift with one draw per replicate shared by all rows, so the rows are compared on the same replicates.

**Why `within` is the primary reading.** Nested subsets keep a class's density almost constant along a chain, so Δln I ≈ Δln N up to sampling noise: in the log form the two regressors are nearly collinear (VIF reported), and in the raw form they are separated mainly by density differences between classes, which are confounded with how hard each class is. Holding shift, class and image step fixed leaves the seed-to-seed variation in instances drawn, which is not confounded with class or image count.

**Reading (descriptive, `instance_summary.json`):** "at a fixed image step, more instances of a class give more AP for that class" when the `within` log coefficient of x_I has a 95% interval above 0; "no detectable instance effect at a fixed image step" when it covers 0; "at a fixed image step, more instances of a class go with less AP gain for that class (no support for the density explanation)" when it is below 0. The pooled specs, their VIF and the out-of-shift RMSE of `instances` vs `images` are reported next to it, without a verdict.

**Figure** `instance_recovery.png` (+ `instance_recovery.csv`): per-class recovery (seed means, E = 3 full-pool reference, minimum denominator 0.01) against images (top row) and against instances of that class (bottom row), one line per shift. If the night pedestrian curve lines up with the other shifts on the instance axis but not on the image axis, the density explanation is visible. Curves with an undefined denominator are left out and named in the panel.

## 2. Seed spread against N (`scripts/analyze_seed_spread.py`)

**Spread.** Per shift, key (reliable and the four classes) and ladder rung: the sample SD of AP over the sampling seeds (ddof 1), its 95% interval from the χ² distribution with ν = seeds − 1 degrees of freedom, the SD in recovery units (÷ the E = 3 denominator, when it exceeds 0.01) and the finite-population factor (M − N)/(M − 1), M = pool size.

**Training-noise floor.** The two E = 3 full-pool runs train on the same images, so their spread is training noise only (order, flips, GPU nondeterminism). Per shift and key, SD with ν = 1, and pooled over the three shifts (mean variance, ν = 3). At a ladder rung the seed changes the subset **and** the training randomness, so SD² ≈ sampling + training noise. That training noise does not depend on N is an assumption, stated with the result.

**Fits** (`seed_spread_fit.csv`), on the ladder rungs (not the full pool), per shift and pooled over shifts (shift intercepts, one slope):

- **Power law:** ln s = a + b ln N by weighted least squares with the sampling variance of ln s known from ν (trigamma(ν/2)/4) and the bias of ln s removed (digamma), slope b with a normal interval, and χ²/df of the standardized residuals as a misfit flag. Flags: `slope_covers_minus_half`, `slope_below_zero`. Two rows: **`power_fpc` (primary)** also subtracts ½ ln(fpc) from the response, because the subsets are drawn without replacement from a finite pool: pure random sampling then gives s ∝ √(fpc/N), and only after this correction is its slope −½ (uncorrected it is about −0.54 to −0.58 on these ladders). `power` is the uncorrected 1/√N form of the review, for reference. The interval is approximate: it treats the rungs as independent, while the rungs of one seed are nested subsets (synthetic nested data: about 89% coverage instead of 95%).
- **Variance model** (per shift only, no pooled row): s² = σ²_floor + c · (1/N − 1/M), the finite-population form of "sampling variance ∝ 1/N plus a floor", by weighted least squares (weights ν / (2 ŝ⁴), one refinement; the OLS-fitted ŝ² is clipped below at 1% of the mean observed s² before it forms the weights and the centre of the parametric redraws). Reported: σ_floor, c, the budget N* where both parts are equal, and σ_floor against the measured training-noise floor. Intervals for σ_floor, c and N* by a parametric bootstrap (1,000 draws of each rung's s² as ŝ² χ²_ν/ν around the fit, refitted; same independence caveat). `floor_above_training_noise` = the lower bound of σ_floor is above the upper bound of the pooled training-noise SD.

**Reading (descriptive).** A `power_fpc` slope interval that covers −1/2 and lies below 0 means the spread shrinks as averaging over randomly drawn images predicts. A fitted floor at the level of the training noise means the spread left at large N is training noise, which no choice of images removes. A floor clearly above the training noise (`floor_above_training_noise`) means some subset-dependent spread remains: room for selection. **Limit, stated with the result:** the SD over random subsets says how far one random draw strays from another. It does not bound what an informed selector can reach, because active selection can beat every random subset. The plot answers "does the choice of random subset matter", not "is active learning useless".

**Figures:** `seed_spread_reliable.png` (one panel per shift: SD against N on log axes with intervals, the fitted power law, a −1/2 reference line, the training-noise band and the full-pool point) and `seed_spread_classes.png` (shift × class grid).

## 3. Source retention (`scripts/analyze_retention.py`)

**Points** (shift, method, N), each a seed mean of runs scored on both the target test set and the source test set:

| method | N | seeds | from |
|---|---|---|---|
| source model | 0 | – | campaign |
| FT | ladder rungs, full pool (E = 3) | 0/1/2; full pool 0/1 | campaign |
| FT, 12 epochs | full pool | 0/1 | revision |
| ALDI++ adapt | 0, 10, 100, 1,000 (+5,000 lighting) | 0/1 (N = 0), 0/1/2 | RQ5 grid |
| replay | 0 (the source-only replay run), 10, 100, 1,000 | 0; +1/2 at N = 10, 100 with the revision | RQ5 grid + revision |
| adapt / replay, matched passes | 1,000 (3,000 updates) | 0/1/2 | revision |

Without `--revision-scored` the revision rows are absent and the rest is unchanged. Revision runs are classified by the existing functions (`analyze_campaign.read_revision`, `analyze_rq5.split_revision` / `merge_replay_seeds`), which also stop on duplicates and on test resamples that are not shared.

**Axes.** Target: gain = mAP_target − mAP(0) and recovery = gain / (FT full pool E = 3 mean − mAP(0)), the reference the RQ5 tables use ("FT full pool, 3 epochs"); with the revision also against the 12-epoch mean, reported as not determined where more than 5% of its drawn replicates are undefined (weather), with the exact expected fraction over every redraw of the reference seeds next to it (as `analyze_rq5.py` does for the matched-pass shares). Source: retention = mAP_source(model) / mAP_source(source model), and the change mAP_source(model) − mAP_source(source model).

**Intervals.** Replicate t takes target resample t and source resample t of every run; a point's seeds are redrawn once and used for both axes; the reference seeds are redrawn once per replicate and shift for the denominator, shared by every point of the shift. The point that is itself the reference (FT full pool for E = 3, FT 12 epochs for E = 12) reuses that draw, so its recovery is 1 in every defined replicate, as the rung at N = pool in `analyze_campaign.py`.

**Pareto front** (`retention_points.csv`, column `pareto`): per shift, a point is on the front when no other point has a target gain and a retention at least as high, one of them strictly. Point estimates only, descriptive. Gain and recovery order the points the same way, so the front does not depend on the denominator.

**Paired comparisons** (`retention_paired.csv`), per shift and N, paired by sampling seed over the seeds both arms have (seed pairs redrawn together): adapt − FT, replay − FT, adapt_matched − FT, replay_matched − FT, and at N = 0 adapt − source model, replay − source model. Each with Δ target mAP (and in recovery units), Δ retention, intervals, and flags `dominates` (both lower bounds above 0), `dominated` (both upper bounds below 0), `trade_off` (both intervals exclude 0, opposite signs). The lecturer's two expectations are rows of this table: dominance at N = 10 and 100, a trade-off at N ≥ 1,000.

**Caveat, stated with the result:** adapt and replay keep one labeled source image in every update (1 source + 1 target), FT trains on target images only. High retention for those arms is source rehearsal by design; the table measures the trade-off of each recipe as run, not of adaptation alone. The replay arm separates the two, as in RQ5.

**Figure** `retention_<shift>.png`: x = recovery (FT full pool, 3 epochs, as reference), y = retention, one line per method through its budgets with N labels, the source model at (0, 1), Pareto points outlined, a dashed line at retention 1.

## Ordered work

1. Branch `alt-1` from `team/main` (done 2026-11-24).
2. This plan (done).
3. The three scripts and their unit tests, synthetic data only; the full suite passes.
4. Independent review of each script against this plan, findings verified and fixed.
5. Commit on `alt-1`. Do not push or run without the user.
6. **Approval gate:** the user approves (a) these rules and (b) running the scripts on `runs/local/...` (on the Linux machine, or after copying the three `scored/` directories here). Then: run, check, record the rules and results in `DECISIONS.md`, write a short Vietnamese result note.

## Validation

- `python -m unittest discover -s tests` passes (existing and new tests).
- Each new test builds synthetic scored trees in a temporary directory; nothing reads `runs/` or `data/`.
- Tests pin down: the subset of a run equals `nested_rows` of the training code; a 0 → N transition never enters a regression; OLS recovers known coefficients; `within` absorbs x_N; the χ² interval and the digamma/trigamma terms against known values; the power-law slope of an exact 1/√N spread is −0.5; Pareto flags on a hand-built example; paired comparisons use common seeds only and the same seed draw on both axes.

## Commands (for after approval; from `scripts/`)

```
python analyze_instances.py --scored ../runs/local/campaign/scored --dataset-root ../data/bdd100k_revised --output-dir ../runs/local/direction1/instances
python analyze_seed_spread.py --scored ../runs/local/campaign/scored --output-dir ../runs/local/direction1/seed_spread
python analyze_retention.py --ft-scored ../runs/local/campaign/scored --da-scored ../runs/local/rq5/scored --revision-scored ../runs/local/revision/scored --output-dir ../runs/local/direction1/retention
```

On the Linux machine use `~/venvs/cv-final/bin/python`. All three are CPU-only and read existing files; `analyze_instances.py` streams the 239 MB train annotation file once.

## Progress

- 2026-11-24: branch `alt-1` created from `team/main` (`ef354cf`); plan written; code in progress.
- 2026-11-25: the three scripts and tests written by a workflow (one implementer per script, two independent reviewers per script, a verifier that checked each finding before fixing). Decisions taken after that review and written into this plan: the finite-pool-corrected power law as the primary slope; the variance model per shift only, with parametric intervals and the `floor_above_training_noise` rule; the slope interval called approximate (nested rungs); a reading for a negative `within` coefficient; partialled collinearity for `both+class`; the expected undefined fraction of the 12-epoch reference; a strict minimum denominator everywhere. New tests: 27 (instances), 35 (seed spread), 24 (retention), synthetic only.
- Full suite on this Windows machine: all pass except `tests/test_revision_runs.py`, which cannot import `fcntl` on Windows (it imports `scripts/local_revision_runs.py`, Linux-only); unrelated to this work. `python -m unittest tests/<file>.py` does not work here because a `tests` package in the user site-packages shadows the folder; use `python -m unittest discover -s tests -p <file>`.
- 2026-11-25: a second, read-only verification workflow (one skeptic per script, one plan-vs-code critic) found no defect in the instances or retention code; it found wording gaps (seed-spread docstring on resampling, plan wording of the strict floor, the clip rule and the shared reference draw not written in the plan) and a missing test of the `--seed` wiring. All fixed.
