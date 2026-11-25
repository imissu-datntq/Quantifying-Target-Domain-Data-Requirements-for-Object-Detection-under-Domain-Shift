# Result tables and figures

A tracked copy of the final analysis outputs, so that the report can be written from a clone. The originals are in `runs/local/campaign/analysis_rev2/` and `runs/local/rq5/analysis_rev2/`, which are not tracked. Copied on 2026-11-23; every file is byte-identical to its original.

The numbers are explained in [`../results_local_rq1_rq4_vi.md`](../results_local_rq1_rq4_vi.md), [`../results_rq5_vi.md`](../results_rq5_vi.md) and, for `direction1/`, [`../results_direction1_vi.md`](../results_direction1_vi.md). The rules behind them are in `.agent/DECISIONS.md` (2026-11-21 onwards).

## `campaign/`: fine-tuning curves, thresholds and ceilings

The column `analysis` separates the ceilings: `full-pool` and `matched-pool` use the 3-epoch ceiling, `full-pool-e12` and `matched-pool-e12` the 12-epoch ceiling.

| File | Content |
| --- | --- |
| `rq1_drop.csv` | Zero-shot drop per shift and class, with CI; 3-epoch ceiling and residual gap |
| `summary.json` | Key `interaction`: compound-vs-sum interaction with its detection limit; reliable thresholds, fits and denominators |
| `recovery_points.csv` | Recovery per rung with CI, and the count of undefined bootstrap replicates |
| `thresholds.csv` | N_BE, N75, N90: point, status, CI band, ladder bracket, fit estimate |
| `fits.csv`, `fit_holdout.csv` | Log-linear fit on the positive region and its held-out check |
| `recovery_denominator.csv` | Ceiling minus zero-shot, with CI and its ratio to the 0.01 floor |
| `ceiling_e12_vs_e3.csv` | Paired difference between the 12-epoch and the 3-epoch ceiling |
| `forgetting.csv`, `forgetting_e12.csv` | Change on the source test set after fine-tuning |
| `controls.csv` | COCO-initialized controls |
| `recovery_<shift>.png` | Recovery against the 3-epoch ceiling. The title only says "the full-pool ceiling": say "3-epoch" in the caption |
| `recovery_<shift>_e12.png` | Recovery against the 12-epoch ceiling, lighting and compound |

`recovery_weather_e12.png` is left out on purpose: weather against the 12-epoch ceiling is reported as not determined.

## `rq5/`: ALDI++, the replay control and fine-tuning

| File | Content |
| --- | --- |
| `rq5_replay_share.csv` | Share of ALDI++'s advantage over fine-tuning that replay reproduces (the headline) |
| `rq5_replay_share_vs_source.csv` | The same share measured from the source model |
| `rq5_vs_zero_shot.csv` | Gain of each arm over the source model |
| `rq5_matched_passes.csv`, `rq5_matched_passes_per_class.csv` | N = 1,000 with matched labeled passes |
| `rq5_replay_share_matched.csv` | Shares under matched passes |
| `rq5_delta.csv`, `rq5_delta_per_class.csv` | ALDI++ minus fine-tuning per budget; absolute mAP |
| `rq5_equivalent_budget.csv` | Label-equivalent budget N_eq |
| `rq5_attribution.csv` | Seed-0 split: adapt − replay and replay − fine-tuning |
| `rq5_forgetting.csv`, `rq5_forgetting_revision.csv` | Change on the source test set |
| `summary.json` | Headline check, matched-pass reading, crossover |
| `rq5_<shift>.png` | The three arms over the budget, with the matched-pass points |

## Refreshing the copy

Rerun the two analysis commands in the repository README ("Reproducing the Runs"), then copy the CSV, JSON and PNG files here again, except `recovery_weather_e12.png`.

## `direction1/`: instances, seed spread and source retention

Copied on 2026-11-25 from `runs/local/direction1/` (not tracked); every file is byte-identical to its original. Rules: `.agent/plans/active/2026-11-24-direction-1-analyses.md`; decision: `.agent/DECISIONS.md` (2026-11-25). Scripts: `scripts/analyze_instances.py`, `scripts/analyze_seed_spread.py`, `scripts/analyze_retention.py`.

| File | Content |
| --- | --- |
| `instances/instance_pool_density.csv` | Instances per image for each class in each train pool |
| `instances/instance_counts.csv`, `instances/instance_transitions.csv` | Instances and images of every run's subset; per-class ΔAP between rungs |
| `instances/instance_regression.csv`, `instances/instance_summary.json` | ΔAP on Δ(instances) and Δ(images): pooled, `within` (the reading), per class; log and raw forms |
| `instances/instance_recovery.csv`, `instances/instance_recovery.png` | Per-class recovery against images and against instances |
| `seed_spread/seed_spread.csv` | SD over sampling seeds per rung, with chi-square CI, finite-population factor, SD in recovery units |
| `seed_spread/seed_spread_floor.csv` | Training noise: SD of the two full-pool E = 3 seeds, per shift and pooled |
| `seed_spread/seed_spread_fit.csv`, `seed_spread/seed_spread_summary.json` | Power law (`power_fpc` primary, `power` uncorrected) and the variance model with a floor |
| `seed_spread/seed_spread_reliable.png`, `seed_spread/seed_spread_classes.png` | SD against N with the fits and the training-noise band |
| `retention/retention_points.csv` | Recovery (E = 3 and E = 12) and source retention per method and N |
| `retention/retention_paired.csv` | Paired differences against FT (or the source model at N = 0) and the dominance flags |
| `retention/retention_summary.json` | References, Pareto front, caveat |
| `retention/retention_<shift>.png` | Recovery against retention, Pareto front circled |
