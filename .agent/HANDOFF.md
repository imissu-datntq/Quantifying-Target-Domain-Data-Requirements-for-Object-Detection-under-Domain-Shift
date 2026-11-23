# Handoff

Updated 2026-11-23, Asia/Saigon.

## 2026-11-23: revision scored and re-analysed; the report is the user's next step (latest)

- **Phase 2 is complete:** all 42 revision runs finished 2026-11-22 20:44, no second crash. The pre-freeze checklist passed for every run (`DECISIONS.md` 2026-11-23).
- **Plan frozen:** `runs/local/revision/final_plan.json`, 42 sets, 84 scoring entries, sha256 `de135bcc…b10282`.
- **One-shot scoring DONE 2026-11-23 10:28:** 84 of 84 entries in `runs/local/revision/scored/`, no error. Never score this plan again.
- **Re-analysis DONE 2026-11-23** (Phase 3c): both scripts take `--revision-scored`; outputs in `runs/local/campaign/analysis_rev2/` and `runs/local/rq5/analysis_rev2/`; 239 tests pass; results and the reporting rules decided afterwards are in `DECISIONS.md` (last 2026-11-23 entry). Commands, from `scripts/`:
  `~/venvs/cv-final/bin/python analyze_campaign.py --scored ../runs/local/campaign/scored --revision-scored ../runs/local/revision/scored --output-dir ../runs/local/campaign/analysis_rev2`
  `~/venvs/cv-final/bin/python analyze_rq5.py --ft-scored ../runs/local/campaign/scored --da-scored ../runs/local/rq5/scored --revision-scored ../runs/local/revision/scored --output-dir ../runs/local/rq5/analysis_rev2`
- **Headline results:** no 12-epoch ceiling is materially higher than its 3-epoch one; the headline claim is supported at N = 10 and N = 100 with three replay seeds; under matched passes fine-tuning stays ahead at N = 1,000 on lighting and compound and the claim is withdrawn on weather.
- **The two result notes are updated** (`docs/results_local_rq1_rq4_vi.md`, `docs/results_rq5_vi.md`); their tables are generated from the CSVs. `docs/bao_cao_ket_qua_vi.docx` is NOT updated: the user writes the report themselves from a template. It still has the Kaggle section, the old thresholds and three mis-rounded interval bounds.
- **For the report (the user's next step). Take every number from the two notes or from `analysis_rev2/`, never from `docs/bao_cao_ket_qua_vi.docx`.** That file is the 2026-11-08 version and differs in: the Kaggle section and comparisons; "test labels opened once" wording; the power fit and N95; the RQ4 table (now two ceiling seeds); the N90 cells; three recovery bounds (lighting N = 1,000 low 0,53 → 0,52; N = 5,000 high 0,83 → 0,82; N = 10,000 low 0,90 → 0,89); and five RQ5 values (compound Δ(100) upper 0,020 → 0,019; ALDI++ compound N = 1,000 mAP 0,241 → 0,240, hence the DA rise 0,003–0,010; N_eq compound N = 1,000 lower 234 → 233; seed-0 adapt − replay lighting N = 1,000 +0,001 → 0,000; seed-0 replay − FT compound N = 1,000 −0,013 → −0,012).
  - **Figures:** `runs/local/campaign/analysis_rev2/recovery_<shift>.png` (3-epoch ceiling; say so in the caption, the title only says "the full-pool ceiling") and `recovery_<shift>_e12.png` (12-epoch ceiling); `runs/local/rq5/analysis_rev2/rq5_<shift>.png`. Do not use `recovery_weather_e12.png`: weather against the 12-epoch ceiling is reported as not determined.
  - **Which CSV holds which table** (all in `analysis_rev2/`). Campaign: `rq1_drop.csv` (RQ1), `summary.json` key `interaction`, `recovery_denominator.csv`, `recovery_points.csv` (column `analysis`: `full-pool`, `matched-pool`, `full-pool-e12`, `matched-pool-e12`), `thresholds.csv`, `fits.csv` and `fit_holdout.csv`, `ceiling_e12_vs_e3.csv`, `forgetting.csv` and `forgetting_e12.csv`, `controls.csv`. RQ5: `rq5_delta.csv` (Δ and absolute mAP), `rq5_equivalent_budget.csv` (N_eq), `rq5_attribution.csv` (seed-0 split), `rq5_replay_share.csv`, `rq5_replay_share_vs_source.csv`, `rq5_vs_zero_shot.csv`, `rq5_matched_passes.csv` and `_per_class.csv`, `rq5_replay_share_matched.csv`, `rq5_forgetting.csv` and `rq5_forgetting_revision.csv`, `rq5_delta_per_class.csv`.
  - **The two notes were audited on 2026-11-23** by three independent agents (about 440 values in the RQ1-RQ4 note alone): every table cell matched its CSV; about 60 prose and labelling findings, including six mis-rounded cells in the older RQ5 tables, were fixed. All tables in both notes are now generated from the CSVs.
- **Kaggle run code removed 2026-11-23** (user): `kaggle/`, `scripts/kaggle_materialize.py`, `docs/kaggle_training.md`, `docs/results_rq1_rq4_vi.md`. The data-provenance check was renamed `scripts/verify_source_mirror.py` (was `kaggle_source.py`). `README.md` and `CLAUDE.md` are updated for local compute, the new framing and the retired N95.
- **No mention of the Kaggle runs in the report, the result notes or the README** (user decision 2026-11-23, no disclosure sentence; the user informs the lecturer). Those documents may say that each frozen plan was scored once; they must not say the held-out labels were opened only once in total. See `DECISIONS.md`.
- **Pushed 2026-11-23** to branch `lecturer-review-revision` (not `main`). The user decides whether to merge or open a pull request.
- **`README.md` rewritten 2026-11-23** around the control-arm framing, with a results section generated from the CSVs and a reproduction section; audited by two agents and fixed.
- **Side effect of that audit:** an auditor ran `scripts/local_rq5_pilot.py --help`; the script has no argument parser and reran its main. All four pilot runs were complete, so nothing trained, but `runs/local/rq5/pilot_summary.json` was rewritten (same four run entries; only the `source` path string can differ). `pilot_lock.json` is untouched. `scripts/local_rq5_pilot.py` and `scripts/local_gate_schedule_pilot.sh` start working on any invocation.
- No user question is open. The two dated correspondence files that mention Kaggle are kept (user, 2026-11-23).
- **The user writes the report themselves from a template** (2026-11-23). Do everything up to that step: finish scoring, re-analyse, update the two result notes, audit. Do not rewrite `docs/bao_cao_ket_qua_vi.docx`, the review-response letter or the README introduction unless asked.
- The two completed plans (2026-10-30 campaign, 2026-11-03 RQ5) were moved to `.agent/plans/archive/` on 2026-11-23.
- Nothing is committed. Deletions are staged by `git rm`; everything else is unstaged.

## 2026-11-21: Lecturer review of the results report; revision started (superseded by the 2026-11-23 section above)

The lecturer reviewed `docs/bao_cao_ket_qua_vi.docx`. The review is in `docs/review_2026-11-21_lecturer_vi.md`, the plan in `.agent/plans/active/2026-11-21-lecturer-review-revision.md`, the rules in `DECISIONS.md` (2026-11-21).

- **Framing changed.** The headline is now the control arm: most of what domain adaptation gives at small labeled budgets (N = 10-100) is not adaptation. Curves are context, thresholds are secondary, weather is the third replication of the decomposition. `CLAUDE.md` still says "the contribution is the curve"; it is rewritten in the plan's Phase 4.
- **Kaggle is dropped as a results source** (user decision). Not yet removed from the docs or the repo; that is Phase 4.
- **Phase 0 done:** review saved, rules pre-registered in `DECISIONS.md`.
- **Phase 1 (analysis only, no new run) is done except the main report.** New outputs are in `runs/local/campaign/analysis_rev1/` and `runs/local/rq5/analysis_rev1/`; the directories `analysis/` next to them are the outputs the 2026-11-08 report cites and stay untouched. `docs/results_local_rq1_rq4_vi.md`, `docs/results_rq5_vi.md` and `docs/training_pipeline.md` are updated. `docs/bao_cao_ket_qua_vi.docx` (and its Claude Doc copy) is NOT touched yet: it is rewritten in Phase 4, after the new runs are scored.
  - Verification: 151 tests pass; every pre-existing table reproduces byte for byte; the code was reviewed by six independent agents and their findings fixed; every new number in the two notes was checked against the CSVs by script. An independent audit of the notes' wording against the rules was started but stopped by the usage limit before returning anything; it has not been rerun.
  - **Finding to carry into the rewrite:** the replay share depends on the baseline at N = 10 (against FT 82-95%, against the source model about 5-9%), and agrees at N = 100 (78-86% and 62-86%). The headline is a statement about the advantage over fine-tuning. See `DECISIONS.md` addendum 2.
- **Do not edit `scripts/bdd_pipeline/` or `scripts/bdd_adapt/`.** Every run recorded the SHA256 of those files. New analysis code goes in `scripts/analyze_campaign.py` and `scripts/analyze_rq5.py`.
- **Machine crash 2026-11-21 ~16:38, queue relaunched 16:51.** The machine froze and rebooted at 16:41 with 19 of 40 runs finished. All 19 were re-verified (manifest, update count, prediction file hashes). Run 20 (`matched_passes/adapt_weather_N1000_s1`) resumed from `last.pt` at step 2,500 of 3,000. The relaunched driver has all 42 runs in its plan, so the chain script is no longer needed.
  - **Likely cause, not fixed:** `nvidia-smi dmon -d 2 -s pucm`, a child of KDE's `ksystemstats` (the System Monitor backend), ballooned to about 54 GB four times that day (kernel OOM kills at 10:37, 11:03, 11:17, 13:37; `journalctl -b -1 -k | grep "Out of memory"`). The journal stops at 16:37 right after System Monitor was opened. A watchdog that kills a runaway `nvidia-smi` was not permitted in the session; the user decides (close System Monitor and GPU widgets, or stop `plasma-ksystemstats.service`, or run a watchdog themselves).
  - **Resume is exact in state, not bit-identical on the GPU.** Two uninterrupted runs with the same seed already differ from step 2 and by 2-3% in per-step loss later (original vs matched `adapt_compound_N1000_s0`, first 2,000 steps). The resumed run deviates from its pre-crash log by the same amount. A resumed run is a valid run; do not expect identical logs.
- **Phase 2 is RUNNING (user go-ahead 2026-11-21, started 10:36).** 40 runs plus 2 chained ones, about 29 h of training plus about 3 h of predictions:
  - Launch: `setsid nohup ~/venvs/cv-final/bin/python scripts/local_revision_runs.py > runs/local/revision.log 2>&1 &` from the repo root. Outputs `runs/local/revision/grid/<block>/<run_id>/`, per-run logs `runs/local/revision/logs/`.
  - Order: `replay_seeds` (12 runs), `matched_passes` (18), `ceiling_e12` (10; the two lighting full-pool ceilings last, about 7 h each).
  - **Two more runs are chained** (DECISIONS addendum, user approved): block `matched_pool_e3` (lighting and weather, 1,738 images, E = 3, seed 1; about 15 min each). `runs/local/revision/chain_matched_pool_e3.sh` waits for the main driver (PID 1440767) and starts `--blocks matched_pool_e3` once the log says `REVISION RUNS COMPLETE`. If the main queue is relaunched by hand instead, the plain command runs that block at the end. The plan to freeze then has **42** prediction sets.
  - Check: `tail -3 runs/local/revision.log`. Finished = the log says `REVISION RUNS COMPLETE`.
  - **Relaunch (after a reboot or a failure):** first make sure nothing is alive: `pgrep -af "scripts/(local_revision_runs|train_adapt|train_frcnn|predict_frcnn)\.py"` must show no Python process (the driver also refuses to start otherwise). Then run the same command with `>>` instead of `>`, so the log is kept. Finished runs are skipped, an interrupted run resumes from `last.pt`. The driver stops on the first failure and names the per-run log.
  - **To stop it, kill the whole process group** (`kill -- -<driver pid>`), not the driver PID alone: the trainer is a child in the same group and would keep running.
  - **While it runs, and until the revision scoring has written `results.json`:** no edit, no new `.py` file and no `git checkout`/`git stash` touching `scripts/bdd_pipeline/`, `scripts/bdd_adapt/`, `scripts/train_*.py`, `scripts/predict_frcnn.py` or `scripts/prepare_revised_bdd100k.py`. Every run is a fresh process that re-imports and re-hashes them. Do not move or rename the repository between freezing the plan and the end of scoring (the plan stores absolute paths).
  - Do not delete the `predictions/` folder of a finished ceiling: the driver then refuses to continue until its `train/` folder is removed by hand, and that means a retrain of up to 7.5 h.
  - Reviewed 2026-11-21 by three independent agents (rules, run control, safety): no finding required stopping or redoing a run. Run 1 matched its original in every recorded field except the seed.
  - `--dry-run` prints the 40 runs and their commands. Each run differs from its original counterpart in exactly one argument (seed, `--floor-updates` or `--epochs`); `tests/test_revision_runs.py` asserts this.
- **After Phase 2 (Phase 3):** freeze ONE plan over the 42 new `prediction_manifest.json` files (glob `runs/local/revision/grid/*/*/predictions/prediction_manifest.json`), score once into a new directory (third opening of held-out labels; the user agreed to it together with the runs), then extend the two analysis scripts. The pre-freeze checklist is in the plan (Phase 3, step 10).
  - **Analysis hazard:** 20 of the 40 revision runs carry the same stage/domain/budget/seed as an original run. Load the three scored directories separately, never merge them on disk and never key on those four values alone. Tell runs apart by `run.epochs` (12 vs 3) and `run.total_updates` (3,000 vs 2,000).
- **Open questions for the user (still unanswered):** keep one disclosure sentence about the 2026-11-03 Kaggle opening of test labels; delete the Kaggle run code and docs from `main`.


## 2026-11-05: Full fresh rerun on the local RTX 3060 (latest; supersedes the Kaggle plan)

Kaggle is dropped. Everything is rerun from scratch on this Linux machine; see `DECISIONS.md` 2026-11-05.

- **2026-11-08: project moved** into the repo `imissu-datntq/Quantifying-Target-Domain-Data-Requirements-for-Object-Detection-under-Domain-Shift` (this folder; the user can push here, not to `YN2TB/CV-final`). Branch `final-project`: the September exploration work sits unchanged in `phase1_initial/`, the CV-final history is merged at the root. `data/` and `runs/` were moved here (not in git). Report: `docs/bao_cao_ket_qua_vi.docx` (also a Claude Doc: https://claude.ai/artifact/6f6s6fy6jvcoirjEufVpqw). Model release package: `runs/local/release/models-local-2026-11/`.
- **2026-11-07 09:40 state (latest): ALL LOCAL WORK DONE.** RQ5 scored once (94/94, `runs/local/rq5/scored/`), analysed (`runs/local/rq5/analysis/`, `scripts/analyze_rq5.py`), reported in `docs/results_rq5_vi.md` (DECISIONS 2026-11-07). Local RQ1–RQ4 report: `docs/results_local_rq1_rq4_vi.md`. No GPU job is running. Remaining: split-script CRLF fix (TODO), commit when the user asks. Optional follow-up if the lecturer asks: a matched-pass RQ5 rerun at N=1000 (FT E=3 vs adapt 2N confound).
- **2026-11-07 07:20 state:** RQ5 grid COMPLETE 07:11 (46/46, no errors). Plan frozen `runs/local/rq5/final_plan.json` (schema 3, 46 sets, bootstrap 1,000 seed 20261030 = FT's, sha256 `3df3d788a57e…b4b7`) before any RQ5 test label was read. **One-shot scoring running (CPU, nice 10)** from `scripts/`: `~/venvs/cv-final/bin/python evaluate_frcnn.py --dataset-root ../data/bdd100k_revised --final-plan ../runs/local/rq5/final_plan.json --output-dir ../runs/local/rq5/scored --threads 4`, log `runs/local/rq5/scoring.log`; if interrupted add `--resume-interrupted` (never a second pass). Then: `cd scripts && ~/venvs/cv-final/bin/python analyze_rq5.py --ft-scored ../runs/local/campaign/scored --da-scored ../runs/local/rq5/scored --output-dir ../runs/local/rq5/analysis` (new script, 8 unit tests in `tests/test_analyze_rq5.py`), then write `docs/results_rq5_vi.md`. Local RQ1–RQ4 report written: `docs/results_local_rq1_rq4_vi.md`.
  - RQ5 disclosure to carry: T_run = max(T, 2N) was pre-registered as 'matching FT's E=2'; locally FT uses E=3, so FT sees each labeled image 3× vs ≥2× for adapt (1 labeled target image per step).
- **Machine:** repo at `/media/orlab/437478a6-2666-4549-903d-18e9d7d1c34f/orlab/Quocdat/Quantifying-Target-Domain-Data-Requirements-for-Object-Detection-under-Domain-Shift` (moved here from `../CV-final` on 2026-11-08 with `data/` and `runs/`), RTX 3060 12 GB. Python env `~/venvs/cv-final` (always use `~/venvs/cv-final/bin/python`). Git needs no `safe.directory` any more (repo chowned to `orlab`).
- **Data (done 00:45):** raw BDD100K hardlinked into `data/raw/` from `../BDD100K-weather-classification/data/raw`; verified `data/raw/kaggle_verified/source_verified.json` (80,002 PASS); split at `data/bdd100k_revised/`, all 7 hashes OK after CRLF conversion.
  - Known issue: `prepare_revised_bdd100k.py` writes platform line endings, so a Linux build differs in bytes from the frozen (Windows) identity. Fix it by writing with `newline="\r\n"` before any rebuild; until then convert as in DECISIONS.
- **Source model DONE 05:04:** source-val reliable mAP50:95 **0.3517** (selected = final step 29,412, still rising slightly; Kaggle 0.352), `runs/local/source/source_seed0/best.pt`. Was: `runs/local/source/run_source.sh`, detached (`setsid nohup`), log `runs/local/source/train.log`, outputs `runs/local/source/source_seed0/`. 29,412 updates, ~0.46 s/update, eval on source val every epoch. ETA ~05:00 on 2026-11-05. If it dies, rerun the same script: it resumes from `last.pt`.
  - Check: `tail -1 runs/local/source/train.log`, `pgrep -af train_frcnn`.
- **Chain (running since 00:53, detached):** `scripts/local_gate_schedule_pilot.sh`, log `runs/local/chain.log`. It waits for the source model, then runs the zero-shot gate (`runs/local/zero_shot_gap/`), N0 on lighting val and 12 schedule-pilot runs (9 round-3 + 3 round-2 seed-0 configs) into `runs/local/schedule_pilot/`, and writes `runs/local/schedule_pilot/summary.json`. It does **not** apply the lock rule. Re-running it is safe (finished steps skipped). Expected end ~08:00–09:00. When the log says `CHAIN COMPLETE`, apply the 2026-10-30 rule by hand and record the lock in `DECISIONS.md`.
- **Chain DONE 07:33.** Gate passed (weather 0.032, CI 0.024–0.041; lighting 0.100; compound 0.129). Schedule rule applied → S=500, **E=3**, LR 1e-3 (DECISIONS 2026-11-05). `bdd_pipeline/campaign.POLICY` epochs 2→3; 44/44 tests pass.
- **Local RQ1–RQ4 analysis DONE 10:10** (`runs/local/campaign/analysis/`, 176 sets → 292 curve points, 136 threshold rows). Reliable, full pool, ladder points:
  - Drops: lighting 0.098 (CI 0.092–0.106), weather 0.022 (0.012–0.032), compound 0.123 (0.112–0.133). Kaggle: 0.097 / 0.021 / 0.123.
  - N_BE: lighting 100, weather 250, compound 100. N75: lighting 5,000, weather **1,000** (Kaggle: full pool), compound 1,000. N90: lighting 10,000, weather/compound full pool.
  - Recovery negative for N ≤ 50 everywhere (weather N10 −3.5, small denominator). Matched pool 1,738: lighting N75 1,000 / N90 1,738, same as compound.
  - Still to do: Vietnamese report for the local replication (cf. `docs/results_rq1_rq4_vi.md`), with a Kaggle-vs-local comparison and the E=3 change disclosed.
- **2026-11-06 10:05 state:** scoring DONE (176/176, `runs/local/campaign/scored/`). Campaign analysis running (`runs/local/campaign/analysis.log`). RQ5 pilot DONE → α 0.9996, T 2,000, gate passed (DECISIONS 2026-11-06). **RQ5 grid running** since 10:03: `~/venvs/cv-final/bin/python scripts/local_rq5_grid.py`, log `runs/local/rq5_grid.log`, outputs `runs/local/rq5/grid/<run_id>/`; rerun the same command if it stops (skips finished, resumes interrupted). ETA roughly 2026-11-07 morning. After it: freeze a plan over the 46 RQ5 prediction files, score once, write `scripts/analyze_rq5.py` (FT side from `runs/local/campaign/scored/`).
- **2026-11-06 04:57 state:** campaign COMPLETE 04:54 (87/87, no errors). Plan frozen `runs/local/campaign/final_plan.json` (schema 3, 87 sets, bootstrap 1,000 seed 20261030, sha256 `a1c5eb9fa457…436a44`).
  - **One-shot scoring running (CPU, nice 10):** from `scripts/`: `~/venvs/cv-final/bin/python evaluate_frcnn.py --dataset-root ../data/bdd100k_revised --final-plan ../runs/local/campaign/final_plan.json --output-dir ../runs/local/campaign/scored --threads 4`; log `runs/local/campaign/scoring.log`. If interrupted, rerun with `--resume-interrupted` (never start a second pass). Then `scripts/analyze_campaign.py --scored runs/local/campaign/scored --output-dir runs/local/campaign/analysis`, and write the local results report.
  - **RQ5 pilot running (GPU):** `scripts/local_rq5_pilot.py`, log `runs/local/rq5_pilot.log`, outputs `runs/local/rq5/pilot/`. 4 runs × 6,000 updates at ~0.61 s/update, ETA ~10:30. Rerun the same command if it stops (resumes). When it prints `RQ5 PILOT COMPLETE`: `scripts/rq5_pilot_lock.py runs/local/rq5/pilot --output runs/local/rq5/pilot_lock.json`, record in DECISIONS, set `bdd_adapt/grid.POLICY`, then run the 46-run grid locally (port `kaggle/bdd_adapt.py` like `scripts/local_campaign.py`).
- **Campaign (ran 2026-11-05 07:36 → 2026-11-06 04:54):** `~/venvs/cv-final/bin/python scripts/local_campaign.py`, log `runs/local/campaign.log`, per-run logs `runs/local/campaign/logs/`, outputs `runs/local/campaign/campaign/<run_id>/`. 87 runs in grid order, ~12.9 h training at 8.45 img/s plus test predictions (~0.083 s/image); expect ~20–24 h total. It stops on the first failure (RuntimeError in the log). Re-running the same command is safe: finished runs are skipped, an interrupted run resumes from `last.pt`.
  - Progress: `grep -c ': {' runs/local/campaign.log` = runs finished.
- **After the campaign:** freeze one prediction plan (`scripts/freeze_evaluation_plan.py`, cf. `runs/campaign/final_plan.json` schema 3 on Kaggle), score once with `scripts/evaluate_frcnn.py --final-plan`, then `scripts/analyze_campaign.py`. The RQ5 pilot (lighting val only) needs only the source model, so it can run after the campaign frees the GPU.
- **Old steps list (kept for reference):**
  1. Source done → check `result.json`, selected step and source-val reliable mAP50:95 (Kaggle got 0.352).
  2. Zero-shot gap gate with the new source model (`scripts/zero_shot_gap.py`, cf. `kaggle/bdd_gap_gate.py`).
  3. Schedule pilot on lighting val only: re-run the round-2 seed-0 configs used by the rule plus all round-3 configs (`kaggle/bdd_schedule_pilot.py`), apply the 2026-10-30 rule unchanged, record the lock in `DECISIONS.md`, set `bdd_pipeline/campaign.POLICY` if it changes.
  4. Campaign 87 runs (`kaggle/bdd_campaign.py` logic, run locally).
  5. RQ5 pilot → lock → 46-run grid.
  6. One frozen plan, scored once (second opening of held-out labels; replication).
- **Rough local timing** (unmeasured beyond source): source ~4.2 h, gate ~1 h, schedule pilot ~2–3 h, campaign ~15–18 h, RQ5 pilot + grid ~25–30 h. Total ~50–60 h.

## 2026-11-04: RQ5 grid running (latest)

- **Pilot locked:** α = 0.9996, T = 2,000, gate passed (`DECISIONS.md` 2026-11-04, `runs/rq5/pilot_lock.json`). `grid.POLICY` is set; 44 tests pass. Cost ~35 T4-h (`runs/rq5/grid.csv`, 0.789 s/update).
- **Both protocol datasets** carry the locked code (versions of 2026-11-04).
- **Running since 2026-11-04 ~02:40 local:**
  - `buihuynhgiahuy/bdd100k-rq5` v1: 32 runs (compound, weather, replay), SESSION_HOURS 11. Quota 30 h at push.
  - `kizhowl/bdd100k-rq5` v1: 14 lighting runs, SESSION_HOURS 8.0 because only 8.83 h quota was left; the rest (mostly N5000) waits for the 2026-11-07 reset.
- **Next session of either:** push v2 of that kernel with the kernel itself added to `kernel_sources` (it restores `rq5/` from the previous output and resumes `last.pt`). Same QUEUE, same code.
- **GitHub:** the repo is pushed to https://github.com/YN2TB/CV-final (public, branch `main`). `data/`, `runs/`, weights and `docs/response/*.jpg` are gitignored.

## 2026-11-03: RQ5 started

- **Plan:** `.agent/plans/active/2026-11-03-rq5-adaptation.md`. Rules are pre-registered in `DECISIONS.md` (2026-11-03).
- **Method:** ALDI++, re-implemented in `scripts/bdd_adapt/`. `bdd_pipeline/` is untouched.
- **Tests and smoke:** 43 tests pass. The CPU smoke resumes bit-identically.
- **Pilot:** `kizhowl/bdd100k-rq5-pilot` v1 was pushed 2026-11-03 and is running. It uses lighting val only: 4 runs of 6,000 updates, with the teacher evaluated every 500 updates.
- **When the pilot is complete:**
  1. Download `rq5_pilot_summary.json` and `rq5_pilot/*/validation_step_*.json`, `training.jsonl` and `result.json` into `runs/rq5/pilot/`.
  2. Apply the α/T rule and check the correctness gate.
  3. Record the result in `DECISIONS.md` and set `scripts/bdd_adapt/grid.POLICY`.
  4. Push the protocol dataset.
  5. Push `kaggle/bdd_adapt.py` as `kizhowl/bdd100k-rq5` with its QUEUE set to the 46-run grid.
     - Kernel sources: `kizhowl/bdd100k-source-train` (later versions also list the kernel itself).
- **Second account:** buihuynhgiahuy is approved for the grid and has its own private copies of the protocol dataset and source model.
  - The grid splits by account: buihuynhgiahuy runs compound, weather and replay; kizhowl runs lighting.
  - Lock the pilot with `python scripts/rq5_pilot_lock.py runs/rq5/pilot --output runs/rq5/pilot_lock.json`.
  - After locking, push the new code to **both** protocol datasets.
- **If the pilot is cut by the session limit:** push v2 with `kizhowl/bdd100k-rq5-pilot` added to its `kernel_sources`; it resumes.
- **Local source model** for smoke tests: `runs/source_model/runs/source_seed0/best.pt`.

## 2026-10-30

This repo copy is on the user's machine at `D:\Personal Projects\CV-research`. It is not a git repository. The data is local in `data/bdd100k_revised/` (hardlinked, all 79,863 images rehashed OK).

The plan for the run campaign is in `.agent/plans/active/2026-10-30-supervised-campaign.md`.

**State:**
- The schedule is locked by the pre-registered round-3 rule: S=500, E=2, LR 1e-3, B_eff 2. See `DECISIONS.md` (2026-10-30) and `runs/schedule_pilot/round3/`.
- The grid is 87 runs, about 12.6 T4-h of training (`runs/campaign/grid.csv`).
- Protocol dataset `kizhowl/bdd100k-revised-protocol` has a new version carrying the campaign code. **`bdd_pipeline/` is frozen from here on:** resume checks implementation hashes.
- Campaign session 1 (`kizhowl/bdd100k-campaign` v1) finished COMPLETE on 2026-10-31. It covered zero-shot, compound, weather and the controls.
  - Training ran at about 6 img/s, update counts match the policy, and every manifest says `test_labels_accessed: False`.
  - All 54 runs finished in about 8.5 h, with no errors (53 trained, plus the zero-shot predictions).
  - The local `runs/campaign/session1/` copy is partial: 34 run folders, because the download was cut off.
  - Get the complete `campaign/` from the final version's output, which carries everything forward.
  - Prediction timing: lighting test takes about 375 s and source test about 203 s per run.
- Session 2 (v2) was pushed 2026-10-31. Its queue is the whole grid: finished runs are skipped and interrupted ones resume. It attaches v1's output.
- Session 2 (v2) finished COMPLETE on 2026-11-01. It trained 27 lighting runs, with no errors.
  - `fine-tune_lighting_N5000_s1` was interrupted by session time; its `last.pt` is on Kaggle, in the v2 output.
  - **Left:** 6 runs: lighting N5000 s1 (resume) and s2, N10000 s0/s1/s2, and the matched-pool N1738 s0. That is about 6 T4-h.
  - kizhowl has 19.6 h of quota left (it resets 2026-11-07).
- Session 3 (v3) was pushed 2026-11-01 for the last 6 lighting runs. It uses the same queue script, so finished runs are skipped. Expect about 6–7 h.
- **Grid complete 2026-11-02:**
  - Session 3 trained the last 6 runs in 4.2 h.
  - All 87 runs and their prediction files are in `runs/campaign/final/campaign/` (9.8 GB).
- **Frozen plan:** `runs/campaign/final_plan.json`, sha256 `8244023f…5ff7a8`. Schema 3, 87 sets, 176 scoring entries, bootstrap 1,000 resamples with seed 20261030.
- **One-shot scoring started 2026-11-02** on the 5060 laptop, CPU only, as a detached below-normal-priority process. It writes to `runs/campaign/scored/`, with logs in `runs/campaign/scoring.{out,err}.txt`.
  - Expect about 7 h.
  - If interrupted, run from `scripts/`: `python evaluate_frcnn.py --dataset-root ../data/bdd100k_revised --final-plan ../runs/campaign/final_plan.json --output-dir ../runs/campaign/scored --threads 4 --resume-interrupted`.
  - The lock records the absolute output path, so another machine needs the same `D:\Personal Projects\CV-research` location.
- **After scoring:** `python scripts/analyze_campaign.py --scored runs/campaign/scored --output-dir runs/campaign/analysis`, then write `docs/results_rq1_rq4_vi.md`.
- **Phase 4 done 2026-11-03:**
  - One-shot scoring completed with 176/176 entries (`runs/campaign/scored/results.json`).
  - Analysis is in `runs/campaign/analysis/`, and the report is `docs/results_rq1_rq4_vi.md`.
  - **Headlines:**
    - Drops: lighting 0.097, weather 0.021, compound 0.123. Compound is additive (interaction +0.005, CI −0.010 to 0.018).
    - Recovery is negative for N ≤ 50, caused by overfitting under the fixed 500-update floor.
    - N_BE is about 100–250. N75: lighting 5,000, compound 1,000, weather full pool.
    - At the matched pool, lighting and compound recover alike.
    - The power fit over-predicts at large N, so the ladder is primary.
  - **Next:** Phase 5 (RQ5) needs its own ExecPlan. Proportional-update reruns at N ≤ 100 are optional.
- **When v3 is COMPLETE:**
  1. Download the full `campaign/` from v3's output (`--file-pattern` for json/log). This is the complete grid, because each version carries the previous one forward.
  2. Check that all 87 runs have `predictions/prediction_manifest.json`.
  3. Freeze one prediction plan and score once. The user plans to do the scoring on their second machine (RTX 3060).
  - The user now has a second machine with an RTX 3060.
  - A Kaggle `last.pt` cannot be resumed locally as is: the config records `/tmp` and `/kaggle` paths, and resume rejects any config difference.
- Lecturer updates are weekly; the draft `docs/lecturer_update_2026-10-30_vi.md` is kept current, not a blocker.

**Next safe action:**
1. When v1 finishes, download `campaign/*/predictions/`, `logs/` and `session_summary.json` into `runs/campaign/`.
2. Check `amp_step_skipped` and the throughput.
3. Push v2 with the remaining runs plus the lighting queue, adding `kizhowl/bdd100k-campaign` to its own `kernel_sources`.
4. Use the `kizhowl` key only, via `KAGGLE_API_TOKEN`. Never use `ngdminh31`.

## 2026-10-19 review amendment (latest)

Six reviewer points were addressed; see `docs/review_response_2026-10-19_vi.md`
and `.agent/DECISIONS.md`. Weather/compound val merged into final test (933/756),
target runs keep the final update, fixed reliable roster pedestrian/car/traffic
light/traffic sign, `scripts/zero_shot_gap.py` pilot gate, per-class
`scripts/recovery_curves.py`, EDA v2 at `runs/eda_revised_v2/` (adds
`domain_occlusion_truncation.csv`, `domain_objects_per_image.csv`, figure 09),
`scripts/label_provenance.py` fingerprint at `runs/label_provenance/`. COCO-proxy gap probe output is at
`runs/zero_shot_gap_coco_proxy/`. Official BDD100K hosts were unreachable, so the
label comparison is pending a user download. Nothing staged or committed.

Kaggle (2026-10-20, see `docs/review_response_2026-10-20_kaggle_vi.md`): source
model trained (source-val reliable mAP50:95 0.352); official zero-shot gate with
it: weather gap 0.031 AP (CI 0.023-0.040) passes, lighting 0.099, compound 0.129.
Work outside the review's six points was removed at the user's request.


## Current state

The user requested a new careful split from the original Kaggle BDD100K source,
EDA exclusively on the new dataset, then a Python training/evaluation pipeline
and a report of its actual verification. The repository was cloned at
52898b14269cf55f3fa06cebf7cccd82de5b8e02. Proposal read in full.

`data/bdd100k_revised/` is complete: 55,000 train / 14,863 held-out test /
10,000 preserved official val. Images are independent copies, annotations are
canonical detection-only JSON. All four main domain counts match README/proposal.
Membership is frozen by the CSV manifests and SHA256 recorded in metadata.

The supervised Faster R-CNN pipeline is implemented in `scripts/bdd_pipeline/`
with `train_frcnn.py`, `pilot_frcnn.py`, `evaluate_frcnn.py` and
`freeze_evaluation_plan.py` entry points. It supports source training, nested
target fine-tuning, full-pool reference and generic-pretrained controls. See
`docs/training_pipeline.md` and the Vietnamese `docs/pipeline_report_vi.md`.
The earlier YOLO prototype is not the current proposal implementation.

Every original input image and both label files matched public Kaggle ZIP sizes
and CRC32. Source verification remains ignored in `data/raw/kaggle_verified/`.
No previous derived subset or model result was used as an EDA input.

Hierarchical Hamilton stratification uses timeofday/weather then scene and seed
20260916. An exact duplicate image pair in original train was reunited using one
same-stratum singleton swap; all stratum quotas remain unchanged. Within-split
extra duplicate images: train 1, test 0, val 2. Official validation membership is
unchanged. Zero cross-split basename and SHA256 overlap. All dimensions 1280×720.

## Verified evidence

- Full construction and structural count/membership validation passed.
- `--validate-only` rehashed all 79,863 new images and passed again.
- Six unit tests passed for deterministic/stratified allocation, strict JSON,
  exact-content leakage, duplicate repair and class-absence probability.
- EDA covers new train/val labels only; held-out detection labels are never read
  by the EDA script. Test metadata provide census/balance only.
- Train has 1,013,542 valid boxes; car 55.29%; small-box share 52.95–55.93%
  across study domains. Joint-stratum train/test TVD is 0.001381 and largest
  absolute share difference is 0.006175 percentage points.
- Local report: `runs/eda_revised/report_vi.html` and `.md`; 8 figures, CSVs,
  exact absence probabilities and nested train-only budgets for seeds 0/1/2.
- Notebook: `notebooks/eda_revised_bdd100k.ipynb`, executed with embedded figures.
  It checks EDA/dataset manifest hash agreement before showing results.
- Pipeline and dataset tests: 14/14 pass; native geometry, normalization,
  class IDs, sanitation/flip, budget nesting, schedule, COCO perfect/empty
  predictions and final-plan guards verified. Syntax checks pass for 17 files.
- PyTorch 2.11.0 CPU, torchvision 0.26.0 CPU and pycocotools 2.0.11 installed.
  No CUDA device. Official COCO pretrained weights downloaded successfully.
- `runs/pipeline_verified/`: native train/val source smoke (3 images, 2 updates),
  stop/resume pair and continuous run. All 295 model tensors, optimizer,
  scaler, RNG and positions match exactly; maximum model difference 0.0.
- Separate validation CLI reload passed; target-weather fine-tune and COCO
  pretrained pilot each passed one update on two train images. Source smoke
  losses 12.5672/3.8292; throughput 0.167 images/sec. These are software checks,
  not research scores. No held-out detection labels accessed in these runs.
- CLI rejects freezing smoke checkpoints and resuming with a different LR.
  Unit-test/guard/resume evidence is recorded beside the ignored run artifacts.

## Boundaries and next action

No full detector training campaign, method selection, held-out performance
evaluation or publication was performed. Content hashes do not exclude near-duplicate,
geographic or video-group dependence. Source ZIP consistency is CRC32/size-based,
not a signed provenance guarantee. Header/format checks cover all images;
brightness EDA fully decodes 800 sampled train images.

Read `docs/revised_dataset_eda.md` for EDA reproduction commands. The next
research step is a representative GPU throughput/memory pilot, verification of
CUDA/AMP and a fixed optimizer/schedule/support policy before full source training.
Native geometry and preprocessing are implemented. S=2000/E=30/LR=0.005,
effective batch 2 and reliable threshold 100 remain provisional. CPU exact resume
is verified only with workers=0. Use only the frozen new manifests.

Final test evaluation requires an explicit frozen checkpoint plan, validates
every checkpoint before label access and creates a one-shot opened marker. No
real final plan or held-out evaluation has been executed. Adaptation methods,
recovery curves and bootstrap intervals are not implemented in this milestone.

Repository now lives directly at `D:\cv`. Preserve the user's existing Git index:
it contains staged deletions/renames from the root reorganization. Implementation
work did not stage or commit files. Dataset, weights and runs remain ignored.
