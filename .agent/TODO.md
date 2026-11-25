# TODO

## Direction 1 of the lecturer's follow-up (2026-11-24), branch `alt-1`

Plan: `.agent/plans/active/2026-11-24-direction-1-analyses.md`.

- [x] Branch `alt-1` from `main`; plan with the analysis rules (2026-11-24).
- [x] `scripts/analyze_instances.py`, `scripts/analyze_seed_spread.py`, `scripts/analyze_retention.py` with synthetic tests; independent review and verification (2026-11-25).
- [x] User approves the run (2026-11-25, on the Windows machine).
- [x] Run on `runs/local/.../scored`; rules and results in `DECISIONS.md` (2026-11-25); note `docs/results_direction1_vi.md`; copy in `docs/results/direction1/`.
- [x] Push `alt-1` to the team repo (2026-11-25, after YN2TB got write access).
- [ ] User reviews the note; decide whether the results go into the report and the paper.

## Revision after the lecturer review of 2026-11-21

Plan: `.agent/plans/active/2026-11-21-lecturer-review-revision.md`.

- [x] Phase 0: save the review (`docs/review_2026-11-21_lecturer_vi.md`), pre-register the rules (`DECISIONS.md` 2026-11-21).
- [x] Phase 1 (no new run), done 2026-11-21: N95 retired and "not determined within pool" status; log-linear fit on the positive region with a held-out check; interaction detection limit; recovery-denominator table; replay share against FT and against the source model; 151 tests; the two result notes and `docs/training_pipeline.md` updated.
- [x] Independent audit of the two notes (numbers, wording against the `DECISIONS.md` rules, completeness), done 2026-11-23 by three agents; all findings applied.
- [x] Runaway `nvidia-smi dmon`: the user chose to leave it alone (2026-11-23).
- [x] Kaggle run code and documents removed from `main` (2026-11-23); the data-provenance check renamed `scripts/verify_source_mirror.py`.
- [x] Kaggle runs removed from the two result notes and the README, no disclosure sentence (user decision 2026-11-23).
- [ ] Phase 4: apply the same to `docs/bao_cao_ket_qua_vi.docx` (section 11 and every comparison) when it is rewritten.
- [x] Phase 2, done 2026-11-22 20:44: 42 runs (`scripts/local_revision_runs.py`, log `runs/local/revision.log`); one crash on 2026-11-21, resumed.
- [x] Phase 3a, 2026-11-23: pre-freeze checklist passed, plan frozen (`runs/local/revision/final_plan.json`, 42 sets).
- [x] Phase 3b: one-shot scoring done 2026-11-23 10:28, 84 of 84 entries (`runs/local/revision/scored/`).
- [x] Phase 3c, done 2026-11-23: both analysis scripts take `--revision-scored`; outputs in `analysis_rev2/`; the two result notes updated from the CSVs.
- [x] English IEEE paper for Overleaf: `docs/overleaf/` written, verified, reviewed (5 lenses), linted; `docs/overleaf_ieee.zip` built (2026-11-23).
- [ ] Paper, user's side: fill the author block; decide whether to cut the 362-word abstract to 150–250; first compile on Overleaf; say whether to commit `docs/overleaf/`.
- [x] Pull of `0bbeabd` finished 2026-11-23 (a windowless Word process held a .docx; stopped with the user's approval).
- [ ] Phase 4, the user's own step: write the report from their template (control arm first; both threshold tables; no Kaggle runs), the reply to the lecturer and the README introduction. Kaggle is already out of the repo and the notes; `CLAUDE.md` is already rewritten.

## Local fresh rerun (RTX 3060, from 2026-11-05; Kaggle dropped)

- [x] Env `~/venvs/cv-final` (torch 2.11.0+cu128), 44/44 tests pass.
- [x] Raw data into `data/raw/` (hardlinks), source verified, split rebuilt, 7/7 hashes OK.
- [x] Source model done 05:04 2026-11-05: source-val reliable mAP50:95 0.3517 (best = final step 29,412; Kaggle 0.352), 8.69 img/s, 3.76 h train.
- [x] Zero-shot gap gate passed 05:39: weather 0.032 (CI 0.024–0.041), lighting 0.100, compound 0.129.
- [x] Schedule pilot done 07:33; rule gives S=500, **E=3**, LR 1e-3 (Kaggle had E=2; E2 missed δ by 0.0001). `campaign.POLICY` set.
- [x] Campaign 87/87 runs done 2026-11-06 04:54 (`runs/local/campaign/campaign/`, 11 GB).
- [x] Plan frozen 04:56 (`runs/local/campaign/final_plan.json`, sha256 `a1c5eb9f…436a44`, 87 sets); scored once, 176/176 entries (log `runs/local/campaign/scoring.log`, output `runs/local/campaign/scored/`). Analysis running → `runs/local/campaign/analysis/`; then write the local results report.
- [x] RQ5 pilot done 10:01: α 0.9996, T 2,000, gate passed (same as Kaggle; POLICY unchanged).
- [x] RQ5 grid 46/46 done 2026-11-07 07:11; plan frozen `runs/local/rq5/final_plan.json`; scored once 09:27, analysed by `scripts/analyze_rq5.py`, report `docs/results_rq5_vi.md` (DECISIONS 2026-11-07).
- [x] Local RQ1–RQ4 report: `docs/results_local_rq1_rq4_vi.md`.
- [x] Freeze one plan, score once, analysis and write-ups (campaign 2026-11-06, RQ5 2026-11-07).
- [x] Make `prepare_revised_bdd100k.py` write CRLF on every platform (commit `7e943f2`).

## Kaggle-era record (superseded by the local rerun)

- [x] Label version decided 2026-10-30: stay on 2018. The local `data/labels_raw/` SHA256 matches the
  Kaggle mirror. A comparison against Berkeley's own file is still pending; disclose this as a limitation.
- [x] Protocol dataset and source weights reachable via the `kizhowl` key (2026-10-30).
- Lecturer gets a weekly update (drafts in `docs/lecturer_update_*_vi.md`); not a blocker.
- [x] GPU throughput/VRAM pilot on Kaggle T4 (AMP ~6.4 img/s, 2.5-4.5 GiB).
- [x] Schedule locked from lighting val only: S=500, E=2, LR 1e-3 (2026-10-30, DECISIONS.md).
- [x] Campaign: 87 runs in 3 Kaggle sessions (2026-10-31 to 2026-11-02).
- [x] Frozen plan scored once (2026-11-03); analysis + `docs/results_rq1_rq4_vi.md`.
- [x] RQ5 (ALDI++; plan `.agent/plans/archive/2026-11-03-rq5-adaptation.md`), completed by the local grid:
  - [x] Port, 43 tests, CPU smoke with exact resume (2026-11-03).
  - [x] Pilot `kizhowl/bdd100k-rq5-pilot`: α 0.9996, T 2000, gate passed (2026-11-04).
  - [x] Set `bdd_adapt/grid.POLICY`; write `runs/rq5/grid.csv`; push both protocol datasets (2026-11-04).
  - [x] The 46-run queue: superseded, run locally and finished 2026-11-07.
  - [x] Freeze, score once, write `scripts/analyze_rq5.py` and `docs/results_rq5_vi.md`.
- [x] Train source on the frozen 9,804-image pool; select only on source val (Kaggle, 2026-10-20).
- [x] Zero-shot gap gate with the source model: weather 0.031 AP (CI 0.023-0.040) passes.
- [x] Implement an established adaptation method after supervised curves, preserving the final-test selection boundary (ALDI++, `scripts/bdd_adapt/`).
- [ ] If required for stronger independence claims, audit near-duplicates and
  geographic/video-group dependence before training; preserve split versioning.
