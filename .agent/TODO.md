# TODO

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
- [ ] Freeze one plan, score once, analysis and write-ups.
- [ ] Make `prepare_revised_bdd100k.py` write CRLF on every platform.

## Kaggle-era record (superseded by the local rerun)

- [x] Label version decided 2026-10-30: stay on 2018. The local `data/labels_raw/` SHA256 matches the
  Kaggle mirror. A comparison against Berkeley's own file is still pending; disclose this as a limitation.
- [x] Protocol dataset and source weights reachable via the `kizhowl` key (2026-10-30).
- Lecturer gets a weekly update (drafts in `docs/lecturer_update_*_vi.md`); not a blocker.
- [x] GPU throughput/VRAM pilot on Kaggle T4 (AMP ~6.4 img/s, 2.5-4.5 GiB).
- [x] Schedule locked from lighting val only: S=500, E=2, LR 1e-3 (2026-10-30, DECISIONS.md).
- [x] Campaign: 87 runs in 3 Kaggle sessions (2026-10-31 to 2026-11-02).
- [x] Frozen plan scored once (2026-11-03); analysis + `docs/results_rq1_rq4_vi.md`.
- [ ] RQ5 (ALDI++; plan `.agent/plans/active/2026-11-03-rq5-adaptation.md`):
  - [x] Port, 43 tests, CPU smoke with exact resume (2026-11-03).
  - [x] Pilot `kizhowl/bdd100k-rq5-pilot`: α 0.9996, T 2000, gate passed (2026-11-04).
  - [x] Set `bdd_adapt/grid.POLICY`; write `runs/rq5/grid.csv`; push both protocol datasets (2026-11-04).
  - [ ] Finish the 46-run queue: bhgh 32 runs, kizhowl 14 lighting runs (resumes after the 2026-11-07 quota reset).
  - [x] Freeze, score once, write `scripts/analyze_rq5.py` and `docs/results_rq5_vi.md`.
- [x] Train source on the frozen 9,804-image pool; select only on source val (Kaggle, 2026-10-20).
- [x] Zero-shot gap gate with the source model: weather 0.031 AP (CI 0.023-0.040) passes.
- [ ] Implement an established adaptation method after supervised curves,
  preserving the final-test selection boundary.
- [ ] If required for stronger independence claims, audit near-duplicates and
  geographic/video-group dependence before training; preserve split versioning.
