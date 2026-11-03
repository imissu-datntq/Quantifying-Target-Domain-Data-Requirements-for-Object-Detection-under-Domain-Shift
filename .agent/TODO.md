# TODO

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
  - [ ] Pilot `kizhowl/bdd100k-rq5-pilot`: lock α and T by the pre-registered rule; check the correctness gate.
  - [ ] Set `bdd_adapt/grid.POLICY`; write `runs/rq5/grid.csv`; push the protocol dataset; run the 46-run queue (`kaggle/bdd_adapt.py`).
  - [ ] Freeze, score once, write `scripts/analyze_rq5.py` and `docs/results_rq5_vi.md`.
- [x] Train source on the frozen 9,804-image pool; select only on source val (Kaggle, 2026-10-20).
- [x] Zero-shot gap gate with the source model: weather 0.031 AP (CI 0.023-0.040) passes.
- [ ] Implement an established adaptation method after supervised curves,
  preserving the final-test selection boundary.
- [ ] If required for stronger independence claims, audit near-duplicates and
  geographic/video-group dependence before training; preserve split versioning.
