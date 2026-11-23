# ExecPlan: revision after the lecturer review of 2026-11-21

- **Status:** Phases 0-3 done (runs finished 2026-11-22; scored and re-analysed 2026-11-23). Phase 4 (the report) is the user's own step, from their template. Created 2026-11-21 (Asia/Saigon); Phases 0-1 approved 2026-11-21 ("bắt đầu đi"), Phase 2 and its scoring approved 2026-11-21 ("đồng ý").
- **Owner:** Claude Code session with the user. Compute: local RTX 3060 12 GB (`~/venvs/cv-final`), per DECISIONS 2026-11-05.
- **Depends on:** the local campaign (`runs/local/campaign/scored/`, 176 entries) and the local RQ5 grid (`runs/local/rq5/scored/`, 94 entries), both scored once.
- **Report under revision:** `docs/bao_cao_ket_qua_vi.docx` (also `docs/results_local_rq1_rq4_vi.md`, `docs/results_rq5_vi.md`).

## Purpose and observable outcome

The lecturer reviewed the results report and raised six technical points plus a reframing of the contribution. The review text is in the appendix.

**Done means:**

- The ceilings are re-run at 12 epochs and every threshold is restated against them, with the old numbers kept as the short-schedule variant.
- RQ5 at N = 1,000 is re-run with labeled image-passes matched to fine-tuning, so the "fine-tuning overtakes ALDI++" claim is either confirmed or withdrawn.
- The report leads with the control-arm finding: *most of what domain adaptation gives at small labeled budgets is not adaptation*. The data-efficiency curves become its measurable context, and the thresholds become secondary practical results.
- Weather is presented as the third replication of the decomposition, with its near-empty recovery cell stated as a detection limit.
- The power fit is redone on the positive, monotone part of the curve. N95 is gone. N90 for weather and compound is reported as not determinable inside the pool.
- Kaggle is removed as a results source (user decision, 2026-11-21).

## Context: what the review found, checked against the data

Each point was checked against `runs/local/*/analysis/` before planning. All six hold.

| # | Review point | Check | Verdict |
|---|---|---|---|
| 1 | The 3-epoch ceiling is undertrained, so thresholds are optimistic | Lighting ceiling: 18,015 × 3 = 54,045 image-passes. Source model: 9,804 × 12 = 117,648. A standard 1× schedule is about 1.44 M. | Holds. The recovery denominator is a schedule ceiling, so N75/N90 are lower bounds. |
| 2 | The RQ5 summary is stronger than the evidence | At N = 1,000 FT sees ~3,000 labeled target passes, adapt/replay ~2,000. `rq5_attribution.csv`: adapt − replay is −0.0004 to +0.0005, replay − FT is −0.003 to −0.013. | Holds. The whole deficit sits in the replay arm, which is the arm with fewer labeled passes. |
| 3 | The weather cell is nearly empty | Ceiling − zero-shot = 0.3355 − 0.3201 = **0.0154** AP (the review's 0.016 comes from rounded table values), 1.5× the 0.01 floor. Traffic sign: 0.0089, below it. Interaction CI −0.012 to +0.016, half-width 0.014, against a weather main effect of 0.022. | Holds, slightly more strongly than stated. |
| 4 | Report Kaggle and local equally | The Kaggle outputs are not on this machine; the two runs used different schedules (E=2 vs E=3); RQ5 exists only locally. | User decision: drop Kaggle instead. See Decisions. |
| 5 | R² is inflated by negative points; the fit is optimistic | Refit of the existing points (no new scoring), see Discoveries. | Holds, and the fix works. |
| 6 | N90 is not determinable for weather and compound; drop N95 | With the ceiling defined as recovery 1 at the full pool, "N90 = full pool" is true by construction. The largest real rung is 1,000 (recovery 0.77 and 0.87). | Holds. |

**Reframing.** The lecturer's sentence: "Hầu hết những gì mà thích ứng miền (DAOD) mang lại với small labeled budgets không phải là thích ứng thực sự." It connects to the thesis of ALDI (Kay et al., CVPR 2024): reported DAOD gains depend heavily on implementation choices and baseline construction.

## Discoveries (from the review pass, 2026-11-21)

**The decomposition already supports the new headline, for labeled budgets.** Share of the ALDI++ advantage over FT that the replay arm (no unlabeled target images, no distillation) reproduces, seed 0:

| N | lighting | weather | compound |
|---:|---:|---:|---:|
| 0 | 39% | 55% | 17% |
| 10 | 87% | 95% | 82% |
| 100 | 83% | 86% | 78% |
| 1,000 (a deficit) | 105% | 100% | 97% |

- At N = 10 and N = 100, replay carries 78–95% of the gain in all three shifts. That is the claim.
- At N = 0 the picture differs: distillation is the larger part for lighting and compound. The claim must be scoped to **labeled budgets (N = 10–100)** and must not be stated for the unsupervised case.
- The replay arm has only seed 0. A headline needs seed-level uncertainty, which is why this plan adds replay seeds.
- A second reading that needs no new run: ALDI++ at N = 10 scores the same as ALDI++ at N = 0 (0.253 on lighting). Against "do not fine-tune at all", its gain at N = 10 is +0.009, not +0.070. The large Δ(10) is mostly FT's overfitting under the 500-update floor.
- **The share depends on the baseline at N = 10** (from `rq5_vs_zero_shot.csv`, 2026-11-21). Against the source model, replay at N = 10 gains +0.0005 / −0.0014 / +0.0010 (lighting / weather / compound, CI containing 0) while ALDI++ gains +0.009 / +0.003 / +0.011. So at N = 10 the control reproduces FT's avoided damage, not a gain: relative to not fine-tuning, the small gain is all distillation and the 10 labels add nothing. At N = 100 FT is at about zero-shot, and both baselines agree: replay reproduces 86% / 61% / 78% of ALDI++'s gain over the source model (83% / 86% / 78% of its gain over FT). The headline is therefore a statement about the advantage over fine-tuning, and N = 100 is its cleanest evidence.

**The power fit is cured by restricting it to the positive region.** Same function and data, different rungs:

| Shift | Rungs | b | R² | RMSE | Fit N75 | Ladder bracket for 0.75 |
|---|---|---:|---:|---:|---:|---|
| lighting | all (10 rungs) | 0.25 | 0.993 | 0.057 | 2,530 | (2,000; 5,000] |
| lighting | N ≥ 250 (6 rungs) | 0.02 | 1.000 | 0.001 | 3,540 | (2,000; 5,000] |
| weather | N ≥ 250 (3 rungs) | 0.12 | 1.000 | 0.001 | 960 | (500; 1,000] |
| compound | N ≥ 250 (3 rungs) | 0.01 | 0.985 | 0.017 | 650 | (500; 1,000] |

- On the positive region b falls to the lower edge of the search grid (0.01–0.02). The b → 0 limit of this form is **log-linear**: Recovery = α + β·ln N. That is the model the data support.
- Out-of-range check on lighting (corrected 2026-11-21 after the code review): the log-linear fit on N = 250, 500, 1,000 predicts 0.663 / 0.812 / 0.924 at N = 2,000 / 5,000 / 10,000, against observed 0.660 / 0.804 / 0.911, and 1.020 at the ceiling rung. The errors grow with N (+0.003, +0.008, +0.013, +0.020). The "three-digit" agreement first written here came from the power form anchored at the ceiling and is withdrawn.
- With only three positive rungs, weather and compound cannot support a fit-based threshold. Report the ladder bracket there.
- All of this is anchored to the 3-epoch ceiling and must be recomputed once the 12-epoch ceilings exist.

**Driver review (2026-11-21, three independent reviewers).** No bug; nothing to stop or redo. Operational risks were closed in the driver (single-instance guard, tail repair, atomic writes, kept ceiling weights) and in `HANDOFF.md`. Two seed gaps the rules leave open:

- The E3 matched-pool ceiling (1,738) for lighting and weather had seed 0 only, while E12 has seeds 0/1. Closed: the user approved two more runs, `fine-tune_{lighting,weather}_N1738_s1` at E = 3 (2,607 updates each), recorded in the `DECISIONS.md` addendum before any scoring. They are block `matched_pool_e3`, chained after the main queue. The plan freezes 42 sets.
- Replay at N = 1,000 at the original length has seed 0 only (see step 13).

**Stale guidance.** `CLAUDE.md` still describes Kaggle as the compute, the RTX 5060 as the local GPU, "the contribution is the curve" and "N95 is secondary". `README.md` and section 14 of the report point to `YN2TB/CV-final`. These are corrected in Phase 4.

## Decisions

Decided:

- **Kaggle is dropped as a results source** (user, 2026-11-21). The report, README and result notes carry local numbers only. This departs from the lecturer's point 4 (report both equally), so the user should tell the lecturer.

Proposed, to be appended to `.agent/DECISIONS.md` on approval and **before any new run**:

- **Keep one disclosure sentence about Kaggle.** Held-out labels were first opened on 2026-11-03 by a Kaggle campaign under a different schedule (E=2). Removing the comparison is fine; removing that fact is not, because it changes how many times the test set has been opened.
- **Ceiling definition.** mAP_c := seed-mean of the full-pool fine-tune at **E = 12** (the source model's epoch count), every other setting as `campaign.POLICY`, final update kept. It replaces the E = 3 ceiling as the recovery denominator **whatever the result**, and recovery stays unclipped. The E = 3 full-pool runs become the top rung of the ladder.
- **"Materially higher"** := paired difference (E12 − E3) ≥ 0.005 AP with the lower 95% bound above 0. This is the δ rule already used for RQ5. Both threshold tables are published either way.
- **Bracket reading.** Ladder rungs stay at E = 3. Thresholds against the E3 ceiling are lower bounds; against the E12 ceiling they are upper bounds, to the extent that large lighting rungs are themselves undertrained. The pilot shows N = 1,000 is saturated at E3 (0.2780, vs 0.2797 at E5 and 0.2772 at E10).
- **RQ5 matched-pass rule.** At N = 1,000: T_run = 3N = 3,000 updates for `adapt` and `replay`, so labeled target image-passes equal FT's 3,000. Seeds 0/1/2 for both arms. Analysis rules of 2026-11-03 unchanged. The original runs stay in the report as the pre-registered result; the matched runs are labeled as the confound control.
- **Replay seeds.** `replay` at N = 10 and 100 gains seeds 1 and 2, so the decomposition has the same seed structure as `adapt`.
- **Thresholds.** N95 is retired. A threshold whose first passing rung is the ceiling itself is reported as "not determined within the pool (> largest rung)".
- **Fit.** Log-linear on rungs at or above the sustained N_BE where the curve is monotone, at least four rungs, R² computed on those rungs only, with a held-out-rung check. Fewer than four rungs: no fit, ladder bracket only.
- **Third opening of held-out labels.** The new prediction sets are scored once from one new frozen plan with bootstrap seed 20261030, so they pair with the existing scored outputs. Existing sets are not rescored.

## Scope and non-goals

**In scope:** the new runs below, one new frozen scoring plan, analysis changes, the rewritten report, and the stale-guidance cleanup.

**Not in scope:**

- Any change to the split, the source model, the ladder runs or the locked E = 3 ladder schedule.
- A second adaptation method.
- Use of weather or compound val for anything.
- Rescoring or re-selecting any existing run.

## Ordered work

### Phase 0: record and pre-register (no GPU, ~1 h)

1. Save the review as `docs/review_2026-11-21_lecturer_vi.md`.
2. Append the decisions above to `.agent/DECISIONS.md`, dated before the first new run.
3. Update `.agent/HANDOFF.md` and `.agent/TODO.md`.

### Phase 1: analysis and text that need no new run (CPU, ~half a day)

These use only the already-scored outputs.

4. **Point 6:** remove N95 from the reported levels, tables and docs (in `scripts/analyze_campaign.py`; `recovery.LEVELS` cannot be edited, see step 5). Add the "not determined within the pool" status to the threshold rows.
5. **Point 5:** add the log-linear fit with the rung restriction and the held-out-rung check. Keep the old power fit out of the report.
   - Checked 2026-11-21: every run hashes all of `scripts/bdd_pipeline/*.py`, `recovery.py` included. The new fit therefore lives in `scripts/analyze_campaign.py`; nothing under `scripts/bdd_pipeline/` or `scripts/bdd_adapt/` is edited.
6. **Point 3:** add the minimum detectable interaction to `analyze_campaign.py`: CI half-width (0.014 AP, 63% of the weather main effect) and the 80%-power value (about 0.020 AP, 90%). Rewrite the claim from "no evidence of super-additivity" to "an interaction smaller than X cannot be detected here".
7. **Point 2, wording only:** soften the RQ5 summary now. The supported statement is that under the pre-registered update rule ALDI++ has no advantage from N = 1,000, and that the deficit sits entirely in the replay arm.
8. Add the replay-share table (Discoveries) to `scripts/analyze_rq5.py` with a bootstrap CI.
9. Tests for 4–6 and 8 in `tests/`; the existing 44 + 8 must still pass.

### Phase 2: new runs (GPU, needs explicit go-ahead)

Rates measured on this machine: FT 8.45 img/s at batch 2, `adapt` ~0.6 s/update, prediction ~0.083 s/image. Replay step time is estimated at 0.35 s.

| Block | Runs | What | GPU time |
|---|---:|---|---:|
| **A. Ceilings at E = 12** (point 1) | 10 | Full pool, seeds 0/1: lighting, weather, compound. Matched pool 1,738, seeds 0/1: lighting, weather (also fixes the 1-seed limitation of RQ4). | ~22 h, of which lighting full pool is 14 h |
| **B. RQ5 matched passes** (point 2) | 18 | `adapt` and `replay` at N = 1,000 with 3,000 updates, 3 shifts × 3 seeds. Needs no code change: `--floor-updates 3000`. | ~9 h |
| **C. Replay seeds** (reframing) | 12 | `replay` seeds 1/2 at N = 10 and 100, 3 shifts, original run length. | ~4 h |
| D. No-floor FT at N ≤ 100 (optional) | 36 | FT with updates proportional to N. Section 13 of the report lists it; Discoveries shows its main message is already available without it. | ~4 h |
| E. Lighting N = 5,000 matched (optional) | 4 | `adapt` × 3 seeds and `replay` × 1 at 15,000 updates. | ~9 h |
| F. Lighting N = 5,000 and 10,000 at E = 12 (optional) | 6 | Closes the bracket in the ceiling decision for the large rungs. | ~18 h |

- **Recommended: A + B + C, about 35 GPU-hours,** roughly two days unattended.
- One driver script in the style of `scripts/local_campaign.py` (skip finished, resume interrupted, stop on first failure), outputs under `runs/local/revision/`.
- Order: C, B, then A with lighting last, so the short blocks finish first.
- Optional sanity read that touches no test label: lighting E12 vs E3 full-pool checkpoints on lighting val. It informs nothing and selects nothing; it only warns early if E = 12 overfits.

### Phase 3: score once and re-analyse (CPU, ~5 h scoring)

10. Pre-freeze checklist, each run validated from its own files (not from `session_summary.json`):
    - the log says `REVISION RUNS COMPLETE` twice (main queue, then the chained `matched_pool_e3` block), and 42 manifests match the glob `runs/local/revision/grid/*/*/predictions/prediction_manifest.json`;
    - each manifest: `run.completed`, `run.global_step == run.total_updates ==` the pre-registered count, `test_labels_accessed: False`;
    - `implementation_sha256` in every `train/config.json` and manifest equals the original grids' set for its package, and `git diff` is empty for `scripts/bdd_pipeline/`, `scripts/bdd_adapt/`, `scripts/train_*.py`, `scripts/predict_frcnn.py`, `scripts/prepare_revised_bdd100k.py`;
    - the 10 ceilings each have a loadable `train/weights.pt`, and no run still has `last.pt`;
    - the new plan's `evaluation_implementation_sha256` equals the one in `runs/local/campaign/final_plan.json`.
11. Freeze `runs/local/revision/final_plan.json` over all new prediction sets. Score once with `evaluate_frcnn.py --final-plan` (`--resume-interrupted` only if interrupted).
12. Extend `analyze_campaign.py` to take the ceiling from the E12 runs and to treat the E3 full-pool runs as a ladder rung. Emit both threshold tables and the paired E12 − E3 difference with CI.
    - 20 of the 40 revision runs share stage/domain/budget/seed with an original run. Load the scored directories separately and split on `run.epochs` and `run.total_updates`; never key on the four values alone.
    - E3 matched-pool ceilings: seed 0 is in `runs/local/campaign/scored/`, seed 1 comes from the revision plan (block `matched_pool_e3`).
13. Extend `analyze_rq5.py`: matched-pass Δ(1,000), three-seed attribution, replay share with CI.
    - Replay at N = 1,000 at the original 2,000-update length stays seed 0 only, so share(1,000) over three seeds exists only under matched passes. Say so.
    - Disclose EMA lag with the matched-pass result: the scored model is the teacher, which after 3,000 updates still carries 0.9996^3000 = 0.30 weight on the source initialization (0.45 after 2,000). Matching labeled passes equalizes the student's exposure, not the teacher's.
14. Recompute the fit and the weather detection limit against the new ceilings.

### Phase 4: rewrite (no GPU, ~1 day)

15. **Report structure** (`docs/bao_cao_ket_qua_vi.docx`, and the Claude Doc copy):
    - Title and summary lead with the control-arm claim, scoped to N = 10–100, tied to ALDI's thesis.
    - Order: decomposition (the result), then Δ(N), crossover and N_eq (how big, in label units), then the FT data-efficiency curves (the context that makes it measurable), then thresholds (secondary, practical).
    - Weather: third replication of the decomposition; its threshold cell carries the detection limit.
    - Section 11 (Kaggle comparison) and every other mention of the Kaggle runs removed, with no disclosure sentence (user decision 2026-11-23). The text may say each frozen plan was scored once; it must not say the labels were opened only once in total.
    - Thresholds: both ceilings, the bracket reading, no N95, "not determined within the pool" where it applies.
    - Section 14 points to this repository.
16. Update `docs/results_local_rq1_rq4_vi.md` and `docs/results_rq5_vi.md` to match; write `docs/review_response_2026-11-21_vi.md` answering the review point by point.
17. **Kaggle cleanup on `main`** (done 2026-11-23): removed `kaggle/`, `scripts/kaggle_materialize.py`, `docs/kaggle_training.md`, `docs/results_rq1_rq4_vi.md`. `scripts/kaggle_source.py` was in the original glob by mistake and is kept as the data-provenance check, renamed `scripts/verify_source_mirror.py`. Nothing imports them (checked); git history keeps them. Keep every mention of the Kaggle *mirror* as the data source, which is provenance, not compute.
18. Rewrite `CLAUDE.md` (compute, framing, thresholds) and the README summary.

## Validation criteria

- All tests pass after each phase.
- Every new run: update count equals the pre-registered count; `test_labels_accessed: False` until the plan is frozen.
- Scoring: one `.opened` lock for the revision plan; entry count equals plan entries; bootstrap vectors align with the existing scored sets (same seed, same image lists).
- Sanity: recovery of the E12 ceiling equals 1; the E3 full-pool rung reproduces the old ceiling mAP (0.296 / 0.336 / 0.257).
- Every number in the rewritten report traces to a CSV under `runs/local/campaign/analysis_rev2/` or `runs/local/rq5/analysis_rev2/`.

## Recovery, interfaces, dependencies

- Runs resume from `last.pt` with the original arguments; re-running the driver is safe.
- If the machine reboots mid-scoring, continue the same pass with `--resume-interrupted`; never start a second pass.
- Disk: 382 GB free; the new runs need about 10 GB.
- Planning approval does not authorize Phase 2 or Phase 3 (`.agent/PLANS.md`). Each needs an explicit go-ahead.

## Risks

- **E = 12 may overfit the small pools.** The pilot at N = 1,000 was flat from E3 to E10, so weather and compound ceilings may barely move or drop slightly. The rule above fixes the reading in advance for either direction.
- **The matched-pass run may not close the gap.** Then "fine-tuning overtakes ALDI++ from N = 1,000" stands on firmer ground. If it closes, the crossover claim is withdrawn and the report says so.
- **Replay seeds 1/2 may shift the shares.** The headline wording is fixed only after Phase 3.
- **Third opening of the test set.** Disclosed; the new sets answer questions fixed before scoring.

## Progress

- 2026-11-21 09:5x: review verified against `runs/local/*/analysis/`; refit and replay-share computed from existing CSVs; plan written.
- 2026-11-21 10:1x: Phase 0 done. Review saved (`docs/review_2026-11-21_lecturer_vi.md`); rules appended to `DECISIONS.md` before any run.
- 2026-11-21 10:36: Phase 2 launched (`scripts/local_revision_runs.py`, 40 runs, `runs/local/revision.log`). Each run differs from its original in one argument (asserted by `tests/test_revision_runs.py`, 12 tests).
- 2026-11-21 10:48: run 1 (`replay_compound_N10_s1`) finished: 2,000 updates, 0.247 s/update, predictions on 756 + 2,650 images, `test_labels_accessed: False`, labeled subset identical to fine-tuning seed 1.
- 2026-11-21 10:52: driver reviewed by three independent agents; no stop, no redo. Driver hardened for relaunch.
- 2026-11-21 16:36: the two result notes and `docs/training_pipeline.md` updated for points 2, 3, 5, 6; every new number checked against the CSVs by script (five rounding errors found and fixed). `analyze_rq5.py` gained the share against the source model and the stricter headline rule of addendum 2 (151 tests).
- 2026-11-21 ~16:38: machine froze and rebooted (16:41) at run 20 of 40. Likely cause: a runaway `nvidia-smi dmon` of KDE's `ksystemstats` (four OOM kills earlier that day).
- 2026-11-21 16:51: 19 finished runs re-verified by hash; queue relaunched with 42 runs in its plan; run 20 resumed from step 2,500.
- 2026-11-22 20:44: all 42 runs complete.
- 2026-11-23 08:5x: Kaggle run code removed (12 files; `scripts/kaggle_source.py` kept), README and CLAUDE.md updated. Pre-freeze checklist passed for all 42 runs; plan frozen (sha256 `de135bcc…b10282`); one-shot scoring started 08:53.
- 2026-11-23 10:28: one-shot scoring done, 84 of 84 entries.
- 2026-11-23 13:3x: Phase 3c done by an 8-agent workflow (paused once by the user and resumed): `--revision-scored` in both analysis scripts, 239 tests, regression re-run by the coordinator. Results and post-hoc reporting rules in `DECISIONS.md`. Both result notes updated with tables generated from the CSVs. The report itself is left to the user (template).
- 2026-11-21 11:35: Phase 1 code done by a 10-agent workflow (2 implementers, 6 reviewers, 2 fixers): `scripts/analyze_campaign.py`, `scripts/analyze_rq5.py`, 54 + 38 tests. Every pre-existing table reproduces byte for byte; new outputs in `runs/local/campaign/analysis_rev1/` and `runs/local/rq5/analysis_rev1/`. Reviewer-raised rule questions settled in `DECISIONS.md` addendum 2.
- 2026-11-21 11:03: user approved the two E3 matched-pool seed-1 runs; `DECISIONS.md` addendum written, block `matched_pool_e3` added (13 driver tests), chained to start after the main queue.

## Outcome

Phases 0-3 complete on 2026-11-23; the report (Phase 4) is left to the user.

- **Point 1 (ceiling):** no 12-epoch ceiling is materially higher than its 3-epoch one; thresholds are reported against both.
- **Point 2 (RQ5 at N = 1,000):** under matched labeled passes fine-tuning stays ahead on lighting and compound; the claim is withdrawn on weather.
- **Points 3, 5, 6:** detection limit of the interaction, log-linear fit on the positive region, N95 retired and "not determined within the pool" introduced.
- **Point 4:** the Kaggle runs are dropped without a disclosure sentence (user decision; the user informs the lecturer).
- **Reframing:** the headline claim is supported at N = 10 and N = 100 with three replay seeds, as a statement about the advantage over fine-tuning.
- **Not done:** `docs/bao_cao_ket_qua_vi.docx`, the reply to the lecturer and the README introduction; optional blocks D, E, F were not run.

## Appendix: the review as received (2026-11-21)

1. Schedule 3 epoch làm ceiling bị undertrain, nên mọi threshold đang optimistic. Ceiling lighting chỉ được tầm 54k image-pass, trong khi source model được gần 120k và schedule 1x tiêu chuẩn là tầm 1,4 triệu. Ceiling thấp -> mẫu số nhỏ -> recovery bị thổi lên -> N75/N90 báo cáo là lower bound của nhu cầu thật. Cần chạy lại ceiling 12 epoch cho compound, weather, lighting. Nếu ceiling 12 epoch cao hơn nhiều thì báo cáo lại threshold; nếu gần bằng thì cũng loại bỏ được phản biện trên.
2. Kết luận RQ5 ở tóm tắt mạnh hơn bằng chứng. Ở 12.3, N = 1.000 fine-tune thấy tầm 3.000 lượt ảnh có nhãn còn ALDI++ chỉ tầm 2.000. Bảng phân tách thì adapt − replay ≈ 0, toàn bộ phần thua nằm ở replay. Nên chạy luôn thí nghiệm đề xuất ở mục 13.
3. Ô Weather cho ra gần như rỗng. Ceiling − zero-shot chỉ 0,016 AP, chỉ gấp 1,6 lần ngưỡng 0,01 mà mình đặt ra để coi là "không xác định". Traffic sign đã rơi xuống dưới (0,0089). Hệ quả: CI của tương tác là (−0,012; +0,016), nửa độ rộng 0,014, trong khi hiệu ứng chính của weather chỉ 0,022. Phép kiểm chỉ phát hiện được tương tác bằng hai phần ba toàn bộ hiệu ứng weather. Do vậy nên ghi "tương tác nhỏ nhất phát hiện được" thay vì kết luận "không có bằng chứng".
4. Báo cáo cả hai lần chạy Kaggle và local như nhau, không để một bên là chính một bên là phụ.
5. R² bị thổi lên nhờ các điểm âm và fit lạc quan (lighting 2.530 so với ladder 5.000). Fit lại chỉ trên vùng dương và đơn điệu.
6. N90 không xác định được cho weather và compound trong phạm vi pool hiện có. Cột N95 bỏ được.

Đóng góp quan trọng là nhánh điều khiển, không phải các con số. Các em diễn đạt lại:
"Hầu hết những gì mà thích ứng miền (Domain Adaptation - DAOD) mang lại với small labeled budgets không phải là thích ứng thực sự."

Luận điểm này liên quan trực tiếp đến luận điểm của ALDI (rằng gain của DAOD bị ảnh hưởng bởi các lựa chọn triển khai). Cái này có thể được kiểm chứng bởi người khác, các đường cong hiệu quả dữ liệu trở thành bối cảnh giúp đo lường được, và các ngưỡng được đặt ở vị trí kết quả thực tiễn thứ cấp. Nó cũng làm giảm bớt ảnh hưởng của nhánh thời tiết gần như bằng không. Với tư cách là replication thứ ba của quá trình phân tích thì nó xứng đáng có vị trí của mình, trong khi với tư cách là phép đo ngưỡng thứ ba, nó lại làm giảm giá trị của bài báo.
