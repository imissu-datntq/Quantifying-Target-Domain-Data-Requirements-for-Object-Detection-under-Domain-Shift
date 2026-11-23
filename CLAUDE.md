# CLAUDE.md

Guidance for Claude Code in this repository. Rewritten 2026-10-30; compute, thresholds and framing updated 2026-11-23 after the local rerun and the lecturer review of 2026-11-21.

## What this is

This is a research project that measures **how much labeled target-domain data a detector needs to recover from domain shift**. It uses BDD100K, with daytime+clear as the source and three targets: lighting (night+clear), weather (daytime+rainy) and compound (night+rainy).

Since the lecturer review of 2026-11-21, the headline is the **control arm**: most of what domain adaptation gives over fine-tuning at small labeled budgets (tested at N = 10 and N = 100) is not adaptation. The **target-data-efficiency curves** (recovery as a function of the budget N) are the context that makes this measurable, and the thresholds are secondary, practical results. Weather is the third replication of the decomposition, not a third threshold measurement. It is *not* "domain shift hurts detectors" and *not* "BDD100K day→night adaptation". The claim rules are in `.agent/DECISIONS.md` (2026-11-21 and its addenda); state the headline only as far as they allow.

## Where the truth lives

| What | Where |
|---|---|
| Protocol in force | `docs/proposal_dataset_aligned.docx` (team-aligned). `docs/proposal.docx` is the last version sent to the lecturer, and is older. |
| Durable decisions | `.agent/DECISIONS.md` |
| Current state and next action | `.agent/HANDOFF.md`, `.agent/TODO.md`, `.agent/plans/active/` |
| Coding, testing and commit rules | `AGENTS.md` |
| Pipeline usage | `docs/training_pipeline.md` |
| Lecturer reviews and our answers | `docs/review_2026-11-21_lecturer_vi.md`, `docs/review_response_*_vi.md` (`docs/response/` screenshots are gitignored) |

Read `AGENTS.md`, then `.agent/HANDOFF.md`, then `TODO.md`, then `DECISIONS.md` before changing anything.

## Protocol essentials

These hold as of the 2026-10-19 amendment, the local rerun of 2026-11-05 and the revision rules of 2026-11-21.

- **Split:** a stratified 55,000 train / 14,863 held-out test / 10,000 official val. Membership is frozen by manifests and SHA256 (`data/bdd100k_revised/`).

| domain | train pool | final test | selection val |
|---|---|---|---|
| source (day+clear) | 9,804 | 2,650 | 1,764 |
| lighting (night+clear) | 18,015 | 4,869 | 3,274 (schedule pilot only) |
| weather (day+rainy) | 1,985 | 933 (test + val) | none |
| compound (night+rainy) | 1,738 | 756 (test + val) | none |

- **Never** use weather or compound val for selection; those images are test. Never open held-out labels outside a frozen, one-shot final plan.
- **Detector:** Faster R-CNN R50-FPN, initialized from COCO, at native 1280×720 (`min_size=720, max_size=1280`). It was chosen because of RQ5, since public DA implementations target it. It was not chosen because of the smoke test.
- **Source model:** trained once on the local RTX 3060 (12 epochs, batch 4, AMP). The checkpoint is selected on source val, with reliable mAP50:95 = 0.352.
- **Target runs** (budget, ceiling, control) all keep the **final update** of one fixed schedule: `max(S updates, E epochs)`, locked by the lighting-val pilot at S = 500, E = 3, LR 1e-3, batch 2 (`DECISIONS.md` 2026-11-05).
- **Primary metric:** mAP50:95 over the fixed roster pedestrian, car, traffic light, traffic sign. It is `null` if any of these classes is missing. The 10-class mean is a sensitivity analysis.
- **Recovery(N)** = (mAP(N) − mAP(0)) / (mAP_c − mAP(0)), where `mAP_c` is the full-pool fine-tune ceiling, **never `mAP_source`**. Recovery is not clipped, and the minimum denominator is 0.01.
  - Since 2026-11-21 the ceiling is the full-pool fine-tune at **E = 12**; the E = 3 full-pool run is the top rung of the ladder. Thresholds are reported against both ceilings side by side, and neither is picked as better; weather against the E = 12 ceiling is reported as not determined (`DECISIONS.md` 2026-11-23).
  - The thresholds are **N_BE, N75 and N90**. N95 is retired. A threshold that only the ceiling rung passes is "not determined within the pool".
  - Recovery is reported per reliable class as well, to separate re-learning the class prior from learning new visual features.
  - RQ4 cross-shift claims rest on matched-pool ceilings at 1,738 images.
- **Zero-shot gate** (passed 2026-11-05). Gaps: weather 0.032 AP (CI 0.024–0.041), lighting 0.100, compound 0.129. Weather stays in the study but must always be reported with CI and the raw change.
- **Labels:** the 2018 BDD100K release (`person`/`bike`/`motor` legacy names, mapped to canonical names). It is officially deprecated in favour of det_20; staying on 2018 is a recorded decision.

## Data and compute

- **Dataset:** the local BDD100K copy under `data/` (raw files hardlinked into `data/raw/`, split in `data/bdd100k_revised/`). `scripts/verify_source_mirror.py` checks the raw files against the public Kaggle mirror `solesensei/solesensei_bdd100k`; that is data provenance and stays.
- **GPU work runs on this machine's RTX 3060 12 GB** (`~/venvs/cv-final/bin/python`, torch 2.11.0+cu128; about 8.4 img/s at batch 2 with AMP). **Ask before starting a GPU job.** Kaggle is no longer used (`DECISIONS.md` 2026-11-05) and its run code was removed on 2026-11-23. **Do not mention the earlier Kaggle runs in the report, the result notes or the README** (user decision 2026-11-23; the user informs the lecturer directly). The Kaggle *mirror* as the source of the dataset is provenance and stays.
- The other machine, an RTX 5060 Laptop (8 GB), is Blackwell (`sm_120`). If it is ever used, install with:

  ```
  python -m pip install <pkg> -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128
  ```

  Then verify with this check; the last statement is the real test:

  ```
  python -c "import torch; torch.zeros(8,device='cuda').sum().item()"
  ```

- `bdd100k_labels_images_train.json` is 1.45 GB. Stream it; never `json.load` it next to a training job.

## Conventions that protect the numbers

- One evaluation path: pycocotools via `scripts/bdd_pipeline/evaluation.py`. `image_id` maps by `file_name`.
- Budgets are nested by seed (N=10 ⊂ N=25), and 3 sampling seeds are fixed. Ceilings use at least 2 seeds.
- Checkpoints are written atomically and resume exactly. Resume uses `last.pt` with the original settings, never a historical best.
- Never edit `scripts/bdd_pipeline/` or `scripts/bdd_adapt/`: every run records the SHA256 of those files, and resume and provenance depend on them. New analysis code goes in `scripts/analyze_*.py`.
- Never commit `data/`, `runs/`, weights or credentials. The one exception is `data/split_reference/` (frozen manifests and hashes, no images or labels; approved 2026-11-05).

## Project layout

This project has not been through the ICM restructure yet. Work in the current layout (`scripts/bdd_pipeline/`, `scripts/bdd_adapt/`, `tests/`, `docs/`, `.agent/`) and don't invent new top-level structure.
