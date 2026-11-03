# CLAUDE.md

Guidance for Claude Code in this repository. Rewritten 2026-10-30. The previous version described a pre-review design (rainy+snowy pooled, val used as test, data under `D:/CV`) that no longer applies.

## What this is

This is a research project that measures **how much labeled target-domain data a detector needs to recover from domain shift**. It uses BDD100K, with daytime+clear as the source and three targets: lighting (night+clear), weather (daytime+rainy) and compound (night+rainy).

The contribution is the **target-data-efficiency curve**: recovery as a function of the budget N, its thresholds, and the point where plain fine-tuning catches up with a real adaptation method. It is *not* "domain shift hurts detectors" and *not* "BDD100K day→night adaptation". Keep every framing, plot and write-up pointed at the curve.

## Where the truth lives

| What | Where |
|---|---|
| Protocol in force | `docs/proposal_dataset_aligned.docx` (team-aligned). `docs/proposal.docx` is the last version sent to the lecturer, and is older. |
| Durable decisions | `.agent/DECISIONS.md` |
| Current state and next action | `.agent/HANDOFF.md`, `.agent/TODO.md`, `.agent/plans/active/` |
| Coding, testing and commit rules | `AGENTS.md` |
| Pipeline usage | `docs/training_pipeline.md`, `docs/kaggle_training.md` |
| Lecturer reviews and our answers | `docs/response/` (screenshots), `docs/review_response_*_vi.md` |

Read `AGENTS.md`, then `.agent/HANDOFF.md`, then `TODO.md`, then `DECISIONS.md` before changing anything.

## Protocol essentials

These hold as of the 2026-10-19 amendment and the 2026-10-20 Kaggle runs.

- **Split:** a stratified 55,000 train / 14,863 held-out test / 10,000 official val. Membership is frozen by manifests and SHA256 (`data/bdd100k_revised/`).

| domain | train pool | final test | selection val |
|---|---|---|---|
| source (day+clear) | 9,804 | 2,650 | 1,764 |
| lighting (night+clear) | 18,015 | 4,869 | 3,274 (schedule pilot only) |
| weather (day+rainy) | 1,985 | 933 (test + val) | none |
| compound (night+rainy) | 1,738 | 756 (test + val) | none |

- **Never** use weather or compound val for selection; those images are test. Never open held-out labels outside a frozen, one-shot final plan.
- **Detector:** Faster R-CNN R50-FPN, initialized from COCO, at native 1280×720 (`min_size=720, max_size=1280`). It was chosen because of RQ5, since public DA implementations target it. It was not chosen because of the smoke test.
- **Source model:** trained once on Kaggle (12 epochs, batch 4, AMP). The checkpoint is selected on source val, with reliable mAP50:95 = 0.352.
- **Target runs** (budget, ceiling, control) all keep the **final update** of one fixed schedule: `max(S updates, E epochs)`. S, E, LR and the batch size are provisional until the lighting-val pilot locks them in `DECISIONS.md`.
- **Primary metric:** mAP50:95 over the fixed roster pedestrian, car, traffic light, traffic sign. It is `null` if any of these classes is missing. The 10-class mean is a sensitivity analysis.
- **Recovery(N)** = (mAP(N) − mAP(0)) / (mAP_c − mAP(0)), where `mAP_c` is the full-pool fine-tune ceiling, **never `mAP_source`**. Recovery is not clipped, and the minimum denominator is 0.01.
  - The primary thresholds are **N_BE, N75 and N90**. N95 is secondary.
  - Recovery is reported per reliable class as well, to separate re-learning the class prior from learning new visual features.
  - RQ4 cross-shift claims rest on matched-pool ceilings at 1,738 images.
- **Zero-shot gate** (passed 2026-10-20). Gaps: weather 0.031 AP (CI 0.023–0.040), lighting 0.099, compound 0.129. Weather stays in the study but must always be reported with CI and the raw change.
- **Labels:** the 2018 BDD100K release (`person`/`bike`/`motor` legacy names, mapped to canonical names). It is officially deprecated in favour of det_20; staying on 2018 is a recorded decision.

## Data and compute

- **Dataset:** the local BDD100K copy, moving into `data/` from `D:/BDD100K-weather-classification/data`. Kaggle kernels instead link images from the public mirror `solesensei/solesensei_bdd100k` and check them against the frozen manifests (`kaggle/common.py`, `scripts/kaggle_materialize.py`).
- **GPU work runs on Kaggle** (account `kizhowl`, private kernels, 30 h/week shared quota, 12 h sessions; long runs resume). A T4 does 6.4 img/s at batch 4 with AMP. **Ask before running anything on the local RTX 5060 Laptop (8 GB).**
- If the local GPU is ever approved, note that it is Blackwell (`sm_120`). Install with:

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
- Never commit `data/`, `runs/`, weights or credentials.

## Project layout

This project has not been through the ICM restructure yet. Work in the current layout (`scripts/bdd_pipeline/`, `kaggle/`, `tests/`, `docs/`, `.agent/`) and don't invent new top-level structure.
