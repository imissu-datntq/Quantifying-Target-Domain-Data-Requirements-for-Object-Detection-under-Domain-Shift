# Revised BDD100K dataset and EDA

Status: complete. Started and completed 2026-10-01 Asia/Saigon.

## Purpose and observable outcome

Create a fresh 55,000/14,863/10,000 image dataset from the Kaggle original and
produce an executed EDA notebook, Vietnamese report, figures and reusable scripts.

## Context

Protocol: docs/proposal_dataset_aligned.docx. User explicitly authorized a new
careful split from the supplied Kaggle original and EDA exclusively on the new
dataset. The existing source image files are in another local project; their
bytes must match the public Kaggle archive before reuse. Existing derived
subsets and training results are not inputs.

## Scope and non-goals

Stratify by timeofday, weather and scene with seed 20260916. Preserve official
validation membership. Exclude the original unlabeled test set and 137 unannotated
training images. EDA uses only newly materialized train and val labels; new test
access is limited to construction, structural integrity and metadata census.
No training, tuning, test performance evaluation, or GitHub publication.

## Progress and discoveries

- Repository cloned at 52898b14269cf55f3fa06cebf7cccd82de5b8e02; proposal read.
- Public Kaggle dataset version 2 exposes HTTP ranges and ZIP size/CRC entries.
- Original local images are distributed among subfolders: flatten by validated
  basename rather than assuming all images are immediate children.
- Full Kaggle download is slow; verify every reusable source file against its
  central directory size and CRC, then copy verified images to the fresh split.
- 2026-10-01 10:52: all 80,002 source image/annotation files passed the public
  Kaggle ZIP size and CRC32 check. Local SHA256 recorded for every file.
- Found three exact duplicate groups in the original source: one train pair
  and two val pairs. The initial train/test metadata draw separated the train
  pair; one deterministic same-stratum singleton swap reunites it while
  preserving every stratum quota. No original train/val content overlap.
- 2026-10-01 11:00: 60,000 newly copied images verified; construction running.
- Six protocol tests passed, including rejection of exact-content leakage,
  grouped duplicate repair, deterministic split, parsing and absence probability.

## Decisions

Use hierarchical largest-remainder apportionment: allocate train quotas by
timeofday/weather, then by scene within each cell. This jointly stratifies all
three metadata variables while achieving exactly 55,000 training images.
Record the algorithm and full membership; a seed alone does not identify a split.
Check SHA256 content overlap as well as basename overlap. Copy images so the new
dataset is independent of the source location.
Retain original within-split duplicates to preserve the requested image totals
and official validation membership, disclose them, and keep each content group
wholly in one split with quota-preserving same-stratum swaps.

## Ordered work

1. Verify source files against Kaggle ZIP directory and record provenance.
2. Inventory annotations; compute deterministic stratified membership.
3. Copy images, write detection annotations and manifests; validate invariants.
4. Analyse new train/val: domain sizes, classes, geometry, scene, image appearance,
   and nested training-only annotation budgets. Document test metadata only.
5. Execute notebook, inspect plots, write report and update project state.

## Validation criteria

Exactly 55,000 train, 14,863 test, 10,000 val; exact image-label membership;
no duplicate basenames or cross-split SHA256 overlaps; unchanged val membership;
train/test proportions within integer-apportionment tolerance for each stratum;
all image files readable with dimensions checked; budgets nested per seed and
exclude val/test. EDA never opens new held-out detection labels.

## Recovery, interfaces and dependencies

Standard-library source verification and split scripts. EDA uses installed numpy,
pandas, scipy, matplotlib, Pillow, nbformat and nbclient. Data and generated outputs
stay ignored. Write a completion marker only after validation. Refuse to overwrite
a completed split whose configuration differs; preserve failed outputs for diagnosis.

## Outcome

Completed construction, full-image rehash validation, EDA and notebook execution.
All 55,000/14,863/10,000 counts and four study-domain counts match the current
protocol. Basename/content overlap across split pairs is zero; within-split
duplicates are disclosed. Maximum joint-stratum allocation error is 0.722543
image; train/test joint-stratum TVD is 0.001381 and largest share difference is
0.006175 percentage points. All images are 1280×720.

Six unit tests passed. All eight figures were visually inspected; two legend
positions were corrected and the report/notebook regenerated. The notebook
executed successfully with embedded output, and checks manifest hashes against
the EDA artifacts. The self-contained Vietnamese HTML/Markdown report, CSVs and
nested budget memberships are in `runs/eda_revised/`.

No training or held-out model-performance evaluation was run. The new dataset
and original-source verification artifacts remain ignored. Scripts, executed
notebook, reproduction documentation and project state are ready for review.
