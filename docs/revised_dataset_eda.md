# Revised dataset and EDA workflow

The new dataset is `data/bdd100k_revised/`. Its inputs are the original labeled
BDD100K train and validation files, verified file by file against the ZIP central
directory of [Kaggle dataset version 2](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k).
Existing derived subsets and model artifacts are not inputs to this workflow.

## Source verification

`scripts/kaggle_source.py` checks the byte count and CRC32 of all 80,000 original
train/val images and both raw annotation files against Kaggle. It computes local
SHA256 hashes for construction and leakage checks. CRC32 establishes content
consistency, not a cryptographic signature. This allows reuse of matching original
files without downloading the entire 7.6 GiB archive. A local source directory may
have images in nested folders; basenames must be unique within the original split.

```powershell
python scripts/kaggle_source.py --images-root <original-100k-images> --labels-root <original-labels> --output-dir data/raw/kaggle_verified --workers 12
```

The verification output contains machine-local input paths and remains ignored.
The ZIP directory is cached locally. To refresh it against the remote dataset,
use a new output directory. Do not treat the local cache as a signed provenance
record. The verified source file list records the Kaggle version and dataset URL.

## Deterministic split

```powershell
python scripts/prepare_revised_bdd100k.py --verified-source data/raw/kaggle_verified/source_verified.json --output-dir data/bdd100k_revised --seed 20260916
python scripts/prepare_revised_bdd100k.py --output-dir data/bdd100k_revised --validate-only
```

Of the 70,000 original training images, only 69,863 have annotation records. The
137 others are listed and excluded. Public original test images are not inputs.
Official validation membership is preserved at 10,000 images.

Hamilton largest-remainder apportionment first assigns exactly 55,000 train
places across `(timeofday, weather)` cells, then across `scene` strata inside
each cell. Ties use sorted metadata keys. Sorted basenames in each stratum are
shuffled using one Python `random.Random(20260916)` generator in sorted stratum
order. The complement forms the 14,863-image held-out test set. Class labels and
box counts do not influence membership. Sparse strata incur integer rounding;
the exact allocation and rounding errors are saved.

Exact-content duplicate images must stay together. If an SHA256 group crosses
train/test after metadata sampling, move its members to one split and swap
singleton images in the identical joint stratum. This preserves every stratum
quota and both global counts; moves and donors are deterministic and recorded.
Within-split duplicates are retained and disclosed so that official validation
membership and the proposal's total image counts remain intact.

The algorithm, manifests and their SHA256 hashes identify this new split. A seed
by itself cannot establish membership equivalence with a previously described
split whose allocation or iteration order was different. Use the generated
domain census for subsequent experiment counts.

Images are copied into independent flat directories and checked against the
verified source SHA256. Pillow checks their format/header and dimensions.
Detection JSON keeps original box coordinates and label attributes; aliases are
canonicalized and lane/drivable-area polygons are excluded. Invalid boxes are
preserved in the dataset for an explicit downstream preprocessing policy.

```text
data/bdd100k_revised/
  COMPLETE
  dataset_metadata.json
  images/{train,test,val}/*.jpg
  annotations/bdd100k_labels_images_{train,test,val}.json
  manifests/{train,test,val}.csv
  manifests/stratification.csv
  manifests/duplicate_content_repairs.json
  manifests/domain_census.csv
  manifests/excluded_unannotated.txt
```

Validation asserts 55,000/14,863/10,000 images, annotation records and manifest
rows, identical membership, unique names, and zero basename/SHA256 overlap across
splits. Content hashes do not identify near-duplicates or shared geographic/video
groups. Existing output directories are never overwritten by construction.
`COMPLETE` is written only after the full construction and validation passes.

## EDA exclusively on the new dataset

```powershell
python scripts/eda_revised_bdd100k.py --dataset-root data/bdd100k_revised --output-dir runs/eda_revised --brightness-per-domain 200 --seeds 0 1 2
python scripts/build_eda_notebook.py
```

The EDA script opens new **train and val detection annotations only**; since the 2026-10-19 amendment it skips weather/compound val labels, which are now final-test images (current output: `runs/eda_revised_v2/`). New held-out
test annotations are accessed during construction/structural validation, never
by EDA; test metadata support only split-balance/domain census tables.

Outputs include environmental counts, class prevalence and instance support,
objects per image, bounding-box sizes/centers, occlusion/truncation, scene and
class composition, 200 deterministic train brightness samples per study domain,
train example images, and nested target-budget class coverage for seeds 0/1/2.
Class-absence probabilities use the exact hypergeometric sampling formula.
The raw coordinate policy is preserved: EDA clips only for geometry diagnostics
and excludes invalid/empty boxes from valid-support counts.

The Vietnamese report is `runs/eda_revised/report_vi.html` (self-contained) and
`runs/eda_revised/report_vi.md`. Eight figures and CSV tables are alongside it.
`notebooks/eda_revised_bdd100k.ipynb` is executed with embedded output. It checks
that the current dataset manifest hashes match those used by the EDA artifacts.

This work produces dataset evidence, not detector metrics or recovery thresholds.
Metric policy, throughput pilot, training and final test performance evaluation
remain later research steps. No adaptation method is selected from EDA.

## Checks

```powershell
python -m unittest discover -s tests -v
python -m py_compile scripts/kaggle_source.py scripts/prepare_revised_bdd100k.py scripts/eda_revised_bdd100k.py scripts/build_eda_notebook.py
```

Dependencies for these scripts are listed in `requirements-eda.txt`. Dataset,
source verification, budget manifests and generated runs remain ignored by Git.
