# Quantifying Target-Domain Data Requirements for Object Detection under Domain Shift

This repository supports a BDD100K study of a practical deployment question: how much labeled target-domain data is needed to recover object-detection performance after domain shift, and when does domain adaptation justify its added complexity over conventional fine-tuning?

The study does not propose a new detector. It measures annotation efficiency under controlled lighting, weather, and compound shifts while keeping the detector, data partitions, evaluation path, and target test sets fixed.

## Research Design

The current proposal uses a materialized, stratified split of the labeled BDD100K images:

- 55,000 images for training;
- 14,863 images for held-out final testing;
- the unchanged official 10,000-image validation split for pilot decisions and checkpoint selection.

The public unlabeled BDD100K test split is excluded. Daytime-clear is the source domain. The target domains are:

| Shift | Target condition | Train pool | Final test | Selection val |
| --- | --- | ---: | ---: | ---: |
| Source | Daytime, clear | 9,804 | 2,650 | 1,764 |
| Lighting | Night, clear | 18,015 | 4,869 | 3,274 |
| Weather | Daytime, rainy | 1,985 | 933 (537 test + 396 val) | 0 |
| Compound | Night, rainy | 1,738 | 756 (470 test + 286 val) | 0 |

**Protocol amendment 2026-10-19** ([review response](docs/review_response_2026-10-19_vi.md)): the weather and compound held-out sets were below the 600-image target, so their preserved-validation images join the final test set; physical folders and training pools are unchanged. Those two domains have no selection validation, so every target run keeps the final update of the common fixed schedule. The primary metric uses one fixed reliable-class roster for every domain (pedestrian, car, traffic light, traffic sign); the ten-class mean is a sensitivity analysis. The pilot measures each shift's zero-shot gap first, so a shift without a gap (Recovery denominator near zero) is flagged before its budget ladder runs.

Target training budgets are nested, repeated across fixed seeds, and sampled from the training partition only. The main ladder is `10, 25, 50, 100, 250, 500, 1,000, full`; the lighting arm also includes `2,000, 5,000, 10,000` because its full pool is much larger.

## Evaluation

The experiment compares source-only inference, supervised target-domain fine-tuning, and one established adaptation method selected after the pilot. Faster R-CNN with a ResNet-50 FPN backbone is the current primary detector candidate because it has broad support in public domain-adaptation implementations.

The primary recovery quantity is, overall and per reliable class:

```text
Recovery(N) = [mAP(N) - mAP(0)] / [mAP_c - mAP(0)]
```

Here, `mAP(0)` is source-only performance on the target test set and `mAP_c` is the empirical full-target-pool ceiling produced by the same fine-tuning procedure. The study estimates break-even, 75%, and 90% recovery budgets; 95% is a sensitivity result. Primary reporting uses COCO-style mAP50:95 over the four fixed reliable classes, with ten-class metrics, seed variation, bootstrap intervals, source-domain retention, and per-class results as diagnostics.

## Repository Status

The research protocol is defined in [`docs/proposal_dataset_aligned.docx`](docs/proposal_dataset_aligned.docx). Project state and execution plans live under [`.agent/`](.agent/README.md).

`train_da_yolo_comparison.py` is an earlier orchestration prototype for SSDA-YOLO, MS-DAYOLO, and SF-YOLO. It assumes a different split and model-comparison design, including a labeled BDD100K test file that is not part of the current proposal. Do not treat it as the executable implementation of the present study without reconciling those differences.

The revised dataset has now been materialized locally at `data/bdd100k_revised/`
from the original BDD100K source files, verified individually against the public
Kaggle archive (solesensei_bdd100k version 2). Its labels are the 2018 BDD100K
detection release (`bdd100k_labels_release`), which the official documentation now
marks as deprecated in favour of Detection 2020 (`det_20`). Kaggle checksums only
show that local files equal the mirror; `scripts/label_provenance.py` freezes a
label sample for comparison with an official download, which is still pending. The 55,000/14,863/10,000 image, annotation, and manifest counts
pass validation, and all four main study-domain counts match the table above.
The split uses seed 20260916, hierarchical largest-remainder allocation over
timeofday/weather/scene, and deterministic same-stratum swaps to keep exact
duplicate content in one split. Cross-split basename and SHA256 overlap are zero.
Official validation membership is preserved; within-split duplicate images are
disclosed in the dataset metadata.

The EDA reads the new train and val detection annotations only. Its executed
[notebook](notebooks/eda_revised_bdd100k.ipynb) covers domain and class support,
box geometry, scene composition, image appearance, and nested training-budget
coverage. The current local Vietnamese report is `runs/eda_revised_v2/report_vi.html` (amended protocol, adds occlusion/truncation and objects-per-image tables by domain);
figures, CSV tables, and a Markdown version are alongside it. See the
[workflow and reproduction commands](docs/revised_dataset_eda.md).

The Python [training/evaluation pipeline](docs/training_pipeline.md) now implements
Faster R-CNN ResNet-50 FPN at native 1280×720, train-only nested budget sampling,
source/target/control initialization, the common step-floor/epoch rule, complete
atomic checkpoint resume and COCO AP/AR with per-class support. CLI entry points
are `scripts/train_frcnn.py`, `scripts/pilot_frcnn.py`,
`scripts/evaluate_frcnn.py` and `scripts/freeze_evaluation_plan.py`.

CPU train/validation smoke and exact resume equivalence have been verified.
The [Vietnamese implementation report](docs/pipeline_report_vi.md) distinguishes
these checks from research performance. Default optimizer/schedule/support
settings are provisional; the next milestone is a representative GPU throughput
and memory pilot and fixed research configuration. Full training runs, final
holdout performance evaluation and adaptation-method selection remain later
research steps.

## GPU Training on Kaggle

GPU runs use private Kaggle kernels on the `kizhowl` account. The frozen protocol (code, annotations, manifests, hashes) is a private Kaggle dataset; images are symlinked from the public mirror and rehashed against the manifests before any run. See [docs/kaggle_training.md](docs/kaggle_training.md).

## Local Checks

The existing prototype uses only the Python standard library:

```powershell
python train_da_yolo_comparison.py --help
python -c "import ast, pathlib; ast.parse(pathlib.Path('train_da_yolo_comparison.py').read_text(encoding='utf-8'))"
```

Datasets, generated manifests, checkpoints, external repositories, and run outputs must remain untracked.

The dataset/EDA checks are:

```powershell
python -m unittest discover -s tests -v
python scripts/prepare_revised_bdd100k.py --output-dir data/bdd100k_revised --validate-only
```

Inspect the new pipeline without training:

```powershell
python scripts/train_frcnn.py --output-dir runs/source_candidate --dry-run
python scripts/pilot_frcnn.py --help
python scripts/evaluate_frcnn.py --help
```

Training dependencies are in `requirements-training.txt`. Final-test evaluation
requires a frozen checkpoint plan; training and ordinary validation evaluation
do not open held-out detection annotations.
