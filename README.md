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

## Current Status (2026-11-05)

| Step | State |
| --- | --- |
| Split, EDA, pipeline | Done. Frozen split in `data/split_reference/`; 44 unit tests pass. |
| Source model | Done. 12 epochs on Kaggle, source-val reliable mAP50:95 = 0.352. |
| Zero-shot gap gate | Passed. Weather 0.031 AP (CI 0.023-0.040), lighting 0.099, compound 0.129. |
| Fine-tuning schedule | Locked by a pre-registered lighting-val pilot: S = 500 updates, E = 2 epochs, LR 1e-3. |
| RQ1-RQ4 (fine-tuning curves) | Done. 87 runs, final test scored once. Report: [`docs/results_rq1_rq4_vi.md`](docs/results_rq1_rq4_vi.md) (Vietnamese). |
| RQ5 (ALDI++ adaptation vs fine-tuning) | Running. Pilot locked EMA α = 0.9996 and T = 2,000 updates; the 46-run grid is on Kaggle (started 2026-11-04). |
| Next | Score RQ5 once, then `scripts/analyze_rq5.py` and `docs/results_rq5_vi.md`. |

The live state, next action and every locked decision are in [`.agent/HANDOFF.md`](.agent/HANDOFF.md), [`.agent/TODO.md`](.agent/TODO.md) and [`.agent/DECISIONS.md`](.agent/DECISIONS.md). The protocol is [`docs/proposal_dataset_aligned.docx`](docs/proposal_dataset_aligned.docx).

`train_da_yolo_comparison.py` is an earlier prototype for SSDA-YOLO, MS-DAYOLO and SF-YOLO with a different split and design. It is not the implementation of this study.

## Getting the Data

The repository holds no images or labels. It holds the frozen split identity in [`data/split_reference/`](data/split_reference/): the membership manifests (`train/test/val.csv`: file name, time of day, weather, scene, image SHA256), the stratification and duplicate-repair records, `dataset_metadata.json`, and `expected_sha256.json` with the hashes of the manifests, metadata and generated annotation files. Labels are the 2018 BDD100K detection release (`bdd100k_labels_release`), which the official documentation marks as deprecated in favour of `det_20`.

1. Download the public Kaggle mirror [`solesensei/solesensei_bdd100k`](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k) (version 2, about 7.6 GiB) and unzip it, for example into `data/raw/`:

   ```powershell
   kaggle datasets download solesensei/solesensei_bdd100k -p data/raw --unzip
   ```

   You need the images under `bdd100k/bdd100k/images/100k/{train,val}/` and the folder that holds `bdd100k_labels_images_train.json` and `bdd100k_labels_images_val.json`. The public test images are not used.

2. Check the files against the mirror (size and CRC32 of all 80,000 images and both label files; SHA256 is computed locally):

   ```powershell
   python scripts/kaggle_source.py --images-root data/raw/bdd100k/bdd100k/images/100k --labels-root <folder with the two label JSONs> --output-dir data/raw/kaggle_verified --workers 12
   ```

3. Build and validate the split (seed 20260916; 55,000 train / 14,863 test / 10,000 official val):

   ```powershell
   python scripts/prepare_revised_bdd100k.py --verified-source data/raw/kaggle_verified/source_verified.json --output-dir data/bdd100k_revised --seed 20260916
   python scripts/prepare_revised_bdd100k.py --output-dir data/bdd100k_revised --validate-only
   ```

4. Confirm that your split is ours: every hash must print `OK`.

   ```powershell
   python -c "import hashlib,json,pathlib; r=pathlib.Path('data/bdd100k_revised'); e=json.loads(pathlib.Path('data/split_reference/expected_sha256.json').read_text()); [print('OK ' if hashlib.sha256((r/k).read_bytes()).hexdigest()==v else 'BAD', k) for k,v in e.items()]"
   ```

How the split is built (hierarchical largest-remainder allocation over time of day / weather / scene, duplicate-content repair, zero cross-split overlap) is documented in [`docs/revised_dataset_eda.md`](docs/revised_dataset_eda.md). Kaggle kernels skip steps 1-3: they link the mirror's images and check them against the same manifests ([`docs/kaggle_training.md`](docs/kaggle_training.md)).

Held-out test labels are opened only by a frozen, one-shot final-evaluation plan. Never use weather or compound validation images for selection: they are part of the final test set.

## GPU Training on Kaggle

GPU runs use private Kaggle kernels on the `kizhowl` account (RQ5 is split with `buihuynhgiahuy`). The frozen protocol (code, annotations, manifests, hashes) is a private Kaggle dataset; images are symlinked from the public mirror and rehashed against the manifests before any run. See [docs/kaggle_training.md](docs/kaggle_training.md).

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
