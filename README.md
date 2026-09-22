# Quantifying Target-Domain Data Requirements for Object Detection under Domain Shift

This repository supports a BDD100K study of a practical deployment question: how much labeled target-domain data is needed to recover object-detection performance after domain shift, and when does domain adaptation justify its added complexity over conventional fine-tuning?

The study does not propose a new detector. It measures annotation efficiency under controlled lighting, weather, and compound shifts while keeping the detector, data partitions, evaluation path, and target test sets fixed.

## Research Design

The current proposal uses a materialized, stratified split of the labeled BDD100K images:

- 55,000 images for training;
- 14,863 images for held-out final testing;
- the unchanged official 10,000-image validation split for pilot decisions and checkpoint selection.

The public unlabeled BDD100K test split is excluded. Daytime-clear is the source domain. The target domains are:

| Shift | Target condition | Train | Test | Validation |
| --- | --- | ---: | ---: | ---: |
| Lighting | Night, clear | 18,015 | 4,869 | 3,274 |
| Weather | Daytime, rainy | 1,985 | 537 | 396 |
| Compound | Night, rainy | 1,738 | 470 | 286 |

Target training budgets are nested, repeated across fixed seeds, and sampled from the training partition only. The main ladder is `10, 25, 50, 100, 250, 500, 1,000, full`; the lighting arm also includes `2,000, 5,000, 10,000` because its full pool is much larger.

## Evaluation

The experiment compares source-only inference, supervised target-domain fine-tuning, and one established adaptation method selected after the pilot. Faster R-CNN with a ResNet-50 FPN backbone is the current primary detector candidate because it has broad support in public domain-adaptation implementations.

The primary recovery quantity is:

```text
Recovery(N) = [mAP(N) - mAP(0)] / [mAP_c - mAP(0)]
```

Here, `mAP(0)` is source-only performance on the target test set and `mAP_c` is the empirical full-target-pool ceiling produced by the same fine-tuning procedure. The study estimates break-even, 75%, and 90% recovery budgets; 95% is a sensitivity result. Primary reporting uses COCO-style mAP50:95 over sufficiently supported classes, with full-class metrics, seed variation, bootstrap intervals, source-domain retention, and per-class results as diagnostics.

## Repository Status

The research protocol is defined in [`docs/proposal_dataset_aligned.docx`](docs/proposal_dataset_aligned.docx). Project state and execution plans live under [`.agent/`](.agent/README.md).

`train_da_yolo_comparison.py` is an earlier orchestration prototype for SSDA-YOLO, MS-DAYOLO, and SF-YOLO. It assumes a different split and model-comparison design, including a labeled BDD100K test file that is not part of the current proposal. Do not treat it as the executable implementation of the present study without reconciling those differences.

The next implementation milestone is a read-only dataset validator followed by a small Faster R-CNN throughput and memory pilot. Large training runs, holdout evaluation, and adaptation-method selection remain separate approval gates.

## Local Checks

The existing prototype uses only the Python standard library:

```powershell
python train_da_yolo_comparison.py --help
python -c "import ast, pathlib; ast.parse(pathlib.Path('train_da_yolo_comparison.py').read_text(encoding='utf-8'))"
```

Datasets, generated manifests, checkpoints, external repositories, and run outputs must remain untracked.
