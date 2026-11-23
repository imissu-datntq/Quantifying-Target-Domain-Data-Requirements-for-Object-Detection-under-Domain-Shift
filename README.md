# Quantifying Target-Domain Data Requirements for Object Detection under Domain Shift

A BDD100K study of what a detector needs after domain shift: how much labeled target-domain data, and how much of the benefit of a domain-adaptation method is adaptation at all.

**Main finding.** At small labeled budgets, most of what domain adaptation gives over plain fine-tuning is not adaptation. A control arm that keeps the adaptation method's training recipe but removes its unlabeled target stream and its distillation reproduces 76–94% of the method's advantage over fine-tuning, on all three shifts, at the two budgets tested (10 and 100 labeled images). This connects to the thesis of ALDI (Kay et al., CVPR 2024) that reported adaptation gains depend on implementation and baseline choices.

The data-efficiency curves (recovery as a function of the labeled budget N) are the context that makes this measurable. The budget thresholds are secondary, practical results. The study does not propose a new detector: the detector, data partitions, evaluation path and target test sets are fixed, and only the labeled budget and the training arm change.

## Results

Numbers are reliable-class mAP50:95 on the held-out final test sets. Full tables, intervals and limitations are in the Vietnamese write-ups [`docs/results_rq5_vi.md`](docs/results_rq5_vi.md) and [`docs/results_local_rq1_rq4_vi.md`](docs/results_local_rq1_rq4_vi.md).

### The control arm

Three arms start from the same source model: plain fine-tuning; ALDI++ ("adapt": a labeled stream of source and target images, plus distillation from an EMA teacher on unlabeled target images); and "replay", the same labeled stream, augmentation and EMA teacher with no unlabeled stream and no distillation. The table gives the share of ALDI++'s gain that replay reproduces, with 95% intervals, three sampling seeds per arm.

| Labeled budget N | Lighting | Weather | Compound |
| ---: | ---: | ---: | ---: |
| 10, against fine-tuning | 0.87 (0.85 to 0.90) | 0.94 (0.91 to 0.96) | 0.83 (0.80 to 0.86) |
| 10, against the source model | 0.05 (−0.09 to 0.21) | not determined | 0.07 (−0.22 to 0.29) |
| 100, against fine-tuning | 0.86 (0.78 to 0.90) | 0.90 (0.81 to 1.04) | 0.76 (0.62 to 0.85) |
| 100, against the source model | 0.87 (0.80 to 0.91) | 0.80 (0.57 to 1.07) | 0.80 (0.72 to 0.88) |

- **Against fine-tuning** the share is above one half in every cell, with the lower bound above one half. This is the headline.
- **Against the source model** the picture at N = 10 is different, and both readings belong together. Fine-tuning on 10 images makes the detector 0.050–0.061 AP worse than not fine-tuning at all, so replay only has to avoid that damage. Measured from the source model, replay gains about nothing and ALDI++ gains +0.003 to +0.011 AP, no more than it gains with zero labels. The 10 labels add nothing.
- **At N = 100** the two readings agree. Fine-tuning is back near the source model on lighting and compound; on weather it is still 0.007 AP below it, so part of the weather advantage over fine-tuning is still avoided damage.
- The rule for stating the headline was written before the second and third replay seeds were scored, but after the seed-0 shares had been seen. It is descriptive; the independent part of the evidence is the two new seeds.

### Larger budgets

From N = 1,000 ALDI++ has no advantage over fine-tuning. In the original grid it also saw fewer labeled target image-passes than fine-tuning (2,000 against 3,000), so the comparison was rerun with matched passes:

| Shift | ALDI++ − fine-tuning, original run length | ALDI++ − fine-tuning, matched passes (95% CI) | Reading |
| --- | ---: | ---: | --- |
| Lighting | −0.0111 | −0.0076 (−0.0091 to −0.0058) | fine-tuning stays ahead |
| Weather | −0.0032 | −0.0021 (−0.0038 to +0.0002) | no meaningful difference |
| Compound | −0.0117 | −0.0074 (−0.0097 to −0.0047) | fine-tuning stays ahead |

Matching the passes closes about a third of the gap. It cannot separate labeled exposure from the lag of the EMA teacher, since both change with the number of updates. Under matched passes ALDI++ is level with or slightly below replay, so distillation contributes nothing on the reliable metric at this budget.

### Data-efficiency curves and thresholds

The source model scores 0.342 on its own test set. Recovery(N) is the share of the gap between zero-shot and a full-pool fine-tune that N labeled images close; N_BE, N75 and N90 are the smallest budgets from which recovery stays above 0, at or above 0.75, and at or above 0.90. Where a cell has two values, the first is against the 3-epoch ceiling and the second against the 12-epoch ceiling.

| Shift | Zero-shot drop (95% CI) | Ceiling, 3 / 12 epochs | N_BE | N75 | N90 |
| --- | ---: | ---: | --- | --- | --- |
| Lighting | 0.098 (0.092 to 0.106) | 0.296 / 0.299 | 100 | 5,000 | 10,000 / full pool (18,015) |
| Weather | 0.022 (0.012 to 0.032) | 0.336 / 0.331 | 250 / not determined | 1,000 / not determined | not determined within the pool (> 1,000) / not determined |
| Compound | 0.123 (0.112 to 0.133) | 0.257 / 0.258 | 100 | 1,000 | not determined within the pool (> 1,000) / full pool (1,738) |

- **Fine-tuning on fewer than about 100 images hurts.** Recovery is negative at N ≤ 50 on every shift, because the fixed schedule gives 10 images 500 updates.
- **The 12-epoch ceiling is not materially higher than the 3-epoch one.** The paired difference is +0.0029 AP on lighting, +0.0005 on compound and −0.0045 on weather, against a pre-set margin of 0.005. Both ceilings are reported and neither is picked as better.
- **Weather is nearly empty as a threshold measurement.** Its recovery denominator is 0.015 AP against the 3-epoch ceiling and 0.011 against the 12-epoch one, with a floor of 0.01, so against the 12-epoch ceiling it is reported as not determined. Weather counts as a third replication of the control-arm result, not as a third threshold measurement.
- **Lighting between the rungs.** A log-linear fit on the positive part of the curve puts N75 at 3,570–4,690 images and N90 at 9,200–12,800, depending on the ceiling. The fit is descriptive.
- **Compound shift.** drop(compound) − drop(lighting) − drop(weather) is +0.002 AP (−0.012 to +0.016). The test can only detect an interaction of about 0.014–0.020 AP, which is most of the weather effect itself, so this is a detection limit and not evidence of additivity.

## Research Design

The study uses a materialized, stratified split of the labeled BDD100K images:

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

- **Evaluation roles** ([amendment of 2026-10-19](docs/review_response_2026-10-19_vi.md)). The weather and compound held-out sets were below the 600-image target, so their preserved-validation images join the final test set. Those two domains have no selection validation, so every target run keeps the final update of one fixed schedule. Lighting validation is used for the schedule pilots only.
- **Detector.** Faster R-CNN with a ResNet-50 FPN backbone at native 1280×720, initialized from COCO, chosen because public domain-adaptation implementations target it.
- **Source model.** 12 epochs on the 9,804 source images; checkpoint selected on source validation (reliable mAP50:95 = 0.352).
- **Fine-tuning.** Every target run starts from the source model and runs `max(500 updates, 3 epochs)` at LR 1e-3, batch 2. The ladder is `10, 25, 50, 100, 250, 500, 1,000` plus `2,000, 5,000, 10,000` for lighting, with three nested sampling seeds per rung.
- **Ceilings.** Full-pool fine-tunes at 3 and at 12 epochs, two seeds each, and the same at a matched pool of 1,738 images for the cross-shift comparison.
- **Adaptation.** ALDI++ (Kay et al., *Align and Distill*, CVPR 2024), re-implemented on the same torchvision detector in `scripts/bdd_adapt/`, with the replay control described above.
- **Metric.** COCO-style mAP50:95 over a fixed roster of four reliable classes (pedestrian, car, traffic light, traffic sign); the ten-class mean is a sensitivity analysis. Intervals come from a hierarchical bootstrap over test images and seeds, with the test resamples shared by all runs.

```text
Recovery(N) = [mAP(N) - mAP(0)] / [mAP_c - mAP(0)]
```

`mAP(0)` is the source model on the target test set and `mAP_c` is the full-pool fine-tuning ceiling, never the source-domain score. Recovery is not clipped, and it is undefined when the denominator does not exceed 0.01 AP.

**Test discipline.** Runs write predictions on the test images without reading labels. A plan listing the hash of every prediction file is frozen, and only then are the labels scored, once per plan. Three plans were scored this way: the fine-tuning campaign (87 runs), the adaptation grid (46 runs) and the revision that followed the lecturer's review (42 runs). Schedules and analysis rules are recorded in [`.agent/DECISIONS.md`](.agent/DECISIONS.md), with a note for each rule on whether it was written before or after the results it reads.

## Repository Layout

- `scripts/bdd_pipeline/` and `scripts/bdd_adapt/`: the training, evaluation and adaptation packages. Every run records the SHA256 of these files, so they are not edited.
- `scripts/`: entry points. Training `train_frcnn.py`, `train_adapt.py`; prediction and scoring `predict_frcnn.py`, `freeze_evaluation_plan.py`, `evaluate_frcnn.py`; queue runners `local_campaign.py`, `local_rq5_grid.py`, `local_revision_runs.py`; analysis `analyze_campaign.py`, `analyze_rq5.py`.
- `tests/`: 239 unit tests (`python -m unittest discover -s tests`).
- `docs/`: the two result write-ups (Vietnamese), the pipeline description [`docs/training_pipeline.md`](docs/training_pipeline.md), the dataset description [`docs/revised_dataset_eda.md`](docs/revised_dataset_eda.md) and the protocol [`docs/proposal_dataset_aligned.docx`](docs/proposal_dataset_aligned.docx). The report `docs/bao_cao_ket_qua_vi.docx` is the 2026-11-08 version and is being rewritten; take numbers from the two write-ups.
- `.agent/`: decisions, hand-over notes and plans.
- `data/split_reference/`: the frozen split identity (manifests and hashes, no images or labels).
- `phase1_initial/`: the initial exploration of September 2026, kept unchanged and not used by the study. `scripts/train_da_yolo_comparison.py` is an earlier YOLO prototype with a different split; it is not the implementation of this study either.

Datasets, run outputs, checkpoints and weights are not tracked.

## Getting the Data

The repository holds no images or labels. It holds the frozen split identity in [`data/split_reference/`](data/split_reference/): the membership manifests (`train/test/val.csv`: file name, time of day, weather, scene, image SHA256), the stratification and duplicate-repair records, `dataset_metadata.json`, and `expected_sha256.json` with the hashes of the manifests, metadata and generated annotation files. Labels are the 2018 BDD100K detection release (`bdd100k_labels_release`), which the official documentation marks as deprecated in favour of `det_20`.

1. Download the public Kaggle mirror [`solesensei/solesensei_bdd100k`](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k) (version 2, about 7.6 GiB) and unzip it, for example into `data/raw/`:

   ```powershell
   kaggle datasets download solesensei/solesensei_bdd100k -p data/raw --unzip
   ```

   You need the images under `bdd100k/bdd100k/images/100k/{train,val}/` and the folder that holds `bdd100k_labels_images_train.json` and `bdd100k_labels_images_val.json`. The public test images are not used.

2. Check the files against the mirror (size and CRC32 of all 80,000 images and both label files; SHA256 is computed locally):

   ```powershell
   python scripts/verify_source_mirror.py --images-root data/raw/bdd100k/bdd100k/images/100k --labels-root <folder with the two label JSONs> --output-dir data/raw/kaggle_verified --workers 12
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

How the split is built (hierarchical largest-remainder allocation over time of day / weather / scene, duplicate-content repair, zero cross-split overlap) is documented in [`docs/revised_dataset_eda.md`](docs/revised_dataset_eda.md).

Never use weather or compound validation images for selection: they are part of the final test set.

## Reproducing the Runs

Everything was trained on one RTX 3060 12 GB (PyTorch 2.11 with CUDA 12.8); dependencies are in `requirements-training.txt`. The queue runners skip finished runs and resume an interrupted run from its last checkpoint.

```powershell
python scripts/local_campaign.py          # 87 fine-tuning runs
python scripts/local_rq5_grid.py          # 46 ALDI++ and replay runs
python scripts/local_revision_runs.py     # 42 revision runs; --dry-run prints them
```

Each grid is then frozen and scored once:

```powershell
cd scripts
python freeze_evaluation_plan.py --dataset-root ../data/bdd100k_revised --prediction-manifests <prediction_manifest.json files> --output <plan.json>
python evaluate_frcnn.py --dataset-root ../data/bdd100k_revised --final-plan <plan.json> --output-dir <scored dir>
python analyze_campaign.py --scored <campaign scored> --revision-scored <revision scored> --output-dir <analysis dir>
python analyze_rq5.py --ft-scored <campaign scored> --da-scored <rq5 scored> --revision-scored <revision scored> --output-dir <analysis dir>
```

A scored plan cannot be scored again; an interrupted pass is finished with `--resume-interrupted`.

## Local Checks

```powershell
python -m unittest discover -s tests
python scripts/prepare_revised_bdd100k.py --output-dir data/bdd100k_revised --validate-only
python scripts/train_frcnn.py --output-dir runs/source_candidate --dry-run
python scripts/local_revision_runs.py --dry-run
```

Training and ordinary validation never open held-out detection annotations; final-test scoring requires a frozen plan.
