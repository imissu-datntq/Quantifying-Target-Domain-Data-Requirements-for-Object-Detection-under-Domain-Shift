# Quantifying Target-Domain Data Requirements for Object Detection under Domain Shift

A BDD100K study of what a detector needs after domain shift: how much labeled target-domain data, and how much of the benefit of a domain-adaptation method is adaptation at all.

**Main finding.** At small labeled budgets, most of what domain adaptation gives over plain fine-tuning is not adaptation. A control arm that keeps the training recipe of the adaptation method tested (ALDI++) but removes its unlabeled target stream and its distillation reproduces 76–94% of the method's advantage over fine-tuning, on all three shifts, at both small labeled budgets tested (10 and 100 images). At 10 images that advantage is mostly damage that fine-tuning does to itself: measured from the source model, the control gains about nothing and ALDI++ gains only +0.003 to +0.011 AP. This connects to the thesis of ALDI (Kay et al., CVPR 2024) that reported adaptation gains depend on implementation and baseline choices.

The data-efficiency curves (recovery as a function of the labeled budget N) are the context that makes this measurable. The budget thresholds are secondary, practical results. The study does not propose a new detector: the detector, data partitions, evaluation path and target test sets are fixed, and only the labeled budget and the training arm change.

## Results

Numbers are reliable-class mAP50:95 on the held-out final test sets. Full tables, intervals and limitations are in the Vietnamese write-ups [`docs/results_rq5_vi.md`](docs/results_rq5_vi.md) and [`docs/results_local_rq1_rq4_vi.md`](docs/results_local_rq1_rq4_vi.md).

### The control arm

Three arms start from the same source model: plain fine-tuning; ALDI++ ("adapt": a labeled stream of source and target images, plus distillation from an EMA teacher on unlabeled target images); and "replay", the same labeled stream, augmentation and EMA teacher with no unlabeled stream and no distillation. The table gives the share of ALDI++'s gain that replay reproduces, with 95% intervals, three sampling seeds per arm.

| Labeled budget N | Lighting | Weather | Compound |
| ---: | ---: | ---: | ---: |
| 10, against fine-tuning | 0.87 (0.85 to 0.90) | 0.94 (0.91 to 0.96) | 0.83 (0.80 to 0.86) |
| 10, against the source model | 0.05 (−0.09 to 0.21) | not determined (ALDI++'s gain, +0.003 AP, is below the 0.005 AP at which a share is defined) | 0.07 (−0.22 to 0.29) |
| 100, against fine-tuning | 0.86 (0.78 to 0.90) | 0.90 (0.81 to 1.04) | 0.76 (0.62 to 0.85) |
| 100, against the source model | 0.87 (0.80 to 0.91) | 0.80 (0.57 to 1.07) | 0.80 (0.72 to 0.88) |

- **Against fine-tuning** the share is above one half in every cell, with the lower bound above one half. This is the headline.
- **Against the source model** the picture at N = 10 is different, and both readings belong together. Fine-tuning on 10 images makes the detector 0.050–0.061 AP worse than not fine-tuning at all, so replay only has to avoid that damage. Measured from the source model, replay gains about nothing and ALDI++ gains +0.003 to +0.011 AP, about what it gains with zero labels (+0.006 to +0.011). The 10 labels add nothing.
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

The source model scores 0.342 on its own test set. Recovery(N) is the share of the gap between zero-shot and a full-pool fine-tune that N labeled images close; N_BE, N75 and N90 are the smallest budgets from which recovery stays above 0, at or above 0.75, and at or above 0.90. Where a cell has two values, the first is against the 3-epoch ceiling and the second against the 12-epoch ceiling. Against the 3-epoch ceiling the full-pool rung is the ceiling itself (recovery 1 by construction), so a threshold first reached there is given as not determined within the pool. Against the 12-epoch ceiling the 3-epoch full-pool run is a measured top rung (recovery 0.95 on lighting, 0.99 on compound), so "full pool" is a measurement.

| Shift | Zero-shot drop (95% CI) | Ceiling, 3 / 12 epochs | N_BE | N75 | N90 |
| --- | ---: | ---: | --- | --- | --- |
| Lighting | 0.098 (0.092 to 0.106) | 0.296 / 0.299 | 100 | 5,000 | 10,000 / full pool (18,015) |
| Weather | 0.022 (0.012 to 0.032) | 0.336 / 0.331 | 250 / not determined | 1,000 / not determined | not determined within the pool (> 1,000) / not determined |
| Compound | 0.123 (0.112 to 0.133) | 0.257 / 0.258 | 100 | 1,000 | not determined within the pool (> 1,000) / full pool (1,738) |

- **Fine-tuning on fewer than about 100 images hurts.** Recovery is negative at N ≤ 50 on every shift, and at N = 100 on weather, because the fixed schedule gives 10 images 500 updates.
- **The 12-epoch ceiling is not materially higher than the 3-epoch one.** The paired difference is +0.0029 AP on lighting, +0.0005 on compound and −0.0045 on weather, against a pre-set margin of 0.005. Both ceilings are reported and neither is picked as better.
- **Weather is nearly empty as a threshold measurement.** Its recovery denominator is 0.015 AP against the 3-epoch ceiling and 0.011 against the 12-epoch one, with a floor of 0.01. Against the 12-epoch ceiling 339 of 1,000 bootstrap replicates fall at or below the floor, so weather is reported there as not determined (a reporting rule set after the results were seen: more than 5% undefined replicates). Weather counts as a third replication of the control-arm result, not as a third threshold measurement.
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

- **Evaluation roles** ([amendment of 2026-10-19](docs/review_response_2026-10-19_vi.md)). The weather and compound held-out sets were below the 600-image target, so their preserved-validation images join the final test set. Those two domains have no selection validation, so every target run keeps the final update of one fixed schedule. Lighting validation is used only for the two pilots: the fine-tuning schedule (updates, epochs, LR) and the ALDI++ settings (EMA rate 0.9996 and run length max(2,000, 2N) updates).
- **Detector.** Faster R-CNN with a ResNet-50 FPN backbone at native 1280×720, initialized from COCO, chosen because public domain-adaptation implementations target it.
- **Source model.** 12 epochs on the 9,804 source images; checkpoint selected on source validation (reliable mAP50:95 = 0.352).
- **Fine-tuning.** Every ladder run starts from the source model and runs `max(500 updates, 3 epochs)` at LR 1e-3, batch 2. Six control runs (N = 100 and 1,000 per shift) start from COCO weights instead. The ladder is `10, 25, 50, 100, 250, 500, 1,000` plus `2,000, 5,000, 10,000` for lighting, with three nested sampling seeds per rung.
- **Ceilings.** Full-pool fine-tunes at 3 and at 12 epochs, two seeds each, and the same at a matched pool of 1,738 images for the cross-shift comparison.
- **Adaptation.** ALDI++ (Kay et al., *Align and Distill*, CVPR 2024), re-implemented on the same torchvision detector in `scripts/bdd_adapt/`, with the replay control described above.
- **Metric.** COCO-style mAP50:95 over a fixed roster of four reliable classes (pedestrian, car, traffic light, traffic sign); the ten-class mean is a sensitivity analysis. Intervals come from a hierarchical bootstrap over test images and seeds, with the test resamples shared by all runs.

```text
Recovery(N) = [mAP(N) - mAP(0)] / [mAP_c - mAP(0)]
```

`mAP(0)` is the source model on the target test set and `mAP_c` is the full-pool fine-tuning ceiling, never the source-domain score. Recovery is not clipped, and it is undefined when the denominator does not exceed 0.01 AP.

**Test discipline.** Runs write predictions on the test images without reading labels. A plan listing the hash of every prediction file is frozen, and only then are the labels scored, once per plan. The three plans behind every number reported here were each scored this way: the fine-tuning campaign (87 runs), the adaptation grid (46 runs) and the revision that followed the lecturer's review (42 runs). Schedules and analysis rules are recorded in [`.agent/DECISIONS.md`](.agent/DECISIONS.md), with a note for each rule on whether it was written before or after the results it reads.

## Repository Layout

- `scripts/bdd_pipeline/` and `scripts/bdd_adapt/`: the training, evaluation and adaptation packages. Each run records the SHA256 of the package it was trained with (`bdd_pipeline/*.py` for fine-tuning, `bdd_adapt/*.py` for ALDI++ and replay). Prediction manifests and frozen plans record `predict_frcnn.py`, `evaluate_frcnn.py` and `bdd_pipeline/{data,evaluation,model}.py`. None of these files is edited.
- `scripts/`: entry points. Data `verify_source_mirror.py`, `prepare_revised_bdd100k.py`; training `train_frcnn.py`, `train_adapt.py`; the zero-shot gate `zero_shot_gap.py` and the pilots that locked the schedules on lighting validation, `local_gate_schedule_pilot.sh`, `local_rq5_pilot.py`, `rq5_pilot_lock.py`; prediction and scoring `predict_frcnn.py`, `freeze_evaluation_plan.py`, `evaluate_frcnn.py`; queue runners `local_campaign.py`, `local_rq5_grid.py`, `local_revision_runs.py`; analysis `analyze_campaign.py`, `analyze_rq5.py`. The locked values are already in the code, so the pilots need not be rerun. `local_gate_schedule_pilot.sh` and `local_rq5_pilot.py` take no arguments and start working on any invocation, `--help` included.
- `tests/`: 239 unit tests (`python -m unittest discover -s tests`).
- `docs/`: the two result write-ups (Vietnamese), the pipeline description [`docs/training_pipeline.md`](docs/training_pipeline.md), the dataset description [`docs/revised_dataset_eda.md`](docs/revised_dataset_eda.md) and the protocol [`docs/proposal_dataset_aligned.docx`](docs/proposal_dataset_aligned.docx). The report `docs/bao_cao_ket_qua_vi.docx` is the 2026-11-08 version and is being rewritten; take numbers from the two write-ups.
- `.agent/`: decisions, hand-over notes and plans.
- `data/split_reference/`: the frozen split identity (manifests and hashes, no images or labels).
- `phase1_initial/`: the initial exploration of September 2026, kept unchanged and not used by the study. `scripts/train_da_yolo_comparison.py` is an earlier YOLO prototype with a different split; it is not the implementation of this study either.

Datasets, run outputs, checkpoints and weights are not tracked.

## Getting the Data

The repository holds no images or labels. It holds the frozen split identity in [`data/split_reference/`](data/split_reference/): the membership manifests (`train/test/val.csv`: file name, time of day, weather, scene, image SHA256), the stratification and duplicate-repair records, `dataset_metadata.json`, and `expected_sha256.json` with the hashes of the manifests, metadata and generated annotation files. Labels are the 2018 BDD100K detection release (`bdd100k_labels_release`), which the official documentation marks as deprecated in favour of `det_20`.

1. Download the public Kaggle mirror [`solesensei/solesensei_bdd100k`](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k) (version 2, about 7.6 GiB) and unzip it into `data/raw/`. The command needs the Kaggle CLI (`pip install kaggle`) with an API token; the ZIP can also be downloaded from the dataset page.

   ```bash
   kaggle datasets download solesensei/solesensei_bdd100k -p data/raw --unzip
   ```

   You need the images under `bdd100k/bdd100k/images/100k/{train,val}/` and the folder that holds `bdd100k_labels_images_train.json` and `bdd100k_labels_images_val.json`. The public test images are not used.

2. Check the files against the mirror (size and CRC32 of all 80,000 images and both label files; SHA256 is computed locally):

   ```bash
   python scripts/verify_source_mirror.py --images-root data/raw/bdd100k/bdd100k/images/100k --labels-root data/raw/bdd100k_labels_release/bdd100k/labels --output-dir data/raw/kaggle_verified --workers 12
   ```

3. Build and validate the split (seed 20260916; 55,000 train / 14,863 test / 10,000 official val):

   ```bash
   python scripts/prepare_revised_bdd100k.py --verified-source data/raw/kaggle_verified/source_verified.json --output-dir data/bdd100k_revised --seed 20260916
   python scripts/prepare_revised_bdd100k.py --output-dir data/bdd100k_revised --validate-only
   ```

4. Confirm that your split is ours: every hash must print `OK`.

   ```bash
   python -c "import hashlib,json,pathlib; r=pathlib.Path('data/bdd100k_revised'); e=json.loads(pathlib.Path('data/split_reference/expected_sha256.json').read_text()); [print('OK ' if hashlib.sha256((r/k).read_bytes()).hexdigest()==v else 'BAD', k) for k,v in e.items()]"
   ```

How the split is built (hierarchical largest-remainder allocation over time of day / weather / scene, duplicate-content repair, zero cross-split overlap) is documented in [`docs/revised_dataset_eda.md`](docs/revised_dataset_eda.md).

Never use weather or compound validation images for selection: they are part of the final test set.

## Reproducing the Runs

Everything was trained on one RTX 3060 12 GB under Linux. Dependencies are in `requirements-training.txt`; install the CUDA builds of torch and torchvision (the runs used torch 2.11.0+cu128 and torchvision 0.26.0+cu128, and the queue runners pass `--device cuda`). The analysis figures also need `matplotlib` (in `requirements-eda.txt`); without it, pass `--no-plots`. `scripts/local_revision_runs.py` needs a POSIX system (`fcntl`, `pgrep`); on Windows it and the 13 tests in `tests/test_revision_runs.py` do not import. All commands below are run from the repository root.

Train the source model first; the queue runners read it from `runs/local/source/source_seed0/best.pt`:

```bash
python scripts/train_frcnn.py --dataset-root data/bdd100k_revised --output-dir runs/local/source/source_seed0 --stage source --domain source --budget full --initialization coco --seed 0 --epochs 12 --floor-steps 1 --batch-size 4 --accumulation 1 --lr 0.005 --warmup-updates 500 --eval-every-updates 2451 --save-every-updates 500 --device cuda --amp --workers 4 --threads 4
```

Then the three grids. The queue runners skip finished runs and resume an interrupted run from its last checkpoint:

```bash
python scripts/local_campaign.py          # 87 runs: source-model test predictions, 80 fine-tunes, 6 COCO-initialized controls
python scripts/local_rq5_grid.py          # 46 ALDI++ and replay runs
python scripts/local_revision_runs.py     # 42 revision runs; --dry-run prints them
```

Each grid is frozen and scored once. For the campaign:

```bash
python scripts/freeze_evaluation_plan.py --prediction-manifests runs/local/campaign/campaign/*/predictions/prediction_manifest.json --output runs/local/campaign/final_plan.json
python scripts/evaluate_frcnn.py --final-plan runs/local/campaign/final_plan.json --output-dir runs/local/campaign/scored
```

The same two commands serve the adaptation grid (`runs/local/rq5/grid/*/predictions/prediction_manifest.json`, plan and scores under `runs/local/rq5/`) and the revision (`runs/local/revision/grid/*/*/predictions/prediction_manifest.json`, under `runs/local/revision/`). A scored plan cannot be scored again; an interrupted pass is finished with `--resume-interrupted`.

Then the analysis, one output directory per script:

```bash
python scripts/analyze_campaign.py --scored runs/local/campaign/scored --revision-scored runs/local/revision/scored --output-dir runs/local/campaign/analysis_rev2
python scripts/analyze_rq5.py --ft-scored runs/local/campaign/scored --da-scored runs/local/rq5/scored --revision-scored runs/local/revision/scored --output-dir runs/local/rq5/analysis_rev2
```

## Local Checks

```bash
python -m unittest discover -s tests
python scripts/prepare_revised_bdd100k.py --output-dir data/bdd100k_revised --validate-only
python scripts/train_frcnn.py --output-dir runs/source_candidate --dry-run   # checks data and config with the CLI defaults, not the study's source settings
python scripts/local_revision_runs.py --dry-run
```

Training and ordinary validation never open held-out detection annotations; final-test scoring requires a frozen plan.
