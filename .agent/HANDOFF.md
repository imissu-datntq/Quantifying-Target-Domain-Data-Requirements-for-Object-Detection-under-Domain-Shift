# Handoff

Updated 2026-10-30, Asia/Saigon.

## 2026-11-03: RQ5 started (latest)

- **Plan:** `.agent/plans/active/2026-11-03-rq5-adaptation.md`. Rules are pre-registered in `DECISIONS.md` (2026-11-03).
- **Method:** ALDI++, re-implemented in `scripts/bdd_adapt/`. `bdd_pipeline/` is untouched.
- **Tests and smoke:** 43 tests pass. The CPU smoke resumes bit-identically.
- **Pilot:** `kizhowl/bdd100k-rq5-pilot` v1 was pushed 2026-11-03 and is running. It uses lighting val only: 4 runs of 6,000 updates, with the teacher evaluated every 500 updates.
- **When the pilot is complete:**
  1. Download `rq5_pilot_summary.json` and `rq5_pilot/*/validation_step_*.json`, `training.jsonl` and `result.json` into `runs/rq5/pilot/`.
  2. Apply the α/T rule and check the correctness gate.
  3. Record the result in `DECISIONS.md` and set `scripts/bdd_adapt/grid.POLICY`.
  4. Push the protocol dataset.
  5. Push `kaggle/bdd_adapt.py` as `kizhowl/bdd100k-rq5` with its QUEUE set to the 46-run grid.
     - Kernel sources: `kizhowl/bdd100k-source-train` (later versions also list the kernel itself).
- **Second account:** buihuynhgiahuy is approved for the grid and has its own private copies of the protocol dataset and source model.
  - The grid splits by account: buihuynhgiahuy runs compound, weather and replay; kizhowl runs lighting.
  - Lock the pilot with `python scripts/rq5_pilot_lock.py runs/rq5/pilot --output runs/rq5/pilot_lock.json`.
  - After locking, push the new code to **both** protocol datasets.
- **If the pilot is cut by the session limit:** push v2 with `kizhowl/bdd100k-rq5-pilot` added to its `kernel_sources`; it resumes.
- **Local source model** for smoke tests: `runs/source_model/runs/source_seed0/best.pt`.

## 2026-10-30

This repo copy is on the user's machine at `D:\Personal Projects\CV-research`. It is not a git repository. The data is local in `data/bdd100k_revised/` (hardlinked, all 79,863 images rehashed OK).

The plan for the run campaign is in `.agent/plans/active/2026-10-30-supervised-campaign.md`.

**State:**
- The schedule is locked by the pre-registered round-3 rule: S=500, E=2, LR 1e-3, B_eff 2. See `DECISIONS.md` (2026-10-30) and `runs/schedule_pilot/round3/`.
- The grid is 87 runs, about 12.6 T4-h of training (`runs/campaign/grid.csv`).
- Protocol dataset `kizhowl/bdd100k-revised-protocol` has a new version carrying the campaign code. **`bdd_pipeline/` is frozen from here on:** resume checks implementation hashes.
- Campaign session 1 (`kizhowl/bdd100k-campaign` v1) finished COMPLETE on 2026-10-31. It covered zero-shot, compound, weather and the controls.
  - Training ran at about 6 img/s, update counts match the policy, and every manifest says `test_labels_accessed: False`.
  - All 54 runs finished in about 8.5 h, with no errors (53 trained, plus the zero-shot predictions).
  - The local `runs/campaign/session1/` copy is partial: 34 run folders, because the download was cut off.
  - Get the complete `campaign/` from the final version's output, which carries everything forward.
  - Prediction timing: lighting test takes about 375 s and source test about 203 s per run.
- Session 2 (v2) was pushed 2026-10-31. Its queue is the whole grid: finished runs are skipped and interrupted ones resume. It attaches v1's output.
- Session 2 (v2) finished COMPLETE on 2026-11-01. It trained 27 lighting runs, with no errors.
  - `fine-tune_lighting_N5000_s1` was interrupted by session time; its `last.pt` is on Kaggle, in the v2 output.
  - **Left:** 6 runs: lighting N5000 s1 (resume) and s2, N10000 s0/s1/s2, and the matched-pool N1738 s0. That is about 6 T4-h.
  - kizhowl has 19.6 h of quota left (it resets 2026-11-07).
- Session 3 (v3) was pushed 2026-11-01 for the last 6 lighting runs. It uses the same queue script, so finished runs are skipped. Expect about 6–7 h.
- **Grid complete 2026-11-02:**
  - Session 3 trained the last 6 runs in 4.2 h.
  - All 87 runs and their prediction files are in `runs/campaign/final/campaign/` (9.8 GB).
- **Frozen plan:** `runs/campaign/final_plan.json`, sha256 `8244023f…5ff7a8`. Schema 3, 87 sets, 176 scoring entries, bootstrap 1,000 resamples with seed 20261030.
- **One-shot scoring started 2026-11-02** on the 5060 laptop, CPU only, as a detached below-normal-priority process. It writes to `runs/campaign/scored/`, with logs in `runs/campaign/scoring.{out,err}.txt`.
  - Expect about 7 h.
  - If interrupted, run from `scripts/`: `python evaluate_frcnn.py --dataset-root ../data/bdd100k_revised --final-plan ../runs/campaign/final_plan.json --output-dir ../runs/campaign/scored --threads 4 --resume-interrupted`.
  - The lock records the absolute output path, so another machine needs the same `D:\Personal Projects\CV-research` location.
- **After scoring:** `python scripts/analyze_campaign.py --scored runs/campaign/scored --output-dir runs/campaign/analysis`, then write `docs/results_rq1_rq4_vi.md`.
- **Phase 4 done 2026-11-03:**
  - One-shot scoring completed with 176/176 entries (`runs/campaign/scored/results.json`).
  - Analysis is in `runs/campaign/analysis/`, and the report is `docs/results_rq1_rq4_vi.md`.
  - **Headlines:**
    - Drops: lighting 0.097, weather 0.021, compound 0.123. Compound is additive (interaction +0.005, CI −0.010 to 0.018).
    - Recovery is negative for N ≤ 50, caused by overfitting under the fixed 500-update floor.
    - N_BE is about 100–250. N75: lighting 5,000, compound 1,000, weather full pool.
    - At the matched pool, lighting and compound recover alike.
    - The power fit over-predicts at large N, so the ladder is primary.
  - **Next:** Phase 5 (RQ5) needs its own ExecPlan. Proportional-update reruns at N ≤ 100 are optional.
- **When v3 is COMPLETE:**
  1. Download the full `campaign/` from v3's output (`--file-pattern` for json/log). This is the complete grid, because each version carries the previous one forward.
  2. Check that all 87 runs have `predictions/prediction_manifest.json`.
  3. Freeze one prediction plan and score once. The user plans to do the scoring on their second machine (RTX 3060).
  - The user now has a second machine with an RTX 3060.
  - A Kaggle `last.pt` cannot be resumed locally as is: the config records `/tmp` and `/kaggle` paths, and resume rejects any config difference.
- Lecturer updates are weekly; the draft `docs/lecturer_update_2026-10-30_vi.md` is kept current, not a blocker.

**Next safe action:**
1. When v1 finishes, download `campaign/*/predictions/`, `logs/` and `session_summary.json` into `runs/campaign/`.
2. Check `amp_step_skipped` and the throughput.
3. Push v2 with the remaining runs plus the lighting queue, adding `kizhowl/bdd100k-campaign` to its own `kernel_sources`.
4. Use the `kizhowl` key only, via `KAGGLE_API_TOKEN`. Never use `ngdminh31`.

## 2026-10-19 review amendment (latest)

Six reviewer points were addressed; see `docs/review_response_2026-10-19_vi.md`
and `.agent/DECISIONS.md`. Weather/compound val merged into final test (933/756),
target runs keep the final update, fixed reliable roster pedestrian/car/traffic
light/traffic sign, `scripts/zero_shot_gap.py` pilot gate, per-class
`scripts/recovery_curves.py`, EDA v2 at `runs/eda_revised_v2/` (adds
`domain_occlusion_truncation.csv`, `domain_objects_per_image.csv`, figure 09),
`scripts/label_provenance.py` fingerprint at `runs/label_provenance/`. COCO-proxy gap probe output is at
`runs/zero_shot_gap_coco_proxy/`. Official BDD100K hosts were unreachable, so the
label comparison is pending a user download. Nothing staged or committed.

Kaggle (2026-10-20, see `docs/review_response_2026-10-20_kaggle_vi.md`): source
model trained (source-val reliable mAP50:95 0.352); official zero-shot gate with
it: weather gap 0.031 AP (CI 0.023-0.040) passes, lighting 0.099, compound 0.129.
Work outside the review's six points was removed at the user's request.


## Current state

The user requested a new careful split from the original Kaggle BDD100K source,
EDA exclusively on the new dataset, then a Python training/evaluation pipeline
and a report of its actual verification. The repository was cloned at
52898b14269cf55f3fa06cebf7cccd82de5b8e02. Proposal read in full.

`data/bdd100k_revised/` is complete: 55,000 train / 14,863 held-out test /
10,000 preserved official val. Images are independent copies, annotations are
canonical detection-only JSON. All four main domain counts match README/proposal.
Membership is frozen by the CSV manifests and SHA256 recorded in metadata.

The supervised Faster R-CNN pipeline is implemented in `scripts/bdd_pipeline/`
with `train_frcnn.py`, `pilot_frcnn.py`, `evaluate_frcnn.py` and
`freeze_evaluation_plan.py` entry points. It supports source training, nested
target fine-tuning, full-pool reference and generic-pretrained controls. See
`docs/training_pipeline.md` and the Vietnamese `docs/pipeline_report_vi.md`.
The earlier YOLO prototype is not the current proposal implementation.

Every original input image and both label files matched public Kaggle ZIP sizes
and CRC32. Source verification remains ignored in `data/raw/kaggle_verified/`.
No previous derived subset or model result was used as an EDA input.

Hierarchical Hamilton stratification uses timeofday/weather then scene and seed
20260916. An exact duplicate image pair in original train was reunited using one
same-stratum singleton swap; all stratum quotas remain unchanged. Within-split
extra duplicate images: train 1, test 0, val 2. Official validation membership is
unchanged. Zero cross-split basename and SHA256 overlap. All dimensions 1280×720.

## Verified evidence

- Full construction and structural count/membership validation passed.
- `--validate-only` rehashed all 79,863 new images and passed again.
- Six unit tests passed for deterministic/stratified allocation, strict JSON,
  exact-content leakage, duplicate repair and class-absence probability.
- EDA covers new train/val labels only; held-out detection labels are never read
  by the EDA script. Test metadata provide census/balance only.
- Train has 1,013,542 valid boxes; car 55.29%; small-box share 52.95–55.93%
  across study domains. Joint-stratum train/test TVD is 0.001381 and largest
  absolute share difference is 0.006175 percentage points.
- Local report: `runs/eda_revised/report_vi.html` and `.md`; 8 figures, CSVs,
  exact absence probabilities and nested train-only budgets for seeds 0/1/2.
- Notebook: `notebooks/eda_revised_bdd100k.ipynb`, executed with embedded figures.
  It checks EDA/dataset manifest hash agreement before showing results.
- Pipeline and dataset tests: 14/14 pass; native geometry, normalization,
  class IDs, sanitation/flip, budget nesting, schedule, COCO perfect/empty
  predictions and final-plan guards verified. Syntax checks pass for 17 files.
- PyTorch 2.11.0 CPU, torchvision 0.26.0 CPU and pycocotools 2.0.11 installed.
  No CUDA device. Official COCO pretrained weights downloaded successfully.
- `runs/pipeline_verified/`: native train/val source smoke (3 images, 2 updates),
  stop/resume pair and continuous run. All 295 model tensors, optimizer,
  scaler, RNG and positions match exactly; maximum model difference 0.0.
- Separate validation CLI reload passed; target-weather fine-tune and COCO
  pretrained pilot each passed one update on two train images. Source smoke
  losses 12.5672/3.8292; throughput 0.167 images/sec. These are software checks,
  not research scores. No held-out detection labels accessed in these runs.
- CLI rejects freezing smoke checkpoints and resuming with a different LR.
  Unit-test/guard/resume evidence is recorded beside the ignored run artifacts.

## Boundaries and next action

No full detector training campaign, method selection, held-out performance
evaluation or publication was performed. Content hashes do not exclude near-duplicate,
geographic or video-group dependence. Source ZIP consistency is CRC32/size-based,
not a signed provenance guarantee. Header/format checks cover all images;
brightness EDA fully decodes 800 sampled train images.

Read `docs/revised_dataset_eda.md` for EDA reproduction commands. The next
research step is a representative GPU throughput/memory pilot, verification of
CUDA/AMP and a fixed optimizer/schedule/support policy before full source training.
Native geometry and preprocessing are implemented. S=2000/E=30/LR=0.005,
effective batch 2 and reliable threshold 100 remain provisional. CPU exact resume
is verified only with workers=0. Use only the frozen new manifests.

Final test evaluation requires an explicit frozen checkpoint plan, validates
every checkpoint before label access and creates a one-shot opened marker. No
real final plan or held-out evaluation has been executed. Adaptation methods,
recovery curves and bootstrap intervals are not implemented in this milestone.

Repository now lives directly at `D:\cv`. Preserve the user's existing Git index:
it contains staged deletions/renames from the root reorganization. Implementation
work did not stage or commit files. Dataset, weights and runs remain ignored.
