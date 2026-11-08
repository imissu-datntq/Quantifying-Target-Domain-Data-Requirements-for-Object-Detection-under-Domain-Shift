#!/usr/bin/env bash
# Local RTX 3060 chain (2026-11-05 fresh rerun, see .agent/DECISIONS.md):
#   wait for the source model -> zero-shot gap gate -> N0 on lighting val -> schedule pilot runs.
# Mirrors kaggle/bdd_gap_gate.py and kaggle/bdd_schedule_pilot.py, plus the round-2 seed-0
# configurations the 2026-10-30 rule compares against. It only runs; it does not apply the
# lock rule. Re-running is safe: finished steps are skipped, interrupted ones restart.
# Reads no held-out label: gate uses source val + target train pools, pilot uses lighting val.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/venvs/cv-final/bin/python
DATA=data/bdd100k_revised
SRC_DIR=runs/local/source/source_seed0
OUT=runs/local
log() { echo "[$(date '+%F %T')] $*"; }

# 1. Wait for the source model.
until [ -f "$SRC_DIR/result.json" ]; do
  if ! pgrep -f "train_frcnn.py.*$SRC_DIR" >/dev/null; then
    log "ERROR: source training is not running and has no result.json"; exit 1
  fi
  sleep 120
done
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open('$SRC_DIR/result.json'))['completed'] else 1)"; then
  log "ERROR: source result.json says not completed"; exit 1
fi
SOURCE="$SRC_DIR/best.pt"
log "source complete: $(cat "$SRC_DIR/result.json" | tr -d '\n' | cut -c1-400)"

# 2. Zero-shot gap gate (same arguments as the Kaggle gate).
if [ ! -f "$OUT/zero_shot_gap/gap_gate.done" ]; then
  log "zero-shot gap gate"
  $PY scripts/zero_shot_gap.py --dataset-root "$DATA" --checkpoint "$SOURCE" \
    --domains weather lighting compound --images-per-domain 2000 --bootstrap 1000 \
    --device cuda --threads 4 --output-dir "$OUT/zero_shot_gap" --resume \
    > "$OUT/zero_shot_gap.log" 2>&1 && touch "$OUT/zero_shot_gap/gap_gate.done" \
    || log "ERROR: gap gate failed, see $OUT/zero_shot_gap.log (continuing with the pilot)"
fi

# 3. N0: source model on lighting val (reference for the pilot table and the N10 flag).
PILOT=$OUT/schedule_pilot
mkdir -p "$PILOT"
if [ ! -f "$PILOT/N0_source_lighting_val/metrics.json" ]; then
  rm -rf "$PILOT/N0_source_lighting_val"
  log "N0 on lighting val"
  $PY scripts/evaluate_frcnn.py --dataset-root "$DATA" --checkpoint "$SOURCE" --domain lighting \
    --output-dir "$PILOT/N0_source_lighting_val" --device cuda --amp --batch-size 4 --workers 4 \
    > "$PILOT/N0_source_lighting_val.log" 2>&1 || log "ERROR: N0 evaluation failed"
fi

# 4. Schedule pilot: name budget floor_S epochs LR seed (round 3 first, then round-2 seed-0 configs).
PILOTS=(
  "N100_S250_E5_lr1e-3_seed0 100 250 5 0.001 0"
  "N100_S250_E5_lr1e-3_seed1 100 250 5 0.001 1"
  "N100_S250_E5_lr1e-3_seed2 100 250 5 0.001 2"
  "N100_S125_E1_lr1e-3_seed0 100 125 1 0.001 0"
  "N10_S250_E5_lr1e-3_seed0 10 250 5 0.001 0"
  "N10_S500_E5_lr1e-3_seed0 10 500 5 0.001 0"
  "N1000_S1_E2_lr1e-3_seed0 1000 1 2 0.001 0"
  "N1000_S1_E3_lr1e-3_seed0 1000 1 3 0.001 0"
  "N1000_S500_E5_lr5e-4_seed0 1000 500 5 0.0005 0"
  "N100_S500_E5_lr1e-3_seed0 100 500 5 0.001 0"
  "N1000_S500_E5_lr1e-3_seed0 1000 500 5 0.001 0"
  "N1000_S500_E10_lr1e-3_seed0 1000 500 10 0.001 0"
)
for row in "${PILOTS[@]}"; do
  read -r name budget floor epochs lr seed <<< "$row"
  eval_dir="$PILOT/${name}_lighting_val"
  [ -f "$eval_dir/metrics.json" ] && { log "skip $name (done)"; continue; }
  rm -rf "$PILOT/$name" "$eval_dir"
  log "train $name"
  if ! $PY scripts/train_frcnn.py --dataset-root "$DATA" --output-dir "$PILOT/$name" \
      --stage fine-tune --domain lighting --budget "$budget" --seed "$seed" --smoke \
      --source-checkpoint "$SOURCE" --floor-steps "$floor" --epochs "$epochs" \
      --batch-size 2 --accumulation 1 --lr "$lr" --warmup-updates 50 \
      --save-every-updates 100000 --device cuda --amp --workers 4 --threads 4 \
      > "$PILOT/$name.log" 2>&1; then
    log "ERROR: training $name failed, see $PILOT/$name.log"; continue
  fi
  log "eval $name"
  $PY scripts/evaluate_frcnn.py --dataset-root "$DATA" --checkpoint "$PILOT/$name/last.pt" --domain lighting \
    --output-dir "$eval_dir" --device cuda --amp --batch-size 4 --workers 4 \
    > "$PILOT/${name}_eval.log" 2>&1 || { log "ERROR: eval $name failed"; continue; }
  rm -f "$PILOT/$name"/*.pt "$eval_dir/predictions.json"
done

# 5. Summary table (scores only; the lock rule is applied by hand per DECISIONS 2026-10-30).
$PY - "$PILOT" <<'EOF'
import json, sys
from pathlib import Path
pilot = Path(sys.argv[1])
summary = {}
for metrics_path in sorted(pilot.glob("*_lighting_val/metrics.json")):
    name = metrics_path.parent.name.removesuffix("_lighting_val")
    metrics = json.loads(metrics_path.read_text())
    result_path = pilot / name / "result.json"
    trained = json.loads(result_path.read_text()) if result_path.exists() else {}
    summary[name] = {"updates": trained.get("global_step"), "training_seconds": trained.get("training_seconds"),
                     "lighting_val_reliable_AP50_95": metrics.get("reliable_AP50_95"),
                     "lighting_val_AP50_95": metrics.get("AP50_95"),
                     "per_class": {row["class"]: row["AP50_95"] for row in metrics.get("per_class", [])}}
(pilot / "summary.json").write_text(json.dumps(summary, indent=2))
for name, row in summary.items():
    print(f"{name:32s} updates={row['updates']} reliable={row['lighting_val_reliable_AP50_95']}")
EOF
log "CHAIN COMPLETE: apply the 2026-10-30 rule to $PILOT/summary.json"
