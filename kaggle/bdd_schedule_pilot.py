"""Kaggle GPU: third target fine-tuning schedule pilot (lighting val only).

Rounds 1-2 (kernels bdd100k-gate-schedule, bdd100k-schedule-pilot) showed fewer updates and a
lower LR scoring better, without reaching a peak. This round brackets S and E downward, checks
LR 5e-4 at N=1000, and measures seed spread at N=100. The decision rule is in .agent/DECISIONS.md
(2026-10-30) and was fixed before this round ran. Runs are smoke-marked.
"""
import json
import os
import sys
from pathlib import Path

PILOTS = [  # name, budget, floor S, epochs E, LR, seed
    ("N100_S250_E5_lr1e-3_seed0", "100", 250, 5, 0.001, 0),
    ("N100_S250_E5_lr1e-3_seed1", "100", 250, 5, 0.001, 1),
    ("N100_S250_E5_lr1e-3_seed2", "100", 250, 5, 0.001, 2),
    ("N100_S125_E1_lr1e-3_seed0", "100", 125, 1, 0.001, 0),
    ("N10_S250_E5_lr1e-3_seed0", "10", 250, 5, 0.001, 0),
    ("N10_S500_E5_lr1e-3_seed0", "10", 500, 5, 0.001, 0),
    ("N1000_S1_E2_lr1e-3_seed0", "1000", 1, 2, 0.001, 0),
    ("N1000_S1_E3_lr1e-3_seed0", "1000", 1, 3, 0.001, 0),
    ("N1000_S500_E5_lr5e-4_seed0", "1000", 500, 5, 0.0005, 0),
]


def locate(marker: str) -> Path:
    for root, directories, files in os.walk("/kaggle/input"):
        if marker in files:
            return Path(root)
        directories[:] = [d for d in directories if d not in ("train", "val", "test", "images")]
    raise FileNotFoundError(marker)


sys.path.insert(0, str(locate("expected_sha256.json") / "code" / "kaggle"))
from common import DATASET, WORK, bootstrap, log, run  # noqa: E402

bootstrap(rehash_sample=2000)
source = None
for root, _, files in os.walk("/kaggle/input"):
    if "best.pt" in files and Path(root).name == "source_seed0":
        source = Path(root) / "best.pt"
        break
if source is None:
    raise FileNotFoundError("Attach the completed kizhowl/bdd100k-source-train output")
summary = {}
for name, budget, floor, epochs, lr, seed in PILOTS:
    output = WORK / "schedule_pilot3" / name
    code = run([sys.executable, "scripts/train_frcnn.py", "--dataset-root", str(DATASET), "--output-dir", str(output),
                "--stage", "fine-tune", "--domain", "lighting", "--budget", budget, "--seed", str(seed), "--smoke",
                "--source-checkpoint", str(source), "--floor-steps", str(floor), "--epochs", str(epochs),
                "--batch-size", "2", "--accumulation", "1", "--lr", str(lr), "--warmup-updates", "50",
                "--save-every-updates", "100000", "--device", "cuda", "--amp", "--workers", "4", "--threads", "4"],
               f"schedule3_{name}.log", check=False)
    if code:
        summary[name] = {"failed": True}
        continue
    evaluation = WORK / "schedule_pilot3" / f"{name}_lighting_val"
    run([sys.executable, "scripts/evaluate_frcnn.py", "--dataset-root", str(DATASET), "--checkpoint", str(output / "last.pt"),
         "--domain", "lighting", "--output-dir", str(evaluation), "--device", "cuda", "--amp", "--batch-size", "4", "--workers", "4"],
        f"schedule3_{name}_eval.log", check=False)
    metrics = json.loads((evaluation / "metrics.json").read_text()) if (evaluation / "metrics.json").exists() else {}
    trained = json.loads((output / "result.json").read_text())
    summary[name] = {"budget": budget, "floor_steps": floor, "epochs": epochs, "lr": lr, "seed": seed,
                     "updates": trained["global_step"], "training_seconds": trained["training_seconds"],
                     "lighting_val_reliable_AP50_95": metrics.get("reliable_AP50_95"), "lighting_val_AP50_95": metrics.get("AP50_95"),
                     "per_class": {row["class"]: row["AP50_95"] for row in metrics.get("per_class", [])}}
    for checkpoint in output.glob("*.pt"):
        checkpoint.unlink()
    (evaluation / "predictions.json").unlink(missing_ok=True)
    log(f"{name}: {json.dumps(summary[name])}")
    (WORK / "schedule_pilot3_summary.json").write_text(json.dumps(summary, indent=2))
log("schedule pilot 3 complete")
