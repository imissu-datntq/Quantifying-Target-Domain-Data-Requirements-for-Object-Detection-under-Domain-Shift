"""Kaggle GPU pilot: dataset rebuild, tests, AMP/batch throughput and memory.

Pilot runs are smoke-marked; no held-out label is read and nothing here is a research score.
"""
import json
import os
import sys
from pathlib import Path


def locate_protocol() -> Path:
    for root, directories, files in os.walk("/kaggle/input"):
        if "expected_sha256.json" in files:
            return Path(root)
        directories[:] = [d for d in directories if d not in ("train", "val", "test", "images")]
    raise FileNotFoundError("protocol dataset not attached")


sys.path.insert(0, str(locate_protocol() / "code" / "kaggle"))
from common import CODE, DATASET, WORK, bootstrap, log, run  # noqa: E402

info = bootstrap(rehash_sample=2000)
run([sys.executable, "-c", "import torch, torchvision, pycocotools, platform; print(platform.python_version(), torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"], "versions.log")
run(["nvidia-smi"], "nvidia_smi.log", check=False)
run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], "unit_tests.log")

pilots = {}
for name, batch, amp in (("b2_amp", 2, True), ("b4_amp", 4, True), ("b2_fp32", 2, False)):
    output = WORK / "pilot" / name
    arguments = [sys.executable, "scripts/train_frcnn.py", "--dataset-root", str(DATASET), "--output-dir", str(output),
                 "--stage", "source", "--domain", "source", "--budget", "full", "--smoke", "--initialization", "coco",
                 "--limit-train", "480", "--limit-val", "100", "--epochs", "1", "--floor-steps", "60",
                 "--batch-size", str(batch), "--accumulation", "1", "--eval-every-updates", "60",
                 "--save-every-updates", "60", "--warmup-updates", "10", "--workers", "4", "--threads", "4",
                 "--device", "cuda"] + (["--amp"] if amp else [])
    if run(arguments, f"pilot_{name}.log", check=False):
        pilots[name] = {"batch_size": batch, "amp": amp, "failed": True}
        continue
    log_rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    result = json.loads((output / "result.json").read_text())
    validation = json.loads((output / "validation_step_60.json").read_text())
    pilots[name] = {"batch_size": batch, "amp": amp, "updates": result["global_step"],
                    "train_images_per_second": result["training_images_per_second"],
                    "peak_cuda_memory_gib": result["peak_cuda_memory_allocated_bytes"] / 2 ** 30,
                    "val_images_per_second": validation["inference_images_per_second"],
                    "final_loss": log_rows[-1]["loss"],
                    "amp_skipped_updates": sum(bool(row.get("amp_step_skipped")) for row in log_rows),
                    "steady_seconds_per_update": sorted(row["seconds"] for row in log_rows[5:])[len(log_rows[5:]) // 2]}
    for checkpoint in output.glob("*.pt"):
        checkpoint.unlink()  # keep pilot outputs small
    log(json.dumps(pilots[name]))

(WORK / "pilot_summary.json").write_text(json.dumps({"bootstrap": info, "pilots": pilots}, indent=2))
log("pilot complete")
