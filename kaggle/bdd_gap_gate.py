"""Kaggle GPU: official zero-shot gap gate with the completed source checkpoint.

The source checkpoint is scored on source val and on each target train pool; no
held-out label is read.
"""
import json
import os
import sys
from pathlib import Path


def locate(marker: str) -> Path:
    for root, directories, files in os.walk("/kaggle/input"):
        if marker in files:
            return Path(root)
        directories[:] = [d for d in directories if d not in ("train", "val", "test", "images")]
    raise FileNotFoundError(marker)


sys.path.insert(0, str(locate("expected_sha256.json") / "code" / "kaggle"))
from common import DATASET, WORK, bootstrap, log, run  # noqa: E402

bootstrap(rehash_sample=2000)
source_dir = None
for root, _, files in os.walk("/kaggle/input"):
    if "best.pt" in files and "result.json" in files and Path(root).name == "source_seed0":
        source_dir = Path(root)
        break
if source_dir is None:
    raise FileNotFoundError("Attach the completed kizhowl/bdd100k-source-train output")
result = json.loads((source_dir / "result.json").read_text())
if not result["completed"]:
    raise ValueError("Source training is not complete")
source = source_dir / "best.pt"
log(f"source checkpoint {source}: {json.dumps(result)}")

run([sys.executable, "scripts/zero_shot_gap.py", "--dataset-root", str(DATASET), "--checkpoint", str(source),
     "--domains", "weather", "lighting", "compound", "--images-per-domain", "2000",
     "--bootstrap", "1000", "--device", "cuda", "--threads", "4", "--output-dir", str(WORK / "zero_shot_gap_source")],
    "gap_gate.log")
log("zero-shot gap gate complete")
