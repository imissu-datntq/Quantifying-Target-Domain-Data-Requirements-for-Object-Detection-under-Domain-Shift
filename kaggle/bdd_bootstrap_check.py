"""CPU-only Kaggle check: rebuild the frozen dataset from the mirror and run the unit tests."""
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
from common import DATASET, bootstrap, log, run  # noqa: E402

bootstrap()
run([sys.executable, "-c", "import torch, torchvision, pycocotools, platform; print(platform.python_version(), torch.__version__, torchvision.__version__)"], "versions.log")
run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], "unit_tests.log")
run([sys.executable, "scripts/train_frcnn.py", "--dataset-root", str(DATASET), "--output-dir", "/tmp/dry",
     "--stage", "source", "--domain", "source", "--budget", "full", "--dry-run"], "dry_run.log")
log("bootstrap check complete")
