"""Atomic artifacts, explicit checkpoint schema and complete random state."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch

from prepare_revised_bdd100k import sha256
from .model import MODEL_CONTRACT

SCHEMA = 1


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("Cannot exactly resume a CUDA run on CPU")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "pycocotools", "numpy", "Pillow")},
            "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
            "torch_threads": torch.get_num_threads()}


def load_checkpoint(path: Path) -> dict:
    # Our local checkpoints contain optimizer/RNG objects, so weights_only is false.
    # Only load trusted local artifacts created by this pipeline.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != SCHEMA or checkpoint.get("model_contract") != MODEL_CONTRACT:
        raise ValueError("Incompatible checkpoint schema/model contract")
    return checkpoint


def checkpoint_fingerprint(path: Path) -> dict:
    checkpoint = load_checkpoint(path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "global_step": checkpoint["global_step"],
            "stage": checkpoint["config"]["stage"], "domain": checkpoint["config"]["domain"],
            "completed": checkpoint.get("training_completed", False),
            "smoke": checkpoint["config"]["smoke"], "manifest_sha256": checkpoint["config"]["manifest_sha256"]}
