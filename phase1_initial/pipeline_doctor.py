"""Read-only readiness checks for the four candidate research pipelines.

The doctor deliberately reports missing environments, weights, auxiliary
assets, unsupported configuration claims, and unsafe label policies instead of
downloading or patching anything.  A nonzero exit means at least one required
gate remains unresolved; it does not mean the host GPU driver is broken.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_EXTERNAL = ROOT / "external"
DEFAULT_SPLIT = ROOT / "data" / "research_split_v1"
DEFAULT_INPUTS = ROOT / "data" / "research_inputs_v1"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail", "pending"}:
            raise ValueError(f"Invalid check status: {self.status}")


@dataclass(frozen=True)
class RepositorySpec:
    name: str
    directory: str
    sha: str
    required_paths: tuple[str, ...]


REPOSITORIES = (
    RepositorySpec(
        "ssda-yolo",
        "SSDA-YOLO",
        "3ab1f506ded2ca67233699a2e9bb8944aea40013",
        ("ssda_yolov5_train.py", "ssda_yolov5_test.py", "requirements.txt"),
    ),
    RepositorySpec(
        "ms-dayolo",
        "MS-DAYOLO",
        "c655ceed3b99e6b499a9c907a3ae9c2deba98c1e",
        ("cfg/ms-dayolo.cfg", "src/detector.c", "Makefile"),
    ),
    RepositorySpec(
        "sf-yolo",
        "sf-yolo",
        "c84dfad79a889b5f172d7192ee0379edfd738835",
        ("train_sf-yolo.py", "TargetAugment_train/train.py", "requirements.txt"),
    ),
    RepositorySpec(
        "adaptive-teacher",
        "adaptive_teacher",
        "5256463ad9ec90fd5ba84ebb8d53bed56bd369df",
        ("train_net.py", "adapteacher/engine/trainer.py", "configs/Base-RCNN-C4.yaml"),
    ),
)


def run_capture(command: Sequence[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return 127, str(error)
    return result.returncode, result.stdout.strip()


def check_repository(
    path: Path, expected_sha: str, required_paths: Iterable[str]
) -> Check:
    if not path.is_dir():
        return Check(f"repository:{path.name}", "fail", f"Repository directory is missing: {path}")
    code, output = run_capture(("git", "rev-parse", "HEAD"), cwd=path)
    if code != 0:
        return Check(
            f"repository:{path.name}", "fail", f"Cannot resolve repository HEAD: {output}"
        )
    if output != expected_sha:
        return Check(
            f"repository:{path.name}",
            "fail",
            f"Revision {output} does not match pinned {expected_sha}",
        )
    missing = [relative for relative in required_paths if not (path / relative).is_file()]
    if missing:
        return Check(
            f"repository:{path.name}",
            "fail",
            f"Pinned checkout is missing required paths: {', '.join(missing)}",
        )
    return Check(f"repository:{path.name}", "pass", f"Pinned revision {expected_sha}")


def check_text_markers(path: Path, markers: Iterable[str]) -> Check:
    if not path.is_file():
        return Check(f"markers:{path.name}", "fail", f"Source file is missing: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        return Check(
            f"markers:{path.name}", "fail", f"Missing expected markers: {', '.join(missing)}"
        )
    return Check(f"markers:{path.name}", "pass", "All expected source markers are present")


def read_json(path: Path) -> dict[str, object] | None:
    """Read a generated evidence artifact without turning stale/malformed runs into passes."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def completed_status(path: Path, *, phase: str | None = None) -> bool:
    value = read_json(path)
    return bool(value and value.get("status") == "completed"
                and (phase is None or value.get("phase") == phase)
                and value.get("selection_evidence") is False)


def sf_runtime_evidence(runs_root: Path) -> bool:
    status = read_json(runs_root / "sf-adapt-120-001/status.json")
    audit = read_json(runs_root / "sf-adapt-120-001/tensor_audit.json")
    return bool(status and audit and status.get("status") == "completed"
                and status.get("candidate") == "sf-yolo"
                and int(status.get("optimizer_updates", 0)) >= 120
                and status.get("nonfinite_log") is False
                and audit.get("audit") == "passed"
                and audit.get("selection_evidence") is False)


def r101_runtime_evidence(runs_root: Path) -> bool:
    source = runs_root / "r101-source-smoke-003/smoke_status.json"
    finetune = runs_root / "r101-finetune-smoke-001/smoke_status.json"
    adaptation = runs_root / "r101-adapt-smoke-004/smoke_status.json"
    exports = (runs_root / "r101-common-compound-001/status.json",
               runs_root / "r101-source-common-compound-001/status.json")
    return (completed_status(source, phase="source")
            and completed_status(finetune, phase="finetune")
            and completed_status(adaptation, phase="adapt")
            and all(completed_status(path) for path in exports))


def check_asset(path: Path, *, required: bool) -> Check:
    if path.is_file() and path.stat().st_size > 0:
        return Check(f"asset:{path.name}", "pass", f"Present ({path.stat().st_size} bytes): {path}")
    status = "fail" if required else "pending"
    qualifier = "Required" if required else "Optional/pilot"
    return Check(f"asset:{path.name}", status, f"{qualifier} asset is missing: {path}")


def check_directory(path: Path, name: str) -> Check:
    if path.is_dir():
        return Check(name, "pass", f"Present: {path}")
    return Check(name, "fail", f"Directory is missing: {path}")


def check_executable(path: Path, name: str) -> Check:
    if path.is_file() and path.stat().st_mode & 0o111:
        return Check(name, "pass", f"Executable is present: {path}")
    return Check(name, "fail", f"Executable is missing or not executable: {path}")


def check_gpu() -> Check:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return Check("gpu", "fail", "nvidia-smi is not on PATH")
    code, output = run_capture((executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"))
    if code != 0:
        return Check(
            "gpu",
            "fail",
            "GPU query failed in this execution context; verify host access separately: " + output,
        )
    return Check("gpu", "pass", output)


def compatibility_checks(external: Path, runs_root: Path | None = None) -> list[Check]:
    runs_root = runs_root or external.parent / "runs"
    checks = [
        check_text_markers(
            external / "SSDA-YOLO" / "ssda_yolov5_train.py",
            ("train_path_target_real", "create_dataloader(train_path_target_real"),
        ),
        check_text_markers(
            external / "MS-DAYOLO" / "src" / "detector.c",
            ("train_images_target", "da_train_network_waitkey"),
        ),
        check_text_markers(
            external / "sf-yolo" / "train_sf-yolo.py",
            (
                "labels = np.concatenate(dataset.labels, 0)",
                "labels_to_class_weights(dataset.labels, nc)",
                "for i, (imgs, targets, paths, _) in pbar",
            ),
        ),
        check_text_markers(
            external / "adaptive_teacher" / "configs" / "faster_rcnn_R101_cross_clipart.yaml",
            ('_BASE_: "./Base-RCNN-C4.yaml"', "DEPTH: 101", 'DIS_TYPE: "res4"'),
        ),
    ]
    if checks[2].status == "pass":
        adapter = external / "sf-yolo" / "utils" / "unlabeled_target.py"
        training = (external / "sf-yolo" / "train_sf-yolo.py").read_text()
        integrated = adapter.is_file() and all(marker in training for marker in (
            "--unlabeled-target", "create_unlabeled_dataloader", "concatenate_pseudo_labels",
        ))
        runtime = integrated and sf_runtime_evidence(runs_root)
        checks.append(
            Check(
                "compatibility:sf-unlabeled-loader",
                "pass" if runtime else ("pending" if integrated else "fail"),
                ("Full TargetAugment/student/teacher runtime completed at the native graph with "
                 "finite audited checkpoints; no selection-quality claim." if runtime else
                 "Image-only adapter markers are present. Full TargetAugment/student/teacher "
                 "runtime and annotation-access verification remain required; markers alone "
                 "do not certify eligibility." if integrated else
                 "Upstream SF-YOLO consumes target label metadata for validation, augmentation, "
                 "auto-anchor/class weights and callbacks. Synthetic boxes are not an admissible "
                 "unlabeled integration."),
            )
        )
        if runtime:
            checks.append(Check(
                "compatibility:sf-exact-restart", "pending",
                "Serialized continuation passed, but upstream RNG/sampler state is absent; "
                "exact restart and full auxiliary preparation are outside the bounded pilot.",
            ))
    if checks[3].status == "pass":
        runtime = r101_runtime_evidence(runs_root)
        checks.append(
            Check(
                "compatibility:faster-rcnn-r101-c4",
                "pass" if runtime else "pending",
                ("Native R101-C4 source, N=100 fine-tune and Adaptive Teacher smokes completed "
                 "with checkpoint reload and common validation exports; no selection-quality claim." if runtime else
                 "User selected Adaptive Teacher R101-C4. Initial native-resolution source/DA "
                 "smokes are documented in docs/adaptive_teacher_execution.md. Final-quality "
                 "training and exact resume remain outside the bounded pilot; "
                 "static source markers alone do not certify runtime readiness."),
            )
        )
        if runtime:
            checks.append(Check(
                "compatibility:faster-rcnn-exact-resume", "pending",
                "Serialized model/optimizer continuation exists, but exact RNG/sampler "
                "restoration remains outside the bounded pilot.",
            ))
    return checks


def environment_and_asset_checks(external: Path, env_root: Path, weights: Path) -> list[Check]:
    checks: list[Check] = []
    for name in ("ssda-yolo", "sf-yolo", "adaptive-teacher"):
        checks.append(check_executable(env_root / name / "bin" / "python", f"environment:{name}"))
    checks.append(check_executable(external / "MS-DAYOLO" / "darknet", "environment:ms-dayolo"))
    for path in (
        weights / "yolov5l-v5.0.pt",
        weights / "yolov4.conv.137",
        weights / "vgg16_ori.pth",
    ):
        checks.append(check_asset(path, required=False))
    checks.extend(
        [
            check_asset(weights / "sf-decoder.pth", required=False),
            check_asset(weights / "sf-fc1.pth", required=False),
            check_asset(weights / "sf-fc2.pth", required=False),
        ]
    )
    return checks


def summarize(checks: Iterable[Check]) -> dict[str, int]:
    summary = {"pass": 0, "fail": 0, "pending": 0}
    for check in checks:
        summary[check.status] += 1
    return summary


def run_doctor(args: argparse.Namespace) -> dict[str, object]:
    checks: list[Check] = [
        check_directory(args.split_root, "data:research-split"),
        check_directory(args.inputs_root, "data:model-inputs"),
        check_gpu(),
    ]
    for spec in REPOSITORIES:
        checks.append(
            check_repository(
                args.external_root / spec.directory,
                expected_sha=spec.sha,
                required_paths=spec.required_paths,
            )
        )
    checks.extend(compatibility_checks(args.external_root, args.runs_root))
    checks.extend(environment_and_asset_checks(args.external_root, args.env_root, args.weights_root))
    return {
        "summary": summarize(checks),
        "checks": [asdict(check) for check in checks],
        "meaning": "fail = required gate unresolved; pending = asset intentionally not acquired yet",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--inputs-root", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--env-root", type=Path, default=ROOT / ".venvs")
    parser.add_argument("--weights-root", type=Path, default=ROOT / "weights")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)
    for name in ("external_root", "split_root", "inputs_root", "env_root", "weights_root", "runs_root"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_doctor(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for check in report["checks"]:
            print(f"[{check['status'].upper():7}] {check['name']}: {check['detail']}")
        print("Summary:", json.dumps(report["summary"], sort_keys=True))
    return 1 if report["summary"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
