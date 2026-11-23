"""The 42 revision runs of 2026-11-21: grid, pre-registered update counts and single-argument parity."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import local_revision_runs as revision  # noqa: E402
from bdd_adapt import grid as adapt_grid  # noqa: E402
from bdd_pipeline import campaign  # noqa: E402

DATASET, TRAIN, SOURCE = Path("data"), Path("out/train"), Path("source.pt")


def flags(command: list[str]) -> dict[str, str | bool]:
    out, index = {}, 2
    while index < len(command):
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            out[command[index]] = command[index + 1]
            index += 2
        else:
            out[command[index]] = True
            index += 1
    return out


class RevisionGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = revision.rows()

    def test_blocks_and_order(self) -> None:
        self.assertEqual(Counter(row["block"] for row in self.rows),
                         {"replay_seeds": 12, "matched_passes": 18, "ceiling_e12": 10, "matched_pool_e3": 2})
        self.assertEqual([row["block"] for row in self.rows],
                         ["replay_seeds"] * 12 + ["matched_passes"] * 18 + ["ceiling_e12"] * 10 + ["matched_pool_e3"] * 2)
        self.assertEqual(len({(row["block"], row["run_id"]) for row in self.rows}), 42)
        self.assertEqual([row["run_id"] for row in self.rows[38:40]], ["fine-tune_lighting_Nfull_s0", "fine-tune_lighting_Nfull_s1"])

    def test_matched_pool_e3_adds_the_missing_seed_under_the_locked_policy(self) -> None:
        block = [row for row in self.rows if row["block"] == "matched_pool_e3"]
        self.assertEqual([row["run_id"] for row in block], ["fine-tune_lighting_N1738_s1", "fine-tune_weather_N1738_s1"])
        original = {row["run_id"] for row in campaign.grid(campaign.POLICY)}
        for row in block:
            self.assertNotIn(row["run_id"], original)
            self.assertIn(row["run_id"].replace("_s1", "_s0"), original)
            self.assertEqual(row["policy"], campaign.POLICY)
            self.assertEqual(revision.updates_of(row), 2607)

    def test_replay_seeds_are_new_and_keep_the_locked_policy(self) -> None:
        block = [row for row in self.rows if row["block"] == "replay_seeds"]
        original = {row["run_id"] for row in adapt_grid.grid(adapt_grid.POLICY)}
        self.assertFalse({row["run_id"] for row in block} & original)
        self.assertEqual({(row["stage"], row["budget"], row["seed"]) for row in block},
                         {("replay", n, s) for n in (10, 100) for s in (1, 2)})
        for row in block:
            self.assertEqual(row["policy"], adapt_grid.POLICY)
            self.assertEqual(revision.updates_of(row), 2000)

    def test_matched_passes_equal_fine_tuning_labeled_passes(self) -> None:
        block = [row for row in self.rows if row["block"] == "matched_passes"]
        self.assertEqual({(row["stage"], row["domain"], row["seed"]) for row in block},
                         {(stage, domain, seed) for stage in ("adapt", "replay") for domain in revision.TARGETS for seed in (0, 1, 2)})
        fine_tune = campaign.estimate({"stage": "fine-tune", "domain": "lighting", "budget": "1000"}, campaign.POLICY)
        for row in block:
            self.assertEqual(row["budget"], 1000)
            self.assertEqual(revision.updates_of(row), 3000)  # one labeled target image per update
            self.assertEqual(revision.updates_of(row), fine_tune["image_passes"])

    def test_ceilings_run_twelve_epochs(self) -> None:
        block = {row["run_id"]: row for row in self.rows if row["block"] == "ceiling_e12"}
        self.assertEqual(set(block), {f"fine-tune_{domain}_N{budget}_s{seed}" for seed in (0, 1) for domain, budget in
                                      (("compound", "full"), ("weather", "full"), ("lighting", "full"),
                                       ("lighting", "1738"), ("weather", "1738"))})
        expected = {"fine-tune_compound_Nfull_s0": 10428, "fine-tune_weather_Nfull_s0": 11916,
                    "fine-tune_lighting_Nfull_s0": 108096, "fine-tune_lighting_N1738_s1": 10428}
        for run_id, updates in expected.items():
            self.assertEqual(revision.updates_of(block[run_id]), updates)
            original = campaign.estimate(block[run_id], campaign.POLICY)["updates"]
            self.assertEqual(updates, original * 12 // campaign.POLICY["epochs"])

    def test_each_run_differs_from_its_original_in_one_argument(self) -> None:
        for row in self.rows:
            new = flags(revision.train_command(row, "python", DATASET, TRAIN, SOURCE))
            if row["block"] == "matched_pool_e3":
                old = flags(["python", "x"] + campaign.train_arguments(dict(row, seed=0), campaign.POLICY, DATASET, TRAIN, SOURCE) + ["--device", "cuda"])
                changed = {"--seed"}
            elif row["kind"] == "fine-tune":
                old = flags(["python", "x"] + campaign.train_arguments(row, campaign.POLICY, DATASET, TRAIN, SOURCE) + ["--device", "cuda"])
                changed = {"--epochs"}
            else:
                reference = dict(row, seed=0) if row["block"] == "replay_seeds" else row
                old = flags(["python", "x"] + adapt_grid.train_arguments(reference, adapt_grid.POLICY, DATASET, TRAIN, SOURCE) + ["--device", "cuda"])
                changed = {"--seed"} if row["block"] == "replay_seeds" else {"--floor-updates"}
            self.assertEqual({flag for flag in new if new[flag] != old.get(flag)}, changed, row["run_id"])
            self.assertEqual(set(new), set(old))
        ceiling = next(row for row in self.rows if row["block"] == "ceiling_e12")
        self.assertEqual(flags(revision.train_command(ceiling, "python", DATASET, TRAIN, SOURCE))["--epochs"], "12")

    def test_frozen_policies_are_not_mutated(self) -> None:
        self.assertEqual(campaign.POLICY["epochs"], 3)
        self.assertEqual(adapt_grid.POLICY["floor_updates"], 2000)
        self.assertEqual(revision.MATCHED_POLICY["floor_updates"], 3000)
        self.assertEqual(revision.CEILING_POLICY["epochs"], 12)


class RunControlTest(unittest.TestCase):
    """Relaunch safety: what is deleted, what is kept and what is repaired."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.train = Path(self.temporary.name) / "train"
        self.train.mkdir()
        self.ceiling = next(row for row in revision.rows() if row["kind"] == "fine-tune")
        self.replay = next(row for row in revision.rows() if row["stage"] == "replay")
        self.summary_path = Path(self.temporary.name) / "session_summary.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_run_stopped_before_first_checkpoint_starts_clean(self) -> None:
        (self.train / "training.jsonl").write_text("{}")
        for row in (self.ceiling, self.replay):
            self.train.mkdir(exist_ok=True)
            revision.prepare_train_dir(row, self.train, "x")
            self.assertFalse(self.train.exists())

    def test_interrupted_run_with_checkpoint_is_kept_for_resume(self) -> None:
        (self.train / "last.pt").write_bytes(b"checkpoint")
        for row in (self.ceiling, self.replay):
            revision.prepare_train_dir(row, self.train, "x")
            self.assertEqual((self.train / "last.pt").read_bytes(), b"checkpoint")

    def test_finished_ceiling_without_predictions_keeps_its_weights(self) -> None:
        (self.train / "result.json").write_text(json.dumps({"completed": True}))
        (self.train / "weights.pt").write_bytes(b"weights")
        with self.assertRaisesRegex(RuntimeError, "by hand"):
            revision.prepare_train_dir(self.ceiling, self.train, "ceiling")
        self.assertEqual((self.train / "weights.pt").read_bytes(), b"weights")
        revision.prepare_train_dir(self.replay, self.train, "replay")  # adapt/replay keep nothing: retrain
        self.assertFalse(self.train.exists())

    def test_finalize_keeps_ceiling_weights_drops_checkpoints_and_records_once(self) -> None:
        import torch
        torch.save({"model": {"w": torch.tensor([1.0, 2.0])}, "optimizer": {"lr": 1}}, self.train / "last.pt")
        torch.save({"model": {}}, self.train / "step_1000.pt")
        (self.train / "result.json").write_text(json.dumps({"global_step": 10428, "training_seconds": 5.0,
                                                            "training_images_per_second": 8.0, "ignored": 1}))
        summary = {"runs": {}}
        with patch.object(revision, "log"):
            revision.finalize(self.ceiling, self.train, "ceiling", summary, self.summary_path)
        self.assertEqual(sorted(path.name for path in self.train.glob("*.pt*")), ["weights.pt"])
        self.assertEqual(torch.load(self.train / "weights.pt")["w"].tolist(), [1.0, 2.0])
        self.assertEqual(summary["runs"]["ceiling"], {"updates": 10428, "training_seconds": 5.0, "training_images_per_second": 8.0})
        self.assertEqual(json.loads(self.summary_path.read_text()), summary)
        self.summary_path.unlink()
        with patch.object(revision, "log"):
            revision.finalize(self.ceiling, self.train, "ceiling", summary, self.summary_path)  # second call: no-op
        self.assertEqual(sorted(path.name for path in self.train.glob("*.pt*")), ["weights.pt"])
        self.assertFalse(self.summary_path.exists())

    def test_finalize_drops_adapt_checkpoint_without_keeping_weights(self) -> None:
        (self.train / "last.pt").write_bytes(b"teacher")
        (self.train / "result.json").write_text(json.dumps({"global_step": 2000, "seconds_per_update": 0.25}))
        summary = {"runs": {}}
        with patch.object(revision, "log"):
            revision.finalize(self.replay, self.train, "replay", summary, self.summary_path)
        self.assertEqual(list(self.train.glob("*.pt")), [])
        self.assertEqual(summary["runs"]["replay"], {"updates": 2000, "seconds_per_update": 0.25})

    def test_other_processes_ignores_itself_only(self) -> None:
        listing = (f"{revision.os.getpid()} python scripts/local_revision_runs.py\n"
                   "4242 /home/u/venv/bin/python scripts/train_adapt.py --stage replay\n"
                   "4300 /bin/bash -c setsid nohup python scripts/local_revision_runs.py\n")  # the launching shell is not a writer
        with patch.object(revision.subprocess, "run") as run:
            run.return_value.stdout = listing
            self.assertEqual(revision.other_processes(), ["4242 /home/u/venv/bin/python scripts/train_adapt.py --stage replay"])
            self.assertIn("train_frcnn", run.call_args.args[0][-1])
            run.return_value.stdout = f"{revision.os.getpid()} python scripts/local_revision_runs.py\n"
            self.assertEqual(revision.other_processes(), [])


if __name__ == "__main__":
    unittest.main()
