"""Static guards for the R101 exact-restart sidecar boundary."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AdaptiveTeacherRestartTests(unittest.TestCase):
    def test_sidecar_captures_all_rng_streams_and_loader_cursor(self):
        source = (ROOT / "run_adaptive_teacher_smoke.py").read_text(encoding="utf-8")
        for marker in (
            "rng_sampler_state.pth",
            "random.getstate()",
            "np.random.get_state()",
            "torch.get_rng_state()",
            "torch.cuda.get_rng_state_all()",
            "sampler_cursor",
            "num_workers",
            "getattr(trainer, '_trainer', trainer)",
            "next(iterator_owner._data_loader_iter)",
            "use_deterministic_algorithms",
            "deterministic_requested",
            "CUBLAS_WORKSPACE_CONFIG",
            "allow_tf32",
        ):
            self.assertIn(marker, source)

    def test_exact_resume_is_not_claimed_without_a_sidecar(self):
        source = (ROOT / "run_adaptive_teacher_smoke.py").read_text(encoding="utf-8")
        self.assertIn("if rng_path.is_file():", source)
        self.assertIn("state['rng_sampler_resume_restored'] = True", source)
        self.assertIn("exact_rng_sampler_resume=False", source)

    def test_restart_audit_separates_same_segment_control(self):
        source = (ROOT / "audit_r101_restart.py").read_text(encoding="utf-8")
        self.assertIn("same_segment_control", source)
        self.assertIn("uninterrupted_vs_resumed", source)
        self.assertIn("exact_restart", source)

    def test_smoke_is_bounded_and_never_uses_full_manifests(self):
        source = (ROOT / "run_adaptive_teacher_smoke.py").read_text(encoding="utf-8")
        self.assertIn("smoke=True", source)
        self.assertIn("--train-images", source)
        self.assertIn("max_training_images_per_pool=args.train_images", source)
        self.assertIn("6 <= args.steps <= 1000", source)
        self.assertNotIn("--campaign", source)

    def test_detector_source_is_copied_into_teacher_and_student(self):
        source = (ROOT / "run_adaptive_teacher_smoke.py").read_text(encoding="utf-8")
        self.assertIn("adapted_state['modelStudent.' + key]", source)
        self.assertIn("adapted_state['modelTeacher.' + key]", source)
        self.assertIn("adapt_init.pth", source)


if __name__ == "__main__":
    unittest.main()
