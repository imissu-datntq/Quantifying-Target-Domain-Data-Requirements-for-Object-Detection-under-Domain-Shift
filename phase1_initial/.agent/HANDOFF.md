# Handoff

## Current state — 2026-09-28

The project model is **Adaptive Teacher + Faster R-CNN R101-C4**. The completed
1,000-image pilot and rationale are documented in
`docs/small_sample_testing.md` and `.agent/DECISIONS.md`.

No training process is live. No held-out test labels were read.

The rerun used exactly 1,000 source and 1,000 target images per candidate and
the same deterministic 1,000-image lighting-validation membership. Numeric
mAP50:95/mAP50 in AP points was SSDA 2.229456/6.614537, SF
1.856307/4.760677 and R101 0.686985/1.936026. These short checkpoints have
different initialization, source history and update semantics, so the order is
diagnostic rather than a defensible accuracy ranking. R101 remains selected for
proposal alignment and the intended Faster R-CNN teacher-student experiment.

R101 completed 1,000 source plus 1,000 adaptation iterations. SSDA and SF each
completed 1,015 traced batches/151 optimizer updates; their earlier 120-update
attempts stopped at 503 batches and were correctly marked incomplete.

An earlier attempt to initialize R101 adaptation from a source detector exposed
a key-layout bug and failed at update 1. `run_adaptive_teacher_smoke.py` now
copies the detector weights into both `modelStudent.*` and `modelTeacher.*`.
The corrected 120-update run and the later 1,000-iteration rerun both completed
with strict checkpoint reload and finite metrics.

The former 60-arm selection campaign, production auxiliary executor, registry,
cost/readiness generators, full YOLO launchers and their tests/docs/artifacts
were removed. Do not recreate or resume that campaign. Retained code is limited
to bounded smoke/integration paths and shared split/evaluation utilities.

Next project work, only when requested: execute the R101-only source,
adaptation and supervised fine-tuning schedule for the proposal budget ladder.
That is separate from the completed model-selection pilot.
