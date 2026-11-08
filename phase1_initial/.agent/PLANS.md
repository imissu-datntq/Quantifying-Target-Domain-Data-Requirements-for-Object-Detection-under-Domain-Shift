# ExecPlan standard

Use an ExecPlan for material cross-module changes, model campaigns, data
migrations, or work requiring multiple approval gates. Small fixes and routine
documentation updates do not need one.

## Required sections

Every active ExecPlan must contain status/timestamps, purpose and observable
outcome, context, scope and non-goals, progress, discoveries, decisions,
ordered work, validation criteria, recovery/interfaces/dependencies, and an
outcome or retrospective when complete.

## Maintenance rules

- Keep one active plan per task or milestone.
- Update progress after each bounded checkpoint with local time and evidence.
- Keep task decisions in the plan and append durable policy decisions to
  `.agent/DECISIONS.md`.
- Move completed or superseded plans to `plans/archive/`; do not rewrite their
  historical progress.
- Do not read archives or history during normal startup.

## Execution boundaries

Planning approval does not authorize data rewrites, large training, holdout
access, packaging, promotion, or publication. Candidate selection and tuning
use chronological development evidence only. Lock before refit; refit uses
train+validation with zero holdout rows. Evaluate a campaign holdout once and
never select a backup after opening it.
