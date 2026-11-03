# Agent state map

This directory separates current operating state from durable decisions and
completed execution plans.

- `HANDOFF.md`: current repository state and exact next safe action.
- `TODO.md`: outstanding work only.
- `DECISIONS.md`: durable project decisions.
- `PLANS.md`: standard for writing and maintaining ExecPlans.
- `plans/active/`: current approved task plans.
- `plans/archive/`: completed or superseded plans; read only when needed.

Normal startup order is `AGENTS.md`, `HANDOFF.md`, `TODO.md`, `DECISIONS.md`,
the relevant active ExecPlan, then `git status` and `git diff`.
