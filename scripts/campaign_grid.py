"""Write the campaign grid with per-run cost, or estimate it under assumed (unlocked) settings.

Launching requires the locked policy in bdd_pipeline.campaign.POLICY. --assume S,E,LR only
prints planning estimates and marks the CSV as unlocked.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from bdd_pipeline.campaign import POLICY, estimate, grid, require_locked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("runs/campaign/grid.csv"))
    parser.add_argument("--assume", help="Planning only: floor_steps,epochs,lr, e.g. 500,5,0.001")
    parser.add_argument("--images-per-second", type=float, default=6.1, help="Measured T4 rate at batch 2, AMP")
    args = parser.parse_args()
    policy = dict(POLICY)
    locked = True
    if args.assume:
        floor, epochs, lr = args.assume.split(",")
        policy.update(floor_steps=int(floor), epochs=int(epochs), lr=float(lr))
        locked = all(POLICY[key] == policy[key] for key in ("floor_steps", "epochs", "lr"))
    require_locked(policy)
    rows = grid(policy)
    totals = defaultdict(lambda: [0, 0])
    for row in rows:
        row.update(estimate(row, policy))
        row["train_hours"] = round(row["image_passes"] / args.images_per_second / 3600, 3)
        row["status"] = "pending" if locked else "UNLOCKED-estimate"
        key = row["domain"] if row["stage"] == "fine-tune" else row["stage"]
        totals[key][0] += 1
        totals[key][1] += row["image_passes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passes = sum(row["image_passes"] for row in rows)
    print(f"{'locked' if locked else 'UNLOCKED estimate'}: S={policy['floor_steps']} E={policy['epochs']} "
          f"LR={policy['lr']} B_eff={policy['batch_size'] * policy['accumulation']}")
    for key, (count, value) in totals.items():
        print(f"  {key:10s} {count:3d} runs {value:>10,d} image-passes {value / args.images_per_second / 3600:6.1f} h")
    print(f"  total      {len(rows):3d} runs {passes:>10,d} image-passes {passes / args.images_per_second / 3600:6.1f} h "
          f"(training only, {args.images_per_second} img/s) -> {args.output}")


if __name__ == "__main__":
    main()
