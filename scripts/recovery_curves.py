"""Overall and per-class recovery curves from evaluation metrics.

Per-class curves separate classes whose recovery mostly re-learns a class prior
(e.g. more traffic lights per night image) from classes that need new visual features.
The index CSV has columns domain,budget,seed,metrics where budget is 0 (source-only),
an integer N or "full" (ceiling), and metrics is an evaluate_frcnn metrics.json path.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from bdd_pipeline.data import EXPECTED_DOMAINS, RELIABLE_CLASSES
from bdd_pipeline.recovery import curve_summary, recovery_table

PALETTE = {"reliable": "#1f2933", "pedestrian": "#2f6b9a", "car": "#2d997c", "traffic light": "#d48041", "traffic sign": "#8163a8"}


def plot(summary: list[dict], output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for domain in sorted({entry["domain"] for entry in summary}):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for name in ("reliable",) + RELIABLE_CLASSES:
            entry = next((item for item in summary if item["domain"] == domain and item["class"] == name), None)
            if entry is None:
                continue
            budgets = [point["budget"] for point in entry["curve"]]
            means = [point["mean"] for point in entry["curve"]]
            sds = [0 if math.isnan(point["sd"]) else point["sd"] for point in entry["curve"]]
            width = 2.6 if name == "reliable" else 1.4
            ax.plot(budgets, means, marker="o", lw=width, color=PALETTE[name], label=name)
            ax.fill_between(budgets, [m - s for m, s in zip(means, sds)], [m + s for m, s in zip(means, sds)],
                            color=PALETTE[name], alpha=0.12, lw=0)
        for level in (0.75, 0.90):
            ax.axhline(level, color="gray", ls=":", lw=1)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Labeled target images N")
        ax.set_ylabel("Recovery(N), seed mean ± SD")
        ax.set_title(f"{domain}: reliable mean and per-class recovery")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        path = output / f"recovery_{domain}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-denominator", type=float, default=0.01,
                        help="Recovery is undefined when mAP_c - mAP(0) does not exceed this AP difference")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    with args.index.open(encoding="utf-8", newline="") as handle:
        records = [dict(row, seed=row["seed"], metrics=json.loads(Path(row["metrics"]).read_text(encoding="utf-8")))
                   for row in csv.DictReader(handle)]
    pools = {domain: counts["train"] for domain, counts in EXPECTED_DOMAINS.items()}
    rows = recovery_table(records, pools, args.min_denominator)
    summary = curve_summary(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "recovery_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "recovery_summary.json").write_text(json.dumps({"min_denominator": args.min_denominator, "curves": summary}, indent=2), encoding="utf-8")
    if not args.no_plots:
        plot(summary, args.output_dir)
    print(f"Wrote {len(rows)} recovery rows and {len(summary)} curves to {args.output_dir}")


if __name__ == "__main__":
    main()
