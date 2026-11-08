"""EDA of the newly materialized BDD100K train/val, with test metadata only.

Weather/compound validation images are held out since the 2026-10-19 protocol
amendment, so their labels are not read either.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import markdown
import numpy as np
import pandas as pd
from PIL import Image

from prepare_revised_bdd100k import CLASSES, DOMAINS, EXPECTED, EXPECTED_DOMAINS, HOLDOUT_VAL_DOMAINS, RELIABLE_CLASSES, stream_records

PALETTE = {"source": "#2f6b9a", "lighting": "#8163a8", "weather": "#2d997c", "compound": "#d48041"}
TITLES = {"source": "Daytime / clear", "lighting": "Night / clear", "weather": "Daytime / rainy", "compound": "Night / rainy"}


def domain_name(timeofday: str, weather: str) -> str:
    return next((key for key, pair in DOMAINS.items() if pair == (timeofday, weather)), "other")


def save_figure(fig, directory: Path, name: str):
    fig.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def md_table(frame: pd.DataFrame, limit=50) -> str:
    values = frame.head(limit).fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |\n"
    rule = "| " + " | ".join("---" for _ in values.columns) + " |\n"
    return header + rule + "\n".join("| " + " | ".join(row) + " |" for row in values.itertuples(index=False, name=None))


def js_divergence(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    def divergence(p):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / m[mask])))
    return (divergence(a) + divergence(b)) / 2


def probability_absent(population: int, positive: int, budget: int) -> float:
    if positive == 0:
        return 1.0
    if budget > population - positive:
        return 0.0
    return math.exp(sum(math.log1p(-positive / (population - i)) for i in range(budget)))


def analyse(dataset: Path, output: Path, brightness_n: int, seeds: list[int]):
    if not (dataset / "COMPLETE").is_file():
        raise ValueError("Dataset construction has not passed validation")
    meta = json.loads((dataset / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert meta["counts"] == EXPECTED and meta["integrity"]["passed"]
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    manifests = pd.concat([pd.read_csv(dataset / "manifests" / f"{split}.csv") for split in EXPECTED], ignore_index=True)
    manifests["domain"] = [domain_name(t, w) for t, w in zip(manifests.timeofday, manifests.weather)]
    assert manifests.groupby("split").size().to_dict() == EXPECTED
    label_images = []
    boxes = []
    quality = Counter()
    index = manifests.set_index("name").to_dict(orient="index")
    # Deliberately never open bdd100k_labels_images_test.json or holdout-val labels in EDA.
    skipped_holdout_val = 0
    for split in ("train", "val"):
        for record in stream_records(dataset / "annotations" / f"bdd100k_labels_images_{split}.json"):
            name = record["name"]
            info = index[name]
            assert info["split"] == split
            if split == "val" and info["domain"] in HOLDOUT_VAL_DOMAINS:
                skipped_holdout_val += 1
                continue
            width, height = info["width"], info["height"]
            counts = Counter()
            occluded, truncated = 0, 0
            for label in record["labels"]:
                category = label["category"]
                quality[f"{split}_raw_detection_boxes"] += 1
                try:
                    x1, y1, x2, y2 = [float(label["box2d"][key]) for key in ("x1", "y1", "x2", "y2")]
                    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                        raise ValueError("nonfinite")
                except (KeyError, ValueError, TypeError):
                    quality[f"{split}_malformed_boxes"] += 1
                    continue
                if x2 <= x1 or y2 <= y1:
                    quality[f"{split}_nonpositive_boxes"] += 1
                    continue
                outside = x1 < 0 or y1 < 0 or x2 > width or y2 > height
                quality[f"{split}_out_of_bounds_boxes"] += int(outside)
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(width, x2), min(height, y2)
                if cx2 <= cx1 or cy2 <= cy1:
                    quality[f"{split}_empty_after_clipping"] += 1
                    continue
                counts[category] += 1
                attrs = label.get("attributes") or {}
                occ, trunc = bool(attrs.get("occluded", False)), bool(attrs.get("truncated", False))
                occluded += occ
                truncated += trunc
                area = (cx2 - cx1) * (cy2 - cy1)
                if info["domain"] != "other":
                    boxes.append((split, info["domain"], category, area, (cx2 - cx1) / (cy2 - cy1), (cx1 + cx2) / (2 * width), (cy1 + cy2) / (2 * height), occ, trunc, name))
            label_images.append(dict(name=name, split=split, domain=info["domain"], boxes=sum(counts.values()), occluded_boxes=occluded, truncated_boxes=truncated, **{category: counts[category] for category in CLASSES}))
        print(f"Analysed new {split} annotations", flush=True)
    images = pd.DataFrame(label_images).merge(manifests[["name", "timeofday", "weather", "scene", "width", "height"]], on="name", validate="one_to_one")
    geometry = pd.DataFrame(boxes, columns=["split", "domain", "class", "area", "aspect_ratio", "cx", "cy", "occluded", "truncated", "name"])
    del boxes
    images.to_csv(tables / "train_val_image_summary.csv", index=False)
    geometry.to_csv(tables / "train_val_domain_boxes.csv", index=False)
    quality["train_empty_detection_images"] = int(((images.split == "train") & (images.boxes == 0)).sum())
    quality["val_empty_detection_images"] = int(((images.split == "val") & (images.boxes == 0)).sum())
    for split in ("train", "val"):
        quality[f"{split}_valid_detection_boxes"] = int(images.loc[images.split == split, "boxes"].sum())
        for suffix in ("malformed_boxes", "nonpositive_boxes", "out_of_bounds_boxes", "empty_after_clipping"):
            quality.setdefault(f"{split}_{suffix}", 0)
    (output / "annotation_quality.json").write_text(json.dumps(dict(quality), indent=2), encoding="utf-8")
    census = manifests.groupby(["split", "timeofday", "weather"]).size().rename("images").reset_index()
    census.to_csv(tables / "domain_census.csv", index=False)
    domain_census = manifests[manifests.domain != "other"].groupby(["domain", "split"]).size().unstack(fill_value=0).reindex(DOMAINS).reset_index()
    domain_census = domain_census[["domain", "train", "test", "val"]]
    holdout = domain_census.domain.isin(HOLDOUT_VAL_DOMAINS)
    domain_census["final_test"] = domain_census.test + np.where(holdout, domain_census.val, 0)
    domain_census["selection_val"] = np.where(holdout, 0, domain_census.val)
    assert {row.domain: {"train": row.train, "final_test": row.final_test, "selection_val": row.selection_val}
            for row in domain_census.itertuples()} == EXPECTED_DOMAINS
    assert skipped_holdout_val == int(domain_census.loc[holdout, "val"].sum())
    domain_census.to_csv(tables / "study_domains.csv", index=False)
    strata = pd.read_csv(dataset / "manifests" / "stratification.csv")
    strata.to_csv(tables / "stratification.csv", index=False)
    tvd = {}
    for columns in (["timeofday"], ["weather"], ["scene"], ["timeofday", "weather", "scene"]):
        counts = manifests[manifests.split != "val"].groupby(columns + ["split"]).size().unstack(fill_value=0)
        delta = counts.train / 55000 - counts.test / 14863
        tvd["/".join(columns)] = {"total_variation_distance": float(abs(delta).sum() / 2), "max_absolute_proportion_difference": float(abs(delta).max())}
    (output / "split_balance.json").write_text(json.dumps(tvd, indent=2), encoding="utf-8")
    support = []
    for (split, domain), group in images.groupby(["split", "domain"]):
        for category in CLASSES:
            support.append({"split": split, "domain": domain, "class": category, "images": len(group), "positive_images": int((group[category] > 0).sum()), "positive_image_pct": round(100 * (group[category] > 0).mean(), 3), "boxes": int(group[category].sum()), "boxes_per_image": round(group[category].mean(), 4)})
    support = pd.DataFrame(support)
    support.to_csv(tables / "class_support.csv", index=False)
    object_summary = images.groupby(["split", "domain"]).boxes.agg(["count", "mean", "median", "max"]).round(3).reset_index()
    object_summary.to_csv(tables / "objects_per_image.csv", index=False)
    geometry["small"] = geometry.area < 32 ** 2
    geometry["medium"] = (geometry.area >= 32 ** 2) & (geometry.area < 96 ** 2)
    geometry["large"] = geometry.area >= 96 ** 2
    box_summary = geometry.groupby(["split", "domain", "class"]).agg(boxes=("area", "size"), median_area=("area", "median"), median_aspect=("aspect_ratio", "median"), small_fraction=("small", "mean"), medium_fraction=("medium", "mean"), large_fraction=("large", "mean"), occluded_fraction=("occluded", "mean"), truncated_fraction=("truncated", "mean")).round(4).reset_index()
    box_summary.to_csv(tables / "box_geometry.csv", index=False)
    train_boxes = geometry[(geometry.split == "train") & (geometry.domain != "other")]
    difficulty = []
    for domain in DOMAINS:
        group = train_boxes[train_boxes.domain == domain]
        for name, subset in [("all classes", group), ("reliable 4", group[group["class"].isin(RELIABLE_CLASSES)])] + [(c, group[group["class"] == c]) for c in RELIABLE_CLASSES]:
            difficulty.append({"domain": domain, "classes": name, "boxes": len(subset), "occluded_fraction": subset.occluded.mean(),
                               "truncated_fraction": subset.truncated.mean(), "occluded_or_truncated_fraction": (subset.occluded | subset.truncated).mean(),
                               "small_fraction": subset.small.mean()})
    difficulty = pd.DataFrame(difficulty)
    source_rates = difficulty[difficulty.domain == "source"].set_index("classes")
    for column in ("occluded_fraction", "truncated_fraction"):
        difficulty[column.replace("_fraction", "_ratio_vs_source")] = difficulty[column] / difficulty.classes.map(source_rates[column])
    difficulty = difficulty.round(4)
    difficulty.to_csv(tables / "domain_occlusion_truncation.csv", index=False)
    train_images = images[(images.split == "train") & (images.domain != "other")]
    prior = []
    for domain in DOMAINS:
        group = train_images[train_images.domain == domain]
        row = {"domain": domain, "images": len(group), "boxes_per_image": group.boxes.mean()}
        row.update({f"{category} per image": group[category].mean() for category in CLASSES})
        row["traffic light image share"] = (group["traffic light"] > 0).mean()
        row["traffic light share of reliable boxes"] = group["traffic light"].sum() / group[list(RELIABLE_CLASSES)].sum().sum()
        prior.append(row)
    prior = pd.DataFrame(prior)
    for category in RELIABLE_CLASSES:
        prior[f"{category} ratio vs source"] = prior[f"{category} per image"] / prior.loc[prior.domain == "source", f"{category} per image"].iloc[0]
    prior = prior.round(4)
    prior.to_csv(tables / "domain_objects_per_image.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cell_table = census[census.split == "train"].pivot(index="timeofday", columns="weather", values="images").fillna(0).astype(int)
    image = axes[0].imshow(np.log1p(cell_table), cmap="Blues", aspect="auto")
    axes[0].set_xticks(range(len(cell_table.columns)), cell_table.columns, rotation=45, ha="right")
    axes[0].set_yticks(range(len(cell_table.index)), cell_table.index)
    for y in range(len(cell_table)):
        for x in range(len(cell_table.columns)):
            axes[0].text(x, y, f"{cell_table.iloc[y,x]:,}", ha="center", va="center", fontsize=8, color="white" if cell_table.iloc[y,x] > 1000 else "black")
    axes[0].set_title("New train: environmental census")
    domain_census.set_index("domain")[["train", "test", "val"]].plot.bar(ax=axes[1], color=["#2f6b9a", "#a6b9c8", "#65a58c"])
    axes[1].set_title("Frozen study-domain image counts")
    axes[1].set_ylabel("Images")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save_figure(fig, figures, "01_domains")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    train_support = support[(support.split == "train") & (support.domain != "other")]
    class_matrix = train_support.pivot(index="class", columns="domain", values="boxes").reindex(index=CLASSES, columns=DOMAINS)
    axes[0].imshow(np.log1p(class_matrix), cmap="Blues", aspect="auto")
    axes[0].set_xticks(range(4), DOMAINS)
    axes[0].set_yticks(range(10), CLASSES)
    for y in range(10):
        for x in range(4):
            value = class_matrix.iloc[y,x]
            axes[0].text(x, y, f"{value:,}", ha="center", va="center", fontsize=8, color="white" if value > 3000 else "black")
    axes[0].set_title("New train: boxes per class (log colour scale)")
    prevalence = train_support.pivot(index="class", columns="domain", values="positive_image_pct").reindex(index=CLASSES, columns=DOMAINS)
    prevalence.plot.barh(ax=axes[1], color=[PALETTE[d] for d in DOMAINS])
    axes[1].set_xlabel("Images containing class (%)")
    axes[1].set_title("Class prevalence in the four train domains")
    fig.tight_layout()
    save_figure(fig, figures, "02_classes")
    scene = images[(images.split == "train") & (images.domain != "other")].groupby(["domain", "scene"]).size().unstack(fill_value=0).reindex(DOMAINS)
    scene.div(scene.sum(axis=1), axis=0).to_csv(tables / "train_scene_proportions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    scene.div(scene.sum(axis=1), axis=0).plot.bar(stacked=True, ax=axes[0], colormap="Set2")
    axes[0].set_ylabel("Proportion of domain")
    axes[0].set_title("Scene composition: potential confounding")
    axes[0].legend(title="scene", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9, frameon=False)
    axes[0].tick_params(axis="x", rotation=0)
    for domain in DOMAINS:
        group = images[(images.split == "train") & (images.domain == domain)]
        axes[1].hist(group.boxes, bins=np.arange(0, 102, 2), histtype="step", density=True, color=PALETTE[domain], label=domain)
    axes[1].set_xlabel("Valid detection boxes per image (0–100 shown)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Annotation workload and object density")
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, figures, "03_scene_density")
    divergences = []
    for domain in list(DOMAINS)[1:]:
        divergences.append({"target": domain, "scene_js_divergence_bits": js_divergence(scene.loc["source"], scene.loc[domain]), "class_box_mix_js_divergence_bits": js_divergence(class_matrix.source, class_matrix[domain])})
    pd.DataFrame(divergences).round(6).to_csv(tables / "composition_divergence.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for domain in DOMAINS:
        group = geometry[(geometry.split == "train") & (geometry.domain == domain)]
        axes[0].hist(np.sqrt(group.area), bins=np.geomspace(0.5, 1100, 50), histtype="step", density=True, color=PALETTE[domain], label=domain)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Square root of clipped box area (pixels)")
    axes[0].set_ylabel("Density")
    axes[0].axvline(32, color="gray", linestyle=":")
    axes[0].axvline(96, color="gray", linestyle=":")
    axes[0].legend()
    axes[0].set_title("Native-resolution object scale")
    size_mix = geometry[geometry.split == "train"].groupby("domain")[["small", "medium", "large"]].mean().reindex(DOMAINS)
    size_mix.plot.bar(stacked=True, ax=axes[1], color=["#c75d5d", "#deb969", "#5c9a8a"])
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].set_ylabel("Fraction of valid boxes")
    axes[1].set_title("Area bands: <32², 32²–96², ≥96²")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.tight_layout()
    save_figure(fig, figures, "04_box_sizes")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True)
    for domain, ax in zip(DOMAINS, axes):
        group = geometry[(geometry.split == "train") & (geometry.domain == domain)]
        ax.hist2d(group.cx, group.cy, bins=40, range=[[0,1],[0,1]], norm=matplotlib.colors.LogNorm(), cmap="Blues")
        ax.invert_yaxis()
        ax.set_title(domain)
        ax.set_xlabel("Normalized x center")
    axes[0].set_ylabel("Normalized y center")
    fig.tight_layout()
    save_figure(fig, figures, "05_box_centers")
    brightness = []
    example_names = {}
    rng = random.Random(20261001)
    for domain in DOMAINS:
        group = images[(images.split == "train") & (images.domain == domain)]
        sampled = rng.sample(sorted(group.name), min(brightness_n, len(group)))
        example_names[domain] = sampled[:3]
        for name in sampled:
            with Image.open(dataset / "images" / "train" / name) as image:
                gray = np.asarray(image.convert("L").resize((320,180)), dtype=float)
            brightness.append({"name": name, "domain": domain, "mean_gray_0_255": float(gray.mean()), "gray_std": float(gray.std()), "dark_pixel_fraction_lt32": float((gray < 32).mean())})
        print(f"Decoded {len(sampled)} new train images for {domain}", flush=True)
    brightness = pd.DataFrame(brightness)
    brightness.to_csv(tables / "brightness_samples.csv", index=False)
    brightness_summary = brightness.groupby("domain").agg(sample_images=("name", "size"), median_gray=("mean_gray_0_255", "median"), median_contrast=("gray_std", "median"), median_dark_fraction=("dark_pixel_fraction_lt32", "median")).reindex(DOMAINS).round(3)
    brightness_summary.to_csv(tables / "brightness_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, column, title in zip(axes, ["mean_gray_0_255", "dark_pixel_fraction_lt32"], ["Mean grayscale: reproducible train sample", "Dark-pixel fraction (<32/255)"]):
        data = [brightness.loc[brightness.domain == domain, column].to_numpy() for domain in DOMAINS]
        violins = ax.violinplot(data, showmedians=True, showextrema=False)
        for body, domain in zip(violins["bodies"], DOMAINS):
            body.set_facecolor(PALETTE[domain])
        ax.set_xticks(range(1,5), DOMAINS)
        ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, figures, "06_brightness")
    fig, axes = plt.subplots(4, 3, figsize=(13, 10))
    for y, domain in enumerate(DOMAINS):
        for x, name in enumerate(example_names[domain]):
            with Image.open(dataset / "images" / "train" / name) as image:
                axes[y,x].imshow(image)
            axes[y,x].set_title(f"{domain} · {index[name]['scene']}\n{name}", fontsize=8)
            axes[y,x].axis("off")
    fig.tight_layout()
    save_figure(fig, figures, "07_train_examples")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    reliable_prior = prior.set_index("domain")[[f"{c} per image" for c in RELIABLE_CLASSES]].reindex(DOMAINS)
    reliable_prior.columns = list(RELIABLE_CLASSES)
    reliable_prior.T.plot.bar(ax=axes[0], color=[PALETTE[d] for d in DOMAINS])
    axes[0].set_ylabel("Boxes per train image")
    axes[0].set_title("Class prior shift: reliable-class boxes per image")
    axes[0].tick_params(axis="x", rotation=0)
    occ = difficulty[difficulty.classes.isin(RELIABLE_CLASSES)].pivot(index="classes", columns="domain", values="occluded_fraction").reindex(index=RELIABLE_CLASSES, columns=DOMAINS)
    trunc = difficulty[difficulty.classes.isin(RELIABLE_CLASSES)].pivot(index="classes", columns="domain", values="truncated_fraction").reindex(index=RELIABLE_CLASSES, columns=DOMAINS)
    x = np.arange(len(RELIABLE_CLASSES))
    for offset, domain in enumerate(DOMAINS):
        axes[1].bar(x + (offset - 1.5) * 0.2, occ[domain], width=0.2, color=PALETTE[domain], label=f"{domain} occluded")
        axes[1].bar(x + (offset - 1.5) * 0.2, trunc[domain], width=0.2, bottom=0, color="none", edgecolor="black", hatch="//", lw=0.6)
    axes[1].set_xticks(x, RELIABLE_CLASSES)
    axes[1].set_ylabel("Fraction of train boxes")
    axes[1].set_title("Occluded (filled) and truncated (hatched) annotations")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, figures, "09_difficulty_class_prior")
    budget_rows, absence_rows, memberships = [], [], {}
    for domain in list(DOMAINS)[1:]:
        pool = images[(images.split == "train") & (images.domain == domain)].set_index("name").sort_index()
        ladder = [10,25,50,100,250,500,1000] + ([2000,5000,10000] if domain == "lighting" else []) + [len(pool)]
        ladder = sorted(set(n for n in ladder if n <= len(pool)))
        for seed in seeds:
            ordering = list(pool.index)
            random.Random(seed).shuffle(ordering)
            previous = set()
            for n in ladder:
                chosen = ordering[:n]
                assert previous <= set(chosen)
                assert all(index[name]["split"] == "train" for name in chosen)
                previous = set(chosen)
                selected = pool.loc[chosen]
                row = {"domain": domain, "seed": seed, "budget": n, "pool_size": len(pool), "pool_fraction": round(n/len(pool),6), "annotation_boxes": int(selected.boxes.sum()), "missing_classes": "; ".join(category for category in CLASSES if selected[category].sum() == 0)}
                row.update({category: int(selected[category].sum()) for category in CLASSES})
                budget_rows.append(row)
                memberships[f"{domain}/seed_{seed}/N_{n}"] = chosen
        for n in ladder:
            for category in CLASSES:
                positive = int((pool[category] > 0).sum())
                absence_rows.append({"domain": domain, "budget": n, "class": category, "pool_images": len(pool), "positive_pool_images": positive, "probability_class_absent": probability_absent(len(pool),positive,n)})
    budgets = pd.DataFrame(budget_rows)
    absence = pd.DataFrame(absence_rows)
    budgets.to_csv(tables / "nested_budget_coverage.csv", index=False)
    absence.to_csv(tables / "class_absence_probability.csv", index=False)
    (output / "nested_budget_membership.json").write_text(json.dumps(memberships), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for domain, ax in zip(list(DOMAINS)[1:], axes):
        table = absence[absence.domain == domain].pivot(index="class", columns="budget", values="probability_class_absent").reindex(CLASSES)
        im = ax.imshow(table, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(table.columns)), table.columns, rotation=60, ha="right")
        ax.set_yticks(range(10), CLASSES)
        ax.set_title(domain)
        ax.set_xlabel("Labeled train-image budget")
    fig.colorbar(im, ax=axes, label="Exact probability of no image with class", shrink=0.8)
    save_figure(fig, figures, "08_budget_class_coverage")
    summary = {"dataset": str(dataset.resolve()), "dataset_manifest_sha256": meta["manifest_sha256"], "source_url": meta["source_dataset"], "split_seed": meta["seed"], "split_counts": EXPECTED, "class_names": CLASSES, "eda_label_splits": ["train", "val (source/lighting only)"], "heldout_labels_opened_by_eda": False, "holdout_val_records_skipped": skipped_holdout_val, "reliable_classes": RELIABLE_CLASSES, "budget_seeds": seeds, "budget_memberships_nested": True, "brightness_seed": 20261001, "brightness_samples_per_domain": brightness_n, "all_image_dimensions": meta["dimensions"], "split_balance": tvd, "annotation_quality": dict(quality), "study_domains": domain_census.to_dict(orient="records"), "brightness": brightness_summary.reset_index().to_dict(orient="records"), "small_box_fraction_train": size_mix.small.to_dict(), "artifacts": {"figures": sorted(p.name for p in figures.glob('*.png')), "tables": sorted(p.name for p in tables.glob('*.csv'))}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_report(output, meta, summary, domain_census, support, object_summary, brightness_summary, divergences, budgets, difficulty, prior)
    print(f"EDA complete: {output}", flush=True)


def make_report(output, meta, summary, domain_census, support, object_summary, brightness_summary, divergences, budgets, difficulty, prior):
    study_support = support[(support.split == "train") & (support.domain != "other")]
    per_class = study_support.pivot(index="class", columns="domain", values="boxes").reindex(index=CLASSES, columns=DOMAINS).reset_index()
    all_train = support[(support.split == "train")]
    total_boxes = int(all_train.boxes.sum())
    car_boxes = int(all_train[all_train['class'] == 'car'].boxes.sum())
    rare = all_train.groupby("class").boxes.sum().sort_values().head(3)
    val_rare = support[(support.split == "val") & (support.domain != "other") & (support.boxes < 100)][["domain", "class", "positive_images", "boxes"]]
    density = object_summary[(object_summary.split == "train") & (object_summary.domain != "other")][["domain", "count", "mean", "median", "max"]]
    max_pp = 100 * summary["split_balance"]["timeofday/weather/scene"]["max_absolute_proportion_difference"]
    lines = ["# EDA BDD100K — dataset mới", "", "Phân tích theo proposal `docs/proposal_dataset_aligned.docx`. Nguồn: " + meta["source_dataset"] + ".", "", "## 1. Bộ dữ liệu và cách chia", "", f"Tạo **55.000 train / 14.863 test / 10.000 validation** với seed **{meta['seed']}** từ 69.863 ảnh train gốc có nhãn. Loại 137 ảnh không có annotation; không đưa 20.000 ảnh test công khai không nhãn vào nghiên cứu. Giữ nguyên danh sách 10.000 ảnh val chính thức.", "", "Nguồn ảnh và hai file nhãn được kiểm tra từng file với kích thước và CRC32 trong archive Kaggle phiên bản 2; SHA256 được tính tại máy. Ảnh của bộ mới là bản sao độc lập. CRC32 kiểm tra tính nhất quán nội dung, không phải chữ ký mật mã.", "", "Phân bổ Hamilton theo `(timeofday, weather)`, sau đó Hamilton theo `scene` trong từng ô. Sắp xếp ô, cảnh, basename và shuffle bằng `random.Random(20260916)`. Phương pháp đảm bảo tổng đúng 55.000 và giới hạn sai số do làm tròn ở các nhóm ít ảnh; metadata quyết định split, không dùng số bbox hay lớp để chọn ảnh.", "", f"Kiểm tra đạt: số ảnh = số record nhãn = số dòng manifest ở cả ba split; không có basename hoặc SHA256 ảnh trùng giữa split; toàn bộ ảnh qua kiểm tra định dạng/header. Kích thước: `{meta['dimensions']}`. Sai lệch tỷ lệ lớn nhất giữa train/test ở một stratum chung là **{max_pp:.4f} điểm phần trăm**; TVD chung **{summary['split_balance']['timeofday/weather/scene']['total_variation_distance']:.6f}**. Chi tiết ở `split_balance.json` và `tables/stratification.csv`.", "", "Manifest và SHA256 được lưu trong dataset. Seed đơn lẻ không đủ khẳng định trùng membership với một bộ chia lịch sử dùng thuật toán khác. Dataset mới này và các manifest của nó là phiên bản thực nghiệm được cố định.", "", "Nhãn mới chỉ giữ detection 10 lớp, chuẩn hóa person→pedestrian, bike→bicycle, motor→motorcycle. Tọa độ và thuộc tính bbox gốc được giữ nguyên; loại polygon lane/drivable area. EDA chỉ đọc annotation train và val của source/lighting; test và val của weather/compound (đã thành held-out theo sửa đổi protocol 2026-10-19) chỉ xuất hiện trong census metadata và kiểm tra cấu trúc.", "", "## 2. Quy mô các domain nghiên cứu", "", md_table(domain_census), "", "**Sửa đổi protocol 2026-10-19.** Test 600 ảnh/domain không đạt với weather (537) và compound (470), nên val của hai domain này được gộp vào test cuối: `final_test` = 933 (weather) và 756 (compound); pool train không đổi. Hai domain này không còn validation để chọn checkpoint: mọi run target dùng checkpoint ở update cuối của schedule cố định; source/lighting val chỉ dùng cho pilot và chọn checkpoint source. Thư mục vật lý vẫn là 55.000/14.863/10.000.", "", "![Domain](figures/01_domains.png)", "", "Source là daytime-clear; lighting là night-clear; weather là daytime-rainy; compound là night-rainy. Điều kiện `undefined` vẫn có trong census toàn bộ dữ liệu nhưng không thuộc các domain chính.", "", "Night-clear có pool lớn hơn hẳn hai domain rainy. Khi triển khai so sánh recovery giữa các shift, dùng thêm ceiling tại pool chung bằng min(pool các target); không suy ra mức khó hay ngân sách recovery chỉ từ số lượng ảnh.", "", "## 3. Mất cân bằng lớp và mức hỗ trợ", "", f"Train có **{total_boxes:,} bbox hợp lệ** sau xử lý hình học chỉ trong EDA. Car chiếm **{100*car_boxes/total_boxes:.2f}%** bbox. Ba lớp có ít bbox nhất: " + "; ".join(f"{name}: {count:,}" for name, count in rare.items()) + ".", "", md_table(per_class), "", "![Class](figures/02_classes.png)", "", "Phần trăm ảnh chứa lớp và số bbox đo hai khía cạnh khác nhau: nhiều bbox có thể tập trung trong một vài ảnh. Dùng `tables/class_support.csv` để xem cả hai. Số bbox trên ảnh cũng là proxy đơn giản cho khối lượng gán nhãn.", "", md_table(density), "", "Các lớp có dưới 100 bbox trong validation được phép đọc (source/lighting):", "", md_table(val_rare), "", "Reliable class được **cố định chung cho mọi domain**: " + ", ".join(RELIABLE_CLASSES) + ". Đây là bốn lớp có nhiều bbox nhất ở cả bốn domain train; dùng chung một tập để mAP các shift so sánh được (RQ4). Mean 10 lớp được giữ làm sensitivity analysis.", "", "## 4. Cảnh đường, đối tượng và yếu tố gây nhiễu", "", "![Scene](figures/03_scene_density.png)", "", md_table(pd.DataFrame(divergences).round(6)), "", "Jensen–Shannon divergence ở đây dùng log cơ số 2, trong [0,1]; đo khác biệt thành phần scene hoặc tỷ lệ bbox theo lớp so với source. Các giá trị mô tả dữ liệu và không đo độ giảm mAP. Lighting/weather được định nghĩa theo metadata, nhưng thành phần scene và lớp có thể đồng thời thay đổi; factorial metadata không tự loại bỏ các yếu tố gây nhiễu này.", "", "## 5. Hình học bbox và chất lượng annotation", "", "![Sizes](figures/04_box_sizes.png)", "", "![Centers](figures/05_box_centers.png)", "", "Các ngưỡng diện tích 32² và 96² pixel được tính ở độ phân giải ảnh gốc, dùng để mô tả kích thước. Không tự động đồng nhất bảng EDA này với AP_small của một evaluator có resize/cấu hình khác.", "", "Tỷ lệ bbox nhỏ trên train theo domain: " + "; ".join(f"{d}: {100*v:.2f}%" for d,v in summary["small_box_fraction_train"].items()) + ".", "", "Kiểm tra hình học:", "", "```json", json.dumps(summary["annotation_quality"], indent=2), "```", "", "EDA bỏ bbox malformed/nonfinite/nonpositive; clip tọa độ vượt biên chỉ để tính diện tích, tâm và số lượng bbox hợp lệ. Dataset giữ nhãn gốc; chưa sửa annotation. Trước training cần áp dụng cùng chính sách sanitize cho mọi domain và ghi lại số bbox bị tác động.", "", "## 6. Độ sáng và ví dụ ảnh train", "", f"Lấy mẫu cố định tối đa {summary['brightness_samples_per_domain']} ảnh mỗi domain từ train mới với seed {summary['brightness_seed']}. Grayscale theo Pillow, resize 320×180, tính trung bình, độ lệch chuẩn và tỷ lệ pixel <32/255. Đây là proxy ảnh đơn giản, không phải đánh giá detector hoặc độ khó nhiệm vụ.", "", md_table(brightness_summary.reset_index()), "", "![Brightness](figures/06_brightness.png)", "", "![Examples](figures/07_train_examples.png)", "", "## 7. Proxy độ khó và dịch chuyển class prior", "", "Độ sáng trung bình chỉ mô tả ảnh. Thuộc tính `occluded`/`truncated` của từng bbox là proxy độ khó gần nhiệm vụ detection hơn. Bảng dưới tính trên bbox hợp lệ của train mới; `ratio_vs_source` = tỷ lệ của domain / tỷ lệ của source cùng nhóm lớp.", "", md_table(difficulty[["domain", "classes", "boxes", "occluded_fraction", "truncated_fraction", "occluded_ratio_vs_source", "truncated_ratio_vs_source"]]), "", "Số đối tượng trên mỗi ảnh train theo domain (cùng cách đếm bbox hợp lệ):", "", md_table(prior[["domain", "images", "boxes_per_image"] + [f"{c} per image" for c in RELIABLE_CLASSES] + ["traffic light image share", "traffic light ratio vs source"]]), "", "![Difficulty](figures/09_difficulty_class_prior.png)", "", "Target không chỉ dịch chuyển ảnh mà còn dịch chuyển **phân phối lớp**: traffic light/ảnh ở night và rainy cao hơn source. Một phần recovery khi fine-tune có thể chỉ là học lại class prior (tần suất, vị trí, kích thước lớp) chứ không phải học đặc trưng ban đêm/mưa. Vì vậy recovery được vẽ thêm **theo từng lớp** (`scripts/recovery_curves.py`): lớp có prior gần như không đổi mà vẫn recovery chậm cho thấy cần đặc trưng mới; lớp recovery nhanh và prior đổi mạnh gợi ý hiệu ứng prior.", "", "## 8. Độ phủ lớp ở ngân sách gán nhãn", "", f"Dùng ladder trong proposal, samples lồng nhau với seeds {summary['budget_seeds']}; cùng seed, tập N nhỏ nằm trong tập N lớn. Chỉ lấy ảnh train của target; không dùng validation/test. Các seed này được ghi rõ để tái lập phân tích lấy mẫu, chưa phải kết quả training.", "", "![Budget](figures/08_budget_class_coverage.png)", "", "Xác suất lớp vắng mặt được tính chính xác theo lấy mẫu không hoàn lại: C(M-K,N)/C(M,N), với M là pool train và K là số ảnh train chứa lớp. Khác với việc đếm số bbox, công thức này phản ánh việc chọn ảnh. Một số lớp có thể không tồn tại trong một domain ngay cả khi dùng full pool.", "", "Ví dụ độ phủ và khối lượng bbox ở N=10,25,100 cho seed đầu:", "", md_table(budgets[(budgets.seed == summary['budget_seeds'][0]) & budgets.budget.isin([10,25,100])][["domain","budget","annotation_boxes","missing_classes"]]), "", "`nested_budget_membership.json` lưu tên ảnh; `tables/nested_budget_coverage.csv` lưu số bbox/lớp theo seed; `tables/class_absence_probability.csv` lưu xác suất từng rung. Số ảnh cố định không đồng nghĩa số đối tượng gán nhãn cố định.", "", "## 9. Điều đã xác nhận và giới hạn", "", "Dataset mới đạt các kiểm tra split và có đủ pool train cho ladder theo protocol. EDA xác định quy mô domain, mức hỗ trợ lớp, annotation workload, hình học và đặc trưng ảnh; chưa có mAP, performance drop, N_BE, N75 hoặc N90 vì chưa huấn luyện/đánh giá detector.", "", "Kiểm tra SHA256 loại trùng nội dung chính xác giữa các split. Chưa kiểm tra near-duplicate, nhóm địa lý hoặc tính phụ thuộc giữa video; basename không cung cấp group ID đã xác nhận. Không khẳng định độc lập thống kê tuyệt đối chỉ từ kiểm tra trùng file. Validation chính thức được giữ nguyên nên không được ép khớp phân bố train; các bảng train/val cho thấy support thực tế.", "", "Đầu vào của mọi biểu đồ và bảng lớp/hình học là train/val của dataset mới. Không dùng subset hoặc kết quả mô hình từ dự án khác.", "", "## 10. Tái lập", "", "```powershell", "python scripts/prepare_revised_bdd100k.py --output-dir data/bdd100k_revised --validate-only", "python scripts/eda_revised_bdd100k.py --dataset-root data/bdd100k_revised --output-dir runs/eda_revised --brightness-per-domain 200 --seeds 0 1 2", "```", "", "Notebook `notebooks/eda_revised_bdd100k.ipynb` đọc các artifact trên và đã được thực thi. Dataset và output chạy được giữ ngoài Git; scripts, notebook và tài liệu hướng dẫn có thể theo dõi trong Git."]
    lines.insert(lines.index("## 2. Quy mô các domain nghiên cứu"), "Ảnh trùng nội dung được giữ trong cùng split bằng hoán đổi singleton trong đúng stratum (quota không đổi). Số ảnh dư do trùng nội dung bên trong mỗi split: " + json.dumps(meta["within_split_repeated_content_images"]) + ". Validation chính thức không bị thay membership. Chi tiết hoán đổi ở `manifests/duplicate_content_repairs.json`.")
    lines.insert(lines.index("## 2. Quy mô các domain nghiên cứu"), "")
    report_text = "\n".join(lines) + "\n"
    (output / "report_vi.md").write_text(report_text, encoding="utf-8")
    # Self-contained report: embeds figures and every substantive table.
    parts = ["<!doctype html><html lang='vi'><meta charset='utf-8'><title>EDA BDD100K mới</title><style>body{font:16px/1.65 system-ui;max-width:1180px;margin:40px auto;padding:0 24px;color:#263545}h1,h2{line-height:1.3;color:#18374b}img{width:100%;height:auto}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:7px 12px;border-bottom:1px solid #dde3e8;text-align:left}th{background:#edf3f7}pre{padding:18px;background:#f1f5f8;overflow:auto}a{color:#22658c}</style><body>"]
    body = markdown.markdown(report_text, extensions=["tables", "fenced_code"])
    for path in sorted((output / "figures").glob("*.png")):
        body = body.replace(f'src="figures/{path.name}"', f'src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"')
    parts.append(body)
    parts.append("</body></html>")
    (output / "report_vi.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--brightness-per-domain", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    args = parser.parse_args()
    if args.brightness_per_domain < 3 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Need at least 3 brightness images per domain and unique seeds")
    analyse(args.dataset_root, args.output_dir, args.brightness_per_domain, args.seeds)
