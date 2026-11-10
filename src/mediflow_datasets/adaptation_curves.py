"""Plot completed camera-adaptation histories without loading models or datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DOMAINS = {"skin": "Microscope skin", "web_skin": "Webcam face", "hair": "Microscope scalp"}
MODELS = {"basic": ("Basic", "#1766A6"), "selected": ("Selected", "#AD421E")}
CONDITIONS = {
    "clean": ("Original", "-", "o"),
    "lighting": ("Lighting", "--", "s"),
    "expanded": ("Expanded", ":", "^"),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def domain_curves(results, domain, stage):
    folder = results / domain
    trials = [(folder / role, role) for role in MODELS] if stage == 1 else [(folder, None)]
    treatment = {1: "lighting", 2: "expanded", 3: "aligned"}[stage]
    curves = []
    for trial, role in trials:
        complete = read_json(trial / "complete.json")
        history = read_json(trial / treatment / "history.json")
        epochs = [row["epoch"] for row in history]
        best = complete["validation"]["epoch"] if stage == 1 else complete["best_epoch"]
        if epochs != list(range(1, len(epochs) + 1)) or best not in epochs:
            raise ValueError(f"Incomplete epoch history: {trial}")
        curves.append((role or complete["model_config"]["role"], epochs, history, best))
    return curves


def save_curves(results, stage, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    if stage not in (1, 2, 3):
        raise ValueError("Expected Stage 1, 2, or 3")
    results, output = Path(results), Path(output)
    data = {domain: domain_curves(results, domain, stage) for domain in DOMAINS}
    conditions = tuple(CONDITIONS)[: 2 if stage == 1 else 3]
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), layout="constrained")
    experiment = {
        1: "Lighting adaptation",
        2: "Expanded camera adaptation",
        3: "Feature alignment",
    }[stage]
    fig.suptitle(f"MediFlow  |  Stage {stage}  ·  {experiment}", x=0.02, ha="left", fontsize=18)
    try:
        for row, (domain, label) in enumerate(DOMAINS.items()):
            loss_axis, f1_axis, accuracy_axis = axes[row]
            loss_axis.set_ylabel(label, fontsize=11, fontweight="bold")
            for role, epochs, history, best in data[domain]:
                model_name, color = MODELS[role]
                loss_axis.plot(
                    epochs,
                    [record["train_loss_mean_per_update"] for record in history],
                    color=color,
                    marker="o",
                    label=model_name,
                )
                for axis, metric in ((f1_axis, "macro_f1"), (accuracy_axis, "accuracy")):
                    for name in conditions:
                        condition, style, marker = CONDITIONS[name]
                        points = [record[name][metric] for record in history]
                        axis.plot(
                            epochs,
                            points,
                            color=color,
                            linestyle=style,
                            marker=marker,
                            markersize=4,
                            label=f"{model_name} · {condition}" if stage == 1 else condition,
                        )
                        axis.scatter(
                            best,
                            points[epochs.index(best)],
                            facecolor="none",
                            edgecolor=color,
                            s=88,
                            linewidth=1.2,
                            zorder=5,
                        )
            for axis, title in zip(
                axes[row],
                ("Training loss", "Validation Macro-F1", "Validation accuracy"),
                strict=True,
            ):
                if row == 0:
                    axis.set_title(title, fontsize=12, fontweight="bold")
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                axis.grid(axis="y", alpha=0.22)
                axis.spines[["top", "right"]].set_visible(False)
                axis.tick_params(labelsize=9)
                if row == len(DOMAINS) - 1:
                    axis.set_xlabel("Epoch")
            if stage != 1 or row == 0:
                loss_axis.legend(loc="best", fontsize=8, frameon=False)
                f1_axis.legend(loc="best", fontsize=7.5, framealpha=0.9)
        fig.supxlabel(
            "Rings mark the saved epoch. Validation axes are zoomed per panel. "
            "Training loss definitions can vary by model.",
            fontsize=9,
            color="#475569",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, facecolor="white")
    finally:
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(save_curves(args.results, args.stage, args.output))


if __name__ == "__main__":
    main()
