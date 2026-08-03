#!/usr/bin/env python3
"""Generate the journal-style per-study LOSO result figure.

The figure uses the donor-primary macro-F1 values stored in the frozen
accuracy_085_push reports. Sample-level fallback values are used only where
the report explicitly marks a fold as a single-donor or mixed-donor contrast.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_REPORT = PROJECT_ROOT / "results/accuracy_085_push/expanded_ensemble_v1.json"
TRANSFER_REPORT = PROJECT_ROOT / "results/accuracy_085_push/transferable_excl_GSE173900.json"
OUTPUT_DIR = PROJECT_ROOT / "results/publication/figures"

STUDY_ORDER = [
    "E-MTAB-4945",
    "GSE158395",
    "GSE181297",
    "GSE181316",
    "GSE190626",
    "GSE92566",
    "Sun_Burns",
    "GSE173900",
]

DISPLAY_NAMES = {
    "E-MTAB-4945": "E-MTAB-4945",
    "GSE158395": "GSE158395",
    "GSE181297": "GSE181297",
    "GSE181316": "GSE181316",
    "GSE190626": "GSE190626",
    "GSE92566": "GSE92566",
    "Sun_Burns": "Sun & Burns",
    "GSE173900": "GSE173900",
}

TRANSFER_COLOR = "#4C78A8"
TRANSFER_EDGE = "#355C7D"
EXCLUDED_COLOR = "#D9DEE3"
EXCLUDED_EDGE = "#7A8792"
MEAN_COLOR = "#D55E00"
GRID_COLOR = "#D9DDE1"
TEXT_COLOR = "#17232D"
MUTED_COLOR = "#5F6B73"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def evaluation_unit_label(fold: dict) -> str:
    if fold["grain"] == "donor":
        count = int(fold["n_donors"])
        unit = "donor" if count == 1 else "donors"
    else:
        count = int(fold["n_test"])
        unit = "profile" if count == 1 else "profiles"
    return f"n = {count} {unit}"


def collect_rows(full_report: dict, transfer_report: dict) -> tuple[list[dict], dict]:
    full_folds = {fold["accession"]: fold for fold in full_report["folds"]}
    transfer_ids = {fold["accession"] for fold in transfer_report["folds"]}

    missing = set(STUDY_ORDER) - set(full_folds)
    if missing:
        raise ValueError(f"Missing LOSO folds: {sorted(missing)}")
    if len(transfer_ids) != int(transfer_report["n_folds"]):
        raise ValueError("Transferable fold count does not match the report")

    rows = []
    for accession in STUDY_ORDER:
        fold = full_folds[accession]
        rows.append(
            {
                "accession": accession,
                "display_name": DISPLAY_NAMES[accession],
                "macro_f1": float(fold["macro_f1"]),
                "evaluation_units": evaluation_unit_label(fold),
                "grain": fold["grain"],
                "n_donors": int(fold["n_donors"]),
                "n_test": int(fold["n_test"]),
                "transferable_subset": accession in transfer_ids,
            }
        )

    expected_mean = sum(row["macro_f1"] for row in rows if row["transferable_subset"]) / len(transfer_ids)
    reported_mean = float(transfer_report["mean_macro_f1"])
    if abs(expected_mean - reported_mean) > 1e-12:
        raise ValueError(f"Computed mean {expected_mean} does not match report {reported_mean}")
    return rows, transfer_report


def write_plot_data(rows: list[dict], report: dict, path: Path) -> None:
    fields = [
        "accession",
        "display_name",
        "macro_f1",
        "evaluation_units",
        "grain",
        "n_donors",
        "n_test",
        "transferable_subset",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "accession": "MEAN_TRANSFERABLE",
                "display_name": "Mean LOSO",
                "macro_f1": report["mean_macro_f1"],
                "evaluation_units": f"{report['n_folds']} studies",
                "grain": "unweighted fold mean",
                "n_donors": "",
                "n_test": "",
                "transferable_subset": True,
            }
        )


def draw_figure(rows: list[dict], report: dict, output_dir: Path, web_export_dir: Path | None) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 17,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    study_x = list(range(7)) + [7.6]
    mean_x = 9.1
    mean = float(report["mean_macro_f1"])
    ci_lo = float(report["bootstrap_ci"]["lo"])
    ci_hi = float(report["bootstrap_ci"]["hi"])

    fig, ax = plt.subplots(figsize=(12.2, 7.2), facecolor="white")
    ax.set_facecolor("white")

    for x, row in zip(study_x, rows):
        excluded = not row["transferable_subset"]
        bar = ax.bar(
            x,
            row["macro_f1"] * 100,
            width=0.68,
            color=EXCLUDED_COLOR if excluded else TRANSFER_COLOR,
            edgecolor=EXCLUDED_EDGE if excluded else TRANSFER_EDGE,
            linewidth=1.2,
            hatch="///" if excluded else None,
            zorder=3,
        )[0]
        value = row["macro_f1"] * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=TEXT_COLOR,
            zorder=5,
        )

    mean_bar = ax.bar(
        mean_x,
        mean * 100,
        width=0.78,
        color=MEAN_COLOR,
        edgecolor="#9E4300",
        linewidth=1.3,
        zorder=4,
    )[0]
    ax.errorbar(
        mean_x,
        mean * 100,
        yerr=[[mean * 100 - ci_lo * 100], [ci_hi * 100 - mean * 100]],
        fmt="none",
        ecolor="#562800",
        elinewidth=1.7,
        capsize=6,
        capthick=1.7,
        zorder=6,
    )
    ax.text(
        mean_bar.get_x() + mean_bar.get_width() / 2,
        mean * 100 - 5.5,
        f"{mean * 100:.1f}",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="white",
        zorder=7,
    )
    ax.text(
        mean_x,
        ci_hi * 100 + 2.4,
        f"95% CI {ci_lo * 100:.1f}–{ci_hi * 100:.1f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=MUTED_COLOR,
    )

    ax.axhline(
        float(report["metric_definition"]["worst_gate"]) * 100,
        color="#69757E",
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    ax.axvline(8.35, color="#BCC3C8", linewidth=1.0, zorder=1)

    tick_positions = study_x + [mean_x]
    tick_labels = [
        f"{row['display_name']}\n{row['evaluation_units']}" for row in rows
    ] + [f"Mean LOSO\n{report['n_folds']} studies"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    for tick, row in zip(ax.get_xticklabels(), rows + [{"transferable_subset": True}]):
        tick.set_color(MUTED_COLOR if row["transferable_subset"] else EXCLUDED_EDGE)
        if not row["transferable_subset"]:
            tick.set_fontweight("bold")
    ax.get_xticklabels()[-1].set_color(MEAN_COLOR)
    ax.get_xticklabels()[-1].set_fontweight("bold")

    ax.set_ylabel("Macro F1 (%)", color=TEXT_COLOR, fontweight="bold")
    ax.set_ylim(0, 113)
    ax.set_xlim(-0.65, 9.75)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", colors=MUTED_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.xaxis.grid(False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#AEB6BC")

    fig.suptitle(
        "Leave-one-study-out performance across public tissue cohorts",
        x=0.075,
        y=0.955,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    fig.text(
        0.075,
        0.900,
        "Primary endpoint: keloid vs unaffected skin · donor-primary macro F1 with documented sample-level fallback",
        ha="left",
        fontsize=10.5,
        color=MUTED_COLOR,
    )

    legend_handles = [
        Patch(facecolor=TRANSFER_COLOR, edgecolor=TRANSFER_EDGE, label="Transferable study"),
        Patch(facecolor=EXCLUDED_COLOR, edgecolor=EXCLUDED_EDGE, hatch="///", label="Non-transferable platform cohort"),
        Patch(facecolor=MEAN_COLOR, edgecolor="#9E4300", label="Mean, transferable subset"),
        Line2D([0], [0], color="#69757E", linewidth=1.1, linestyle=(0, (4, 3)), label="Minimum-fold criterion (75%)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=1.8,
        columnspacing=1.7,
    )

    fig.text(
        0.075,
        0.035,
        "LOSO, leave-one-study-out. GSE173900 is displayed for completeness but excluded from the 7-study mean because of documented non-transferable platform behavior.",
        ha="left",
        fontsize=8.6,
        color=MUTED_COLOR,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.80)

    stem = "fig_loso_per_study_journal"
    outputs = []
    for suffix, kwargs in [
        ("png", {"dpi": 300}),
        ("pdf", {}),
        ("svg", {}),
    ]:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, facecolor="white", bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)

    data_path = output_dir / f"{stem}.csv"
    write_plot_data(rows, report, data_path)
    outputs.append(data_path)

    if web_export_dir is not None:
        web_export_dir.mkdir(parents=True, exist_ok=True)
        for path in outputs[:3]:
            shutil.copy2(path, web_export_dir / f"keloidbench-loso-journal.{path.suffix.lstrip('.')}")

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-report", type=Path, default=FULL_REPORT)
    parser.add_argument("--transfer-report", type=Path, default=TRANSFER_REPORT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--web-export-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_report = load_json(args.full_report)
    transfer_report = load_json(args.transfer_report)
    rows, transfer_report = collect_rows(full_report, transfer_report)
    outputs = draw_figure(rows, transfer_report, args.output_dir, args.web_export_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
