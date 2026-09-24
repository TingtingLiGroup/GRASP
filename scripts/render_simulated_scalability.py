#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATASET_ORDER = ["simulated1", "simulated2", "simulated3", "merfish_u2os"]
DATASET_LABELS = {
    "simulated1": "Simulated 1\n10 cells, 80 genes\n800 graphs",
    "simulated2": "Simulated 2\n100 cells, 25 genes\n2,500 graphs",
    "simulated3": "Simulated 3\n50 cells, 400 genes\n15,000 graphs",
    "merfish_u2os": "MERFISH U2OS\n989 cells, 135 genes\n113,909 graphs",
}
RESOLUTION_LABELS = {
    "n20_m10": "GRASP (20×10)",
    "n30_m15": "GRASP (30×15)",
}
RESOLUTION_COLORS = {
    "n20_m10": "#4C78A8",
    "n30_m15": "#E39C37",
}
PROTOCOL_LABELS = {
    "full_pipeline": "Full pipeline",
    "train_only": "Train only",
}
PROTOCOL_LINESTYLES = {
    "full_pipeline": "-",
    "train_only": "--",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the four-panel simulated GRASP scalability figure."
    )
    parser.add_argument("suite_dir", type=Path)
    return parser


def ordered_values(
    frame: pd.DataFrame,
    *,
    resolution: str,
    protocol: str,
    column: str,
) -> list[float]:
    subset = frame[
        (frame["resolution"] == resolution)
        & (frame["protocol"] == protocol)
    ].set_index("dataset")
    missing = set(DATASET_ORDER) - set(subset.index)
    if missing:
        raise ValueError(
            f"Missing {protocol}/{resolution} values for: {sorted(missing)}"
        )
    return [float(subset.loc[dataset, column]) for dataset in DATASET_ORDER]


def add_runtime_reference_lines(axis: plt.Axes) -> None:
    for minutes, label in ((10, "10 min"), (60, "1 h")):
        axis.axhline(
            minutes,
            color="#A33A3A",
            linestyle=":",
            linewidth=0.8,
            alpha=0.75,
            zorder=0,
        )
        axis.text(
            0.55 if minutes == 60 else 0.99,
            minutes,
            label,
            color="#A33A3A",
            fontsize=7,
            va="bottom",
            ha="right",
            transform=axis.get_yaxis_transform(),
        )


def style_axis(
    axis: plt.Axes,
    *,
    title: str,
    ylabel: str,
) -> None:
    axis.set_facecolor("#FAFAFA")
    axis.set_title(title, fontsize=10, fontweight="bold")
    axis.set_xlabel("Dataset scale", fontsize=9, fontweight="bold")
    axis.set_ylabel(ylabel, fontsize=9, fontweight="bold")
    axis.set_xticks(range(len(DATASET_ORDER)))
    axis.set_xticklabels(
        [DATASET_LABELS[dataset] for dataset in DATASET_ORDER],
        fontsize=7,
    )
    axis.tick_params(axis="y", labelsize=8)
    axis.set_yscale("log")
    axis.grid(axis="y", which="both", linewidth=0.45, alpha=0.3)
    sns.despine(ax=axis)


def plot_runtime_panel(
    axis: plt.Axes,
    *,
    frame: pd.DataFrame,
    protocol: str,
    title: str,
) -> None:
    for resolution in RESOLUTION_LABELS:
        axis.plot(
            range(len(DATASET_ORDER)),
            ordered_values(
                frame,
                resolution=resolution,
                protocol=protocol,
                column="total_wall_minutes",
            ),
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=RESOLUTION_COLORS[resolution],
            label=RESOLUTION_LABELS[resolution],
        )
    add_runtime_reference_lines(axis)
    style_axis(
        axis,
        title=title,
        ylabel="End-to-end runtime (min, log scale)",
    )
    axis.legend(frameon=False, fontsize=7, loc="upper left")


def plot_memory_panel(
    axis: plt.Axes,
    *,
    frame: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    add_gpu_limit: bool = False,
) -> None:
    for resolution in RESOLUTION_LABELS:
        for protocol in PROTOCOLS:
            axis.plot(
                range(len(DATASET_ORDER)),
                ordered_values(
                    frame,
                    resolution=resolution,
                    protocol=protocol,
                    column=column,
                ),
                marker=("o" if protocol == "full_pipeline" else "x"),
                markersize=5,
                linewidth=1.6,
                linestyle=PROTOCOL_LINESTYLES[protocol],
                color=RESOLUTION_COLORS[resolution],
                label=(
                    f"{RESOLUTION_LABELS[resolution]} — "
                    f"{PROTOCOL_LABELS[protocol]}"
                ),
            )
    if add_gpu_limit:
        axis.axhline(
            24 * 1024,
            color="#666666",
            linestyle=":",
            linewidth=0.9,
            zorder=0,
        )
        axis.text(
            0.01,
            24 * 1024,
            "24 GiB GPU limit",
            color="#555555",
            fontsize=7,
            va="bottom",
            ha="left",
            transform=axis.get_yaxis_transform(),
        )
    style_axis(axis, title=title, ylabel=ylabel)
    axis.legend(frameon=False, fontsize=6.3, loc="best")


PROTOCOLS = ("full_pipeline", "train_only")


def render_figure(frame: pd.DataFrame, output_stem: Path) -> Dict[str, str]:
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "Arial"
    sns.set_theme(style="ticks")

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 8),
        constrained_layout=True,
    )
    figure.get_layout_engine().set(
        w_pad=0.03,
        h_pad=0.03,
        wspace=0.05,
        hspace=0.06,
    )

    plot_runtime_panel(
        axes[0, 0],
        frame=frame,
        protocol="full_pipeline",
        title="A  Full preprocessing + 200-epoch training",
    )
    plot_runtime_panel(
        axes[0, 1],
        frame=frame,
        protocol="train_only",
        title="B  Existing PKL + 200-epoch training",
    )
    plot_memory_panel(
        axes[1, 0],
        frame=frame,
        column="peak_tree_rss_mib",
        title="C  Peak CPU RAM",
        ylabel="Peak process-tree RSS (MiB, log scale)",
    )
    plot_memory_panel(
        axes[1, 1],
        frame=frame,
        column="peak_process_vram_mib",
        title="D  Peak GPU VRAM",
        ylabel="Peak process VRAM (MiB, log scale)",
        add_gpu_limit=True,
    )

    outputs = {
        "png": str(output_stem.with_suffix(".png")),
        "pdf": str(output_stem.with_suffix(".pdf")),
        "svg": str(output_stem.with_suffix(".svg")),
    }
    figure.savefig(outputs["png"], dpi=620, bbox_inches="tight")
    figure.savefig(outputs["pdf"], bbox_inches="tight")
    figure.savefig(outputs["svg"], bbox_inches="tight")
    plt.close(figure)
    return outputs


def markdown_table(
    headers: Iterable[str],
    rows: Iterable[Iterable[object]],
) -> str:
    header_list = list(headers)
    lines = [
        "| " + " | ".join(header_list) + " |",
        "|" + "|".join("---" for _ in header_list) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def render_report(
    *,
    suite_dir: Path,
    tasks: pd.DataFrame,
    stages: pd.DataFrame,
    environment: Dict[str, object],
    figure_outputs: Dict[str, str],
) -> Path:
    ordered = tasks.copy()
    ordered["dataset_order"] = ordered["dataset"].map(
        {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    )
    ordered["resolution_order"] = ordered["resolution"].map(
        {resolution: index for index, resolution in enumerate(RESOLUTION_LABELS)}
    )
    ordered["protocol_order"] = ordered["protocol"].map(
        {"full_pipeline": 0, "train_only": 1}
    )
    ordered = ordered.sort_values(
        ["dataset_order", "resolution_order", "protocol_order"]
    )

    result_rows = [
        (
            DATASET_LABELS[row.dataset].replace("\n", " / "),
            RESOLUTION_LABELS[row.resolution],
            PROTOCOL_LABELS[row.protocol],
            f"{row.total_wall_minutes:.3f}",
            f"{row.peak_tree_rss_mib:.1f}",
            f"{row.peak_process_vram_mib:.1f}",
            f"{row.final_loss:.6f}",
        )
        for row in ordered.itertuples()
    ]
    stage_rows = [
        (
            row.task_id,
            row.stage,
            f"{row.wall_seconds:.3f}",
            f"{row.peak_tree_rss_kb / 1024.0:.1f}",
            f"{row.peak_process_vram_mib:.1f}",
        )
        for row in stages.itertuples()
    ]
    gpu_devices = ", ".join(
        str(int(device))
        for device in sorted(tasks["cuda_device"].dropna().unique())
    )

    report = f"""# Optimized GRASP scalability benchmark

## Scope

- Code: `{environment["grasp_source"]}`
- Python: `{environment["python"]}`
- Git commit: `{environment["git_commit"]}`
- GPU devices: `{gpu_devices}`; all formal tasks used an exclusive NVIDIA GeForce RTX 4090.
- Formal repeats: one run per task; values are descriptive and have no error bars.
- Training: 200 epochs, lr=0.001, batch size=64, four positives, seed=2025, TF32 high, immutable batch cache, asynchronous cached H2D, foreach momentum.
- Full pipeline starts from `df_registered` and independently includes Portrait, pair-filtered Partition, Augment, Build-train-pkl, training initialization, 200 epochs, and final outputs.
- Train-only starts from the matching validated step3 PKL and includes loading, positive/cache preparation, 200 epochs, and final outputs.
- MERFISH U2OS was added as a later supplemental benchmark under the same command, validation, and resource-measurement contract; supplemental run metadata are retained in `environment.json`.
- U2OS has no benchmark label file, so final clustering evaluation is disabled, matching the existing optimized U2OS training configuration.

## End-to-end results

{markdown_table(
    ["Dataset", "Resolution", "Protocol", "Runtime (min)", "Peak CPU RSS (MiB)", "Peak GPU VRAM (MiB)", "Final loss"],
    result_rows,
)}

## Stage-level measurements

{markdown_table(
    ["Task", "Stage", "Wall (s)", "Peak CPU RSS (MiB)", "Peak GPU VRAM (MiB)"],
    stage_rows,
)}

## Measurement definition

- Wall time uses `time.perf_counter()` around each complete subprocess, including process startup and artifact writes.
- CPU RAM is the maximum sampled sum of RSS across the complete subprocess tree.
- GPU VRAM is the maximum `nvidia-smi compute-apps` memory attributed only to PIDs in that subprocess tree.
- Sampling intervals and GNU time comparison values are retained in `stages.csv` and each `task_summary.json`.
- OS page cache was not reset. RSS may double-count shared pages across multiprocessing workers, and sub-sampling can miss very short peaks.
- The historical figure timed mainly the training loop and used manually recorded memory. These new values are not direct replacements under the old measurement definition.

## Files

- Figure PNG: `{figure_outputs["png"]}`
- Figure PDF: `{figure_outputs["pdf"]}`
- Figure SVG: `{figure_outputs["svg"]}`
- Task summary: `{suite_dir / "tasks.csv"}`
- Stage summary: `{suite_dir / "stages.csv"}`
- Environment: `{suite_dir / "environment.json"}`
"""
    report_path = suite_dir / "BENCHMARK_REPORT.md"
    report_path.write_text(report)
    return report_path


def main() -> int:
    args = build_parser().parse_args()
    suite_dir = args.suite_dir.resolve()
    tasks_path = suite_dir / "tasks.csv"
    stages_path = suite_dir / "stages.csv"
    environment_path = suite_dir / "environment.json"
    if not tasks_path.is_file() or not stages_path.is_file():
        raise FileNotFoundError("Suite tasks.csv and stages.csv are required")

    tasks = pd.read_csv(tasks_path)
    stages = pd.read_csv(stages_path)
    expected_tasks = len(DATASET_ORDER) * len(RESOLUTION_LABELS) * len(PROTOCOLS)
    if len(tasks) != expected_tasks:
        raise ValueError(
            f"Expected {expected_tasks} completed tasks, found {len(tasks)}"
        )
    if tasks["task_id"].nunique() != expected_tasks:
        raise ValueError("Task IDs are not unique")
    if (tasks[["total_wall_minutes", "peak_tree_rss_mib", "peak_process_vram_mib"]] <= 0).any().any():
        raise ValueError("Runtime and memory metrics must be positive")

    environment = json.loads(environment_path.read_text())
    figure_outputs = render_figure(
        tasks,
        suite_dir / "optimized_grasp_simulated_scalability",
    )
    report_path = render_report(
        suite_dir=suite_dir,
        tasks=tasks,
        stages=stages,
        environment=environment,
        figure_outputs=figure_outputs,
    )
    (suite_dir / "render_summary.json").write_text(
        json.dumps(
            {"figures": figure_outputs, "report": str(report_path)},
            indent=2,
        )
    )
    print(f"Figure and report written to: {suite_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
