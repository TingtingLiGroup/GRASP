#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import psutil
import torch

from benchmark_pipeline import run_stage, verify_graph_outputs


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE_ROOT = REPO_ROOT.parent
PYTHON_MODULE = [sys.executable, "-m", "grasp_tool"]
PROTOCOLS = ("full_pipeline", "train_only")


@dataclass(frozen=True)
class Resolution:
    slug: str
    n_sectors: int
    m_rings: int
    k_neighbor: int


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    display_name: str
    cli_dataset: str
    registered_pkl: Path
    pairs_csv: Path
    label_csv: Optional[Path]
    js_file: Path
    train_pkls: Dict[str, Optional[Path]]
    graph_count: int
    cell_count: int
    gene_count: int
    num_clusters: int


RESOLUTIONS = {
    "n20_m10": Resolution("n20_m10", 20, 10, 20),
    "n30_m15": Resolution("n30_m15", 30, 15, 45),
}


def build_dataset_specs(workspace_root: Path) -> Dict[str, DatasetSpec]:
    return {
        "simulated1": DatasetSpec(
            slug="simulated1",
            display_name="Simulated 1",
            cli_dataset="data1_simulated1",
            registered_pkl=workspace_root / "1_input/pkl_data/simulated1_data_dict.pkl",
            pairs_csv=workspace_root / "1_input/label/data1_simulated1_label.csv",
            label_csv=workspace_root / "1_input/label/data1_simulated1_label.csv",
            js_file=workspace_root
            / "data1_simulated1/step2_js/js_distances_bin0.0100_count10_threshold0.05.csv",
            train_pkls={
                "n20_m10": workspace_root
                / "data1_simulated1/step3_graph_data/simulated1_n20_m10_cell10_gene80_graph800.pkl",
                "n30_m15": workspace_root
                / "data1_simulated1/step3_graph_data/simulated1_n30_m15_cell10_gene80_graph800.pkl",
            },
            graph_count=800,
            cell_count=10,
            gene_count=80,
            num_clusters=8,
        ),
        "simulated2": DatasetSpec(
            slug="simulated2",
            display_name="Simulated 2",
            cli_dataset="data1_simulated2",
            registered_pkl=workspace_root / "1_input/pkl_data/simulated2_data_dict.pkl",
            pairs_csv=workspace_root / "1_input/label/data1_simulated2_label.csv",
            label_csv=workspace_root / "1_input/label/data1_simulated2_label.csv",
            js_file=workspace_root
            / "data1_simulated2/step2_js/js_distances_bin0.0100_count10_threshold0.05.csv",
            train_pkls={
                "n20_m10": workspace_root
                / "data1_simulated2/step3_graph_data/simulated2_n20_m10_cell100_gene25_graph2500.pkl",
                "n30_m15": workspace_root
                / "data1_simulated2/step3_graph_data/simulated2_n30_m15_cell100_gene25_graph2500.pkl",
            },
            graph_count=2_500,
            cell_count=100,
            gene_count=25,
            num_clusters=5,
        ),
        "simulated3": DatasetSpec(
            slug="simulated3",
            display_name="Simulated 3",
            cli_dataset="data1_simulated3",
            registered_pkl=workspace_root
            / "1_input/pkl_data/simulated3_data_dict_filtered_cell50_gene400.pkl",
            pairs_csv=workspace_root / "1_input/label/data1_simulated3_label.csv",
            label_csv=workspace_root / "1_input/label/data1_simulated3_label.csv",
            js_file=workspace_root
            / "data1_simulated3/step2_js/js_distances_bin0.0100_count10_threshold0.05.csv",
            train_pkls={
                "n20_m10": workspace_root
                / "data1_simulated3/step3_graph_data/simulated3_n20_m10_cell50_gene400_graph15000.pkl",
                "n30_m15": workspace_root
                / "data1_simulated3/step3_graph_data/simulated3_n30_m15_cell50_gene400_graph15000.pkl",
            },
            graph_count=15_000,
            cell_count=50,
            gene_count=400,
            num_clusters=8,
        ),
        "merfish_u2os": DatasetSpec(
            slug="merfish_u2os",
            display_name="MERFISH U2OS",
            cli_dataset="data2_merfish_u2os",
            registered_pkl=workspace_root
            / "1_input/pkl_data/merfish_u2os_data_dict.pkl",
            pairs_csv=workspace_root / "data2_merfish_u2os/load_graph_data.csv",
            label_csv=None,
            js_file=workspace_root
            / "data2_merfish_u2os/step2_js/js_distances_bin0.0100_count10_threshold0.05.csv",
            train_pkls={
                "n20_m10": workspace_root
                / "data2_merfish_u2os/step3_graph_data/merfish_u2os_n20_m10_cell989_gene135_graph113909.pkl",
                "n30_m15": None,
            },
            graph_count=113_909,
            cell_count=989,
            gene_count=135,
            num_clusters=8,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark optimized GRASP across simulated and real datasets."
    )
    parser.add_argument(
        "--workspace_root",
        type=Path,
        default=DEFAULT_WORKSPACE_ROOT,
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT
        / "outputs/perf_benchmark/simulated_200epoch_end_to_end",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--resume_run",
        type=Path,
        default=None,
        help="Resume a suite directory and rerun only incomplete tasks.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["simulated1", "simulated2", "simulated3", "merfish_u2os"],
        default=["simulated1", "simulated2", "simulated3"],
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        choices=list(RESOLUTIONS),
        default=list(RESOLUTIONS),
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=list(PROTOCOLS),
        default=list(PROTOCOLS),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--cuda_device", type=int, default=None)
    parser.add_argument(
        "--allowed_cuda_devices",
        nargs="+",
        type=int,
        default=None,
        help="Restrict automatic GPU switching to these physical device indices.",
    )
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--portrait_threads", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--wait_for_gpu",
        action="store_true",
        help="Wait for or switch to an exclusive idle GPU between tasks.",
    )
    parser.add_argument("--gpu_poll_seconds", type=float, default=60.0)
    parser.add_argument(
        "--cleanup_intermediates",
        action="store_true",
        help="Delete generated portraits, graph CSVs, and train PKLs after validation.",
    )
    return parser


def run_command(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()


def gpu_inventory() -> List[Dict[str, object]]:
    gpu_lines = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    active_uuids = set()
    process_output = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in process_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2:
            active_uuids.add(fields[0])

    inventory = []
    for line in gpu_lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            continue
        inventory.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_used_mib": float(fields[3]),
                "memory_total_mib": float(fields[4]),
                "utilization_percent": float(fields[5]),
                "has_compute_process": fields[1] in active_uuids,
            }
        )
    return inventory


def choose_exclusive_gpu(requested_device: Optional[int]) -> int:
    inventory = gpu_inventory()
    if requested_device is not None:
        matches = [gpu for gpu in inventory if gpu["index"] == requested_device]
        if not matches:
            raise RuntimeError(f"GPU {requested_device} was not found")
        if matches[0]["has_compute_process"]:
            raise RuntimeError(f"GPU {requested_device} already has a compute process")
        return requested_device

    candidates = [
        gpu
        for gpu in inventory
        if not gpu["has_compute_process"] and gpu["memory_used_mib"] < 1024
    ]
    if not candidates:
        raise RuntimeError("No exclusive idle GPU is available")
    return int(min(candidates, key=lambda gpu: gpu["memory_used_mib"])["index"])


def acquire_exclusive_gpu(
    *,
    preferred_device: Optional[int],
    allowed_devices: Optional[Sequence[int]],
    wait: bool,
    poll_seconds: float,
) -> int:
    while True:
        candidates = []
        if preferred_device is not None:
            candidates.append(preferred_device)
        if allowed_devices is not None:
            candidates.extend(
                device
                for device in allowed_devices
                if device not in candidates
            )

        for device in candidates:
            try:
                return choose_exclusive_gpu(device)
            except RuntimeError:
                continue
        if allowed_devices is None:
            try:
                return choose_exclusive_gpu(None)
            except RuntimeError:
                pass
        if not wait:
            allowed_message = (
                f" among {list(allowed_devices)}"
                if allowed_devices is not None
                else ""
            )
            raise RuntimeError(
                f"No exclusive idle GPU is available{allowed_message}"
            )
        print(
            f"No allowed exclusive GPU is available; "
            f"retrying in {poll_seconds:.0f}s",
            flush=True,
        )
        time.sleep(poll_seconds)


def verify_inputs(
    specs: Dict[str, DatasetSpec],
    dataset_names: Iterable[str],
    resolution_names: Iterable[str],
    protocols: Iterable[str] = PROTOCOLS,
) -> None:
    selected_protocols = set(protocols)
    missing = []
    for dataset_name in dataset_names:
        spec = specs[dataset_name]
        required_paths = []
        if "full_pipeline" in selected_protocols:
            required_paths.extend((spec.registered_pkl, spec.pairs_csv))
        if "train_only" in selected_protocols:
            required_paths.append(spec.js_file)
        if spec.label_csv is not None:
            required_paths.append(spec.label_csv)
        for path in required_paths:
            if not path.is_file():
                missing.append(path)
        if (
            "train_only" in selected_protocols
            and "full_pipeline" not in selected_protocols
        ):
            for resolution_name in resolution_names:
                path = spec.train_pkls.get(resolution_name)
                if path is None or not path.is_file():
                    missing.append(
                        path
                        or Path(
                            f"<missing train PKL for {dataset_name}/{resolution_name}>"
                        )
                    )
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required benchmark inputs are missing:\n{joined}")


def collect_environment(
    *,
    workspace_root: Path,
    cuda_device: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workspace_root": str(workspace_root),
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(),
        "host_memory_bytes": psutil.virtual_memory().total,
        "cuda_device": cuda_device,
        "gpu_inventory": gpu_inventory(),
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_command(["git", "status", "--short"]),
        "grasp_source": str(REPO_ROOT / "grasp_tool"),
        "arguments": vars(args) | {"workspace_root": str(workspace_root)},
        "measurement_notes": [
            "Single-run benchmark; no error bars.",
            "CPU RAM is the sampled sum of RSS across the complete subprocess tree.",
            "GPU VRAM is sampled from nvidia-smi compute-apps for subprocess-tree PIDs only.",
            "RSS can double-count shared pages across multiprocessing workers.",
            "Sampling can miss peaks shorter than the sampling interval.",
            "OS page-cache state is not reset between tasks.",
        ],
    }


def train_command(
    *,
    spec: DatasetSpec,
    resolution: Resolution,
    train_pkl: Path,
    js_file: Path,
    output_dir: Path,
    epochs: int,
    cuda_device: int,
    seed: int,
) -> List[str]:
    command = [
        *PYTHON_MODULE,
        "train-moco",
        "--dataset",
        spec.cli_dataset,
        "--pkl",
        str(train_pkl),
        "--js",
        "1",
        "--js_file",
        str(js_file),
        "--n",
        str(resolution.n_sectors),
        "--m",
        str(resolution.m_rings),
        "--a",
        "0.3",
        "--b",
        "0.7",
        "--temperature",
        "0.07",
        "--batch_size",
        "64",
        "--num_positive",
        "4",
        "--num_epoch",
        str(epochs),
        "--lrs",
        "0.001",
        "--k",
        "512",
        "--cuda_device",
        str(cuda_device),
        "--seed",
        str(seed),
        "--output_dir",
        str(output_dir),
        "--cache_train_batches",
        "1",
        "--pipeline_cached_h2d",
        "1",
        "--cached_h2d_prefetch_batches",
        "2",
        "--foreach_momentum",
        "1",
        "--matmul_precision",
        "high",
        "--eval_freq",
        "0",
        "--eval_at_start",
        "0",
        "--visualize",
        "0",
    ]
    if spec.label_csv is not None:
        command.extend(
            [
                "--num_clusters",
                str(spec.num_clusters),
                "--label_file",
                str(spec.label_csv),
            ]
        )
    return command


def tensors_are_finite(value: object) -> bool:
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            return bool(torch.isfinite(value).all())
        return True
    if isinstance(value, dict):
        return all(tensors_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensors_are_finite(item) for item in value)
    return True


def validate_train_pkl(path: Path, expected_graphs: int) -> None:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    for key in (
        "original_graphs",
        "augmented_graphs",
        "gene_labels",
        "cell_labels",
    ):
        if key not in payload or len(payload[key]) != expected_graphs:
            actual = len(payload.get(key, ()))
            raise RuntimeError(
                f"{path}: {key} has {actual} entries, expected {expected_graphs}"
            )


def validate_training_output(
    *,
    output_root: Path,
    expected_graphs: int,
    expected_epochs: int,
) -> Dict[str, object]:
    completed_markers = list(output_root.rglob("ALL_COMPLETED.txt"))
    histories = list(output_root.rglob("training_history_lr0.001.csv"))
    if len(completed_markers) != 1 or len(histories) != 1:
        raise RuntimeError(
            f"{output_root}: completion markers={len(completed_markers)}, histories={len(histories)}"
        )

    run_dir = completed_markers[0].parent
    history = pd.read_csv(histories[0])
    if len(history) != expected_epochs:
        raise RuntimeError(
            f"{histories[0]} has {len(history)} rows, expected {expected_epochs}"
        )
    if history["epoch"].tolist() != list(range(1, expected_epochs + 1)):
        raise RuntimeError(f"{histories[0]} has an invalid epoch sequence")
    if not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError(f"{histories[0]} contains NaN or Inf")

    embedding_path = run_dir / f"epoch{expected_epochs}_lr0.001_embedding.csv"
    checkpoint_path = run_dir / f"epoch_{expected_epochs}_lr_0.001_checkpoint.pth"
    if not embedding_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError(f"{run_dir}: missing final embedding or checkpoint")

    embedding_rows = 0
    for chunk in pd.read_csv(embedding_path, chunksize=50_000):
        feature_columns = [
            column for column in chunk.columns if column.startswith("feature_")
        ]
        if len(feature_columns) != 128:
            raise RuntimeError(
                f"{embedding_path}: expected 128 features, got {len(feature_columns)}"
            )
        if not np.isfinite(
            chunk[feature_columns].to_numpy(dtype=np.float32, copy=False)
        ).all():
            raise RuntimeError(f"{embedding_path} contains NaN or Inf")
        if chunk["cell"].isna().any() or chunk["gene"].isna().any():
            raise RuntimeError(f"{embedding_path} contains missing labels")
        embedding_rows += len(chunk)
    if embedding_rows != expected_graphs:
        raise RuntimeError(
            f"{embedding_path} has {embedding_rows} rows, expected {expected_graphs}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not tensors_are_finite(checkpoint):
        raise RuntimeError(f"{checkpoint_path} contains NaN or Inf")

    return {
        "run_dir": str(run_dir),
        "history": str(histories[0]),
        "history_rows": len(history),
        "final_loss": float(history.iloc[-1]["total_loss"]),
        "embedding": str(embedding_path),
        "embedding_rows": embedding_rows,
        "checkpoint": str(checkpoint_path),
        "finite": True,
    }


def stage_result(
    *,
    name: str,
    command: Sequence[str],
    task_dir: Path,
    artifact_path: Path,
    cuda_device: int,
) -> Dict[str, object]:
    return run_stage(
        name=name,
        command=command,
        repo_root=REPO_ROOT,
        run_dir=task_dir,
        artifact_path=artifact_path,
        profile=False,
        gpu_device=cuda_device,
    )


def run_task(
    *,
    suite_dir: Path,
    spec: DatasetSpec,
    resolution: Resolution,
    protocol: str,
    epochs: int,
    cuda_device: int,
    processes: int,
    portrait_threads: int,
    seed: int,
    train_pkl_override: Optional[Path] = None,
) -> Dict[str, object]:
    task_id = f"{spec.slug}_{resolution.slug}_{protocol}"
    task_dir = suite_dir / "tasks" / task_id
    summary_path = task_dir / "task_summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text())
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    stages = []
    if protocol == "full_pipeline":
        portrait_dir = task_dir / "portrait"
        graph_root = task_dir / "graphs"
        generated_train_pkl = task_dir / "train.pkl"
        training_output = task_dir / "training"
        generated_js = (
            portrait_dir
            / "js_distances_bin0.0100_count10_threshold0.05.csv"
        )

        portrait_dir.mkdir()
        stages.append(
            stage_result(
                name="portrait",
                command=[
                    *PYTHON_MODULE,
                    "portrait",
                    "--pkl_file",
                    str(spec.registered_pkl),
                    "--output_dir",
                    str(portrait_dir),
                    "--max_count",
                    "10",
                    "--transcript_window",
                    "30",
                    "--bin_size",
                    "0.01",
                    "--threshold",
                    "0.05",
                    "--r_min",
                    "0.01",
                    "--r_max",
                    "0.6",
                    "--r_step",
                    "0.03",
                    "--num_threads",
                    str(portrait_threads),
                    "--visualize_top_n",
                    "0",
                    "--log_file",
                    str(task_dir / "portrait_internal.log"),
                ],
                task_dir=task_dir,
                artifact_path=portrait_dir,
                cuda_device=cuda_device,
            )
        )
        if not generated_js.is_file():
            raise RuntimeError(f"Portrait did not create {generated_js}")

        stages.append(
            stage_result(
                name="partition",
                command=[
                    *PYTHON_MODULE,
                    "partition-graphs",
                    "--pkl",
                    str(spec.registered_pkl),
                    "--graph_root",
                    str(graph_root),
                    "--pairs_csv",
                    str(spec.pairs_csv),
                    "--n_sectors",
                    str(resolution.n_sectors),
                    "--m_rings",
                    str(resolution.m_rings),
                    "--k_neighbor",
                    str(resolution.k_neighbor),
                    "--write_distance_matrix",
                    "0",
                ],
                task_dir=task_dir,
                artifact_path=graph_root,
                cuda_device=cuda_device,
            )
        )
        stages.append(
            stage_result(
                name="augmentation",
                command=[
                    *PYTHON_MODULE,
                    "augment-graphs",
                    "--graph_root",
                    str(graph_root),
                    "--dropout_ratio",
                    "0.1",
                    "--seed",
                    str(seed),
                ],
                task_dir=task_dir,
                artifact_path=graph_root,
                cuda_device=cuda_device,
            )
        )
        verify_graph_outputs(graph_root, spec.graph_count)

        stages.append(
            stage_result(
                name="build_train_pkl",
                command=[
                    *PYTHON_MODULE,
                    "build-train-pkl",
                    "--pairs_csv",
                    str(spec.pairs_csv),
                    "--graph_root",
                    str(graph_root),
                    "--output_pkl",
                    str(generated_train_pkl),
                    "--dataset",
                    spec.cli_dataset,
                    "--n_sectors",
                    str(resolution.n_sectors),
                    "--m_rings",
                    str(resolution.m_rings),
                    "--k_neighbor",
                    str(resolution.k_neighbor),
                    "--processes",
                    str(processes),
                ],
                task_dir=task_dir,
                artifact_path=generated_train_pkl,
                cuda_device=cuda_device,
            )
        )
        train_pkl = generated_train_pkl
        js_file = generated_js
    elif protocol == "train_only":
        training_output = task_dir / "training"
        train_pkl = train_pkl_override or spec.train_pkls.get(resolution.slug)
        if train_pkl is None or not train_pkl.is_file():
            raise FileNotFoundError(
                f"No train PKL is available for {spec.slug}/{resolution.slug}"
            )
        js_file = spec.js_file
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    stages.append(
        stage_result(
            name="train_moco",
            command=train_command(
                spec=spec,
                resolution=resolution,
                train_pkl=train_pkl,
                js_file=js_file,
                output_dir=training_output,
                epochs=epochs,
                cuda_device=cuda_device,
                seed=seed,
            ),
            task_dir=task_dir,
            artifact_path=training_output,
            cuda_device=cuda_device,
        )
    )
    validation = validate_training_output(
        output_root=training_output,
        expected_graphs=spec.graph_count,
        expected_epochs=epochs,
    )
    validate_train_pkl(train_pkl, spec.graph_count)

    total_wall_seconds = sum(float(stage["wall_seconds"]) for stage in stages)
    summary = {
        "task_id": task_id,
        "dataset": spec.slug,
        "dataset_display": spec.display_name,
        "resolution": resolution.slug,
        "n_sectors": resolution.n_sectors,
        "m_rings": resolution.m_rings,
        "k_neighbor": resolution.k_neighbor,
        "protocol": protocol,
        "graph_count": spec.graph_count,
        "cell_count": spec.cell_count,
        "gene_count": spec.gene_count,
        "epochs": epochs,
        "cuda_device": cuda_device,
        "train_pkl": str(train_pkl),
        "js_file": str(js_file),
        "total_wall_seconds": total_wall_seconds,
        "peak_tree_rss_kb": max(
            int(stage.get("peak_tree_rss_kb", 0)) for stage in stages
        ),
        "peak_process_vram_mib": max(
            float(stage.get("peak_process_vram_mib", 0.0)) for stage in stages
        ),
        "stages": stages,
        "validation": validation,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for _ in range(3):
        try:
            choose_exclusive_gpu(cuda_device)
            break
        except RuntimeError:
            time.sleep(2)
    else:
        raise RuntimeError(
            f"GPU {cuda_device} gained a foreign compute process during {task_id}; "
            "discard this task and rerun it on an exclusive GPU"
        )
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def refresh_suite_outputs(suite_dir: Path) -> List[Dict[str, object]]:
    summaries = [
        json.loads(path.read_text())
        for path in sorted((suite_dir / "tasks").glob("*/task_summary.json"))
    ]
    task_rows = []
    stage_rows = []
    for summary in summaries:
        task_rows.append(
            {
                "task_id": summary["task_id"],
                "dataset": summary["dataset"],
                "dataset_display": summary["dataset_display"],
                "resolution": summary["resolution"],
                "protocol": summary["protocol"],
                "graph_count": summary["graph_count"],
                "epochs": summary["epochs"],
                "cuda_device": summary.get("cuda_device"),
                "total_wall_seconds": summary["total_wall_seconds"],
                "total_wall_minutes": summary["total_wall_seconds"] / 60.0,
                "peak_tree_rss_kb": summary["peak_tree_rss_kb"],
                "peak_tree_rss_mib": summary["peak_tree_rss_kb"] / 1024.0,
                "peak_process_vram_mib": summary["peak_process_vram_mib"],
                "final_loss": summary["validation"]["final_loss"],
                "training_run_dir": summary["validation"]["run_dir"],
            }
        )
        for stage in summary["stages"]:
            stage_rows.append(
                {
                    "task_id": summary["task_id"],
                    "dataset": summary["dataset"],
                    "resolution": summary["resolution"],
                    "protocol": summary["protocol"],
                    "stage": stage["stage"],
                    "wall_seconds": stage["wall_seconds"],
                    "peak_tree_rss_kb": stage.get("peak_tree_rss_kb", 0),
                    "peak_process_vram_mib": stage.get(
                        "peak_process_vram_mib", 0.0
                    ),
                    "gnu_time_max_rss_kb": stage.get("max_rss_kb"),
                    "resource_cpu_samples": stage.get(
                        "resource_cpu_samples", 0
                    ),
                    "resource_gpu_samples": stage.get(
                        "resource_gpu_samples", 0
                    ),
                    "command": json.dumps(stage["command"]),
                    "log": stage["log"],
                }
            )
    write_csv(suite_dir / "tasks.csv", task_rows)
    write_csv(suite_dir / "stages.csv", stage_rows)
    (suite_dir / "suite_summary.json").write_text(
        json.dumps({"tasks": summaries}, indent=2)
    )
    return summaries


def cleanup_task_intermediates(summary: Dict[str, object]) -> List[str]:
    if summary["protocol"] != "full_pipeline":
        return []
    task_dir = Path(summary["stages"][0]["log"]).parent
    removed = []
    for path in (task_dir / "portrait", task_dir / "graphs", task_dir / "train.pkl"):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))
        elif path.is_file():
            path.unlink()
            removed.append(str(path))
    return removed


def generated_train_pkl(
    suite_dir: Path,
    spec: DatasetSpec,
    resolution: Resolution,
) -> Path:
    return (
        suite_dir
        / "tasks"
        / f"{spec.slug}_{resolution.slug}_full_pipeline"
        / "train.pkl"
    )


def requested_task_ids(args: argparse.Namespace) -> set[str]:
    return {
        f"{dataset}_{resolution}_{protocol}"
        for dataset in args.datasets
        for resolution in args.resolutions
        for protocol in args.protocols
    }


def append_supplement_environment(
    *,
    suite_dir: Path,
    workspace_root: Path,
    cuda_device: int,
    args: argparse.Namespace,
) -> None:
    environment_path = suite_dir / "environment.json"
    if not environment_path.is_file():
        raise FileNotFoundError(f"Missing suite environment: {environment_path}")
    environment = json.loads(environment_path.read_text())
    supplement = collect_environment(
        workspace_root=workspace_root,
        cuda_device=cuda_device,
        args=args,
    )
    environment.setdefault("supplemental_runs", []).append(supplement)
    environment_path.write_text(json.dumps(environment, indent=2, default=str))


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    workspace_root = args.workspace_root.resolve()
    specs = build_dataset_specs(workspace_root)
    verify_inputs(specs, args.datasets, args.resolutions, args.protocols)
    cuda_device = acquire_exclusive_gpu(
        preferred_device=args.cuda_device,
        allowed_devices=args.allowed_cuda_devices,
        wait=args.wait_for_gpu,
        poll_seconds=args.gpu_poll_seconds,
    )

    if args.resume_run is not None:
        suite_dir = args.resume_run.resolve()
        if not suite_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {suite_dir}")
        append_supplement_environment(
            suite_dir=suite_dir,
            workspace_root=workspace_root,
            cuda_device=cuda_device,
            args=args,
        )
    else:
        run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
        suite_dir = (args.output_root / run_name).resolve()
        if suite_dir.exists():
            raise FileExistsError(f"Suite directory already exists: {suite_dir}")
        (suite_dir / "tasks").mkdir(parents=True)
        environment = collect_environment(
            workspace_root=workspace_root,
            cuda_device=cuda_device,
            args=args,
        )
        (suite_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, default=str)
        )

    expected_task_ids = requested_task_ids(args)
    for dataset_name in args.datasets:
        spec = specs[dataset_name]
        for resolution_name in args.resolutions:
            resolution = RESOLUTIONS[resolution_name]
            for protocol in args.protocols:
                cuda_device = acquire_exclusive_gpu(
                    preferred_device=cuda_device,
                    allowed_devices=args.allowed_cuda_devices,
                    wait=args.wait_for_gpu,
                    poll_seconds=args.gpu_poll_seconds,
                )
                print(
                    f"Running {spec.slug} {resolution.slug} {protocol} "
                    f"for {args.epochs} epochs on GPU {cuda_device}",
                    flush=True,
                )
                train_pkl_override = None
                if protocol == "train_only" and "full_pipeline" in args.protocols:
                    candidate = generated_train_pkl(suite_dir, spec, resolution)
                    if candidate.is_file():
                        train_pkl_override = candidate
                run_task(
                    suite_dir=suite_dir,
                    spec=spec,
                    resolution=resolution,
                    protocol=protocol,
                    epochs=args.epochs,
                    cuda_device=cuda_device,
                    processes=args.processes,
                    portrait_threads=args.portrait_threads,
                    seed=args.seed,
                    train_pkl_override=train_pkl_override,
                )
                refresh_suite_outputs(suite_dir)

    summaries = refresh_suite_outputs(suite_dir)
    completed_task_ids = {summary["task_id"] for summary in summaries}
    missing_task_ids = expected_task_ids - completed_task_ids
    if missing_task_ids:
        raise RuntimeError(
            f"Requested benchmark tasks are incomplete: {sorted(missing_task_ids)}"
        )

    cleanup_records = []
    if args.cleanup_intermediates:
        for summary in summaries:
            removed = cleanup_task_intermediates(summary)
            cleanup_records.append(
                {"task_id": summary["task_id"], "removed": removed}
            )
        (suite_dir / "cleanup.json").write_text(
            json.dumps(cleanup_records, indent=2)
        )

    (suite_dir / "ALL_COMPLETED.txt").write_text(
        f"Completed {len(summaries)} tasks at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Latest requested tasks: {', '.join(sorted(expected_task_ids))}\n"
    )
    print(f"Suite completed: {suite_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
