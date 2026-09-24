#!/usr/bin/env python

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import pickle
import resource
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psutil


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the packaged GRASP pipeline on deterministic synthetic data."
    )
    parser.add_argument("--graph_count", type=int, default=None)
    parser.add_argument("--cell_count", type=int, default=None)
    parser.add_argument("--gene_count", type=int, default=None)
    parser.add_argument("--transcripts_per_graph", type=int, default=40)
    parser.add_argument("--n_sectors", type=int, default=20)
    parser.add_argument("--m_rings", type=int, default=10)
    parser.add_argument("--k_neighbor", type=int, default=5)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--num_clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs/perf_benchmark"),
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Run directory name (default: timestamp plus graph count).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap each pipeline command with cProfile.",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Measure preprocessing stages only.",
    )
    parser.add_argument(
        "--train_profile_pkl",
        type=Path,
        default=None,
        help=(
            "Profile only the steady-state training loop from an existing train.pkl; "
            "skips preprocessing, final embeddings, and startup in the timed region."
        ),
    )
    parser.add_argument(
        "--train_profile_subset",
        type=int,
        default=0,
        help="Use the first N graphs from --train_profile_pkl; 0 uses all graphs.",
    )
    parser.add_argument(
        "--train_profile_js_file",
        type=Path,
        default=None,
        help="Use production JS distances to build positive indices.",
    )
    parser.add_argument(
        "--train_profile_active_batches",
        type=int,
        default=10,
        help="Number of active batches captured by torch.profiler.",
    )
    parser.add_argument(
        "--train_profile_prefetch_batches",
        type=int,
        default=0,
        help="Bounded training prefetch depth used by train-only profiling.",
    )
    parser.add_argument(
        "--train_profile_pin_memory",
        type=int,
        default=0,
        choices=[0, 1],
        help="Pin bounded prefetched batches during train-only profiling.",
    )
    parser.add_argument(
        "--train_profile_wall_only",
        action="store_true",
        help="Measure complete steady-state epochs without torch.profiler overhead.",
    )
    parser.add_argument(
        "--train_profile_epochs",
        type=int,
        default=1,
        help="Complete epochs measured by --train_profile_wall_only (default: 1).",
    )
    parser.add_argument(
        "--train_profile_warmup_epochs",
        type=int,
        default=0,
        help="Unmeasured warmup epochs before steady-state wall timing.",
    )
    parser.add_argument(
        "--train_profile_cache_batches",
        type=int,
        default=0,
        choices=[0, 1],
        help="Prebuild immutable CPU training batches before the timed region.",
    )
    parser.add_argument(
        "--train_profile_pipeline_cached_h2d",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use bounded pinned staging and asynchronous H2D for cached batches.",
    )
    parser.add_argument(
        "--train_profile_h2d_prefetch_batches",
        type=int,
        default=2,
        help="Pinned staging depth for cached asynchronous H2D.",
    )
    parser.add_argument(
        "--train_profile_zero_grad_set_to_none",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use Adam zero_grad(set_to_none=True) in train-only benchmarks.",
    )
    parser.add_argument(
        "--train_profile_foreach_momentum",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use foreach kernels for momentum-encoder EMA updates.",
    )
    parser.add_argument(
        "--train_profile_matmul_precision",
        choices=["highest", "high", "medium"],
        default="highest",
        help="Float32 matrix multiplication precision.",
    )
    parser.add_argument(
        "--train_profile_fuse_positive_encoders",
        type=int,
        default=0,
        choices=[0, 1],
        help="Encode all positive views in one key-encoder call.",
    )
    parser.add_argument(
        "--train_profile_num_positive",
        type=int,
        default=4,
        help="Number of positive views used by train-only benchmarks.",
    )
    parser.add_argument(
        "--train_profile_reconstruction_negative_ratio",
        type=int,
        default=0,
        help="Sampled reconstruction negatives per positive; 0 keeps dense BCE.",
    )
    parser.add_argument(
        "--train_profile_record_stages",
        type=int,
        default=0,
        choices=[0, 1],
        help="Add CUDA event stage timing to wall-only benchmarks.",
    )
    return parser


def choose_layout(graph_count: int) -> Tuple[int, int]:
    if graph_count <= 0:
        raise ValueError("--graph_count must be positive")

    preferred_gene_count = min(25, max(2, int(math.sqrt(graph_count / 2))))
    for gene_count in range(preferred_gene_count, 0, -1):
        if graph_count % gene_count == 0:
            return graph_count // gene_count, gene_count
    return graph_count, 1


def generate_synthetic_inputs(
    *,
    output_dir: Path,
    graph_count: int,
    cell_count: int,
    gene_count: int,
    transcripts_per_graph: int,
    seed: int,
) -> Dict[str, Path]:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    if cell_count <= 0 or gene_count <= 0:
        raise ValueError("cell_count and gene_count must be positive")
    if cell_count * gene_count < graph_count:
        raise ValueError("cell_count * gene_count must be at least graph_count")

    cells = [f"cell_{index:04d}" for index in range(cell_count)]
    genes = [f"gene_{index:03d}" for index in range(gene_count)]
    all_pair_indices = rng.choice(
        cell_count * gene_count,
        size=graph_count,
        replace=False,
    )
    selected_pairs = sorted(
        (
            cells[int(pair_index) // gene_count],
            genes[int(pair_index) % gene_count],
        )
        for pair_index in all_pair_indices
    )

    transcript_frames = []
    pair_records = []
    label_records = []
    for cell, gene in selected_pairs:
        cell_index = int(cell.split("_")[-1])
        gene_index = int(gene.split("_")[-1])
        radii = np.sqrt(rng.uniform(0.0025, 0.9801, transcripts_per_graph))
        angles = rng.uniform(-np.pi, np.pi, transcripts_per_graph)
        transcript_frames.append(
            pd.DataFrame(
                {
                    "cell": cell,
                    "gene": gene,
                    "x_c_s": radii * np.cos(angles),
                    "y_c_s": radii * np.sin(angles),
                }
            )
        )
        pair_records.append({"cell": cell, "gene": gene})
        label_records.append(
            {
                "cell": cell,
                "gene": gene,
                "groundtruth": f"class_{(cell_index + gene_index) % 5}",
            }
        )

    boundary_angles = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    boundary_frames = [
        pd.DataFrame(
            {
                "cell": cell,
                "x_c_s": 0.45 * np.cos(boundary_angles),
                "y_c_s": 0.45 * np.sin(boundary_angles),
            }
        )
        for cell in cells
    ]

    payload = {
        "df_registered": pd.concat(transcript_frames, ignore_index=True),
        "nuclear_boundary_df_registered": pd.concat(boundary_frames, ignore_index=True),
        "cell_radii": {cell: 1.0 for cell in cells},
        "meta": {
            "synthetic_benchmark": True,
            "seed": seed,
            "graph_count": graph_count,
            "transcripts_per_graph": transcripts_per_graph,
            "cell_count": cell_count,
            "gene_count": gene_count,
        },
    }

    input_pkl = output_dir / "registered.pkl"
    pairs_csv = output_dir / "pairs.csv"
    labels_csv = output_dir / "labels.csv"
    with input_pkl.open("wb") as handle:
        pickle.dump(payload, handle)
    pd.DataFrame(pair_records).to_csv(pairs_csv, index=False)
    pd.DataFrame(label_records).to_csv(labels_csv, index=False)

    return {
        "input_pkl": input_pkl,
        "pairs_csv": pairs_csv,
        "labels_csv": labels_csv,
    }


def path_metrics(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return 1, path.stat().st_size

    file_count = 0
    total_bytes = 0
    for child in path.rglob("*"):
        if child.is_file():
            file_count += 1
            total_bytes += child.stat().st_size
    return file_count, total_bytes


def parse_gnu_time_report(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}

    wanted_keys = {
        "User time (seconds)": ("user_seconds", float),
        "System time (seconds)": ("system_seconds", float),
        "Percent of CPU this job got": (
            "cpu_percent",
            lambda value: float(value.rstrip("%")),
        ),
        "Maximum resident set size (kbytes)": ("max_rss_kb", int),
        "File system inputs": ("filesystem_inputs", int),
        "File system outputs": ("filesystem_outputs", int),
    }
    result: Dict[str, object] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        for label, (key, converter) in wanted_keys.items():
            prefix = f"{label}:"
            if stripped.startswith(prefix):
                result[key] = converter(stripped[len(prefix) :].strip())
                break
    return result


def resolve_gpu_uuid(device: int) -> Optional[str]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) == 2 and fields[0] == str(device):
            return fields[1]
    return None


def collect_process_tree_usage(root_pid: int) -> Tuple[int, set[int]]:
    try:
        root = psutil.Process(root_pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0, set()

    rss_bytes = 0
    process_ids = set()
    for process in processes:
        try:
            rss_bytes += process.memory_info().rss
            process_ids.add(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss_bytes, process_ids


def query_process_gpu_memory_mib(
    *,
    gpu_uuid: str,
    process_ids: set[int],
) -> float:
    if not process_ids:
        return 0.0
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0

    if completed.returncode != 0:
        return 0.0

    used_memory_mib = 0.0
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3 or fields[0] != gpu_uuid:
            continue
        try:
            process_id = int(fields[1])
            memory_mib = float(fields[2])
        except ValueError:
            continue
        if process_id in process_ids:
            used_memory_mib += memory_mib
    return used_memory_mib


class ProcessResourceMonitor:
    """Sample the complete subprocess tree without counting unrelated GPU jobs."""

    def __init__(
        self,
        *,
        root_pid: int,
        gpu_device: Optional[int],
        cpu_interval_seconds: float = 0.1,
        gpu_interval_seconds: float = 0.5,
    ):
        self.root_pid = root_pid
        self.gpu_uuid = resolve_gpu_uuid(gpu_device) if gpu_device is not None else None
        self.cpu_interval_seconds = cpu_interval_seconds
        self.gpu_interval_seconds = gpu_interval_seconds
        self.peak_tree_rss_bytes = 0
        self.peak_process_vram_mib = 0.0
        self.cpu_samples = 0
        self.gpu_samples = 0
        self._known_process_ids: set[int] = set()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    def _sample(self) -> None:
        next_gpu_sample = 0.0
        while not self._stop_event.is_set():
            sampled_at = time.monotonic()
            rss_bytes, process_ids = collect_process_tree_usage(self.root_pid)
            self._known_process_ids.update(process_ids)
            self.peak_tree_rss_bytes = max(self.peak_tree_rss_bytes, rss_bytes)
            self.cpu_samples += 1

            if self.gpu_uuid is not None and sampled_at >= next_gpu_sample:
                memory_mib = query_process_gpu_memory_mib(
                    gpu_uuid=self.gpu_uuid,
                    process_ids=self._known_process_ids,
                )
                self.peak_process_vram_mib = max(
                    self.peak_process_vram_mib,
                    memory_mib,
                )
                self.gpu_samples += 1
                next_gpu_sample = sampled_at + self.gpu_interval_seconds

            self._stop_event.wait(self.cpu_interval_seconds)

    def summary(self) -> Dict[str, object]:
        return {
            "resource_sampling_interval_seconds": self.cpu_interval_seconds,
            "gpu_sampling_interval_seconds": self.gpu_interval_seconds,
            "resource_cpu_samples": self.cpu_samples,
            "resource_gpu_samples": self.gpu_samples,
            "peak_tree_rss_kb": int(self.peak_tree_rss_bytes / 1024),
            "peak_process_vram_mib": self.peak_process_vram_mib,
        }


def run_stage(
    *,
    name: str,
    command: Sequence[str],
    repo_root: Path,
    run_dir: Path,
    artifact_path: Path,
    profile: bool,
    gpu_device: Optional[int] = None,
) -> Dict[str, object]:
    stage_command = list(command)
    profile_path = run_dir / f"{name}.prof"
    if profile:
        if stage_command[:3] != [sys.executable, "-m", "grasp_tool"]:
            raise ValueError("Profiling expects a 'python -m grasp_tool' command")
        stage_command = [
            sys.executable,
            "-m",
            "cProfile",
            "-o",
            str(profile_path),
            "-m",
            "grasp_tool",
            *stage_command[3:],
        ]

    time_report = run_dir / f"{name}.time.txt"
    log_path = run_dir / f"{name}.log"
    timed_command = ["/usr/bin/time", "-v", "-o", str(time_report), *stage_command]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    )

    before_files, before_bytes = path_metrics(artifact_path)
    started_at = time.perf_counter()
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            timed_command,
            cwd=repo_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        resource_monitor = ProcessResourceMonitor(
            root_pid=process.pid,
            gpu_device=gpu_device,
        )
        resource_monitor.start()
        try:
            returncode = process.wait()
        finally:
            resource_monitor.stop()
    wall_seconds = time.perf_counter() - started_at

    after_files, after_bytes = path_metrics(artifact_path)
    result: Dict[str, object] = {
        "stage": name,
        "returncode": returncode,
        "wall_seconds": wall_seconds,
        "artifact_files_added": max(0, after_files - before_files),
        "artifact_bytes_added": max(0, after_bytes - before_bytes),
        "command": stage_command,
        "log": str(log_path),
    }
    result.update(parse_gnu_time_report(time_report))
    result.update(resource_monitor.summary())
    if profile:
        result["profile"] = str(profile_path)

    if returncode != 0:
        log_lines = log_path.read_text(errors="replace").splitlines()
        excerpt = "\n".join(log_lines[-40:])
        raise RuntimeError(f"Stage {name} failed:\n{excerpt}")

    print(
        f"{name}: {wall_seconds:.2f}s, "
        f"files +{result['artifact_files_added']}, "
        f"bytes +{result['artifact_bytes_added']}"
    )
    return result


def verify_graph_outputs(graph_root: Path, expected_graphs: int) -> None:
    original_nodes = [path for path in graph_root.glob("*/*_node_matrix.csv") if not path.parent.name.endswith("_aug")]
    augmented_nodes = list(graph_root.glob("*_aug/*_node_matrix.csv"))
    if len(original_nodes) != expected_graphs:
        raise RuntimeError(f"Expected {expected_graphs} original graphs, found {len(original_nodes)}")
    if len(augmented_nodes) != expected_graphs:
        raise RuntimeError(f"Expected {expected_graphs} augmented graphs, found {len(augmented_nodes)}")


class CudaStageRecorder:
    """Collect asynchronous CUDA event timings without synchronizing each batch."""

    def __init__(self, torch_module: Any, enabled: bool):
        self.torch = torch_module
        self.enabled = enabled
        self.cpu_seconds: Dict[str, float] = {}
        self.cuda_events: Dict[str, List[Tuple[Any, Any]]] = {}

    @contextmanager
    def __call__(self, name: str):
        cpu_started = time.perf_counter()
        cuda_started = None
        cuda_finished = None
        if self.enabled:
            cuda_started = self.torch.cuda.Event(enable_timing=True)
            cuda_finished = self.torch.cuda.Event(enable_timing=True)
            cuda_started.record()
        try:
            yield
        finally:
            if cuda_finished is not None:
                cuda_finished.record()
                self.cuda_events.setdefault(name, []).append((cuda_started, cuda_finished))
            self.cpu_seconds[name] = self.cpu_seconds.get(name, 0.0) + (time.perf_counter() - cpu_started)

    def summarize(self) -> Dict[str, Dict[str, float]]:
        if self.enabled:
            self.torch.cuda.synchronize()
        summary = {}
        for name, cpu_seconds in self.cpu_seconds.items():
            events = self.cuda_events.get(name, ())
            cuda_milliseconds = sum(start.elapsed_time(finish) for start, finish in events) if events else 0.0
            summary[name] = {
                "calls": len(events) if events else 0,
                "cpu_wall_ms": cpu_seconds * 1000.0,
                "cuda_ms": cuda_milliseconds,
            }
        return summary


def _profile_event_device_time(event: Any) -> float:
    for attribute in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, attribute, None)
        if value is not None:
            return float(value)
    return 0.0


def _profile_event_total_device_time(event: Any) -> float:
    for attribute in ("device_time_total", "cuda_time_total"):
        value = getattr(event, attribute, None)
        if value is not None:
            return float(value)
    return 0.0


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered_values = sorted(values)
    position = (len(ordered_values) - 1) * percentile_value
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered_values[lower_index]
    fraction = position - lower_index
    return ordered_values[lower_index] * (1.0 - fraction) + ordered_values[upper_index] * fraction


def summarize_cached_batch_shapes(cached_batches: Sequence[Any]) -> Dict[str, Any]:
    if not cached_batches:
        return {}

    signatures = Counter()
    query_node_counts = []
    query_edge_counts = []
    for query_batch, positive_batches in cached_batches:
        query_node_counts.append(query_batch.x.size(0))
        query_edge_counts.append(query_batch.edge_index.size(1))
        signature = (
            query_batch.x.size(0),
            query_batch.edge_index.size(1),
            *(
                dimension
                for positive_batch in positive_batches
                for dimension in (
                    positive_batch.x.size(0),
                    positive_batch.edge_index.size(1),
                )
            ),
        )
        signatures[signature] += 1

    return {
        "unique_five_view_shapes": len(signatures),
        "query_nodes_min": min(query_node_counts),
        "query_nodes_max": max(query_node_counts),
        "query_edges_min": min(query_edge_counts),
        "query_edges_max": max(query_edge_counts),
        "most_common_signatures": [
            {"signature": list(signature), "batches": count} for signature, count in signatures.most_common(10)
        ],
    }


def run_training_profile(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch

    from grasp_tool.cli import train_moco

    train_moco._lazy_import_training_deps()
    if not args.train_profile_pkl.exists():
        raise FileNotFoundError(f"Training PKL not found: {args.train_profile_pkl}")
    if args.train_profile_active_batches <= 0:
        raise ValueError("--train_profile_active_batches must be positive")
    if args.train_profile_prefetch_batches < 0:
        raise ValueError("--train_profile_prefetch_batches must be non-negative")
    if args.train_profile_epochs <= 0:
        raise ValueError("--train_profile_epochs must be positive")
    if args.train_profile_warmup_epochs < 0:
        raise ValueError("--train_profile_warmup_epochs must be non-negative")
    if args.train_profile_h2d_prefetch_batches <= 0:
        raise ValueError("--train_profile_h2d_prefetch_batches must be positive")
    if args.train_profile_pipeline_cached_h2d and not args.train_profile_cache_batches:
        raise ValueError("--train_profile_pipeline_cached_h2d requires " "--train_profile_cache_batches=1")
    if args.train_profile_js_file is not None and not args.train_profile_js_file.exists():
        raise FileNotFoundError(f"JS distance file not found: {args.train_profile_js_file}")
    if args.train_profile_num_positive <= 0:
        raise ValueError("--train_profile_num_positive must be positive")
    if args.train_profile_reconstruction_negative_ratio < 0:
        raise ValueError("--train_profile_reconstruction_negative_ratio must be non-negative")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{timestamp}_train_profile"
    run_dir = (args.output_root / run_name).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    with args.train_profile_pkl.open("rb") as handle:
        training_data = pickle.load(handle)
    original_graphs = training_data["original_graphs"]
    augmented_graphs = training_data["augmented_graphs"]
    gene_labels = training_data["gene_labels"]
    cell_labels = training_data["cell_labels"]
    if args.train_profile_subset > 0:
        subset_size = min(args.train_profile_subset, len(original_graphs))
        original_graphs = original_graphs[:subset_size]
        augmented_graphs = augmented_graphs[:subset_size]
        gene_labels = gene_labels[:subset_size]
        cell_labels = cell_labels[:subset_size]
    if len(original_graphs) < 2:
        raise ValueError("Training profile requires at least two graphs")

    train_moco.set_seed(args.seed)
    train_moco.configure_matmul_precision(args.train_profile_matmul_precision)
    positive_started_at = time.perf_counter()
    if args.train_profile_js_file is not None:
        import pandas as pd

        js_distances = pd.read_csv(args.train_profile_js_file)
        positive_samples = train_moco.gcl.MoCoMultiPositive.generate_samples_js(
            original_graphs,
            augmented_graphs,
            gene_labels,
            cell_labels,
            num_positive=args.train_profile_num_positive,
            js_distances_df=js_distances,
        )
        positive_mode = "js"
    else:
        # Positive selection is outside the steady-state training target. Fixed
        # self indices preserve five-view batch sizes for synthetic smoke tests.
        positive_samples = [
            (graph_index, [graph_index] * args.train_profile_num_positive)
            for graph_index in range(len(original_graphs))
        ]
        positive_mode = "self"
    positive_preparation_seconds = time.perf_counter() - positive_started_at
    cached_training_batches = None
    estimated_cache_bytes = 0
    actual_cache_bytes = 0
    cache_build_seconds = 0.0
    if args.train_profile_cache_batches:
        cache_started_at = time.perf_counter()
        estimated_cache_bytes = train_moco.gcl.MoCoMultiPositive.estimate_cached_training_batch_bytes(
            original_graphs,
            augmented_graphs,
            positive_samples,
            args.batch_size,
        )
        cached_training_batches = train_moco.gcl.MoCoMultiPositive.build_cached_training_batches(
            original_graphs,
            augmented_graphs,
            positive_samples,
            args.batch_size,
        )
        actual_cache_bytes = train_moco.gcl.MoCoMultiPositive.cached_training_batch_bytes(cached_training_batches)
        cache_build_seconds = time.perf_counter() - cache_started_at
    cached_batch_shapes = summarize_cached_batch_shapes(cached_training_batches or ())
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    feature_dim = original_graphs[0].x.shape[1]
    base_encoder = train_moco.gcl.GATEncoder(
        in_channels=feature_dim,
        hidden_channels=64,
        out_channels=128,
    ).to(device)
    model = train_moco.gcl.MoCoMultiPositive(
        base_encoder,
        dim=128,
        K=512,
        m=0.999,
        T=0.07,
    ).to(device)
    model.weighted_recon_loss = False
    model.profile_training = True
    optimization_args = SimpleNamespace(
        foreach_momentum=bool(args.train_profile_foreach_momentum),
        fuse_positive_encoders=bool(args.train_profile_fuse_positive_encoders),
        reconstruction_negative_ratio=args.train_profile_reconstruction_negative_ratio,
        seed=args.seed,
    )
    train_moco.configure_model_optimizations(model, optimization_args)
    optimizer = torch.optim.Adam(
        list(model.encoder_q.parameters()),
        lr=0.001,
        weight_decay=1e-5,
    )
    training_args = SimpleNamespace(
        batch_size=args.batch_size,
        use_clustering=False,
        spectral_loss=False,
        dist_type="uniform",
        forward_method="default",
        num_clusters=8,
        a=0.5,
        b=0.5,
        c=0.0,
        use_gradient_clipping=1,
        gradient_clip_norm=3.0,
        profile_training=True,
        prefetch_batches=args.train_profile_prefetch_batches,
        pin_prefetched_batches=bool(args.train_profile_pin_memory),
        pipeline_cached_h2d=bool(args.train_profile_pipeline_cached_h2d),
        cached_h2d_prefetch_batches=args.train_profile_h2d_prefetch_batches,
        zero_grad_set_to_none=bool(args.train_profile_zero_grad_set_to_none),
    )

    full_batches, remainder = divmod(len(original_graphs), args.batch_size)
    available_batches = full_batches + (1 if remainder >= 2 else 0)
    requested_batches = args.train_profile_active_batches + 2
    if not args.train_profile_wall_only and available_batches < requested_batches:
        raise ValueError(
            "Profile subset is too small: "
            f"{available_batches} batches available, {requested_batches} required "
            "(one wait, one warmup, and active batches)"
        )

    cuda_enabled = device.type == "cuda"
    if cuda_enabled:
        torch.cuda.reset_peak_memory_stats(device)
    record_stages = not args.train_profile_wall_only or bool(args.train_profile_record_stages)
    stage_recorder = CudaStageRecorder(torch, cuda_enabled) if record_stages else None
    if stage_recorder is not None:
        model.training_stage_recorder = stage_recorder
    profile_metadata: Dict[str, int] = {}
    operator_hotspots = []
    cpu_operator_hotspots = []
    profiler_stage_totals = {}
    trace_path = None
    table_path = None
    cpu_table_path = None
    epoch_wall_seconds = []
    warmup_epoch_wall_seconds = []

    if args.train_profile_wall_only:
        model.profile_training = False
        training_args.profile_training = False
        profile_metadata.update(batch_count=0, graph_count=0)
        for warmup_epoch in range(1, args.train_profile_warmup_epochs + 1):
            if cuda_enabled:
                torch.cuda.synchronize(device)
            warmup_started_at = time.perf_counter()
            train_moco.train_epoch(
                model,
                original_graphs,
                augmented_graphs,
                positive_samples,
                optimizer,
                device,
                training_args,
                epoch=warmup_epoch,
                prepared_batches=cached_training_batches,
                stage_recorder=stage_recorder,
            )
            if cuda_enabled:
                torch.cuda.synchronize(device)
            warmup_epoch_wall_seconds.append(time.perf_counter() - warmup_started_at)

        if cuda_enabled:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        for measured_epoch in range(1, args.train_profile_epochs + 1):
            epoch_metadata: Dict[str, int] = {}
            epoch_started_at = time.perf_counter()
            losses = train_moco.train_epoch(
                model,
                original_graphs,
                augmented_graphs,
                positive_samples,
                optimizer,
                device,
                training_args,
                epoch=args.train_profile_warmup_epochs + measured_epoch,
                prepared_batches=cached_training_batches,
                stage_recorder=stage_recorder,
                profile_metadata=epoch_metadata,
            )
            if cuda_enabled:
                torch.cuda.synchronize(device)
            epoch_wall_seconds.append(time.perf_counter() - epoch_started_at)
            profile_metadata["batch_count"] += epoch_metadata["batch_count"]
            profile_metadata["graph_count"] += epoch_metadata["graph_count"]
        wall_seconds = time.perf_counter() - started_at
    else:
        if cuda_enabled:
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda_enabled:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        trace_path = run_dir / "train_trace.json"
        profiler_schedule = torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=args.train_profile_active_batches,
            repeat=1,
        )
        with torch.profiler.profile(
            activities=activities,
            schedule=profiler_schedule,
            on_trace_ready=lambda current_profiler: current_profiler.export_chrome_trace(str(trace_path)),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            losses = train_moco.train_epoch(
                model,
                original_graphs,
                augmented_graphs,
                positive_samples,
                optimizer,
                device,
                training_args,
                epoch=1,
                prepared_batches=cached_training_batches,
                max_batches=requested_batches,
                profiler=profiler,
                stage_recorder=stage_recorder,
                profile_metadata=profile_metadata,
            )

        key_averages = profiler.key_averages()
        table_path = run_dir / "train_profile_table.txt"
        table_path.write_text(
            key_averages.table(
                sort_by=("self_cuda_time_total" if cuda_enabled else "self_cpu_time_total"),
                row_limit=80,
            )
        )
        cpu_table_path = run_dir / "train_profile_cpu_table.txt"
        cpu_table_path.write_text(key_averages.table(sort_by="self_cpu_time_total", row_limit=80))
        for event in sorted(
            key_averages,
            key=(_profile_event_device_time if cuda_enabled else lambda item: float(item.self_cpu_time_total)),
            reverse=True,
        )[:30]:
            operator_hotspots.append(
                {
                    "name": event.key,
                    "calls": event.count,
                    "self_cpu_ms": float(event.self_cpu_time_total) / 1000.0,
                    "self_device_ms": _profile_event_device_time(event) / 1000.0,
                }
            )
        cpu_operator_hotspots = [
            {
                "name": event.key,
                "calls": event.count,
                "self_cpu_ms": float(event.self_cpu_time_total) / 1000.0,
                "cpu_total_ms": float(event.cpu_time_total) / 1000.0,
            }
            for event in sorted(
                key_averages,
                key=lambda item: float(item.self_cpu_time_total),
                reverse=True,
            )[:30]
        ]
        custom_stage_names = {
            "batch_collate_query",
            "batch_collate_positive",
            "train_h2d",
            "train_forward",
            "moco_query_encoder",
            "moco_key_encoder",
            "moco_contrastive",
            "moco_reconstruction",
            "moco_queue_update",
            "train_zero_grad",
            "train_backward",
            "train_gradient_clipping",
            "train_optimizer_step",
        }
        profiler_stage_totals = {
            event.key: {
                "calls": event.count,
                "cpu_total_ms": float(event.cpu_time_total) / 1000.0,
                "device_total_ms": (_profile_event_total_device_time(event) / 1000.0),
            }
            for event in key_averages
            if event.key in custom_stage_names
        }
        if cuda_enabled:
            torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started_at

    stage_summary = stage_recorder.summarize() if stage_recorder is not None else {}

    top_level_stages = (
        "train_h2d",
        "train_forward",
        "train_zero_grad",
        "train_backward",
        "train_gradient_clipping",
        "train_optimizer_step",
    )
    gpu_active_ms = sum(stage_summary.get(name, {}).get("cuda_ms", 0.0) for name in top_level_stages)
    graph_count = profile_metadata["graph_count"]
    summary = {
        "mode": ("train_wall_only" if args.train_profile_wall_only else "train_profile_only"),
        "repo_root": str(repo_root),
        "training_pkl": str(args.train_profile_pkl.resolve()),
        "positive_mode": positive_mode,
        "js_file": (str(args.train_profile_js_file.resolve()) if args.train_profile_js_file is not None else None),
        "positive_preparation_seconds": positive_preparation_seconds,
        "device": str(device),
        "graphs_available": len(original_graphs),
        "graphs_measured": graph_count,
        "batches_measured": profile_metadata["batch_count"],
        "epochs": (args.train_profile_epochs if args.train_profile_wall_only else 1),
        "warmup_epochs": (args.train_profile_warmup_epochs if args.train_profile_wall_only else 0),
        "warmup_epoch_wall_seconds": warmup_epoch_wall_seconds,
        "epoch_wall_seconds": epoch_wall_seconds,
        "epoch_wall_median_seconds": (statistics.median(epoch_wall_seconds) if epoch_wall_seconds else None),
        "epoch_wall_p95_seconds": (percentile(epoch_wall_seconds, 0.95) if epoch_wall_seconds else None),
        "active_batches": (0 if args.train_profile_wall_only else args.train_profile_active_batches),
        "batch_size": args.batch_size,
        "prefetch_batches": args.train_profile_prefetch_batches,
        "pin_memory": bool(args.train_profile_pin_memory),
        "cache_batches": bool(args.train_profile_cache_batches),
        "pipeline_cached_h2d": bool(args.train_profile_pipeline_cached_h2d),
        "cached_h2d_prefetch_batches": (args.train_profile_h2d_prefetch_batches),
        "zero_grad_set_to_none": bool(args.train_profile_zero_grad_set_to_none),
        "foreach_momentum": bool(args.train_profile_foreach_momentum),
        "matmul_precision": args.train_profile_matmul_precision,
        "fuse_positive_encoders": bool(args.train_profile_fuse_positive_encoders),
        "num_positive": args.train_profile_num_positive,
        "reconstruction_negative_ratio": (args.train_profile_reconstruction_negative_ratio),
        "record_stages": record_stages,
        "estimated_cache_bytes": estimated_cache_bytes,
        "actual_cache_bytes": actual_cache_bytes,
        "cache_build_seconds": cache_build_seconds,
        "cached_batch_shapes": cached_batch_shapes,
        "wall_seconds": wall_seconds,
        "milliseconds_per_batch": wall_seconds * 1000.0 / profile_metadata["batch_count"],
        "graphs_per_second": graph_count / wall_seconds,
        "gpu_active_percent": (min(100.0, gpu_active_ms / (wall_seconds * 10.0)) if cuda_enabled else 0.0),
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_vram_mib": (torch.cuda.max_memory_allocated(device) / (1024**2) if cuda_enabled else 0.0),
        "losses": losses,
        "stage_timings": stage_summary,
        "profiler_stage_totals": profiler_stage_totals,
        "operator_hotspots": operator_hotspots,
        "cpu_operator_hotspots": cpu_operator_hotspots,
        "trace": str(trace_path) if trace_path is not None else None,
        "table": str(table_path) if table_path is not None else None,
        "cpu_table": (str(cpu_table_path) if cpu_table_path is not None else None),
    }
    summary_path = run_dir / "train_profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Training profile: {summary_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.train_profile_pkl is not None:
        return run_training_profile(args)
    if args.graph_count is None:
        raise ValueError("--graph_count is required unless --train_profile_pkl is specified")
    if (args.cell_count is None) != (args.gene_count is None):
        raise ValueError("--cell_count and --gene_count must be specified together")
    if args.cell_count is None:
        cell_count, gene_count = choose_layout(args.graph_count)
    else:
        cell_count, gene_count = args.cell_count, args.gene_count

    repo_root = Path(__file__).resolve().parents[1]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode = "clustered" if args.num_clusters is not None else "no_clustering"
    run_name = args.run_name or f"{timestamp}_graphs_{args.graph_count}_{mode}"
    run_dir = (args.output_root / run_name).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    inputs = generate_synthetic_inputs(
        output_dir=run_dir,
        graph_count=args.graph_count,
        cell_count=cell_count,
        gene_count=gene_count,
        transcripts_per_graph=args.transcripts_per_graph,
        seed=args.seed,
    )
    graph_root = run_dir / "graphs"
    train_pkl = run_dir / "train.pkl"
    training_output = run_dir / "training"
    python_module = [sys.executable, "-m", "grasp_tool"]

    results = []
    results.append(
        run_stage(
            name="partition",
            command=[
                *python_module,
                "partition-graphs",
                "--pkl",
                str(inputs["input_pkl"]),
                "--graph_root",
                str(graph_root),
                "--n_sectors",
                str(args.n_sectors),
                "--m_rings",
                str(args.m_rings),
                "--k_neighbor",
                str(args.k_neighbor),
            ],
            repo_root=repo_root,
            run_dir=run_dir,
            artifact_path=graph_root,
            profile=args.profile,
        )
    )
    results.append(
        run_stage(
            name="augmentation",
            command=[
                *python_module,
                "augment-graphs",
                "--graph_root",
                str(graph_root),
                "--dropout_ratio",
                "0.1",
                "--seed",
                str(args.seed),
            ],
            repo_root=repo_root,
            run_dir=run_dir,
            artifact_path=graph_root,
            profile=args.profile,
        )
    )
    verify_graph_outputs(graph_root, args.graph_count)

    results.append(
        run_stage(
            name="build_train_pkl",
            command=[
                *python_module,
                "build-train-pkl",
                "--pairs_csv",
                str(inputs["pairs_csv"]),
                "--graph_root",
                str(graph_root),
                "--output_pkl",
                str(train_pkl),
                "--dataset",
                f"synthetic_{args.graph_count}",
                "--n_sectors",
                str(args.n_sectors),
                "--m_rings",
                str(args.m_rings),
                "--k_neighbor",
                str(args.k_neighbor),
                "--processes",
                str(args.processes),
            ],
            repo_root=repo_root,
            run_dir=run_dir,
            artifact_path=train_pkl,
            profile=args.profile,
        )
    )
    if not train_pkl.exists():
        raise RuntimeError("build-train-pkl did not create train.pkl")

    if not args.skip_training:
        training_command = [
            *python_module,
            "train-moco",
            "--dataset",
            f"synthetic_{args.graph_count}",
            "--pkl",
            str(train_pkl),
            "--num_epoch",
            str(args.train_epochs),
            "--batch_size",
            str(args.batch_size),
            "--lrs",
            "0.001",
            "--cuda_device",
            str(args.cuda_device),
            "--seed",
            str(args.seed),
            "--output_dir",
            str(training_output),
            "--visualize",
            "0",
        ]
        if args.num_clusters is not None:
            training_command.extend(
                [
                    "--num_clusters",
                    str(args.num_clusters),
                    "--label_file",
                    str(inputs["labels_csv"]),
                ]
            )

        results.append(
            run_stage(
                name="train_moco",
                command=training_command,
                repo_root=repo_root,
                run_dir=run_dir,
                artifact_path=training_output,
                profile=args.profile,
                gpu_device=args.cuda_device,
            )
        )
        if not list(training_output.rglob("ALL_COMPLETED.txt")):
            raise RuntimeError("train-moco did not create ALL_COMPLETED.txt")

    for stage_result in results:
        processed_graphs = args.graph_count
        throughput_basis = "graphs"
        if stage_result["stage"] == "train_moco":
            processed_graphs *= args.train_epochs
            throughput_basis = "graph_epochs"
        stage_result["graphs_per_second"] = processed_graphs / stage_result["wall_seconds"]
        stage_result["throughput_basis"] = throughput_basis

    summary = {
        "run_dir": str(run_dir),
        "python": sys.executable,
        "graph_count": args.graph_count,
        "cell_count": cell_count,
        "gene_count": gene_count,
        "transcripts_per_graph": args.transcripts_per_graph,
        "nodes_per_graph": args.n_sectors * args.m_rings,
        "train_epochs": 0 if args.skip_training else args.train_epochs,
        "num_clusters": args.num_clusters,
        "profiled": args.profile,
        "stages": results,
    }
    summary_path = run_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
