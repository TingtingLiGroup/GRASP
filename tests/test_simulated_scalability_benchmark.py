from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
benchmark_pipeline = importlib.import_module("benchmark_pipeline")
benchmark_suite = importlib.import_module("benchmark_simulated_scalability")
benchmark_renderer = importlib.import_module("render_simulated_scalability")


def test_process_resource_monitor_counts_descendant_rss():
    child_code = "import time; payload=bytearray(40*1024*1024); time.sleep(1.5)"
    parent_code = (
        "import subprocess,sys,time; "
        "payload=bytearray(24*1024*1024); "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        "time.sleep(1.5); child.wait()"
    )
    process = subprocess.Popen([sys.executable, "-c", parent_code])
    monitor = benchmark_pipeline.ProcessResourceMonitor(
        root_pid=process.pid,
        gpu_device=None,
        cpu_interval_seconds=0.02,
    )
    monitor.start()
    try:
        assert process.wait(timeout=10) == 0
    finally:
        monitor.stop()

    summary = monitor.summary()
    assert summary["resource_cpu_samples"] > 1
    assert summary["peak_tree_rss_kb"] > 64 * 1024


def test_process_gpu_memory_filters_uuid_and_process_tree(monkeypatch):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            "GPU-target, 101, 400\n"
            "GPU-target, 102, 600\n"
            "GPU-target, 999, 900\n"
            "GPU-other, 101, 700\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(
        benchmark_pipeline.subprocess,
        "run",
        lambda *args, **kwargs: completed,
    )

    used_memory = benchmark_pipeline.query_process_gpu_memory_mib(
        gpu_uuid="GPU-target",
        process_ids={101, 102},
    )
    assert used_memory == 1000.0


def test_training_command_locks_benchmark_contract(tmp_path):
    spec = benchmark_suite.DatasetSpec(
        slug="simulated1",
        display_name="Simulated 1",
        cli_dataset="data1_simulated1",
        registered_pkl=tmp_path / "registered.pkl",
        pairs_csv=tmp_path / "labels.csv",
        label_csv=tmp_path / "labels.csv",
        js_file=tmp_path / "distances.csv",
        train_pkls={"n20_m10": tmp_path / "train.pkl"},
        graph_count=800,
        cell_count=10,
        gene_count=80,
        num_clusters=8,
    )
    command = benchmark_suite.train_command(
        spec=spec,
        resolution=benchmark_suite.RESOLUTIONS["n20_m10"],
        train_pkl=tmp_path / "train.pkl",
        js_file=tmp_path / "distances.csv",
        output_dir=tmp_path / "training",
        epochs=200,
        cuda_device=2,
        seed=2025,
    )

    joined = " ".join(command)
    assert "--lrs 0.001" in joined
    assert "--num_epoch 200" in joined
    assert "--n 20 --m 10" in joined
    assert "--cache_train_batches 1" in joined
    assert "--pipeline_cached_h2d 1" in joined
    assert "--foreach_momentum 1" in joined
    assert "--matmul_precision high" in joined
    assert "--eval_freq 0 --eval_at_start 0 --visualize 0" in joined


def test_u2os_spec_and_command_support_unlabeled_full_dataset(tmp_path):
    spec = benchmark_suite.build_dataset_specs(tmp_path)["merfish_u2os"]
    command = benchmark_suite.train_command(
        spec=spec,
        resolution=benchmark_suite.RESOLUTIONS["n30_m15"],
        train_pkl=tmp_path / "train.pkl",
        js_file=tmp_path / "distances.csv",
        output_dir=tmp_path / "training",
        epochs=200,
        cuda_device=1,
        seed=2025,
    )

    joined = " ".join(command)
    assert spec.graph_count == 113_909
    assert spec.train_pkls["n30_m15"] is None
    assert "--dataset data2_merfish_u2os" in joined
    assert "--n 30 --m 15" in joined
    assert "--label_file" not in command
    assert "--num_clusters" not in command


def test_generated_train_pkl_links_full_pipeline_to_train_only(tmp_path):
    spec = benchmark_suite.build_dataset_specs(tmp_path)["merfish_u2os"]
    path = benchmark_suite.generated_train_pkl(
        tmp_path,
        spec,
        benchmark_suite.RESOLUTIONS["n30_m15"],
    )

    assert path == (
        tmp_path
        / "tasks/merfish_u2os_n30_m15_full_pipeline/train.pkl"
    )


def test_four_panel_renderer_writes_all_formats(tmp_path):
    rows = []
    for dataset_index, dataset in enumerate(benchmark_renderer.DATASET_ORDER, start=1):
        for resolution_index, resolution in enumerate(
            benchmark_renderer.RESOLUTION_LABELS,
            start=1,
        ):
            for protocol_index, protocol in enumerate(
                benchmark_renderer.PROTOCOLS,
                start=1,
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "resolution": resolution,
                        "protocol": protocol,
                        "total_wall_minutes": (
                            dataset_index * resolution_index * protocol_index
                        ),
                        "peak_tree_rss_mib": (
                            1000 * dataset_index * resolution_index
                        ),
                        "peak_process_vram_mib": (
                            500 * resolution_index * protocol_index
                        ),
                    }
                )
    outputs = benchmark_renderer.render_figure(
        pd.DataFrame(rows),
        tmp_path / "scalability",
    )

    assert set(outputs) == {"png", "pdf", "svg"}
    assert all(Path(path).is_file() for path in outputs.values())
