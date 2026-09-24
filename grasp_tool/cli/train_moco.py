from __future__ import annotations

import os

# =========================================================================
# IMPORTANT: set thread-related env vars before importing numpy/torch.
# This avoids OpenBLAS/OMP errors such as "too many memory regions".
# =========================================================================
os.environ["OPENBLAS_NUM_THREADS"] = "8"  # Tune for your CPU (e.g., 1/8/16)
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

# NOTE: `grasp_tool.gnn.plot_refined` imports `umap`, which may import TensorFlow
# and emit noisy INFO/WARN logs even for `--help`. Suppress them by default.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import glob
import importlib
import json
import pickle
import random
import shutil
import time
import traceback
import uuid
import warnings
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _lazy_import_training_deps() -> None:
    """Import torch/pyg (and friends) only when training is actually executed.

    This keeps `grasp-tool train-moco --help` working in a base install where
    torch/pyg are intentionally NOT declared as PyPI dependencies.
    """

    global torch, Batch, ReduceLROnPlateau, gcl, vis

    try:
        torch = importlib.import_module("torch")
        torch_geometric_data = importlib.import_module("torch_geometric.data")
        Batch = getattr(torch_geometric_data, "Batch")
        lr_scheduler = importlib.import_module("torch.optim.lr_scheduler")
        ReduceLROnPlateau = getattr(lr_scheduler, "ReduceLROnPlateau")
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", "")
        if missing == "torch" or missing.startswith("torch_geometric"):
            raise ModuleNotFoundError(
                "Missing training dependencies: torch and torch-geometric.\n"
                "Install them first (conda or pip wheels), then re-run: grasp-tool train-moco ..."
            ) from e
        raise

    gcl = importlib.import_module("grasp_tool.gnn.gat_moco_final")
    vis = importlib.import_module("grasp_tool.gnn.plot_refined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoCo Training for Graph Neural Networks")

    # Dataset inputs
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., data1_simulated1)",
    )
    parser.add_argument("--pkl", type=str, required=True, help="Path to training PKL")
    parser.add_argument(
        "--js",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use JS distances: 0=no, 1=yes (default: 0)",
    )
    parser.add_argument(
        "--js_file",
        type=str,
        default=None,
        help="Path to JS distances CSV (required when --js=1)",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of sectors (n_sectors) (default: 20)")
    parser.add_argument("--m", type=int, default=10, help="Number of rings (m_rings) (default: 10)")
    parser.add_argument("--a", type=float, default=0.5, help="Reconstruction loss weight (default: 0.5)")
    parser.add_argument("--b", type=float, default=0.5, help="Contrastive loss weight (default: 0.5)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Contrastive temperature (default: 0.07)",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=0,
        help=(
            "Number of upcoming five-view CPU batches built in a background "
            "thread; 0 disables prefetch (default: 0)"
        ),
    )
    parser.add_argument(
        "--pin_prefetched_batches",
        type=int,
        default=0,
        choices=[0, 1],
        help=("Pin only bounded prefetched batches and use non-blocking H2D copies: " "0=no, 1=yes (default: 0)"),
    )
    parser.add_argument(
        "--cache_train_batches",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Prebuild immutable CPU x/edge_index/batch tensors for reuse across "
            "epochs and learning rates: 0=no, 1=yes (default: 0)"
        ),
    )
    parser.add_argument(
        "--pipeline_cached_h2d",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Stage cached batches through bounded pinned memory and overlap H2D "
            "with GPU compute: 0=no, 1=yes (default: 0)"
        ),
    )
    parser.add_argument(
        "--cached_h2d_prefetch_batches",
        type=int,
        default=2,
        help="Pinned staging depth for --pipeline_cached_h2d (default: 2)",
    )
    parser.add_argument(
        "--foreach_momentum",
        type=int,
        default=0,
        choices=[0, 1],
        help=("Batch momentum-encoder EMA tensor updates with foreach kernels: " "0=no, 1=yes (default: 0)"),
    )
    parser.add_argument(
        "--matmul_precision",
        choices=["highest", "high", "medium"],
        default="highest",
        help="Float32 matrix multiplication precision (default: highest)",
    )
    parser.add_argument(
        "--fuse_positive_encoders",
        type=int,
        default=0,
        choices=[0, 1],
        help="Encode all positive views in one key-encoder call: 0=no, 1=yes (default: 0)",
    )
    parser.add_argument(
        "--reconstruction_negative_ratio",
        type=int,
        default=0,
        help=(
            "Sample this many reconstruction negatives per positive; " "0 keeps the original dense BCE (default: 0)"
        ),
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Learning rate list (one or more values, e.g. --lrs 0.001 0.002). "
            "If omitted, uses the built-in default list."
        ),
    )
    parser.add_argument("--num_positive", type=int, default=4, help="Number of positives (default: 4)")
    parser.add_argument("--num_epoch", type=int, default=300, help="Number of epochs (default: 300)")
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=None,
        help="Enable clustering eval with this number of clusters (e.g., 5, 8)",
    )
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed (default: 2025)")
    parser.add_argument(
        "--use_gradient_clipping",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use gradient clipping: 0=no, 1=yes (default: 1)",
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=3.0,
        help="Gradient clipping max_norm (default: 3.0)",
    )
    parser.add_argument("--k", type=int, default=512, help="Queue size (default: 512)")
    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help=(
            "Optional ground-truth label CSV path (absolute or relative). "
            "Used by clustering evaluation when --num_clusters is set. "
            "The label CSV must contain columns: cell, gene, and one label column (e.g. groundtruth). "
            "Recognized label column names include: groundtruth_wzx, groundtruth, label, location, cluster, category, type. "
            "If omitted, tries to auto-discover label files under common project paths."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output root directory (default: ./outputs/<dataset>/step5_embedding)",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        choices=[0, 1],
        help="Create t-SNE/UMAP plots during evaluation: 0=no, 1=yes (default: 0)",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=0,
        help="Evaluate every N epochs; 0 evaluates only the final epoch (default: 0)",
    )
    parser.add_argument(
        "--eval_at_start",
        type=int,
        default=0,
        choices=[0, 1],
        help="Evaluate before training at epoch 0: 0=no, 1=yes (default: 0)",
    )

    args = parser.parse_args()

    if args.js == 1 and not args.js_file:
        parser.error("--js_file must be specified when using --js=1")
    if args.eval_freq < 0:
        parser.error("--eval_freq must be >= 0")
    if args.prefetch_batches < 0:
        parser.error("--prefetch_batches must be >= 0")
    if args.cached_h2d_prefetch_batches <= 0:
        parser.error("--cached_h2d_prefetch_batches must be positive")
    if args.pipeline_cached_h2d and not args.cache_train_batches:
        parser.error("--pipeline_cached_h2d requires --cache_train_batches=1")
    if args.num_positive <= 0:
        parser.error("--num_positive must be positive")
    if args.reconstruction_negative_ratio < 0:
        parser.error("--reconstruction_negative_ratio must be >= 0")

    args.n_sectors = args.n
    args.m_rings = args.m
    args.positive_sample_method = "js" if args.js == 1 else "random_window"
    args.js_distances_file = args.js_file
    args.pkl_file = args.pkl
    args.model = "gat"
    args.layer = "layer2"
    args.dist_type = "uniform"
    if args.lrs is None:
        args.lrs = [0.001, 0.002, 0.005, 0.01]
    args.c = 0.0
    args.use_clustering = False
    args.visualize = bool(args.visualize)
    args.eval_at_start = bool(args.eval_at_start)
    args.pin_prefetched_batches = bool(args.pin_prefetched_batches)
    args.cache_train_batches = bool(args.cache_train_batches)
    args.pipeline_cached_h2d = bool(args.pipeline_cached_h2d)
    args.foreach_momentum = bool(args.foreach_momentum)
    args.fuse_positive_encoders = bool(args.fuse_positive_encoders)

    if args.num_clusters is not None:
        args.clustering = True
        print(f"Clustering evaluation enabled. num_clusters={args.num_clusters}")
    else:
        args.clustering = False
        args.num_clusters = 8

    args.reduce_dims = True
    args.forward_method = "default"
    # use_gradient_clipping / gradient_clip_norm come from CLI
    args.print_freq = 10
    args.checkpoint_freq = 20
    args.save_best_only = False
    args.early_stopping = 0
    args.weighted = False
    args.window_size = 5
    args.optimizer = "adam"
    args.weight_decay = 1e-5
    args.lr_scheduler = "plateau"
    args.lr_patience = 10
    args.clustering_methods = [
        "KMeans",
        "Agglomerative",
        "SpectralClustering",
        "GaussianMixture",
    ]
    args.spectral_loss = False
    args.tsne_perplexity = 30.0
    args.umap_n_neighbors = 15
    args.umap_min_dist = 0.2
    args.size = 20
    args.graphs_number = None
    args.cell_numbers = None
    args.gene_numbers = None
    args.tissue = None
    args.experiment_id = None
    args.no_timestamp = False
    args.vis_methods = None

    if args.label_file is not None and os.path.exists(args.label_file):
        print(f"Using label file: {args.label_file}")

    return args


def load_data(
    args: argparse.Namespace,
) -> Tuple[List, List, List, List, pd.DataFrame, Optional[pd.DataFrame]]:
    save_file = args.pkl
    if not os.path.exists(save_file):
        raise FileNotFoundError(f"PKL file not found: {save_file}")

    print(f"Loading data from: {save_file}")
    with open(save_file, "rb") as f:
        data = pickle.load(f)

    original_graphs = data["original_graphs"]
    augmented_graphs = data["augmented_graphs"]
    gene_labels = data["gene_labels"]
    cell_labels = data["cell_labels"]

    args.graphs_number = len(original_graphs)
    args.cell_numbers = len(set(cell_labels)) if cell_labels else 0
    args.gene_numbers = len(set(gene_labels)) if gene_labels else 0

    gw_distances_df = pd.DataFrame(
        columns=pd.Index(
            [
                "target_cell",
                "target_gene",
                "cell",
                "gene",
                "num_real_nodes",
                "gw_distance",
            ]
        )
    )

    js_distances_df = None
    if args.js == 1:
        js_file = args.js_file
        if js_file and os.path.exists(js_file):
            js_distances_df = pd.read_csv(js_file)
            print(f"Loaded JS distances from: {js_file}")
        else:
            print(f"ERROR: JS distance file not found: {js_file}")
            print("Falling back to random_window method")
            args.positive_sample_method = "random_window"

    return (
        original_graphs,
        augmented_graphs,
        gene_labels,
        cell_labels,
        gw_distances_df,
        js_distances_df,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def configure_matmul_precision(precision: str) -> None:
    if precision not in {"highest", "high", "medium"}:
        raise ValueError(f"Unsupported matmul precision: {precision}")
    torch.set_float32_matmul_precision(precision)


def configure_model_optimizations(model: Any, args: argparse.Namespace) -> None:
    model.foreach_momentum = bool(getattr(args, "foreach_momentum", False))
    model.fuse_positive_encoders = bool(getattr(args, "fuse_positive_encoders", False))
    model.reconstruction_negative_ratio = int(getattr(args, "reconstruction_negative_ratio", 0))
    model.reconstruction_sampling_seed = int(getattr(args, "seed", 2025))


def setup_training(args):
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    timestamp = time.strftime("%m%d_%H%M")
    js_flag = "js" if args.js == 1 else "nojs"
    folder_name = (
        f"n{args.n}_m{args.m}_{js_flag}_"
        f"a{args.a}_b{args.b}_t{args.temperature}_"
        f"bs{args.batch_size}_neg{args.num_positive}_{timestamp}"
    )
    base_output_dir = args.output_dir or f"./outputs/{args.dataset}/step5_embedding"
    save_path = os.path.join(base_output_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Results will be saved to: {save_path}")
    return save_path, device


@contextmanager
def _training_stage_scope(
    name: str,
    *,
    profile_training: bool,
    stage_recorder: Optional[Any],
):
    if not profile_training and stage_recorder is None:
        with nullcontext():
            yield
        return

    with ExitStack() as stack:
        if profile_training:
            stack.enter_context(torch.profiler.record_function(name))
        if stage_recorder is not None:
            stack.enter_context(stage_recorder(name))
        yield


def train_epoch(
    model: Any,
    original_graphs: List,
    augmented_graphs: List,
    positive_samples: List,
    optimizer: Any,
    device: Any,
    args: argparse.Namespace,
    epoch: int,
    *,
    prepared_batches: Optional[Any] = None,
    max_batches: Optional[int] = None,
    profiler: Optional[Any] = None,
    stage_recorder: Optional[Any] = None,
    profile_metadata: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    model.train()
    loss_totals = torch.zeros(4, device=device, dtype=torch.float64)
    batch_count = 0
    graph_count = 0

    use_clustering = getattr(args, "use_clustering", True)
    spectral_loss = getattr(args, "spectral_loss", False)
    dist_type = "spectral" if spectral_loss else args.dist_type
    forward_method = getattr(args, "forward_method", "default")
    profile_training = getattr(args, "profile_training", False)
    prefetch_batches = getattr(args, "prefetch_batches", 0)
    pin_prefetched_batches = getattr(args, "pin_prefetched_batches", False) and device.type == "cuda"
    pipeline_cached_h2d = (
        getattr(args, "pipeline_cached_h2d", False) and prepared_batches is not None and device.type == "cuda"
    )
    batches_on_device = False

    if prepared_batches is None:
        batch_generator = gcl.MoCoMultiPositive.prepare_multi_positive_batch(
            original_graphs,
            augmented_graphs,
            positive_samples,
            args.batch_size,
            profile_training=profile_training,
            prefetch_batches=prefetch_batches,
            pin_memory=pin_prefetched_batches,
        )
    elif pipeline_cached_h2d:
        batch_generator = gcl.MoCoMultiPositive.prepare_cached_cuda_batches(
            prepared_batches,
            device,
            prefetch_batches=getattr(args, "cached_h2d_prefetch_batches", 2),
        )
        batches_on_device = True
    else:
        # The full cache intentionally remains pageable to avoid locking all
        # cached tensor memory; only bounded producer batches are pinned.
        pin_prefetched_batches = False
        batch_generator = iter(prepared_batches)

    for query_batch, positive_batches in batch_generator:
        batch_count += 1
        graph_count += query_batch.num_graphs
        if not batches_on_device:
            with _training_stage_scope(
                "train_h2d",
                profile_training=profile_training,
                stage_recorder=stage_recorder,
            ):
                query_batch = query_batch.to(device, non_blocking=pin_prefetched_batches)
                positive_batches = [
                    batch.to(device, non_blocking=pin_prefetched_batches) for batch in positive_batches
                ]
        im_q, edge_index_q, batch = (
            query_batch.x,
            query_batch.edge_index,
            query_batch.batch,
        )
        im_k_list = [pos_batch.x for pos_batch in positive_batches]
        edge_index_k_list = [pos_batch.edge_index for pos_batch in positive_batches]

        with _training_stage_scope(
            "train_forward",
            profile_training=profile_training,
            stage_recorder=stage_recorder,
        ):
            if forward_method == "supcon":
                loss, reconstruction_loss, contrastive_loss, clustering_loss, _, _, _ = model.forward_supcon(
                    im_q,
                    im_k_list,
                    edge_index_q,
                    edge_index_k_list,
                    batch,
                    args.num_clusters,
                    dist_type,
                    args.a,
                    args.b,
                    args.c,
                    use_clustering,
                )
            elif forward_method == "avg":
                loss, reconstruction_loss, contrastive_loss, clustering_loss, _, _, _ = model.forward_avg(
                    im_q,
                    im_k_list,
                    edge_index_q,
                    edge_index_k_list,
                    batch,
                    args.num_clusters,
                    dist_type,
                    args.a,
                    args.b,
                    args.c,
                    use_clustering,
                )
            else:
                loss, reconstruction_loss, contrastive_loss, clustering_loss, _, _, _ = model(
                    im_q,
                    im_k_list,
                    edge_index_q,
                    edge_index_k_list,
                    batch,
                    args.num_clusters,
                    dist_type,
                    args.a,
                    args.b,
                    args.c,
                    use_clustering,
                )

        with _training_stage_scope(
            "train_zero_grad",
            profile_training=profile_training,
            stage_recorder=stage_recorder,
        ):
            optimizer.zero_grad(set_to_none=getattr(args, "zero_grad_set_to_none", False))
        with _training_stage_scope(
            "train_backward",
            profile_training=profile_training,
            stage_recorder=stage_recorder,
        ):
            loss.backward()

        if hasattr(args, "use_gradient_clipping") and args.use_gradient_clipping:
            with _training_stage_scope(
                "train_gradient_clipping",
                profile_training=profile_training,
                stage_recorder=stage_recorder,
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_norm)

        with _training_stage_scope(
            "train_optimizer_step",
            profile_training=profile_training,
            stage_recorder=stage_recorder,
        ):
            optimizer.step()
        loss_totals += torch.stack(
            (
                loss.detach(),
                reconstruction_loss.detach(),
                contrastive_loss.detach(),
                clustering_loss.detach(),
            )
        )
        if profiler is not None:
            profiler.step()
        if max_batches is not None and batch_count >= max_batches:
            break

    if profile_metadata is not None:
        profile_metadata.update(batch_count=batch_count, graph_count=graph_count)

    if batch_count > 0:
        (
            total_loss,
            total_reconstruction_loss,
            total_contrastive_loss,
            total_clustering_loss,
        ) = (loss_totals / batch_count).tolist()
    else:
        (
            total_loss,
            total_reconstruction_loss,
            total_contrastive_loss,
            total_clustering_loss,
        ) = (
            0.0,
            0.0,
            0.0,
            0.0,
        )

    return {
        "total_loss": total_loss,
        "reconstruction_loss": total_reconstruction_loss,
        "contrastive_loss": total_contrastive_loss,
        "clustering_loss": total_clustering_loss,
    }


def train_model(args: argparse.Namespace) -> None:
    save_path, device = None, None
    try:
        (
            original_graphs,
            augmented_graphs,
            gene_labels,
            cell_labels,
            gw_distances_df,
            js_distances_df,
        ) = load_data(args)
        save_path, device = setup_training(args)
        configure_matmul_precision(getattr(args, "matmul_precision", "highest"))
        positive_sample_method = getattr(args, "positive_sample_method", "gw")

        if positive_sample_method == "gw":
            positive_samples = gcl.MoCoMultiPositive.generate_samples_gw(
                original_graphs,
                augmented_graphs,
                gene_labels,
                cell_labels,
                args.num_positive,
                gw_distances_df,
            )
        elif positive_sample_method == "js":
            positive_samples = gcl.MoCoMultiPositive.generate_samples_js(
                original_graphs,
                augmented_graphs,
                gene_labels,
                cell_labels,
                args.num_positive,
                js_distances_df,
            )
        else:
            positive_samples = gcl.MoCoMultiPositive.generate_samples_random_window(
                original_graphs,
                augmented_graphs,
                gene_labels,
                cell_labels,
                args.num_positive,
                args.window_size,
            )

        cached_training_batches = None
        if getattr(args, "cache_train_batches", False):
            estimated_cache_bytes = gcl.MoCoMultiPositive.estimate_cached_training_batch_bytes(
                original_graphs,
                augmented_graphs,
                positive_samples,
                args.batch_size,
            )
            print(
                "Building immutable CPU training batch cache; "
                f"estimated additional RAM: {estimated_cache_bytes / (1024**3):.2f} GiB"
            )
            cache_started_at = time.perf_counter()
            cached_training_batches = gcl.MoCoMultiPositive.build_cached_training_batches(
                original_graphs,
                augmented_graphs,
                positive_samples,
                args.batch_size,
            )
            actual_cache_bytes = gcl.MoCoMultiPositive.cached_training_batch_bytes(cached_training_batches)
            args.train_batch_cache_estimated_bytes = estimated_cache_bytes
            args.train_batch_cache_bytes = actual_cache_bytes
            print(
                f"Cached {len(cached_training_batches)} training batches in "
                f"{time.perf_counter() - cache_started_at:.2f}s "
                f"({actual_cache_bytes / (1024**3):.2f} GiB)"
            )

        config_path = f"{save_path}/1_training_config.json"
        config_dict = {k: str(v) if isinstance(v, (np.ndarray, torch.Tensor)) else v for k, v in vars(args).items()}
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

        visualize, clustering = (
            getattr(args, "visualize", True),
            getattr(args, "clustering", True),
        )
        print_freq, checkpoint_freq = (
            getattr(args, "print_freq", 10),
            getattr(args, "checkpoint_freq", 20),
        )
        eval_freq = getattr(args, "eval_freq", 0)
        eval_at_start = getattr(args, "eval_at_start", False)
        save_best_only, early_stopping = (
            getattr(args, "save_best_only", False),
            getattr(args, "early_stopping", 0),
        )

        for lr in args.lrs:
            print(f"Starting training with lr: {lr}")
            experiment_base_name = os.path.basename(save_path)

            feature_dim = 16
            try:
                feature_dim = original_graphs[0].x.shape[1]
                print(f"Detected feature_dim: {feature_dim}")
            except (IndexError, AttributeError):
                pass

            base_encoder = gcl.GATEncoder(in_channels=feature_dim, hidden_channels=64, out_channels=128).to(device)
            model = gcl.MoCoMultiPositive(base_encoder, dim=128, K=args.k, m=0.999, T=args.temperature).to(device)

            if getattr(args, "spectral_loss", False):
                model.k_neighbors, model.sigma = args.k_neighbors, args.sigma
                torch.autograd.set_detect_anomaly(True)

            model.weighted_recon_loss = getattr(args, "weighted", False)
            configure_model_optimizations(model, args)
            optimizer = torch.optim.Adam(
                list(model.encoder_q.parameters()),
                lr=lr,
                weight_decay=getattr(args, "weight_decay", 1e-5),
            )

            lr_scheduler_type = getattr(args, "lr_scheduler", "plateau").lower()
            if lr_scheduler_type == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=args.lr_patience,
                    min_lr=1e-6,
                )
            else:
                scheduler = None

            best_metrics = {}
            if clustering:
                methods = [
                    "KMeans",
                    "Agglomerative",
                    "SpectralClustering",
                    "GaussianMixture",
                ]
                best_metrics = {
                    k: {m: {"best_epoch": 0, "metrics": None} for m in methods}
                    for k in ["basic", "scaler", "pca", "select"]
                }

            early_stop_counter, best_loss = 0, float("inf")
            training_history = []

            if eval_at_start:
                model.eval()
                with torch.no_grad():
                    _, initial_figures = evaluate_and_visualize(
                        model,
                        original_graphs,
                        device,
                        save_path,
                        0,
                        lr,
                        args,
                        visualize=visualize,
                        clustering=clustering,
                        specific_label_file=args.label_file,
                    )
                _close_figures(initial_figures)

            for epoch in range(1, args.num_epoch + 1):
                epoch_started_at = time.perf_counter()
                losses = train_epoch(
                    model,
                    original_graphs,
                    augmented_graphs,
                    positive_samples,
                    optimizer,
                    device,
                    args,
                    epoch,
                    prepared_batches=cached_training_batches,
                )
                if scheduler:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(losses["total_loss"])
                    else:
                        scheduler.step()
                training_history.append(
                    {
                        "epoch": epoch,
                        **losses,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch_seconds": time.perf_counter() - epoch_started_at,
                    }
                )

                if epoch % print_freq == 0:
                    print(
                        f"Epoch [{epoch}/{args.num_epoch}], Loss: {losses['total_loss']:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )

                stop_training = False
                if early_stopping > 0:
                    if losses["total_loss"] < best_loss:
                        best_loss, early_stop_counter = losses["total_loss"], 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= early_stopping:
                            stop_training = True

                should_save_checkpoint = epoch % checkpoint_freq == 0 or epoch == args.num_epoch or stop_training
                should_evaluate = (
                    epoch == args.num_epoch or (eval_freq > 0 and epoch % eval_freq == 0) or stop_training
                )

                if should_save_checkpoint:
                    if not save_best_only:
                        torch.save(
                            {"epoch": epoch, "model_state_dict": model.state_dict()},
                            os.path.join(save_path, f"epoch_{epoch}_lr_{lr}_checkpoint.pth"),
                        )

                if should_evaluate:
                    model.eval()
                    with torch.no_grad():
                        current_metrics, current_figs = evaluate_and_visualize(
                            model,
                            original_graphs,
                            device,
                            save_path,
                            epoch,
                            lr,
                            args,
                            visualize=visualize,
                            clustering=clustering,
                            specific_label_file=args.label_file,
                        )

                    best_model_found = False
                    if clustering:
                        for vis_method, vis_results in current_metrics.items():
                            for cluster_method, cluster_results in vis_results.items():
                                if (
                                    best_metrics[vis_method][cluster_method]["metrics"] is None
                                    or cluster_results["F1-Score"]
                                    > best_metrics[vis_method][cluster_method]["metrics"]["F1-Score"]
                                ):
                                    best_metrics[vis_method][cluster_method].update(
                                        {
                                            "best_epoch": epoch,
                                            "metrics": cluster_results,
                                        }
                                    )
                                    best_model_found = True
                                    if visualize and vis_method in current_figs and current_figs[vis_method]:
                                        current_figs[vis_method].savefig(
                                            f"{save_path}/best_{vis_method}_{cluster_method}_lr{lr}.png",
                                            bbox_inches="tight",
                                        )
                    elif losses["total_loss"] < best_loss:
                        best_loss, best_model_found = losses["total_loss"], True

                    if save_best_only and best_model_found:
                        torch.save(
                            {"model_state_dict": model.state_dict()},
                            os.path.join(save_path, f"best_model_epoch_{epoch}_lr_{lr}.pth"),
                        )
                    _close_figures(current_figs)
                if stop_training:
                    break

            pd.DataFrame(training_history).to_csv(
                f"{save_path}/training_history_lr{lr}.csv",
                index=False,
            )
            if clustering:
                with open(f"{save_path}/best_metrics_lr{lr}.json", "w") as f:
                    json.dump(convert_to_serializable(best_metrics), f, indent=4)
            print(f"Completed training for lr: {lr}")

        with open(f"{save_path}/ALL_COMPLETED.txt", "w") as f:
            f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        if save_path and os.path.isdir(save_path):
            shutil.rmtree(save_path)


def evaluate_and_visualize(
    model,
    original_graphs,
    device,
    save_path,
    epoch,
    lr,
    args,
    visualize=True,
    clustering=True,
    specific_label_file=None,
    tsne_perplexity=None,
    umap_n_neighbors=None,
    umap_min_dist=None,
):
    if tsne_perplexity is None:
        tsne_perplexity = getattr(args, "tsne_perplexity", 30.0)
    if umap_n_neighbors is None:
        umap_n_neighbors = getattr(args, "umap_n_neighbors", 15)
    if umap_min_dist is None:
        umap_min_dist = getattr(args, "umap_min_dist", 0.1)

    model.eval()
    graph_representations = []
    embedding_batch_size = max(1, int(getattr(args, "batch_size", 64)))
    with torch.no_grad():
        for graph_offset in range(0, len(original_graphs), embedding_batch_size):
            source_graphs = original_graphs[graph_offset : graph_offset + embedding_batch_size]
            graph_batch = Batch.from_data_list(source_graphs)
            graph_batch = graph_batch.to(device)
            _, representations = model.encoder_q(
                graph_batch.x,
                graph_batch.edge_index,
                batch=graph_batch.batch,
            )
            representation_rows = representations.cpu().numpy().tolist()
            graph_representations.extend(
                representation + [graph.cell, graph.gene]
                for representation, graph in zip(representation_rows, source_graphs)
            )

    if not graph_representations:
        return {}, {}

    df = pd.DataFrame(
        graph_representations,
        columns=pd.Index([f"feature_{i + 1}" for i in range(len(graph_representations[0]) - 2)] + ["cell", "gene"]),
    )
    df.to_csv(f"{save_path}/epoch{epoch}_lr{lr}_embedding.csv", index=False)

    if not clustering:
        if visualize:
            figure = vis.plot_embeddings_only(df, save_path, epoch, lr, visualize=True)
            _close_figures(figure)
        return {}, {}

    all_metrics, figures_dict = vis.evaluate_and_visualize(
        dataset=args.dataset,
        df=df,
        save_path=save_path,
        num_epochs=epoch,
        lr=lr,
        n_clusters=args.num_clusters,
        visualize=visualize,
        clustering_methods=args.clustering_methods,
        specific_label_file=specific_label_file,
        tsne_perplexity=tsne_perplexity,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_metrics, figures_dict


def _close_figures(figures) -> None:
    if isinstance(figures, dict):
        for figure in figures.values():
            _close_figures(figure)
        return
    if isinstance(figures, (list, tuple)):
        for figure in figures:
            _close_figures(figure)
        return
    if figures is not None:
        plt.close(figures)


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    try:
        args = parse_args()
    except SystemExit as e:
        # argparse uses SystemExit for --help/-h and parse errors.
        code = getattr(e, "code", 0)
        return int(code) if isinstance(code, int) else 0

    try:
        _lazy_import_training_deps()
    except ModuleNotFoundError as e:
        print(str(e))
        return 1
    set_seed(args.seed)
    train_model(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
