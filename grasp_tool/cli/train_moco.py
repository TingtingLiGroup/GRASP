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

    global torch, ReduceLROnPlateau, gcl, vis

    try:
        torch = importlib.import_module("torch")
        torch_geometric_loader = importlib.import_module("torch_geometric.loader")
        _ = getattr(torch_geometric_loader, "DataLoader")
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
    parser = argparse.ArgumentParser(
        description="MoCo Training for Graph Neural Networks"
    )

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
    parser.add_argument(
        "--n", type=int, default=20, help="Number of sectors (n_sectors) (default: 20)"
    )
    parser.add_argument(
        "--m", type=int, default=10, help="Number of rings (m_rings) (default: 10)"
    )
    parser.add_argument(
        "--a", type=float, default=0.5, help="Reconstruction loss weight (default: 0.5)"
    )
    parser.add_argument(
        "--b", type=float, default=0.5, help="Contrastive loss weight (default: 0.5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Contrastive temperature (default: 0.07)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size (default: 64)"
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
    parser.add_argument(
        "--num_positive", type=int, default=4, help="Number of positives (default: 4)"
    )
    parser.add_argument(
        "--num_epoch", type=int, default=300, help="Number of epochs (default: 300)"
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=None,
        help="Enable clustering eval with this number of clusters (e.g., 5, 8)",
    )
    parser.add_argument(
        "--cuda_device", type=int, default=0, help="CUDA device index (default: 0)"
    )
    parser.add_argument(
        "--seed", type=int, default=2025, help="Random seed (default: 2025)"
    )
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

    args = parser.parse_args()

    if args.js == 1 and not args.js_file:
        parser.error("--js_file must be specified when using --js=1")

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
    args.visualize = True

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


def setup_training(args):
    device = torch.device(
        f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    )
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


def train_epoch(
    model: Any,
    original_graphs: List,
    augmented_graphs: List,
    positive_samples: List,
    optimizer: Any,
    device: Any,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    (
        total_loss,
        total_reconstruction_loss,
        total_contrastive_loss,
        total_clustering_loss,
    ) = 0.0, 0.0, 0.0, 0.0
    batch_count = 0

    use_clustering = getattr(args, "use_clustering", True)
    spectral_loss = getattr(args, "spectral_loss", False)
    dist_type = "spectral" if spectral_loss else args.dist_type
    forward_method = getattr(args, "forward_method", "default")

    batch_generator = gcl.MoCoMultiPositive.prepare_multi_positive_batch(
        original_graphs, augmented_graphs, positive_samples, args.batch_size
    )

    for query_batch, positive_batches in batch_generator:
        batch_count += 1
        query_batch = query_batch.to(device)
        positive_batches = [batch.to(device) for batch in positive_batches]
        im_q, edge_index_q, batch = (
            query_batch.x,
            query_batch.edge_index,
            query_batch.batch,
        )
        im_k_list = [pos_batch.x for pos_batch in positive_batches]
        edge_index_k_list = [pos_batch.edge_index for pos_batch in positive_batches]

        if forward_method == "supcon":
            loss, reconstruction_loss, contrastive_loss, clustering_loss, _, _, _ = (
                model.forward_supcon(
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
            )
        elif forward_method == "avg":
            loss, reconstruction_loss, contrastive_loss, clustering_loss, _, _, _ = (
                model.forward_avg(
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
            )
        else:
            loss, reconstruction_loss, contrastive_loss, clustering_loss, _, _, _ = (
                model(
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
            )

        optimizer.zero_grad()
        loss.backward()

        if hasattr(args, "use_gradient_clipping") and args.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.gradient_clip_norm
            )

        optimizer.step()
        total_loss += loss.item()
        total_reconstruction_loss += reconstruction_loss.item()
        total_contrastive_loss += contrastive_loss.item()
        total_clustering_loss += clustering_loss.item()

    if batch_count > 0:
        total_loss /= batch_count
        total_reconstruction_loss /= batch_count
        total_contrastive_loss /= batch_count
        total_clustering_loss /= batch_count

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

        config_path = f"{save_path}/1_training_config.json"
        config_dict = {
            k: str(v) if isinstance(v, (np.ndarray, torch.Tensor)) else v
            for k, v in vars(args).items()
        }
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

            base_encoder = gcl.GATEncoder(
                in_channels=feature_dim, hidden_channels=64, out_channels=128
            ).to(device)
            model = gcl.MoCoMultiPositive(
                base_encoder, dim=128, K=args.k, m=0.999, T=args.temperature
            ).to(device)

            if getattr(args, "spectral_loss", False):
                model.k_neighbors, model.sigma = args.k_neighbors, args.sigma
                torch.autograd.set_detect_anomaly(True)

            model.weighted_recon_loss = getattr(args, "weighted", False)
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

            # Epoch 0 evaluation
            model.eval()
            with torch.no_grad():
                evaluate_and_visualize(
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

            for epoch in range(1, args.num_epoch + 1):
                losses = train_epoch(
                    model,
                    original_graphs,
                    augmented_graphs,
                    positive_samples,
                    optimizer,
                    device,
                    args,
                    epoch,
                )
                if scheduler:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(losses["total_loss"])
                    else:
                        scheduler.step()

                if epoch % print_freq == 0:
                    print(
                        f"Epoch [{epoch}/{args.num_epoch}], Loss: {losses['total_loss']:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )

                should_save_checkpoint = (epoch % checkpoint_freq == 0) or (
                    epoch == args.num_epoch
                )
                if early_stopping > 0:
                    if losses["total_loss"] < best_loss:
                        best_loss, early_stop_counter = losses["total_loss"], 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= early_stopping:
                            break

                if should_save_checkpoint:
                    if not save_best_only:
                        torch.save(
                            {"epoch": epoch, "model_state_dict": model.state_dict()},
                            os.path.join(
                                save_path, f"epoch_{epoch}_lr_{lr}_checkpoint.pth"
                            ),
                        )

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
                                    best_metrics[vis_method][cluster_method]["metrics"]
                                    is None
                                    or cluster_results["F1-Score"]
                                    > best_metrics[vis_method][cluster_method][
                                        "metrics"
                                    ]["F1-Score"]
                                ):
                                    best_metrics[vis_method][cluster_method].update(
                                        {
                                            "best_epoch": epoch,
                                            "metrics": cluster_results,
                                        }
                                    )
                                    best_model_found = True
                                    if (
                                        visualize
                                        and vis_method in current_figs
                                        and current_figs[vis_method]
                                    ):
                                        current_figs[vis_method].savefig(
                                            f"{save_path}/best_{vis_method}_{cluster_method}_lr{lr}.png",
                                            bbox_inches="tight",
                                        )
                    elif losses["total_loss"] < best_loss:
                        best_loss, best_model_found = losses["total_loss"], True

                    if save_best_only and best_model_found:
                        torch.save(
                            {"model_state_dict": model.state_dict()},
                            os.path.join(
                                save_path, f"best_model_epoch_{epoch}_lr_{lr}.pth"
                            ),
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
    with torch.no_grad():
        for graph in original_graphs:
            try:
                graph = graph.to(device)
                _, rep = model.encoder_q(graph.x, graph.edge_index, batch=None)
                graph_representations.append(
                    rep.cpu().numpy().tolist() + [graph.cell, graph.gene]
                )
            except:
                continue

    if not graph_representations:
        return {}, {}

    df = pd.DataFrame(
        graph_representations,
        columns=pd.Index(
            [f"feature_{i + 1}" for i in range(len(graph_representations[0]) - 2)]
            + ["cell", "gene"]
        ),
    )
    df.to_csv(f"{save_path}/epoch{epoch}_lr{lr}_embedding.csv", index=False)

    if not clustering:
        if visualize:
            vis.plot_embeddings_only(df, save_path, epoch, lr, visualize=True)
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
