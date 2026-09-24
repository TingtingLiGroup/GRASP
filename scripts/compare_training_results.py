#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Compare baseline and optimized GRASP runs against baseline seed-to-seed " "variation.")
    )
    parser.add_argument("--baseline_runs", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate_runs", type=Path, nargs="+", required=True)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def find_single_result(run_dir: Path, pattern: str) -> Path:
    matches = list(run_dir.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one result matching {pattern!r} under {run_dir}, " f"found {len(matches)}")
    return matches[0]


def load_embedding(run_dir: Path, epoch: int, lr: float) -> pd.DataFrame:
    path = find_single_result(run_dir, f"epoch{epoch}_lr{lr}_embedding.csv")
    frame = pd.read_csv(path)
    feature_columns = [column for column in frame.columns if column.startswith("feature_")]
    if not feature_columns or not {"cell", "gene"}.issubset(frame.columns):
        raise ValueError(f"Unexpected embedding columns in {path}")
    if frame.duplicated(["cell", "gene"]).any():
        raise ValueError(f"Duplicate cell/gene rows in {path}")
    return frame.set_index(["cell", "gene"]).sort_index()


def load_history(run_dir: Path, lr: float, max_epoch: int) -> pd.DataFrame:
    path = find_single_result(run_dir, f"training_history_lr{lr}.csv")
    history = pd.read_csv(path)
    required_columns = {"epoch", "total_loss"}
    if not required_columns.issubset(history.columns):
        raise ValueError(f"Unexpected training history columns in {path}")
    history = history.loc[history["epoch"] <= max_epoch].copy()
    if history.empty:
        raise ValueError(f"No history rows through epoch {max_epoch} in {path}")
    return history


def centered_linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    cross_covariance = left_centered.T @ right_centered
    left_covariance = left_centered.T @ left_centered
    right_covariance = right_centered.T @ right_centered
    denominator = np.linalg.norm(left_covariance) * np.linalg.norm(right_covariance)
    if denominator == 0:
        return 0.0
    return float(np.linalg.norm(cross_covariance) ** 2 / denominator)


def neighbor_indices(values: np.ndarray, neighbors: int) -> np.ndarray:
    neighbor_count = min(neighbors + 1, len(values))
    model = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    indices = model.fit(values).kneighbors(return_distance=False)
    return indices[:, 1:]


def mean_neighbor_overlap(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[1] == 0:
        return 1.0
    overlaps = [len(set(left_row).intersection(right_row)) / left.shape[1] for left_row, right_row in zip(left, right)]
    return float(np.mean(overlaps))


def loss_curve_metrics(left: pd.DataFrame, right: pd.DataFrame) -> Dict[str, float]:
    merged = left[["epoch", "total_loss"]].merge(
        right[["epoch", "total_loss"]],
        on="epoch",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    left_loss = merged["total_loss_left"].to_numpy()
    right_loss = merged["total_loss_right"].to_numpy()
    scale = max(float(np.ptp(left_loss)), abs(float(left_loss[-1])), 1e-12)
    return {
        "loss_correlation": float(np.corrcoef(left_loss, right_loss)[0, 1]),
        "normalized_loss_rmse": float(np.sqrt(np.mean((left_loss - right_loss) ** 2)) / scale),
        "final_loss_relative_difference": float(abs(left_loss[-1] - right_loss[-1]) / max(abs(left_loss[-1]), 1e-12)),
    }


def pair_metrics(
    left_embedding: np.ndarray,
    right_embedding: np.ndarray,
    left_neighbors: np.ndarray,
    right_neighbors: np.ndarray,
    left_clusters: np.ndarray,
    right_clusters: np.ndarray,
    left_history: pd.DataFrame,
    right_history: pd.DataFrame,
) -> Dict[str, float]:
    return {
        "linear_cka": centered_linear_cka(left_embedding, right_embedding),
        "knn_overlap": mean_neighbor_overlap(left_neighbors, right_neighbors),
        "ari": float(adjusted_rand_score(left_clusters, right_clusters)),
        "nmi": float(normalized_mutual_info_score(left_clusters, right_clusters)),
        **loss_curve_metrics(left_history, right_history),
    }


def compare_prepared_runs(
    left_run: Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    right_run: Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
) -> Dict[str, float]:
    left_embedding, left_neighbors, left_clusters, left_history = left_run
    right_embedding, right_neighbors, right_clusters, right_history = right_run
    return pair_metrics(
        left_embedding,
        right_embedding,
        left_neighbors,
        right_neighbors,
        left_clusters,
        right_clusters,
        left_history,
        right_history,
    )


def prepare_runs(
    run_dirs: Sequence[Path],
    *,
    epoch: int,
    lr: float,
    sample_index: pd.MultiIndex,
    neighbors: int,
    clusters: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]]:
    prepared = []
    for run_dir in run_dirs:
        frame = load_embedding(run_dir, epoch, lr)
        if not sample_index.isin(frame.index).all():
            raise ValueError(f"Embedding rows do not align for {run_dir}")
        values = frame.loc[sample_index].filter(like="feature_").to_numpy(dtype=np.float64)
        neighbor_values = neighbor_indices(values, neighbors)
        cluster_values = KMeans(
            n_clusters=clusters,
            random_state=0,
            n_init=10,
        ).fit_predict(values)
        prepared.append(
            (
                values,
                neighbor_values,
                cluster_values,
                load_history(run_dir, lr, max_epoch=epoch),
            )
        )
    return prepared


def metric_extrema(pair_results: Sequence[Dict[str, float]]) -> Dict[str, float]:
    similarity_metrics = ("linear_cka", "knn_overlap", "ari", "nmi", "loss_correlation")
    distance_metrics = ("normalized_loss_rmse", "final_loss_relative_difference")
    extrema = {f"minimum_{metric}": min(result[metric] for result in pair_results) for metric in similarity_metrics}
    extrema.update(
        {f"maximum_{metric}": max(result[metric] for result in pair_results) for metric in distance_metrics}
    )
    return extrema


def passes_baseline_variation_gate(
    candidate_results: Sequence[Dict[str, float]],
    baseline_extrema: Dict[str, float],
) -> Tuple[bool, Dict[str, bool]]:
    checks = {}
    for metric in ("linear_cka", "knn_overlap", "ari", "nmi", "loss_correlation"):
        checks[metric] = min(result[metric] for result in candidate_results) >= baseline_extrema[f"minimum_{metric}"]
    for metric in ("normalized_loss_rmse", "final_loss_relative_difference"):
        checks[metric] = max(result[metric] for result in candidate_results) <= baseline_extrema[f"maximum_{metric}"]
    return all(checks.values()), checks


def main() -> int:
    args = build_parser().parse_args()
    if len(args.baseline_runs) < 2:
        raise ValueError("At least two baseline runs are required")
    if len(args.candidate_runs) != len(args.baseline_runs):
        raise ValueError("Baseline and candidate run counts must match")
    reference = load_embedding(args.baseline_runs[0], args.epoch, args.lr)
    sample_size = min(args.sample_size, len(reference))
    rng = np.random.default_rng(args.seed)
    sample_positions = np.sort(rng.choice(len(reference), size=sample_size, replace=False))
    sample_index = reference.index[sample_positions]

    baseline_runs = prepare_runs(
        args.baseline_runs,
        epoch=args.epoch,
        lr=args.lr,
        sample_index=sample_index,
        neighbors=args.neighbors,
        clusters=args.clusters,
    )
    candidate_runs = prepare_runs(
        args.candidate_runs,
        epoch=args.epoch,
        lr=args.lr,
        sample_index=sample_index,
        neighbors=args.neighbors,
        clusters=args.clusters,
    )

    baseline_pairs = []
    for left_index in range(len(baseline_runs)):
        for right_index in range(left_index + 1, len(baseline_runs)):
            baseline_pairs.append(
                {
                    "left": str(args.baseline_runs[left_index]),
                    "right": str(args.baseline_runs[right_index]),
                    **compare_prepared_runs(
                        baseline_runs[left_index],
                        baseline_runs[right_index],
                    ),
                }
            )
    matched_candidate_pairs = [
        {
            "baseline": str(baseline_dir),
            "candidate": str(candidate_dir),
            **compare_prepared_runs(baseline_run, candidate_run),
        }
        for baseline_dir, candidate_dir, baseline_run, candidate_run in zip(
            args.baseline_runs,
            args.candidate_runs,
            baseline_runs,
            candidate_runs,
        )
    ]
    baseline_extrema = metric_extrema(baseline_pairs)
    passed, checks = passes_baseline_variation_gate(
        matched_candidate_pairs,
        baseline_extrema,
    )
    summary = {
        "sample_size": sample_size,
        "neighbors": args.neighbors,
        "clusters": args.clusters,
        "baseline_seed_pairs": baseline_pairs,
        "matched_candidate_pairs": matched_candidate_pairs,
        "baseline_variation_gate": baseline_extrema,
        "gate_checks": checks,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Comparison summary: {args.output.resolve()}")
    print(f"Passed baseline-variation gate: {passed}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
