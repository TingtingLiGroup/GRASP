from __future__ import annotations

import copy
import pickle
import random
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from grasp_tool.cli import train_moco
from grasp_tool.cli.main import (
    _run_partition_graphs,
    build_parser as build_cli_parser,
)
from grasp_tool.gnn.gat_moco_final import GATEncoder, MoCoMultiPositive
from grasp_tool.preprocessing.partition import (
    count_points_in_areas_same,
    save_node_data_to_csv,
)

launcher_spec = spec_from_file_location(
    "train_multi_lr",
    Path(__file__).resolve().parents[1] / "scripts" / "train_multi_lr.py",
)
assert launcher_spec is not None and launcher_spec.loader is not None
launcher_module = module_from_spec(launcher_spec)
launcher_spec.loader.exec_module(launcher_module)
build_train_command = launcher_module.build_train_command

comparison_spec = spec_from_file_location(
    "compare_training_results",
    Path(__file__).resolve().parents[1] / "scripts" / "compare_training_results.py",
)
assert comparison_spec is not None and comparison_spec.loader is not None
comparison_module = module_from_spec(comparison_spec)
comparison_spec.loader.exec_module(comparison_module)


def legacy_count_points_in_areas_same(df, n_sectors, m_rings, radius_limit):
    df = df.copy()
    df["theta"] = np.arctan2(df["y_c_s"], df["x_c_s"])
    df["radius"] = np.sqrt(df["x_c_s"] ** 2 + df["y_c_s"] ** 2)
    count_matrix = np.zeros((m_rings, n_sectors))
    theta_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    radius_edges = np.linspace(0, radius_limit, m_rings + 1)
    center_points = []
    point_counts = []
    is_virtual = []
    is_edge = []

    for ring_index in range(m_rings):
        for sector_index in range(n_sectors):
            points_in_ring = df[
                (df["radius"] > radius_edges[ring_index]) & (df["radius"] <= radius_edges[ring_index + 1])
            ]
            points_in_sector = points_in_ring[
                (points_in_ring["theta"] >= theta_edges[sector_index])
                & (points_in_ring["theta"] < theta_edges[sector_index + 1])
            ]
            count = len(points_in_sector)
            count_matrix[ring_index, sector_index] = count
            point_counts.append(count)
            theta_center = (theta_edges[sector_index] + theta_edges[sector_index + 1]) / 2
            radius_center = (radius_edges[ring_index] + radius_edges[ring_index + 1]) / 2
            center_points.append(
                (
                    radius_center * np.cos(theta_center),
                    radius_center * np.sin(theta_center),
                )
            )
            is_virtual.append(count == 0)
            is_edge.append(ring_index in (m_rings - 2, m_rings - 1))

    return count_matrix, center_points, point_counts, is_virtual, is_edge


def legacy_graph_matrices(center_points, is_virtual, k):
    num_nodes = len(center_points)
    distance_values = np.zeros((num_nodes, num_nodes))
    for source_index in range(num_nodes):
        for target_index in range(num_nodes):
            if source_index == target_index:
                distance_values[source_index, target_index] = 0
            elif is_virtual[source_index] or is_virtual[target_index]:
                distance_values[source_index, target_index] = 1e6
            else:
                distance_values[source_index, target_index] = np.linalg.norm(
                    np.array(center_points[source_index]) - np.array(center_points[target_index])
                )

    adjacency_values = np.zeros((num_nodes, num_nodes), dtype=int)
    distance_frame = pd.DataFrame(distance_values)
    for source_index in range(num_nodes):
        if is_virtual[source_index]:
            continue
        nearest_indices = np.argsort(distance_frame[source_index])[: k + 1]
        for target_index in nearest_indices:
            if not is_virtual[target_index]:
                adjacency_values[source_index, target_index] = 1
    np.fill_diagonal(adjacency_values, 0)
    return distance_values, adjacency_values


def legacy_generate_samples_js(
    original_graphs,
    gene_labels,
    cell_labels,
    num_positive,
    js_distances_df,
):
    positive_samples = []
    for query_index in range(len(original_graphs)):
        current_positives = [query_index]
        filtered_distances = js_distances_df[
            (js_distances_df["target_cell"] == cell_labels[query_index])
            & (js_distances_df["target_gene"] == gene_labels[query_index])
        ]
        closest_samples = filtered_distances.nsmallest(num_positive - 1, "js_distance")
        for _, row in closest_samples.iterrows():
            for candidate_index in range(len(original_graphs)):
                if cell_labels[candidate_index] == row["cell"] and gene_labels[candidate_index] == row["gene"]:
                    current_positives.append(candidate_index)
                    break
        while len(current_positives) < num_positive:
            current_positives.append(query_index)
        positive_samples.append((query_index, current_positives))
    return positive_samples


def test_vectorized_partition_matches_legacy_boundaries():
    coordinates = pd.DataFrame(
        {
            "x_c_s": [0.0, 1.0, 0.0, -1.0, -1.0, 2.0, 2.1, -1.0],
            "y_c_s": [0.0, 0.0, 1.0, 0.0, -0.0, 0.0, 0.0, 1.0],
        }
    )
    legacy = legacy_count_points_in_areas_same(coordinates, 4, 2, 2.0)
    optimized = count_points_in_areas_same(coordinates, 4, 2, 2.0)

    np.testing.assert_array_equal(optimized[0], legacy[0])
    np.testing.assert_allclose(optimized[1], legacy[1], rtol=0, atol=1e-15)
    assert optimized[2:] == legacy[2:]


def test_vectorized_graph_matrices_match_legacy(tmp_path):
    center_points = [
        (-1.0, 0.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 2.0),
        (0.0, -2.0),
    ]
    is_virtual = [False, False, False, True, False]
    is_edge = [False, False, False, True, True]
    expected_distance, expected_adjacency = legacy_graph_matrices(center_points, is_virtual, k=2)

    save_node_data_to_csv(
        center_points=center_points,
        is_virtual=is_virtual,
        is_edge=is_edge,
        plot_dir=str(tmp_path),
        gene="G",
        node_counts=[1, 2, 3, 0, 4],
        k=2,
        nuclear_positions=["inside", "inside", "outside", "edge", "edge"],
        write_distance_matrix=True,
    )

    actual_distance = pd.read_csv(tmp_path / "G_dis_matrix.csv").to_numpy()
    actual_adjacency = pd.read_csv(tmp_path / "G_adj_matrix.csv").to_numpy()
    np.testing.assert_allclose(actual_distance, expected_distance)
    np.testing.assert_array_equal(actual_adjacency, expected_adjacency)


def test_all_virtual_graph_has_no_edges(tmp_path):
    save_node_data_to_csv(
        center_points=[(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        is_virtual=[True, True, True],
        is_edge=[False, True, True],
        plot_dir=str(tmp_path),
        gene="virtual",
        node_counts=[0, 0, 0],
        k=2,
        nuclear_positions=["outside", "edge", "edge"],
        write_distance_matrix=False,
    )

    adjacency = pd.read_csv(tmp_path / "virtual_adj_matrix.csv").to_numpy()
    assert not adjacency.any()
    assert not (tmp_path / "virtual_dis_matrix.csv").exists()


@pytest.mark.parametrize("graph_count", [50, 200, 1000, 2500])
def test_indexed_js_samples_match_legacy(graph_count):
    original_graphs = [object()] * graph_count
    gene_labels = [f"gene_{index % 17}" for index in range(graph_count)]
    cell_labels = [f"cell_{index}" for index in range(graph_count)]
    rows = []
    for index in range(graph_count):
        target = {
            "target_cell": cell_labels[index],
            "target_gene": gene_labels[index],
        }
        candidates = [
            ("missing_cell", "missing_gene", 0.05),
            (
                cell_labels[(index + 1) % graph_count],
                gene_labels[(index + 1) % graph_count],
                0.1,
            ),
            (
                cell_labels[(index + 2) % graph_count],
                gene_labels[(index + 2) % graph_count],
                0.1,
            ),
            (
                cell_labels[(index + 3) % graph_count],
                gene_labels[(index + 3) % graph_count],
                0.2,
            ),
        ]
        rows.extend(
            {
                **target,
                "cell": cell,
                "gene": gene,
                "js_distance": distance,
            }
            for cell, gene, distance in candidates
        )
    js_distances = pd.DataFrame(rows)

    expected = legacy_generate_samples_js(
        original_graphs,
        gene_labels,
        cell_labels,
        num_positive=4,
        js_distances_df=js_distances,
    )
    actual = MoCoMultiPositive.generate_samples_js(
        original_graphs,
        original_graphs,
        gene_labels,
        cell_labels,
        num_positive=4,
        js_distances_df=js_distances,
    )
    assert actual == expected


class EncoderModel(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder_q = encoder


def test_batched_embeddings_match_single_graph_embeddings(tmp_path):
    torch.manual_seed(7)
    graphs = []
    for index, node_count in enumerate((3, 5, 4, 6, 2)):
        source = torch.arange(node_count, dtype=torch.long)
        target = torch.roll(source, shifts=-1)
        graphs.append(
            Data(
                x=torch.randn(node_count, 3),
                edge_index=torch.stack((source, target)),
                cell=f"cell_{index}",
                gene=f"gene_{index}",
            )
        )

    encoder = GATEncoder(3, 5, 4, dropout=0.0)
    model = EncoderModel(encoder).eval()
    with torch.no_grad():
        expected = np.stack([encoder(graph.x, graph.edge_index, batch=None)[1].numpy() for graph in graphs])

    train_moco._lazy_import_training_deps()
    args = SimpleNamespace(batch_size=2)
    rng_state_before_evaluation = torch.random.get_rng_state()
    train_moco.evaluate_and_visualize(
        model=model,
        original_graphs=graphs,
        device=torch.device("cpu"),
        save_path=str(tmp_path),
        epoch=3,
        lr=0.001,
        args=args,
        visualize=False,
        clustering=False,
    )
    assert torch.equal(torch.random.get_rng_state(), rng_state_before_evaluation)

    embedding_frame = pd.read_csv(tmp_path / "epoch3_lr0.001_embedding.csv")
    actual = embedding_frame.filter(like="feature_").to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert embedding_frame["cell"].tolist() == [graph.cell for graph in graphs]
    assert embedding_frame["gene"].tolist() == [graph.gene for graph in graphs]


def test_training_cli_defaults_to_final_embedding_only(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train-moco", "--dataset", "demo", "--pkl", "demo.pkl"],
    )
    args = train_moco.parse_args()
    assert args.visualize is False
    assert args.eval_at_start is False
    assert args.eval_freq == 0
    assert args.prefetch_batches == 0
    assert args.pin_prefetched_batches is False
    assert args.cache_train_batches is False
    assert args.pipeline_cached_h2d is False
    assert args.cached_h2d_prefetch_batches == 2
    assert args.foreach_momentum is False
    assert args.matmul_precision == "highest"
    assert args.fuse_positive_encoders is False
    assert args.num_positive == 4
    assert args.reconstruction_negative_ratio == 0


@pytest.mark.parametrize(
    ("graph_count", "expected_sizes"),
    [
        (1, []),
        (2, [2]),
        (63, [63]),
        (64, [64]),
        (65, [65]),
        (66, [64, 2]),
        (127, [64, 63]),
        (128, [64, 64]),
        (129, [64, 65]),
    ],
)
def test_training_batch_ranges_preserve_tail_semantics(graph_count, expected_sizes):
    batch_ranges = list(MoCoMultiPositive.iter_training_batch_ranges(graph_count, batch_size=64))
    assert [end - start for start, end in batch_ranges] == expected_sizes
    assert [start for start, _ in batch_ranges] == [
        sum(expected_sizes[:index]) for index in range(len(expected_sizes))
    ]


def _make_training_views(graph_count, fixed_node_count=None):
    original_graphs = []
    augmented_graphs = []
    for graph_index in range(graph_count):
        node_count = fixed_node_count if fixed_node_count is not None else graph_index % 4 + 2
        source = torch.arange(node_count, dtype=torch.long)
        target = torch.roll(source, shifts=-1)
        edge_index = torch.stack((source, target))
        base_features = torch.arange(node_count * 3, dtype=torch.float32).reshape(node_count, 3)
        original_graphs.append(
            Data(
                x=base_features + graph_index * 100,
                edge_index=edge_index,
                cell=f"cell_{graph_index}",
                gene=f"gene_{graph_index}",
            )
        )
        augmented_graphs.append(
            Data(
                x=base_features + graph_index * 100 + 10_000,
                edge_index=edge_index,
                cell=f"cell_{graph_index}",
                gene=f"gene_{graph_index}",
            )
        )
    positive_samples = [
        (
            graph_index,
            [
                graph_index,
                (graph_index + 1) % graph_count,
                (graph_index + 2) % graph_count,
                (graph_index + 3) % graph_count,
            ],
        )
        for graph_index in range(graph_count)
    ]
    return original_graphs, augmented_graphs, positive_samples


def _assert_batch_groups_equal(expected_group, actual_group):
    expected_query, expected_positives = expected_group
    actual_query, actual_positives = actual_group
    assert len(expected_positives) == len(actual_positives) == 4
    for expected_batch, actual_batch in zip(
        [expected_query, *expected_positives],
        [actual_query, *actual_positives],
    ):
        assert torch.equal(expected_batch.x, actual_batch.x)
        assert torch.equal(expected_batch.edge_index, actual_batch.edge_index)
        assert torch.equal(expected_batch.batch, actual_batch.batch)


def test_prefetch_preserves_five_views_order_and_rng_state():
    original_graphs, augmented_graphs, positive_samples = _make_training_views(131)
    sequential_batches = list(
        MoCoMultiPositive.prepare_multi_positive_batch(
            original_graphs,
            augmented_graphs,
            positive_samples,
            batch_size=64,
            prefetch_batches=0,
        )
    )

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    prefetched_batches = list(
        MoCoMultiPositive.prepare_multi_positive_batch(
            original_graphs,
            augmented_graphs,
            positive_samples,
            batch_size=64,
            prefetch_batches=2,
        )
    )

    assert len(sequential_batches) == len(prefetched_batches) == 3
    for sequential_group, prefetched_group in zip(sequential_batches, prefetched_batches):
        _assert_batch_groups_equal(sequential_group, prefetched_group)
        query_batch = prefetched_group[0]
        assert "cell" not in query_batch
        assert "gene" not in query_batch
    assert random.getstate() == python_state
    actual_numpy_state = np.random.get_state()
    assert actual_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_numpy_state[1], numpy_state[1])
    assert actual_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_immutable_batch_cache_matches_uncached_batches_and_estimate():
    original_graphs, augmented_graphs, positive_samples = _make_training_views(131)
    uncached_batches = list(
        MoCoMultiPositive.prepare_multi_positive_batch(
            original_graphs,
            augmented_graphs,
            positive_samples,
            batch_size=64,
            prefetch_batches=0,
        )
    )
    estimated_bytes = MoCoMultiPositive.estimate_cached_training_batch_bytes(
        original_graphs,
        augmented_graphs,
        positive_samples,
        batch_size=64,
    )
    cached_batches = MoCoMultiPositive.build_cached_training_batches(
        original_graphs,
        augmented_graphs,
        positive_samples,
        batch_size=64,
    )
    actual_bytes = MoCoMultiPositive.cached_training_batch_bytes(cached_batches)

    assert estimated_bytes == actual_bytes
    for uncached_group, cached_group in zip(uncached_batches, cached_batches):
        _assert_batch_groups_equal(uncached_group, cached_group)
    cached_query = cached_batches[0][0]
    transferred_query = cached_query.to(torch.device("cpu"))
    assert transferred_query is not cached_query
    assert cached_query.x.device.type == "cpu"


def test_training_batches_support_configurable_positive_count():
    original_graphs, augmented_graphs, positive_samples = _make_training_views(6)
    two_positive_samples = [(query_index, positive_indices[:2]) for query_index, positive_indices in positive_samples]

    query_batch, positive_batches = next(
        MoCoMultiPositive.prepare_multi_positive_batch(
            original_graphs,
            augmented_graphs,
            two_positive_samples,
            batch_size=2,
        )
    )

    assert query_batch.num_graphs == 2
    assert len(positive_batches) == 2
    assert torch.equal(
        positive_batches[0].x,
        torch.cat([augmented_graphs[0].x, augmented_graphs[1].x]),
    )


def test_sampled_reconstruction_uses_separate_rng_and_has_finite_gradients():
    torch.manual_seed(37)
    model = MoCoMultiPositive(
        GATEncoder(3, 5, 4, dropout=0.0),
        dim=4,
        K=16,
        m=0.999,
        T=0.07,
    )
    model.reconstruction_negative_ratio = 10
    model.reconstruction_sampling_seed = 71
    node_embeddings = torch.randn(6, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 4]])
    global_rng_state = torch.random.get_rng_state()
    dense_loss = model._compute_dense_basic_reconstruction_loss(
        node_embeddings.detach(),
        edge_index,
        num_nodes=6,
    )

    loss = model._compute_basic_reconstruction_loss(
        node_embeddings,
        edge_index,
        num_nodes=6,
    )
    loss.backward()

    assert torch.equal(torch.random.get_rng_state(), global_rng_state)
    assert torch.isfinite(loss)
    assert torch.isfinite(node_embeddings.grad).all()

    positive_flat_indices = torch.unique(edge_index[0] * 6 + edge_index[1])
    sampled_negatives = model._sample_negative_flat_indices(
        positive_flat_indices,
        total_pairs=36,
        sample_count=100,
        device=torch.device("cpu"),
    )
    assert not torch.isin(sampled_negatives, positive_flat_indices).any()

    model._reconstruction_generators.clear()
    model.reconstruction_negative_ratio = 5000
    high_sample_loss = model._compute_basic_reconstruction_loss(
        node_embeddings.detach(),
        edge_index,
        num_nodes=6,
    )
    assert high_sample_loss.item() == pytest.approx(
        dense_loss.item(),
        rel=0.03,
        abs=0.03,
    )


def test_fused_positive_encoder_path_preserves_output_contract():
    train_moco._lazy_import_training_deps()
    original_graphs, augmented_graphs, positive_samples = _make_training_views(
        4,
        fixed_node_count=3,
    )
    query_batch, positive_batches = next(
        MoCoMultiPositive.prepare_multi_positive_batch(
            original_graphs,
            augmented_graphs,
            positive_samples,
            batch_size=2,
        )
    )
    torch.manual_seed(41)
    model = MoCoMultiPositive(
        GATEncoder(3, 5, 4, dropout=0.0),
        dim=4,
        K=16,
        m=0.999,
        T=0.07,
    )
    model.weighted_recon_loss = False
    model.fuse_positive_encoders = True

    outputs = model(
        query_batch.x,
        [positive_batch.x for positive_batch in positive_batches],
        query_batch.edge_index,
        [positive_batch.edge_index for positive_batch in positive_batches],
        query_batch.batch,
        use_clustering=False,
    )

    assert len(outputs) == 7
    assert all(torch.isfinite(output) for output in outputs)
    assert model.queue_ptr.item() == query_batch.num_graphs


def test_training_result_comparison_metrics_are_identity_invariant():
    values = np.arange(24, dtype=np.float64).reshape(6, 4)
    neighbors = comparison_module.neighbor_indices(values, neighbors=2)

    assert comparison_module.centered_linear_cka(values, values) == pytest.approx(1.0)
    assert comparison_module.mean_neighbor_overlap(neighbors, neighbors) == 1.0

    clusters = np.array([0, 0, 1, 1, 2, 2])
    history = pd.DataFrame({"epoch": [1, 2], "total_loss": [2.0, 1.0]})
    prepared_run = (values, neighbors, clusters, history)
    metrics = comparison_module.compare_prepared_runs(prepared_run, prepared_run)
    assert metrics["linear_cka"] == pytest.approx(1.0)
    assert metrics["knn_overlap"] == 1.0
    assert metrics["ari"] == 1.0
    assert metrics["nmi"] == 1.0
    assert metrics["loss_correlation"] == pytest.approx(1.0)
    assert metrics["normalized_loss_rmse"] == 0.0
    assert metrics["final_loss_relative_difference"] == 0.0


def test_baseline_variation_gate_accepts_within_seed_spread():
    baseline_pairs = [
        {
            "linear_cka": 0.91,
            "knn_overlap": 0.62,
            "ari": 0.55,
            "nmi": 0.58,
            "loss_correlation": 0.97,
            "normalized_loss_rmse": 0.12,
            "final_loss_relative_difference": 0.08,
        },
        {
            "linear_cka": 0.88,
            "knn_overlap": 0.59,
            "ari": 0.51,
            "nmi": 0.54,
            "loss_correlation": 0.95,
            "normalized_loss_rmse": 0.15,
            "final_loss_relative_difference": 0.11,
        },
    ]
    candidate_pairs = [
        {
            "linear_cka": 0.89,
            "knn_overlap": 0.60,
            "ari": 0.52,
            "nmi": 0.55,
            "loss_correlation": 0.96,
            "normalized_loss_rmse": 0.14,
            "final_loss_relative_difference": 0.09,
        }
    ]
    extrema = comparison_module.metric_extrema(baseline_pairs)
    passed, checks = comparison_module.passes_baseline_variation_gate(candidate_pairs, extrema)
    assert passed
    assert all(checks.values())

    drifted_pairs = [
        {
            **candidate_pairs[0],
            "linear_cka": 0.80,
            "normalized_loss_rmse": 0.20,
        }
    ]
    drifted_passed, drifted_checks = comparison_module.passes_baseline_variation_gate(
        drifted_pairs,
        extrema,
    )
    assert drifted_passed is False
    assert drifted_checks["linear_cka"] is False
    assert drifted_checks["normalized_loss_rmse"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cached_cuda_pipeline_preserves_batch_order_and_tensor_values():
    original_graphs, augmented_graphs, positive_samples = _make_training_views(7, fixed_node_count=3)
    cached_batches = MoCoMultiPositive.build_cached_training_batches(
        original_graphs,
        augmented_graphs,
        positive_samples,
        batch_size=2,
    )
    expected_batches = [
        (
            query_batch.to(torch.device("cuda")),
            tuple(batch.to(torch.device("cuda")) for batch in positive_batches),
        )
        for query_batch, positive_batches in cached_batches
    ]
    torch_state = torch.random.get_rng_state()

    actual_batches = list(
        MoCoMultiPositive.prepare_cached_cuda_batches(
            cached_batches,
            torch.device("cuda"),
            prefetch_batches=2,
        )
    )
    torch.cuda.synchronize()

    assert len(actual_batches) == len(expected_batches)
    for expected_group, actual_group in zip(expected_batches, actual_batches):
        _assert_batch_groups_equal(expected_group, actual_group)
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_foreach_momentum_update_is_cpu_bitwise_equivalent():
    torch.manual_seed(31)
    initial_model = MoCoMultiPositive(
        GATEncoder(3, 5, 4, dropout=0.1),
        dim=4,
        K=16,
        m=0.999,
        T=0.07,
    )
    with torch.no_grad():
        for parameter in initial_model.encoder_q.parameters():
            parameter.add_(torch.randn_like(parameter))
        for parameter in initial_model.projector_q.parameters():
            parameter.add_(torch.randn_like(parameter))
    baseline_model = copy.deepcopy(initial_model)
    foreach_model = copy.deepcopy(initial_model)
    foreach_model.foreach_momentum = True
    torch_state = torch.random.get_rng_state()

    baseline_model._momentum_update_key_encoder()
    foreach_model._momentum_update_key_encoder()

    for baseline_parameter, foreach_parameter in zip(
        baseline_model.encoder_k.parameters(),
        foreach_model.encoder_k.parameters(),
    ):
        assert torch.equal(baseline_parameter, foreach_parameter)
    for baseline_parameter, foreach_parameter in zip(
        baseline_model.projector_k.parameters(),
        foreach_model.projector_k.parameters(),
    ):
        assert torch.equal(baseline_parameter, foreach_parameter)
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def _run_two_cuda_epochs(
    initial_model,
    original_graphs,
    augmented_graphs,
    positive_samples,
    cached_batches,
    *,
    pipeline_cached_h2d=False,
    foreach_momentum=False,
):
    model = copy.deepcopy(initial_model).cuda()
    model.foreach_momentum = foreach_momentum
    optimizer = torch.optim.Adam(
        model.encoder_q.parameters(),
        lr=0.001,
        weight_decay=1e-5,
    )
    args = SimpleNamespace(
        batch_size=2,
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
        profile_training=False,
        prefetch_batches=0,
        pin_prefetched_batches=False,
        pipeline_cached_h2d=pipeline_cached_h2d,
        cached_h2d_prefetch_batches=2,
        zero_grad_set_to_none=False,
    )
    losses = []
    for epoch in (1, 2):
        losses.append(
            train_moco.train_epoch(
                model,
                original_graphs,
                augmented_graphs,
                positive_samples,
                optimizer,
                torch.device("cuda"),
                args,
                epoch,
                prepared_batches=cached_batches,
            )
        )
    torch.cuda.synchronize()
    return losses, model.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "candidate_options",
    [
        {"pipeline_cached_h2d": True},
        {"foreach_momentum": True},
        {
            "pipeline_cached_h2d": True,
            "foreach_momentum": True,
        },
    ],
    ids=[
        "pipeline_cached_h2d",
        "foreach_momentum",
        "combined",
    ],
)
def test_cuda_optimizations_preserve_training_within_tolerance(
    candidate_options,
):
    train_moco._lazy_import_training_deps()
    original_graphs, augmented_graphs, positive_samples = _make_training_views(6, fixed_node_count=3)
    torch.manual_seed(43)
    initial_model = MoCoMultiPositive(
        GATEncoder(3, 5, 4, dropout=0.1),
        dim=4,
        K=16,
        m=0.999,
        T=0.07,
    )
    initial_model.weighted_recon_loss = False
    cached_batches = MoCoMultiPositive.build_cached_training_batches(
        original_graphs,
        augmented_graphs,
        positive_samples,
        batch_size=2,
    )
    cuda_rng_state = torch.cuda.get_rng_state()

    torch.cuda.set_rng_state(cuda_rng_state)
    baseline_losses, baseline_state = _run_two_cuda_epochs(
        initial_model,
        original_graphs,
        augmented_graphs,
        positive_samples,
        cached_batches,
    )
    torch.cuda.set_rng_state(cuda_rng_state)
    optimized_losses, optimized_state = _run_two_cuda_epochs(
        initial_model,
        original_graphs,
        augmented_graphs,
        positive_samples,
        cached_batches,
        **candidate_options,
    )

    for baseline_epoch, optimized_epoch in zip(baseline_losses, optimized_losses):
        for loss_name, baseline_loss in baseline_epoch.items():
            assert optimized_epoch[loss_name] == pytest.approx(
                baseline_loss,
                rel=1e-6,
                abs=1e-6,
            )
    for parameter_name, baseline_tensor in baseline_state.items():
        torch.testing.assert_close(
            optimized_state[parameter_name],
            baseline_tensor,
            rtol=1e-6,
            atol=1e-6,
            msg=lambda message: f"{parameter_name}: {message}",
        )


def _run_two_cpu_epochs(
    initial_model,
    original_graphs,
    augmented_graphs,
    positive_samples,
    *,
    prefetch_batches,
    zero_grad_set_to_none,
    cached_batches=None,
):
    model = copy.deepcopy(initial_model)
    optimizer = torch.optim.Adam(
        list(model.encoder_q.parameters()),
        lr=0.001,
        weight_decay=1e-5,
    )
    args = SimpleNamespace(
        batch_size=2,
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
        profile_training=False,
        prefetch_batches=prefetch_batches,
        pin_prefetched_batches=False,
        zero_grad_set_to_none=zero_grad_set_to_none,
    )
    losses = []
    for epoch in (1, 2):
        losses.append(
            train_moco.train_epoch(
                model,
                original_graphs,
                augmented_graphs,
                positive_samples,
                optimizer,
                torch.device("cpu"),
                args,
                epoch,
                prepared_batches=cached_batches,
            )
        )
    return losses, model.state_dict()


def test_cpu_two_epoch_training_is_exact_with_prefetch_cache_and_zero_grad():
    train_moco._lazy_import_training_deps()
    original_graphs, augmented_graphs, positive_samples = _make_training_views(6, fixed_node_count=3)
    torch.manual_seed(29)
    initial_model = MoCoMultiPositive(
        GATEncoder(3, 5, 4, dropout=0.1),
        dim=4,
        K=16,
        m=0.999,
        T=0.07,
    )
    initial_model.weighted_recon_loss = False
    training_rng_state = torch.random.get_rng_state()
    cached_batches = MoCoMultiPositive.build_cached_training_batches(
        original_graphs,
        augmented_graphs,
        positive_samples,
        batch_size=2,
    )

    torch.random.set_rng_state(training_rng_state)
    baseline_losses, baseline_state = _run_two_cpu_epochs(
        initial_model,
        original_graphs,
        augmented_graphs,
        positive_samples,
        prefetch_batches=0,
        zero_grad_set_to_none=False,
    )
    torch.random.set_rng_state(training_rng_state)
    prefetched_losses, prefetched_state = _run_two_cpu_epochs(
        initial_model,
        original_graphs,
        augmented_graphs,
        positive_samples,
        prefetch_batches=2,
        zero_grad_set_to_none=True,
    )
    torch.random.set_rng_state(training_rng_state)
    cached_losses, cached_state = _run_two_cpu_epochs(
        initial_model,
        original_graphs,
        augmented_graphs,
        positive_samples,
        prefetch_batches=0,
        zero_grad_set_to_none=True,
        cached_batches=cached_batches,
    )

    assert prefetched_losses == baseline_losses
    assert cached_losses == baseline_losses
    for parameter_name, baseline_tensor in baseline_state.items():
        assert torch.equal(prefetched_state[parameter_name], baseline_tensor), parameter_name
        assert torch.equal(cached_state[parameter_name], baseline_tensor), parameter_name


def test_partition_cli_skips_legacy_distance_matrix_by_default():
    args = build_cli_parser().parse_args(
        [
            "partition-graphs",
            "--pkl",
            "registered.pkl",
            "--graph_root",
            "graphs",
        ]
    )
    assert args.write_distance_matrix == 0


def test_partition_cli_filters_to_pairs_csv(tmp_path):
    registered = pd.DataFrame(
        {
            "cell": ["cell_a"] * 8,
            "gene": ["keep"] * 4 + ["drop"] * 4,
            "x_c_s": [0.1, -0.1, 0.2, -0.2] * 2,
            "y_c_s": [0.2, -0.2, -0.1, 0.1] * 2,
        }
    )
    registered_path = tmp_path / "registered.pkl"
    with registered_path.open("wb") as handle:
        pickle.dump(
            {"df_registered": registered, "cell_radii": {"cell_a": 1.0}},
            handle,
        )
    pairs_path = tmp_path / "pairs.csv"
    pd.DataFrame({"cell": ["cell_a"], "gene": ["keep"]}).to_csv(
        pairs_path,
        index=False,
    )
    graph_root = tmp_path / "graphs"

    assert (
        _run_partition_graphs(
            pkl_path=str(registered_path),
            graph_root=str(graph_root),
            pairs_csv=str(pairs_path),
            n_sectors=2,
            m_rings=2,
            k_neighbor=2,
            processes=1,
            fixed_radius=1.0,
            epsilon=0.1,
            cells_arg=None,
            genes_arg=None,
            write_distance_matrix=False,
        )
        == 0
    )
    assert (graph_root / "cell_a/keep_node_matrix.csv").is_file()
    assert not (graph_root / "cell_a/drop_node_matrix.csv").exists()


def test_multi_gpu_launcher_manages_per_run_options(tmp_path):
    command = build_train_command(
        learning_rate=0.002,
        cuda_device=3,
        output_dir=tmp_path / "lr_0p002",
        train_args=["--", "--dataset", "demo", "--pkl", "train.pkl"],
    )
    assert command[-6:] == [
        "--lrs",
        "0.002",
        "--cuda_device",
        "3",
        "--output_dir",
        str(tmp_path / "lr_0p002"),
    ]
