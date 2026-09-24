from __future__ import annotations

import pickle

import pandas as pd

from grasp_tool.gnn.gat_moco_final import MoCoMultiPositive
from grasp_tool.preprocessing.portrait import (
    _build_pair_indices_by_gene,
    analyze_transcript_distribution,
    calculate_js_distances,
    precompute_portraits_for_gene,
)


def _sparse_categorical_frame() -> pd.DataFrame:
    coordinates = {
        ("c0", "g0"): [(0.0, 0.0), (0.10, 0.0)],
        ("c1", "g0"): [(0.0, 0.0), (0.10, 0.0)],
        ("c2", "g0"): [(0.0, 0.0), (0.10, 0.0), (0.0, 0.10)],
        ("c0", "g1"): [(0.0, 0.0), (0.20, 0.0)],
        ("c1", "g1"): [(0.0, 0.0), (0.20, 0.0), (0.0, 0.20)],
    }
    rows = [
        {"cell": cell, "gene": gene, "x_c_s": x, "y_c_s": y}
        for (cell, gene), points in coordinates.items()
        for x, y in points
    ]
    frame = pd.DataFrame(rows)
    frame["cell"] = pd.Categorical(
        frame["cell"],
        categories=["c0", "c1", "c2", "c_unused"],
    )
    frame["gene"] = pd.Categorical(
        frame["gene"],
        categories=["g0", "g1", "g_unused"],
    )
    return frame


def _sorted_distances(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["target_gene", "target_cell", "cell"],
        kind="stable",
    ).reset_index(drop=True)


def test_sparse_categorical_diagnostics_count_only_observed_pairs():
    stats = analyze_transcript_distribution(_sparse_categorical_frame())

    assert stats["gene_stats"] == {
        "total_genes": 2,
        "transcript_per_gene_mean": 6.0,
        "transcript_per_gene_median": 6.0,
        "transcript_per_gene_std": 2**0.5,
        "transcript_per_gene_min": 5,
        "transcript_per_gene_max": 7,
    }
    assert stats["cell_stats"] == {
        "total_cells": 3,
        "transcript_per_cell_mean": 4.0,
        "transcript_per_cell_median": 4.0,
        "transcript_per_cell_std": 1.0,
        "transcript_per_cell_min": 3,
        "transcript_per_cell_max": 5,
    }
    assert stats["pair_stats"] == {
        "total_cell_gene_pairs": 5,
        "transcript_per_pair_mean": 2.4,
        "transcript_per_pair_median": 2.0,
        "transcript_per_pair_std": 0.3**0.5,
        "single_transcript_pairs": 0,
        "multi_transcript_pairs": 5,
    }


def test_indexed_gene_portraits_match_legacy_iteration_exactly():
    frame = _sparse_categorical_frame()
    cell_list = sorted(frame["cell"].unique())
    pair_indices_by_gene = _build_pair_indices_by_gene(frame)
    parameters = {
        "threshold": 0.05,
        "bin_size": 0.01,
        "r_min": 0.01,
        "r_max": 0.6,
        "r_step": 0.03,
        "use_same_r": False,
        "use_vectorized": True,
    }

    for gene in sorted(frame["gene"].unique()):
        legacy = precompute_portraits_for_gene(
            gene,
            frame,
            cell_list,
            **parameters,
        )
        indexed = precompute_portraits_for_gene(
            gene,
            frame,
            cell_list,
            **parameters,
            pair_indices=pair_indices_by_gene[gene],
        )
        assert indexed == legacy


def test_portrait_indexing_preserves_legacy_csv_and_positive_indices(tmp_path):
    input_path = tmp_path / "registered.pkl"
    output_dir = tmp_path / "portrait"
    with input_path.open("wb") as handle:
        pickle.dump({"df_registered": _sparse_categorical_frame()}, handle)

    actual = calculate_js_distances(
        pkl_file=str(input_path),
        output_dir=str(output_dir),
        max_count=2,
        transcript_window=30,
        bin_size=0.01,
        threshold=0.05,
        r_min=0.01,
        r_max=0.6,
        r_step=0.03,
        num_threads=1,
        use_same_r=False,
        visualize_top_n=0,
        use_vectorized=True,
    )
    expected = pd.DataFrame(
        [
            ("c0", "g0", "c1", "g0", 2, 0.0, 0, 0.1, 0.1),
            ("c0", "g0", "c2", "g0", 3, 0.3728780188096281, 1, 0.1, 0.1),
            ("c1", "g0", "c0", "g0", 2, 0.0, 0, 0.1, 0.1),
            ("c1", "g0", "c2", "g0", 3, 0.3728780188096281, 1, 0.1, 0.1),
            ("c2", "g0", "c0", "g0", 2, 0.3728780188096281, 1, 0.1, 0.1),
            ("c2", "g0", "c1", "g0", 2, 0.3728780188096281, 1, 0.1, 0.1),
            ("c0", "g1", "c1", "g1", 3, 0.3728780188096281, 1, 0.2, 0.2),
            ("c1", "g1", "c0", "g1", 2, 0.3728780188096281, 1, 0.2, 0.2),
        ],
        columns=[
            "target_cell",
            "target_gene",
            "cell",
            "gene",
            "num_transcripts",
            "js_distance",
            "transcript_diff",
            "target_r",
            "other_r",
        ],
    )
    pd.testing.assert_frame_equal(
        _sorted_distances(actual),
        expected,
        check_exact=True,
    )

    graph_pairs = [
        ("c0", "g0"),
        ("c1", "g0"),
        ("c2", "g0"),
        ("c0", "g1"),
        ("c1", "g1"),
    ]
    positive_samples = MoCoMultiPositive.generate_samples_js(
        original_graphs=[object()] * len(graph_pairs),
        augmented_graphs=[object()] * len(graph_pairs),
        gene_labels=[gene for _, gene in graph_pairs],
        cell_labels=[cell for cell, _ in graph_pairs],
        num_positive=3,
        js_distances_df=actual,
    )
    assert positive_samples == [
        (0, [0, 1, 2]),
        (1, [1, 0, 2]),
        (2, [2, 0, 1]),
        (3, [3, 4, 3]),
        (4, [4, 3, 4]),
    ]
