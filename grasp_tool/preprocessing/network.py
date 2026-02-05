"""Misc network/analysis helpers.

This module is not part of the main GRASP training pipeline. Keep imports minimal
so that `import grasp_tool` and other modules do not pull heavy optional deps.
"""

from __future__ import annotations

import os

import pandas as pd


def read_data(
    dataset,
    target_cells,
    df_registered,
    n_sectors,
    m_rings,
    k_neighbor,
    if_same,
    partition_root=None,
):
    """Locate per-cell graph files on disk.

    Historical code had hard-coded absolute paths. To keep this usable across
    machines, the root directory must be provided.

    Args:
        partition_root: Base directory containing partition outputs.
            If None, will use env var `GRASP_PARTITION_ROOT`.
    """
    if partition_root is None:
        partition_root = os.environ.get("GRASP_PARTITION_ROOT")
    if not partition_root:
        raise ValueError("partition_root is required (or set env GRASP_PARTITION_ROOT)")
    unique_genes = df_registered["gene"].unique()
    target_genes = unique_genes.tolist()
    # print(target_genes)
    matching_paths = []
    for cell in target_cells:
        if if_same == "yes":
            cell_dir = os.path.join(partition_root, f"4_{dataset}_partition_same", cell)
            if not os.path.exists(cell_dir):
                print(f"Directory {cell_dir} does not exist.")
                continue
        else:
            cell_dir = os.path.join(partition_root, f"4_{dataset}_partition", cell)
            if not os.path.exists(cell_dir):
                print(f"Directory {cell_dir} does not exist.")
                continue

        for root, dirs, files in os.walk(cell_dir):
            if f"{n_sectors}_{m_rings}_k{k_neighbor}" in root:
                for file in files:
                    gene_in_file = any(gene in file for gene in target_genes)
                    if gene_in_file and file.endswith("distance_matrix.csv"):
                        file_path = os.path.join(root, file)
                        # file_path = file_path.replace("_adjacency_matrix.csv", "")
                        file_path = file_path.replace("_distance_matrix.csv", "")
                        matching_paths.append(file_path)
    return matching_paths


def load_graph_data(paths):
    graphs = {}
    for path in paths:
        # NOTE: assumes path layout: .../<cell>/<...>/<gene>
        cell_name = path.split("/")[-3]
        gene_name = path.split("/")[-1]
        graph_name = f"{cell_name}_{gene_name}"
        data = pd.read_csv(f"{path}_nodes.csv")
        features = data[["x", "y", "is_virtual"]].to_numpy()
        adj_matrix = pd.read_csv(f"{path}_adjacency_matrix.csv")
        # adj_matrix = pd.read_csv(f"{path}_distance_matrix.csv")
        weights = data["count"].to_numpy()
        # name = path.split('/')[-1]
        graphs[graph_name] = (features, adj_matrix, weights)
    return graphs
