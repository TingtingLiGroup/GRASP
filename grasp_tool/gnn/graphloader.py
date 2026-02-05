import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np
import random
from sklearn.cluster import KMeans
import warnings
from . import embedding as emb
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
import multiprocessing
from multiprocessing import Pool, cpu_count


def process_cell_gene(
    cell, gene, dataset, path, n_sectors, m_rings, k_neighbor, base_path
):
    graphs = []
    aug_graphs = []

    raw_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}"
    aug_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}_aug"

    # df_file = f"{base_path}/{gene}_distances_filter_new.csv"
    df_file = f"{base_path}/{gene}_distances_filtered.csv"
    if not os.path.exists(df_file):
        # print(f"1. Skipping {gene} in {cell} (file not found).")
        return graphs, aug_graphs

    df = pd.read_csv(df_file)
    filtered_df = df[
        (df["gene"] == gene) & (df["cell"] == cell) & (df["location"] == "other")
    ]
    if filtered_df.empty:
        # print(f"Skipping {gene} in cell {cell} filtered_df is empty")
        return graphs, aug_graphs

    # Original graph
    nodes_file = f"{raw_path}/{gene}_node_matrix.csv"
    adj_file = f"{raw_path}/{gene}_adj_matrix.csv"
    if not os.path.exists(nodes_file) or not os.path.exists(adj_file):
        # print(f"1. Skipping {gene} in {cell} (file not found).")
        return graphs, aug_graphs

    node_features = pd.read_csv(nodes_file, usecols=[1, 2, 3, 4, 5])
    # if len(node_features) <= 5:
    #     # print(f"2. Skipping {gene} in {cell} (too few points).")
    #     return graphs, aug_graphs

    node_features["nuclear_position"] = (
        node_features["nuclear_position"]
        .map({"inside": 0, "outside": 1, "boundary": 2, "edge": 3})
        .fillna(4)
        .astype(int)
    )
    count_features = (
        node_features["count"]
        .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12))
        .tolist()
    )
    count_features = pd.DataFrame(
        count_features, columns=[f"dim_{i}" for i in range(12)]
    )
    position_one_hot = pd.get_dummies(
        node_features["nuclear_position"], prefix="pos"
    ).astype(int)
    node_features = pd.concat([count_features, position_one_hot], axis=1)
    node_features_tensor = torch.tensor(node_features.values, dtype=torch.float)
    adj_matrix = pd.read_csv(adj_file)
    # edge_index = torch.tensor(adj_matrix.values.nonzero(), dtype=torch.long)
    edge_index = torch.tensor(np.array(adj_matrix.values.nonzero()), dtype=torch.long)
    graph = Data(x=node_features_tensor, edge_index=edge_index, cell=cell, gene=gene)
    graphs.append(graph)

    # Augmented graph
    nodes_file = f"{aug_path}/{gene}_node_matrix.csv"
    adj_file = f"{aug_path}/{gene}_adj_matrix.csv"
    if not os.path.exists(nodes_file) or not os.path.exists(adj_file):
        # print(f"Skipping {gene} in {cell} (augmented file not found).")
        return graphs, aug_graphs

    aug_node_features = pd.read_csv(nodes_file, usecols=[1, 2, 3, 4, 5])
    aug_node_features["nuclear_position"] = (
        aug_node_features["nuclear_position"]
        .map({"inside": 0, "outside": 1, "boundary": 2, "edge": 3})
        .fillna(4)
        .astype(int)
    )
    aug_count_features = (
        aug_node_features["count"]
        .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12))
        .tolist()
    )
    aug_count_features = pd.DataFrame(
        aug_count_features, columns=[f"dim_{i}" for i in range(12)]
    )
    aug_position_one_hot = pd.get_dummies(
        aug_node_features["nuclear_position"], prefix="pos"
    ).astype(int)
    aug_node_features = pd.concat([aug_count_features, aug_position_one_hot], axis=1)
    aug_node_features_tensor = torch.tensor(aug_node_features.values, dtype=torch.float)
    aug_adj_matrix = pd.read_csv(adj_file)
    # aug_edge_index = torch.tensor(aug_adj_matrix.values.nonzero(), dtype=torch.long)
    aug_edge_index = torch.tensor(
        np.array(aug_adj_matrix.values.nonzero()), dtype=torch.long
    )
    aug_graph = Data(
        x=aug_node_features_tensor, edge_index=aug_edge_index, cell=cell, gene=gene
    )
    aug_graphs.append(aug_graph)

    return graphs, aug_graphs


def generate_graph_data_parallel(
    dataset,
    cell_list,
    gene_list,
    path,
    base_path,
    n_sectors,
    m_rings,
    k_neighbor,
    n_jobs=4,
):
    # tqdm progress bar
    all_cells_genes = [(cell, gene) for cell in cell_list for gene in gene_list]
    # Run in parallel with progress
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cell_gene)(
            cell, gene, dataset, path, n_sectors, m_rings, k_neighbor, base_path
        )
        for cell, gene in tqdm(
            all_cells_genes,
            desc="Processing cells and genes",
            total=len(all_cells_genes),
        )
    )
    # Merge results
    graphs = []
    aug_graphs = []
    for result in results:
        g, ag = result
        graphs.extend(g)
        aug_graphs.extend(ag)

    return graphs, aug_graphs


## simulated_data1 / simulated_data2
def process_cell_gene_nofiltered(
    cell, gene, dataset, path, n_sectors, m_rings, k_neighbor
):
    graphs = []
    aug_graphs = []
    raw_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}"
    aug_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}_aug"

    if dataset == "merfish_u2os" or dataset == "seqfish_fibroblast":
        # Optional filtering lists (historically hard-coded).
        # If GRASP_FILTER_ROOT is unset or files are missing, skip filtering.
        filter_root = os.environ.get("GRASP_FILTER_ROOT")
        if filter_root:
            list1_path = os.path.join(filter_root, dataset, "all_low_points_list.csv")
            list2_path = os.path.join(filter_root, dataset, "all_no_points_list.csv")

            if os.path.exists(list1_path) and os.path.exists(list2_path):
                list1 = pd.read_csv(list1_path)
                list2 = pd.read_csv(list2_path)

                # Check whether (cell, gene) is in the filter lists.
                is_in_list1 = ((list1["cell"] == cell) & (list1["gene"] == gene)).any()
                is_in_list2 = ((list2["cell"] == cell) & (list2["gene"] == gene)).any()
                if is_in_list1 or is_in_list2:
                    print(f"Skipping {gene} in {cell} (found in filter list).")
                    return graphs, aug_graphs

    # Original graph
    nodes_file = f"{raw_path}/{gene}_node_matrix.csv"
    adj_file = f"{raw_path}/{gene}_adj_matrix.csv"
    if not os.path.exists(nodes_file) or not os.path.exists(adj_file):
        # print(f"1. Skipping {gene} in {cell} (file not found).")
        return graphs, aug_graphs

    node_features = pd.read_csv(nodes_file, usecols=[1, 2, 3, 4, 5])
    total_count = node_features["count"].sum()
    node_features["count_ratio"] = node_features["count"] / total_count
    # print(node_features.head(5))
    node_features["nuclear_position"] = (
        node_features["nuclear_position"]
        .map({"inside": 0, "outside": 1, "boundary": 2, "edge": 3})
        .fillna(4)
        .astype(int)
    )
    # count_features = node_features['count'].apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12)).tolist()
    count_features = (
        node_features["count_ratio"]
        .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12))
        .tolist()
    )
    count_features = pd.DataFrame(
        count_features, columns=[f"dim_{i}" for i in range(12)]
    )
    position_one_hot = pd.get_dummies(
        node_features["nuclear_position"], prefix="pos"
    ).astype(int)
    node_features = pd.concat([count_features, position_one_hot], axis=1)
    node_features_tensor = torch.tensor(node_features.values, dtype=torch.float)
    adj_matrix = pd.read_csv(adj_file)
    # edge_index = torch.tensor(adj_matrix.values.nonzero(), dtype=torch.long)
    edge_index = torch.tensor(np.array(adj_matrix.values.nonzero()), dtype=torch.long)
    graph = Data(x=node_features_tensor, edge_index=edge_index, cell=cell, gene=gene)
    graphs.append(graph)

    # Augmented graph
    nodes_file = f"{aug_path}/{gene}_node_matrix.csv"
    adj_file = f"{aug_path}/{gene}_adj_matrix.csv"
    if not os.path.exists(nodes_file) or not os.path.exists(adj_file):
        # print(f"Skipping {gene} in {cell} (augmented file not found).")
        return graphs, aug_graphs

    aug_node_features = pd.read_csv(nodes_file, usecols=[1, 2, 3, 4, 5])
    total_count = aug_node_features["count"].sum()
    aug_node_features["count_ratio"] = aug_node_features["count"] / total_count
    # print(aug_node_features.head(5))
    aug_node_features["nuclear_position"] = (
        aug_node_features["nuclear_position"]
        .map({"inside": 0, "outside": 1, "boundary": 2, "edge": 3})
        .fillna(4)
        .astype(int)
    )
    # aug_count_features = aug_node_features['count'].apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12)).tolist()
    aug_count_features = (
        aug_node_features["count_ratio"]
        .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12))
        .tolist()
    )

    aug_count_features = pd.DataFrame(
        aug_count_features, columns=[f"dim_{i}" for i in range(12)]
    )
    aug_position_one_hot = pd.get_dummies(
        aug_node_features["nuclear_position"], prefix="pos"
    ).astype(int)
    aug_node_features = pd.concat([aug_count_features, aug_position_one_hot], axis=1)
    aug_node_features_tensor = torch.tensor(aug_node_features.values, dtype=torch.float)
    aug_adj_matrix = pd.read_csv(adj_file)
    # aug_edge_index = torch.tensor(aug_adj_matrix.values.nonzero(), dtype=torch.long)
    aug_edge_index = torch.tensor(
        np.array(aug_adj_matrix.values.nonzero()), dtype=torch.long
    )
    aug_graph = Data(
        x=aug_node_features_tensor, edge_index=aug_edge_index, cell=cell, gene=gene
    )
    aug_graphs.append(aug_graph)

    return graphs, aug_graphs


def generate_graph_data_parallel_nofiltered(
    dataset, cell_list, gene_list, path, n_sectors, m_rings, k_neighbor, n_jobs=4
):
    # tqdm progress bar
    all_cells_genes = [(cell, gene) for cell in cell_list for gene in gene_list]
    # Run in parallel with progress
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cell_gene_nofiltered)(
            cell, gene, dataset, path, n_sectors, m_rings, k_neighbor
        )
        for cell, gene in tqdm(
            all_cells_genes,
            desc="Processing cells and genes",
            total=len(all_cells_genes),
        )
    )
    # Merge results
    graphs = []
    aug_graphs = []
    for result in results:
        g, ag = result
        graphs.extend(g)
        aug_graphs.extend(ag)
    return graphs, aug_graphs

    graphs = []
    aug_graphs = []
    # df = df[df['groundtruth_yyzh'].isin(['Nuclear', 'Nuclear_edge', 'Cytoplasmic', 'Cell_edge', 'Random'])].reset_index(drop=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Graphs"):
        cell = row["cell"]
        gene = row["gene"]

        raw_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}"
        aug_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}_aug"

        # Original graph
        nodes_file = f"{raw_path}/{gene}_node_matrix.csv"
        adj_file = f"{raw_path}/{gene}_adj_matrix.csv"

        node_features = pd.read_csv(nodes_file, usecols=[1, 2, 3, 4, 5])
        total_count = node_features["count"].sum()
        node_features["count_ratio"] = node_features["count"] / total_count
        # print(node_features.head(5))

        node_features["nuclear_position"] = (
            node_features["nuclear_position"]
            .map({"inside": 0, "outside": 1, "boundary": 2, "edge": 3})
            .fillna(4)
            .astype(int)
        )
        # count_features = node_features['count'].apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12)).tolist()
        count_features = (
            node_features["count_ratio"]
            .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12))
            .tolist()
        )
        count_features = pd.DataFrame(
            count_features, columns=[f"dim_{i}" for i in range(12)]
        )
        position_one_hot = pd.get_dummies(
            node_features["nuclear_position"], prefix="pos"
        ).astype(int)
        node_features = pd.concat([count_features, position_one_hot], axis=1)
        node_features_tensor = torch.tensor(node_features.values, dtype=torch.float)
        adj_matrix = pd.read_csv(adj_file)
        # edge_index = torch.tensor(adj_matrix.values.nonzero(), dtype=torch.long)
        edge_index = torch.tensor(
            np.array(adj_matrix.values.nonzero()), dtype=torch.long
        )
        graph = Data(
            x=node_features_tensor, edge_index=edge_index, cell=cell, gene=gene
        )
        graphs.append(graph)
        # Augmented graph
        nodes_file = f"{aug_path}/{gene}_node_matrix.csv"
        adj_file = f"{aug_path}/{gene}_adj_matrix.csv"

        aug_node_features = pd.read_csv(nodes_file, usecols=[1, 2, 3, 4, 5])
        total_count = aug_node_features["count"].sum()
        aug_node_features["count_ratio"] = aug_node_features["count"] / total_count
        # print(aug_node_features.head(5))
        aug_node_features["nuclear_position"] = (
            aug_node_features["nuclear_position"]
            .map({"inside": 0, "outside": 1, "boundary": 2, "edge": 3})
            .fillna(4)
            .astype(int)
        )
        # aug_count_features = aug_node_features['count'].apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12)).tolist()
        aug_count_features = (
            aug_node_features["count_ratio"]
            .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12))
            .tolist()
        )
        aug_count_features = pd.DataFrame(
            aug_count_features, columns=[f"dim_{i}" for i in range(12)]
        )
        aug_position_one_hot = pd.get_dummies(
            aug_node_features["nuclear_position"], prefix="pos"
        ).astype(int)
        aug_node_features = pd.concat(
            [aug_count_features, aug_position_one_hot], axis=1
        )
        aug_node_features_tensor = torch.tensor(
            aug_node_features.values, dtype=torch.float
        )
        aug_adj_matrix = pd.read_csv(adj_file)
        # aug_edge_index = torch.tensor(aug_adj_matrix.values.nonzero(), dtype=torch.long)
        aug_edge_index = torch.tensor(
            np.array(aug_adj_matrix.values.nonzero()), dtype=torch.long
        )
        aug_graph = Data(
            x=aug_node_features_tensor, edge_index=aug_edge_index, cell=cell, gene=gene
        )
        aug_graphs.append(aug_graph)
    return graphs, aug_graphs


def generate_graph_data_target_weight(
    dataset,
    df,
    path,
    n_sectors,
    m_rings,
    k_neighbor,
    knn_k_for_similarity_graph=None,
    knn_k_ratio=0.25,
    knn_k_min=3,
):
    graphs = []
    aug_graphs = []
    # df = df[df['groundtruth_yyzh'].isin(['Nuclear', 'Nuclear_edge', 'Cytoplasmic', 'Cell_edge', 'Random'])].reset_index(drop=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Graphs"):
        cell = row["cell"]
        gene = row["gene"]

        raw_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}"
        aug_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}_aug"

        # Original graph
        nodes_file = f"{raw_path}/{gene}_node_matrix.csv"

        node_data_df = pd.read_csv(
            nodes_file, usecols=["count", "is_edge", "nuclear_position", "is_virtual"]
        )

        # --- Filter out virtual nodes ---
        if "is_virtual" in node_data_df.columns:
            node_data_df = node_data_df[node_data_df["is_virtual"] == 0].reset_index(
                drop=True
            )
        # --- End filtering ---

        if node_data_df.empty:  # If no real nodes left
            continue

        total_count = node_data_df["count"].sum()
        if (
            total_count == 0
        ):  # Should not happen if only real nodes with counts are kept, but good for safety
            node_data_df["count_ratio"] = 0
        else:
            node_data_df["count_ratio"] = node_data_df["count"] / total_count

        count_features_embedded = (
            node_data_df["count_ratio"]
            .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=11))
            .tolist()
        )
        count_features_final_df = pd.DataFrame(
            count_features_embedded,
            columns=[f"count_dim_{i}" for i in range(11)],
            index=node_data_df.index,
        )  # Use filtered df index

        is_edge_one_hot_df = pd.get_dummies(
            node_data_df["is_edge"], prefix="edge", dtype=int
        )
        # Ensure consistent columns for is_edge (edge_0, edge_1)
        for val in [0, 1]:
            col_name = f"edge_{val}"
            if col_name not in is_edge_one_hot_df.columns:
                is_edge_one_hot_df[col_name] = 0
        is_edge_one_hot_df = is_edge_one_hot_df[
            is_edge_one_hot_df.columns.intersection(["edge_0", "edge_1"])
        ]  # Ensure only existing columns selected
        if "edge_0" not in is_edge_one_hot_df.columns:
            is_edge_one_hot_df["edge_0"] = 0
        if "edge_1" not in is_edge_one_hot_df.columns:
            is_edge_one_hot_df["edge_1"] = 0
        is_edge_one_hot_df = is_edge_one_hot_df[["edge_0", "edge_1"]]

        nuclear_position_one_hot_df = pd.get_dummies(
            node_data_df["nuclear_position"], prefix="pos", dtype=int
        )
        expected_pos_categories = ["inside", "outside", "boundary"]
        for cat in expected_pos_categories:
            col_name = f"pos_{cat}"
            if col_name not in nuclear_position_one_hot_df.columns:
                nuclear_position_one_hot_df[col_name] = 0
        # Ensure correct order and presence of all expected columns
        nuclear_position_one_hot_df = nuclear_position_one_hot_df.reindex(
            columns=[f"pos_{cat}" for cat in expected_pos_categories], fill_value=0
        )

        final_node_features_df = pd.concat(
            [count_features_final_df, is_edge_one_hot_df, nuclear_position_one_hot_df],
            axis=1,
        )

        # No need to set virtual node features to 0 as they are already removed

        node_features_tensor = torch.tensor(
            final_node_features_df.values, dtype=torch.float
        )

        num_nodes = node_features_tensor.shape[0]
        edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=node_features_tensor.device
        )

        if num_nodes > 1:
            norm_node_features = F.normalize(node_features_tensor, p=2, dim=1)
            similarity_matrix = torch.matmul(norm_node_features, norm_node_features.t())

            # Set diagonal to a very small value to prevent self-loops from being top-k
            similarity_matrix.fill_diagonal_(float("-inf"))

            # --- Dynamically determine k ---
            # 1) If the caller explicitly passes a positive integer (knn_k_for_similarity_graph),
            #    we will respect it but still cap it with (num_nodes-1).
            # 2) If the arg is None or a non-positive value, we compute k as
            #       k = max(1, int(num_nodes * knn_k_ratio))
            #    and again cap it with (num_nodes-1).
            if (
                isinstance(knn_k_for_similarity_graph, int)
                and knn_k_for_similarity_graph > 0
            ):
                current_knn_k = min(knn_k_for_similarity_graph, num_nodes - 1)
            else:
                adaptive_k = max(knn_k_min, int(num_nodes * knn_k_ratio))
                current_knn_k = min(adaptive_k, num_nodes - 1)

            if current_knn_k > 0:
                # Get top-k similar nodes for each node
                top_k_vals, top_k_indices = torch.topk(
                    similarity_matrix, k=current_knn_k, dim=1
                )

                # Create adjacency matrix from top-k indices
                adj = torch.zeros_like(similarity_matrix, dtype=torch.bool)
                row_indices = (
                    torch.arange(num_nodes, device=node_features_tensor.device)
                    .unsqueeze(1)
                    .expand(-1, current_knn_k)
                )
                adj[row_indices, top_k_indices] = True

                # Symmetrize the adjacency matrix
                adj = adj | adj.t()

                # No need to remove virtual node edges as virtual nodes themselves are removed
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()

        graph = Data(
            x=node_features_tensor, edge_index=edge_index, cell=cell, gene=gene
        )
        graphs.append(graph)

        # Augmented graph
        nodes_file_aug = f"{aug_path}/{gene}_node_matrix.csv"
        aug_node_data_df = pd.read_csv(
            nodes_file_aug,
            usecols=["count", "is_edge", "nuclear_position", "is_virtual"],
        )

        # --- Filter out virtual nodes for augmented graph ---
        if "is_virtual" in aug_node_data_df.columns:
            aug_node_data_df = aug_node_data_df[
                aug_node_data_df["is_virtual"] == 0
            ].reset_index(drop=True)
        # --- End filtering ---

        if aug_node_data_df.empty:  # If no real nodes left in augmented graph
            # If original graph was added, but augmented has no real nodes, we might still want the original.
            # Current behavior: skip adding an augmented graph, original graph is already in the list.
            # If you require pairs, you might need to remove the original graph too or handle this case differently.
            continue

        aug_total_count = aug_node_data_df["count"].sum()
        if aug_total_count == 0:
            aug_node_data_df["count_ratio"] = 0
        else:
            aug_node_data_df["count_ratio"] = (
                aug_node_data_df["count"] / aug_total_count
            )

        aug_count_features_embedded = (
            aug_node_data_df["count_ratio"]
            .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=11))
            .tolist()
        )
        aug_count_features_final_df = pd.DataFrame(
            aug_count_features_embedded,
            columns=[f"count_dim_{i}" for i in range(11)],
            index=aug_node_data_df.index,
        )  # Use filtered df index

        aug_is_edge_one_hot_df = pd.get_dummies(
            aug_node_data_df["is_edge"], prefix="edge", dtype=int
        )
        for val in [0, 1]:
            col_name = f"edge_{val}"
            if col_name not in aug_is_edge_one_hot_df.columns:
                aug_is_edge_one_hot_df[col_name] = 0
        aug_is_edge_one_hot_df = aug_is_edge_one_hot_df[
            aug_is_edge_one_hot_df.columns.intersection(["edge_0", "edge_1"])
        ]
        if "edge_0" not in aug_is_edge_one_hot_df.columns:
            aug_is_edge_one_hot_df["edge_0"] = 0
        if "edge_1" not in aug_is_edge_one_hot_df.columns:
            aug_is_edge_one_hot_df["edge_1"] = 0
        aug_is_edge_one_hot_df = aug_is_edge_one_hot_df[["edge_0", "edge_1"]]

        aug_nuclear_position_one_hot_df = pd.get_dummies(
            aug_node_data_df["nuclear_position"], prefix="pos", dtype=int
        )
        for cat in expected_pos_categories:
            col_name = f"pos_{cat}"
            if col_name not in aug_nuclear_position_one_hot_df.columns:
                aug_nuclear_position_one_hot_df[col_name] = 0
        aug_nuclear_position_one_hot_df = aug_nuclear_position_one_hot_df.reindex(
            columns=[f"pos_{cat}" for cat in expected_pos_categories], fill_value=0
        )

        aug_final_node_features_df = pd.concat(
            [
                aug_count_features_final_df,
                aug_is_edge_one_hot_df,
                aug_nuclear_position_one_hot_df,
            ],
            axis=1,
        )

        # No need to set virtual node features to 0 as they are already removed

        aug_node_features_tensor = torch.tensor(
            aug_final_node_features_df.values, dtype=torch.float
        )

        num_aug_nodes = aug_node_features_tensor.shape[0]
        aug_edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=aug_node_features_tensor.device
        )

        if num_aug_nodes > 1:
            norm_aug_node_features = F.normalize(aug_node_features_tensor, p=2, dim=1)
            aug_similarity_matrix = torch.matmul(
                norm_aug_node_features, norm_aug_node_features.t()
            )
            aug_similarity_matrix.fill_diagonal_(float("-inf"))

            # Same adaptive-k logic for augmented graphs
            if (
                isinstance(knn_k_for_similarity_graph, int)
                and knn_k_for_similarity_graph > 0
            ):
                current_knn_k_aug = min(knn_k_for_similarity_graph, num_aug_nodes - 1)
            else:
                adaptive_k_aug = max(knn_k_min, int(num_aug_nodes * knn_k_ratio))
                current_knn_k_aug = min(adaptive_k_aug, num_aug_nodes - 1)

            if current_knn_k_aug > 0:
                aug_top_k_vals, aug_top_k_indices = torch.topk(
                    aug_similarity_matrix, k=current_knn_k_aug, dim=1
                )
                aug_adj = torch.zeros_like(aug_similarity_matrix, dtype=torch.bool)
                aug_row_indices = (
                    torch.arange(num_aug_nodes, device=aug_node_features_tensor.device)
                    .unsqueeze(1)
                    .expand(-1, current_knn_k_aug)
                )
                aug_adj[aug_row_indices, aug_top_k_indices] = True
                aug_adj = aug_adj | aug_adj.t()

                # No need to remove virtual node edges as virtual nodes themselves are removed
                aug_edge_index = aug_adj.nonzero(as_tuple=False).t().contiguous()

        aug_graph = Data(
            x=aug_node_features_tensor, edge_index=aug_edge_index, cell=cell, gene=gene
        )
        aug_graphs.append(aug_graph)
    return graphs, aug_graphs


def generate_graph_data_target_weight2(
    dataset, df, path, n_sectors, m_rings, k_neighbor
):
    graphs = []
    aug_graphs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Graphs"):
        cell = row["cell"]
        gene = row["gene"]

        raw_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}"
        aug_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}_aug"

        # Original graph
        nodes_file = f"{raw_path}/{gene}_node_matrix.csv"
        adj_file = f"{raw_path}/{gene}_adj_matrix.csv"

        try:
            node_features_df = pd.read_csv(
                nodes_file,
                usecols=["count", "is_edge", "nuclear_position", "is_virtual"],
            )
        except FileNotFoundError:
            continue
        if (
            node_features_df.empty
            or "count" not in node_features_df.columns
            or "nuclear_position" not in node_features_df.columns
        ):
            continue
        if "is_virtual" not in node_features_df.columns:
            node_features_df["is_virtual"] = 0

        df_for_feature_calc = node_features_df.copy()
        total_count = df_for_feature_calc["count"].sum()
        df_for_feature_calc["count_ratio"] = (
            0 if total_count == 0 else df_for_feature_calc["count"] / total_count
        )

        # Assumes emb.nonlinear_transform_embedding is available
        count_features_list = (
            df_for_feature_calc["count_ratio"]
            .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12))
            .tolist()
        )
        count_features_final_df = pd.DataFrame(
            count_features_list,
            columns=[f"dim_{i}" for i in range(12)],
            index=df_for_feature_calc.index,
        )

        df_for_feature_calc["nuclear_position_mapped"] = (
            df_for_feature_calc["nuclear_position"]
            .map({"inside": 0, "outside": 1, "boundary": 2, "edge": 3})
            .astype(np.int64)
        )
        position_one_hot_df = pd.get_dummies(
            df_for_feature_calc["nuclear_position_mapped"], prefix="pos", dtype=int
        )

        expected_pos_cols = [f"pos_{i}" for i in range(4)]
        for col in expected_pos_cols:
            if col not in position_one_hot_df.columns:
                position_one_hot_df[col] = 0
        position_one_hot_df = position_one_hot_df[expected_pos_cols]
        final_node_features_df = pd.concat(
            [count_features_final_df, position_one_hot_df], axis=1
        ).astype(float)
        virtual_node_mask = node_features_df["is_virtual"] == 1
        final_node_features_df.loc[virtual_node_mask, :] = 0.0
        final_node_features_df = final_node_features_df.astype(float)
        node_features_tensor = torch.tensor(
            final_node_features_df.values, dtype=torch.float
        )

        # adj_matrix = pd.read_csv(adj_file)
        # edge_index = torch.tensor(np.array(adj_matrix.values.nonzero()), dtype=torch.long)
        # graph = Data(x=node_features_tensor, edge_index=edge_index, cell=cell, gene=gene)
        # graphs.append(graph)

        # Augmented graph
        nodes_file_aug = f"{aug_path}/{gene}_node_matrix.csv"
        adj_file_aug = f"{aug_path}/{gene}_adj_matrix.csv"

        try:
            aug_node_features_df = pd.read_csv(nodes_file_aug)
        except FileNotFoundError:
            continue
        if (
            aug_node_features_df.empty
            or "count" not in aug_node_features_df.columns
            or "nuclear_position" not in aug_node_features_df.columns
        ):
            continue
        if "is_virtual" not in aug_node_features_df.columns:
            aug_node_features_df["is_virtual"] = 0

        aug_df_for_feature_calc = aug_node_features_df.copy()
        aug_total_count = aug_df_for_feature_calc["count"].sum()
        aug_df_for_feature_calc["count_ratio"] = (
            0
            if aug_total_count == 0
            else aug_df_for_feature_calc["count"] / aug_total_count
        )
        aug_count_features_list = (
            aug_df_for_feature_calc["count_ratio"]
            .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12))
            .tolist()
        )
        aug_count_features_final_df = pd.DataFrame(
            aug_count_features_list,
            columns=[f"dim_{i}" for i in range(12)],
            index=aug_df_for_feature_calc.index,
        )
        aug_df_for_feature_calc["nuclear_position_mapped"] = (
            aug_df_for_feature_calc["nuclear_position"]
            .map({"inside": 0, "outside": 1, "boundary": 2, "edge": 3})
            .astype(np.int64)
        )
        aug_position_one_hot_df = pd.get_dummies(
            aug_df_for_feature_calc["nuclear_position_mapped"], prefix="pos", dtype=int
        )

        for col in expected_pos_cols:
            if col not in aug_position_one_hot_df.columns:
                aug_position_one_hot_df[col] = 0
        aug_position_one_hot_df = aug_position_one_hot_df[expected_pos_cols]
        aug_final_node_features_df = pd.concat(
            [aug_count_features_final_df, aug_position_one_hot_df], axis=1
        ).astype(float)
        aug_virtual_node_mask = aug_node_features_df["is_virtual"] == 1
        aug_final_node_features_df.loc[aug_virtual_node_mask, :] = 0.0
        aug_final_node_features_df = aug_final_node_features_df.astype(float)
        aug_node_features_tensor = torch.tensor(
            aug_final_node_features_df.values, dtype=torch.float
        )

        # aug_adj_matrix = pd.read_csv(adj_file_aug)
        # aug_edge_index = torch.tensor(np.array(aug_adj_matrix.values.nonzero()), dtype=torch.long)
        # aug_graph = Data(x=aug_node_features_tensor, edge_index=aug_edge_index, cell=cell, gene=gene)
        # aug_graphs.append(aug_graph)

    return graphs, aug_graphs


# def process_graph_row_safe(args):
#     row, path, n_sectors, m_rings, k_neighbor = args
#     cell = row['cell']
#     gene = row['gene']
#     graph_data = []

#     for is_aug in [False, True]:
#         base_path = f"{path}/{cell}"
#         if is_aug:
#             base_path += "_aug"

#         nodes_file = f'{base_path}/{gene}_node_matrix.csv'
#         adj_file = f'{base_path}/{gene}_adj_matrix.csv'

#         try:
#             node_df = pd.read_csv(nodes_file, usecols=[1, 2, 3, 4, 5])
#             total_count = node_df['count'].sum()
#             if total_count == 0:
#                 return None

#             count_ratio = (node_df['count'] / total_count).to_numpy()
#             count_embed_np = np.array([emb.nonlinear_transform_embedding(x, dim=12) for x in count_ratio])
#             pos_int = node_df['nuclear_position'].map(
#                 {'inside': 0, 'outside': 1, 'boundary': 2, 'edge': 3}
#             ).fillna(4).astype(int).to_numpy()
#             pos_one_hot = np.eye(4)[pos_int]
#             node_features_np = np.hstack([count_embed_np, pos_one_hot])

#             adj_matrix = pd.read_csv(adj_file).values
#             edge_index_np = np.array(np.nonzero(adj_matrix))

#             graph_data.append({
#                 'x': node_features_np,
#                 'edge_index': edge_index_np,
#                 'cell': cell,
#                 'gene': gene
#             })

#         except Exception as e:
#             print(f"Error processing {nodes_file}: {e}")
#             return None

#     return graph_data[0], graph_data[1]

# def generate_graph_data_target_parallel(dataset, df, path, n_sectors, m_rings, k_neighbor):
#     args_list = [(row, path, n_sectors, m_rings, k_neighbor) for _, row in df.iterrows()]

#     multiprocessing.set_start_method("spawn", force=True)

#     with Pool(processes=4) as pool:
#         results = list(tqdm(pool.imap_unordered(process_graph_row_safe, args_list, chunksize=10), total=len(df), desc="Parallel Processing Graphs"))

#     original_graphs, augmented_graphs = [], []

#     for result in results:
#         if result is None:
#             continue
#         for idx, (graph_list, target) in enumerate(zip([original_graphs, augmented_graphs], result)):
#             data = Data(
#                 x=torch.tensor(target['x'], dtype=torch.float),
#                 edge_index=torch.tensor(target['edge_index'], dtype=torch.long),
#                 cell=target['cell'],
#                 gene=target['gene']
#             )
#             graph_list.append(data)

#     return original_graphs, augmented_graphs

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm
import multiprocessing
from multiprocessing import Pool
import functools


def process_single_gene_pair(args):
    """Process one (cell, gene) pair inside a worker process."""
    row, path, n_sectors, m_rings, k_neighbor = args
    cell = row["cell"]
    gene = row["gene"]

    results = {}

    for suffix in ["", "_aug"]:
        is_aug = suffix == "_aug"
        base_path = f"{path}/{cell}{suffix}"
        nodes_file = f"{base_path}/{gene}_node_matrix.csv"
        adj_file = f"{base_path}/{gene}_adj_matrix.csv"

        try:
            # Read node features.
            try:
                node_df = pd.read_csv(nodes_file)
            except FileNotFoundError:
                return None

            required_cols = ["count", "nuclear_position"]
            if not all(col in node_df.columns for col in required_cols):
                return None

            # Feature computation (count ratio embedding + position one-hot).
            total_count = node_df["count"].sum()
            if total_count == 0:
                return None

            count_ratio = node_df["count"] / total_count
            count_embed_list = [
                emb.nonlinear_transform_embedding(x, dim=12) for x in count_ratio
            ]
            count_embed_np = np.array(count_embed_list)

            # Position one-hot.
            pos_map = {"inside": 0, "outside": 1, "boundary": 2, "edge": 3}
            node_df["pos_mapped"] = (
                node_df["nuclear_position"].map(pos_map).fillna(4).astype(int)
            )

            # NOTE: enforce dtype=int to avoid mixed bool/int columns which become
            # dtype=object when converting to numpy.
            pos_dummies = pd.get_dummies(node_df["pos_mapped"], prefix="pos", dtype=int)
            expected_cols = [f"pos_{i}" for i in range(4)]
            pos_dummies = pos_dummies.reindex(columns=expected_cols, fill_value=0)
            pos_features_np = pos_dummies.to_numpy()

            # Merge features.
            node_features_np = np.hstack([count_embed_np, pos_features_np])

            # 4. Adjacency handling (check empty matrices)
            try:
                adj_df = pd.read_csv(adj_file)
                if adj_df.empty:
                    edge_index_np = np.empty((2, 0))
                else:
                    edge_index_np = np.array(adj_df.values.nonzero())
            except FileNotFoundError:
                return None

            # Store in a temporary dict
            key = "aug" if is_aug else "orig"
            results[key] = {
                "x": node_features_np,
                "edge_index": edge_index_np,
                "cell": cell,
                "gene": gene,
            }

        except Exception as e:
            # Log errors for debugging; keep output minimal in parallel runs
            # print(f"Error in {cell}-{gene}: {e}")
            return None

    # Only return when both original and augmented graphs succeed
    if "orig" in results and "aug" in results:
        return results["orig"], results["aug"]
    return None


def generate_graph_data_target_parallel(
    dataset, df, path, n_sectors, m_rings, k_neighbor, processes=8
):
    # Build argument list
    args_list = [
        (row, path, n_sectors, m_rings, k_neighbor) for _, row in df.iterrows()
    ]

    # Ensure spawn mode works on some platforms
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    original_graphs = []
    augmented_graphs = []

    print(f"Starting parallel processing with {processes} cores...")

    with Pool(processes=processes) as pool:
        # imap_unordered is efficient because it does not preserve task order
        iterator = pool.imap_unordered(
            process_single_gene_pair, args_list, chunksize=10
        )

        for result in tqdm(
            iterator, total=len(args_list), desc="Generating Target Graphs"
        ):
            if result is None:
                continue

            orig_dict, aug_dict = result

            # Convert to torch_geometric Data in the parent process (safer)
            for target_list, data_dict in zip(
                [original_graphs, augmented_graphs], [orig_dict, aug_dict]
            ):
                data = Data(
                    x=torch.tensor(data_dict["x"], dtype=torch.float),
                    edge_index=torch.tensor(data_dict["edge_index"], dtype=torch.long),
                    cell=data_dict["cell"],
                    gene=data_dict["gene"],
                )
                target_list.append(data)

    return original_graphs, augmented_graphs


## AD/merfish mouse brain (df is the dataframe to load)
def generate_graph_data_target_noposition(
    dataset, df, path, n_sectors, m_rings, k_neighbor
):
    graphs = []
    aug_graphs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Graphs"):
        cell = row["cell"]
        gene = row["gene"]

        raw_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}"
        aug_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}_aug"
        # Original graph
        nodes_file = f"{raw_path}/{gene}_node_matrix.csv"
        adj_file = f"{raw_path}/{gene}_adj_matrix.csv"

        node_features = pd.read_csv(nodes_file, usecols=[1, 2, 3, 4])
        total_count = node_features["count"].sum()
        node_features["count_ratio"] = node_features["count"] / total_count
        count_features = (
            node_features["count_ratio"]
            .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=16))
            .tolist()
        )
        count_features = pd.DataFrame(
            count_features, columns=[f"dim_{i}" for i in range(16)]
        )
        node_features_tensor = torch.tensor(count_features.values, dtype=torch.float)
        adj_matrix = pd.read_csv(adj_file)
        edge_index = torch.tensor(
            np.array(adj_matrix.values.nonzero()), dtype=torch.long
        )
        graph = Data(
            x=node_features_tensor, edge_index=edge_index, cell=cell, gene=gene
        )
        graphs.append(graph)
        # Augmented graph
        nodes_file = f"{aug_path}/{gene}_node_matrix.csv"
        adj_file = f"{aug_path}/{gene}_adj_matrix.csv"

        aug_node_features = pd.read_csv(nodes_file, usecols=[1, 2, 3, 4])
        total_count = aug_node_features["count"].sum()
        aug_node_features["count_ratio"] = aug_node_features["count"] / total_count
        aug_count_features = (
            aug_node_features["count_ratio"]
            .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=16))
            .tolist()
        )
        aug_count_features = pd.DataFrame(
            aug_count_features, columns=[f"dim_{i}" for i in range(16)]
        )
        aug_node_features_tensor = torch.tensor(
            aug_count_features.values, dtype=torch.float
        )
        aug_adj_matrix = pd.read_csv(adj_file)
        aug_edge_index = torch.tensor(
            np.array(aug_adj_matrix.values.nonzero()), dtype=torch.long
        )
        aug_graph = Data(
            x=aug_node_features_tensor, edge_index=aug_edge_index, cell=cell, gene=gene
        )
        aug_graphs.append(aug_graph)
    return graphs, aug_graphs


def process_cell_gene_noposition(
    cell, gene, dataset, path, n_sectors, m_rings, k_neighbor, base_path
):
    graphs = []
    aug_graphs = []
    # base_path = f"/Volumes/hyydisk/GCN_CL/3_filter/1_{dataset}_Wasserstein_Distance/"

    raw_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}"
    aug_path = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}_aug"

    df_file = f"{base_path}/{gene}_distances_filter_new.csv"
    if not os.path.exists(df_file):
        print(f"1. Skipping {gene} in {cell} (file not found).")
        return graphs, aug_graphs

    df = pd.read_csv(df_file)
    filtered_df = df[
        (df["gene"] == gene) & (df["cell"] == cell) & (df["location"] == "other")
    ]
    if filtered_df.empty:
        # print(f"Skipping {gene} in cell {cell} filtered_df is empty")
        return graphs, aug_graphs

    # Original graph
    nodes_file = f"{raw_path}/{gene}_node_matrix.csv"
    adj_file = f"{raw_path}/{gene}_adj_matrix.csv"
    if not os.path.exists(nodes_file) or not os.path.exists(adj_file):
        # print(f"1. Skipping {gene} in {cell} (file not found).")
        return graphs, aug_graphs

    node_features = pd.read_csv(nodes_file, usecols=[1, 2, 3, 4])
    if len(node_features) <= 5:
        # print(f"2. Skipping {gene} in {cell} (too few points).")
        return graphs, aug_graphs

    total_count = node_features["count"].sum()
    node_features["count_ratio"] = node_features["count"] / total_count
    count_features = (
        node_features["count_ratio"]
        .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=16))
        .tolist()
    )
    count_features = pd.DataFrame(
        count_features, columns=[f"dim_{i}" for i in range(8)]
    )
    node_features_tensor = torch.tensor(node_features.values, dtype=torch.float)
    adj_matrix = pd.read_csv(adj_file)
    edge_index = torch.tensor(np.array(adj_matrix.values.nonzero()), dtype=torch.long)
    graph = Data(x=node_features_tensor, edge_index=edge_index, cell=cell, gene=gene)
    graphs.append(graph)

    # Augmented graph
    nodes_file = f"{aug_path}/{gene}_node_matrix.csv"
    adj_file = f"{aug_path}/{gene}_adj_matrix.csv"
    if not os.path.exists(nodes_file) or not os.path.exists(adj_file):
        # print(f"Skipping {gene} in {cell} (augmented file not found).")
        return graphs, aug_graphs

    aug_node_features = pd.read_csv(nodes_file, usecols=[1, 2, 3, 4])
    aug_count_features = (
        aug_node_features["count"]
        .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=8))
        .tolist()
    )
    aug_count_features = pd.DataFrame(
        aug_count_features, columns=[f"dim_{i}" for i in range(8)]
    )
    aug_node_features_tensor = torch.tensor(aug_node_features.values, dtype=torch.float)
    aug_adj_matrix = pd.read_csv(adj_file)
    aug_edge_index = torch.tensor(
        np.array(aug_adj_matrix.values.nonzero()), dtype=torch.long
    )
    aug_graph = Data(
        x=aug_node_features_tensor, edge_index=aug_edge_index, cell=cell, gene=gene
    )
    aug_graphs.append(aug_graph)
    return graphs, aug_graphs


def generate_graph_data_parallel_noposition(
    dataset,
    cell_list,
    gene_list,
    path,
    base_path,
    n_sectors,
    m_rings,
    k_neighbor,
    n_jobs=4,
):
    # tqdm progress bar
    all_cells_genes = [(cell, gene) for cell in cell_list for gene in gene_list]
    # Run in parallel with progress
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cell_gene_noposition)(
            cell, gene, dataset, path, n_sectors, m_rings, k_neighbor, base_path
        )
        for cell, gene in tqdm(
            all_cells_genes,
            desc="Processing cells and genes",
            total=len(all_cells_genes),
        )
    )

    # Merge results
    graphs = []
    aug_graphs = []
    for result in results:
        g, ag = result
        graphs.extend(g)
        aug_graphs.extend(ag)

    return graphs, aug_graphs


def generate_graph_data_target(dataset, df, path, n_sectors, m_rings, k_neighbor):
    graphs = []
    aug_graphs = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Processing Graphs generate_graph_data_target",
    ):
        cell = row["cell"]
        gene = row["gene"]

        # --- Process Original Graph ---
        raw_path_base = f"{path}/{cell}"
        nodes_file_orig = f"{raw_path_base}/{gene}_node_matrix.csv"
        adj_file_orig = f"{raw_path_base}/{gene}_adj_matrix.csv"
        skip_current_pair = False
        original_graph_data = None

        try:
            # Read only necessary columns by name
            try:
                # Try reading with is_virtual first
                node_features_orig_df_raw = pd.read_csv(
                    nodes_file_orig, usecols=["count", "nuclear_position", "is_virtual"]
                )
            except ValueError:  # If is_virtual column doesn't exist
                try:
                    node_features_orig_df_raw = pd.read_csv(
                        nodes_file_orig, usecols=["count", "nuclear_position"]
                    )
                except FileNotFoundError:
                    print(
                        f"Original node_matrix file not found: {nodes_file_orig}. Skipping pair."
                    )
                    skip_current_pair = True
                    raise  # Re-raise to be caught by outer try-except
                except ValueError:  # If count or nuclear_position are missing
                    print(
                        f"Original node_matrix {nodes_file_orig} missing 'count' or 'nuclear_position'. Skipping pair."
                    )
                    skip_current_pair = True
                    raise

            if node_features_orig_df_raw.empty:
                print(
                    f"Original node_matrix {nodes_file_orig} is empty. Skipping pair."
                )
                skip_current_pair = True
                raise FileNotFoundError  # Treat as if file not found for outer catch

            node_features_orig_df = node_features_orig_df_raw.copy()

            total_count_orig = node_features_orig_df["count"].sum()
            node_features_orig_df["count_ratio"] = (
                0.0
                if total_count_orig == 0
                else node_features_orig_df["count"] / total_count_orig
            )

            count_features_orig_list = (
                node_features_orig_df["count_ratio"]
                .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12))
                .tolist()
            )
            count_features_orig_embedded_df = pd.DataFrame(
                count_features_orig_list,
                columns=[f"dim_{i}" for i in range(12)],
                index=node_features_orig_df.index,
            )

            node_features_orig_df["nuclear_position_mapped"] = (
                node_features_orig_df["nuclear_position"]
                .map({"inside": 0, "outside": 1, "boundary": 2, "edge": 3})
                .fillna(4)
                .astype(int)
            )
            position_orig_one_hot_df = pd.get_dummies(
                node_features_orig_df["nuclear_position_mapped"],
                prefix="pos",
                dtype=int,
            )

            # Ensure all graphs have the same position feature dimensionality (pos_0..pos_3)
            expected_pos_cols = ["pos_0", "pos_1", "pos_2", "pos_3"]
            for col in expected_pos_cols:
                if col not in position_orig_one_hot_df.columns:
                    position_orig_one_hot_df[col] = 0
            position_orig_one_hot_df = position_orig_one_hot_df[expected_pos_cols]

            final_node_features_orig_df = pd.concat(
                [count_features_orig_embedded_df, position_orig_one_hot_df], axis=1
            )
            node_features_orig_tensor = torch.tensor(
                final_node_features_orig_df.values, dtype=torch.float
            )

            try:
                adj_matrix_orig_df = pd.read_csv(adj_file_orig)
            except FileNotFoundError:
                print(
                    f"Original adj_matrix file not found: {adj_file_orig}. Skipping pair."
                )
                skip_current_pair = True
                raise

            if adj_matrix_orig_df.empty and node_features_orig_tensor.shape[0] > 0:
                print(
                    f"Warning: Original adj_matrix {adj_file_orig} is empty for a graph with {node_features_orig_tensor.shape[0]} nodes."
                )
                edge_index_orig = torch.empty((2, 0), dtype=torch.long)
            elif node_features_orig_tensor.shape[0] == 0:
                edge_index_orig = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index_orig = torch.tensor(
                    np.array(adj_matrix_orig_df.values.nonzero()), dtype=torch.long
                )

            original_graph_data = Data(
                x=node_features_orig_tensor,
                edge_index=edge_index_orig,
                cell=cell,
                gene=gene,
            )

        except (
            FileNotFoundError,
            ValueError,
            KeyError,
        ) as e_orig:  # Catch specific errors for original graph processing
            # This block now effectively handles and reports on the re-raised errors from inner try-excepts.
            # The specific print statements inside the inner blocks provide more detail.
            pass  # Continue to next pair in the main loop
        except Exception as e_general_orig:
            print(
                f"Unexpected error processing original graph for {cell}-{gene}: {e_general_orig}. Skipping pair."
            )
            skip_current_pair = True
            pass

        if skip_current_pair:
            continue  # Move to the next cell-gene pair

        # --- Process Augmented Graph ---
        aug_path_base = f"{path}/{cell}_aug"
        nodes_file_aug = f"{aug_path_base}/{gene}_node_matrix.csv"
        adj_file_aug = f"{aug_path_base}/{gene}_adj_matrix.csv"
        augmented_graph_data = None

        try:
            try:
                node_features_aug_df_raw = pd.read_csv(
                    nodes_file_aug, usecols=["count", "nuclear_position", "is_virtual"]
                )
            except ValueError:
                try:
                    node_features_aug_df_raw = pd.read_csv(
                        nodes_file_aug, usecols=["count", "nuclear_position"]
                    )
                except FileNotFoundError:
                    print(
                        f"Augmented node_matrix file not found: {nodes_file_aug}. Original graph will be kept if processed."
                    )
                    raise  # Re-raise to be caught by outer try-except for augmented graph
                except ValueError:
                    print(
                        f"Augmented node_matrix {nodes_file_aug} missing 'count' or 'nuclear_position'. Original graph will be kept."
                    )
                    raise

            if node_features_aug_df_raw.empty:
                print(
                    f"Augmented node_matrix {nodes_file_aug} is empty. Original graph will be kept."
                )
                raise FileNotFoundError

            node_features_aug_df = node_features_aug_df_raw.copy()

            total_count_aug = node_features_aug_df["count"].sum()
            node_features_aug_df["count_ratio"] = (
                0.0
                if total_count_aug == 0
                else node_features_aug_df["count"] / total_count_aug
            )

            count_features_aug_list = (
                node_features_aug_df["count_ratio"]
                .apply(lambda x: emb.nonlinear_transform_embedding(x, dim=12))
                .tolist()
            )
            count_features_aug_embedded_df = pd.DataFrame(
                count_features_aug_list,
                columns=[f"dim_{i}" for i in range(12)],
                index=node_features_aug_df.index,
            )

            node_features_aug_df["nuclear_position_mapped"] = (
                node_features_aug_df["nuclear_position"]
                .map({"inside": 0, "outside": 1, "boundary": 2, "edge": 3})
                .fillna(4)
                .astype(int)
            )
            position_aug_one_hot_df = pd.get_dummies(
                node_features_aug_df["nuclear_position_mapped"], prefix="pos", dtype=int
            )

            # Ensure augmented graphs also have the same position feature dimensionality (pos_0..pos_3)
            expected_pos_cols = ["pos_0", "pos_1", "pos_2", "pos_3"]
            for col in expected_pos_cols:
                if col not in position_aug_one_hot_df.columns:
                    position_aug_one_hot_df[col] = 0
            position_aug_one_hot_df = position_aug_one_hot_df[expected_pos_cols]

            final_node_features_aug_df = pd.concat(
                [count_features_aug_embedded_df, position_aug_one_hot_df], axis=1
            )
            node_features_aug_tensor = torch.tensor(
                final_node_features_aug_df.values, dtype=torch.float
            )

            try:
                adj_matrix_aug_df = pd.read_csv(adj_file_aug)
            except FileNotFoundError:
                print(
                    f"Augmented adj_matrix file not found: {adj_file_aug}. Original graph will be kept."
                )
                raise

            if adj_matrix_aug_df.empty and node_features_aug_tensor.shape[0] > 0:
                print(
                    f"Warning: Augmented adj_matrix {adj_file_aug} is empty for a graph with {node_features_aug_tensor.shape[0]} nodes."
                )
                edge_index_aug = torch.empty((2, 0), dtype=torch.long)
            elif node_features_aug_tensor.shape[0] == 0:
                edge_index_aug = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index_aug = torch.tensor(
                    np.array(adj_matrix_aug_df.values.nonzero()), dtype=torch.long
                )

            augmented_graph_data = Data(
                x=node_features_aug_tensor,
                edge_index=edge_index_aug,
                cell=cell,
                gene=gene,
            )

        except (FileNotFoundError, ValueError, KeyError) as e_aug:
            print(
                f"Failed to process augmented graph for {cell}-{gene}: {e_aug}. Original graph will be kept if available."
            )
            pass  # Keep original graph if it was successfully processed
        except Exception as e_general_aug:
            print(
                f"Unexpected error processing augmented graph for {cell}-{gene}: {e_general_aug}. Original graph will be kept if available."
            )
            pass

        # Add to lists if both original and augmented were processed successfully
        if original_graph_data and augmented_graph_data:
            graphs.append(original_graph_data)
            aug_graphs.append(augmented_graph_data)
        elif original_graph_data:  # Only original was successful
            print(
                f"Only original graph processed for {cell}-{gene}. Augmented failed or was missing."
            )
            # Decide if you want to add originals even if augmented is missing.
            # For now, we are only adding pairs. To change this, uncomment the lines below:
            # graphs.append(original_graph_data)
            # aug_graphs.append(original_graph_data) # Or some placeholder for aug_graph
            pass

    return graphs, aug_graphs


def generate_graph_data_target_weight3(
    dataset, df, path, n_sectors, m_rings, k_neighbor
):
    """
    Robustly loads original and augmented graph data for a target dataframe.
    This version uses sinusoidal embedding for count features.
    This version creates a 16-dimensional node feature vector:
    - 12 dims for sinusoidal count embedding.
    - 3 dims for one-hot encoded nuclear position ('inside', 'outside', 'boundary').
    - 1 dim for the 'is_edge' flag.
    Virtual nodes are processed normally with their actual features.
    """
    graphs = []
    aug_graphs = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Processing Graphs for generate_graph_data_target_weight3",
    ):
        cell = row["cell"]
        gene = row["gene"]

        # --- Process Original Graph ---
        raw_path_base = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}"
        nodes_file_orig = f"{raw_path_base}/{gene}_node_matrix.csv"
        adj_file_orig = f"{raw_path_base}/{gene}_adj_matrix.csv"
        original_graph_data = None

        try:
            # Step 1: Robustly read node data
            try:
                node_features_orig_df_raw = pd.read_csv(
                    nodes_file_orig,
                    usecols=["count", "nuclear_position", "is_edge", "is_virtual"],
                )
            except ValueError:
                try:
                    node_features_orig_df_raw = pd.read_csv(
                        nodes_file_orig,
                        usecols=["count", "nuclear_position", "is_edge"],
                    )
                    node_features_orig_df_raw["is_virtual"] = (
                        0  # Assume not virtual if column is missing
                    )
                except ValueError:
                    try:
                        node_features_orig_df_raw = pd.read_csv(
                            nodes_file_orig, usecols=["count", "nuclear_position"]
                        )
                        node_features_orig_df_raw["is_edge"] = (
                            0  # Assume not edge if column is missing
                        )
                        node_features_orig_df_raw["is_virtual"] = 0
                    except (FileNotFoundError, ValueError) as e:
                        print(
                            f"Original node_matrix file error for {nodes_file_orig}: {e}. Skipping pair."
                        )
                        raise e from None

            if node_features_orig_df_raw.empty:
                print(
                    f"Original node_matrix {nodes_file_orig} is empty. Skipping pair."
                )
                raise FileNotFoundError

            node_features_orig_df = node_features_orig_df_raw.copy()

            # Step 2: Feature Engineering
            # Count embedding (12-dim)
            total_count_orig = node_features_orig_df["count"].sum()
            node_features_orig_df["count_ratio"] = (
                0.0
                if total_count_orig == 0
                else node_features_orig_df["count"] / total_count_orig
            )
            count_features_list = (
                node_features_orig_df["count_ratio"]
                .apply(
                    lambda x: emb.get_sinusoidal_embedding_for_continuous_value(
                        x, dim=12
                    )
                )
                .tolist()
            )
            count_features_embedded_df = pd.DataFrame(
                count_features_list,
                columns=[f"dim_{i}" for i in range(12)],
                index=node_features_orig_df.index,
            )

            # Nuclear position one-hot (3-dim)
            expected_pos_cats = ["inside", "outside", "boundary"]
            node_features_orig_df["nuclear_position_cat"] = pd.Categorical(
                node_features_orig_df["nuclear_position"], categories=expected_pos_cats
            )
            position_one_hot_df = pd.get_dummies(
                node_features_orig_df["nuclear_position_cat"], prefix="pos", dtype=int
            )

            # is_edge feature (1-dim)
            is_edge_df = node_features_orig_df[["is_edge"]]

            # Combine features
            final_node_features_df = pd.concat(
                [count_features_embedded_df, position_one_hot_df, is_edge_df], axis=1
            )

            # Virtual nodes keep their original features (no longer set to zero)
            node_features_orig_tensor = torch.tensor(
                final_node_features_df.values, dtype=torch.float
            )

            # Step 3: Load Adjacency Matrix
            try:
                adj_matrix_orig_df = pd.read_csv(adj_file_orig)
            except FileNotFoundError:
                print(
                    f"Original adj_matrix file not found: {adj_file_orig}. Skipping pair."
                )
                raise

            if adj_matrix_orig_df.empty and node_features_orig_tensor.shape[0] > 0:
                edge_index_orig = torch.empty((2, 0), dtype=torch.long)
            elif node_features_orig_tensor.shape[0] == 0:
                edge_index_orig = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index_orig = torch.tensor(
                    np.array(adj_matrix_orig_df.values.nonzero()), dtype=torch.long
                )

            original_graph_data = Data(
                x=node_features_orig_tensor,
                edge_index=edge_index_orig,
                cell=cell,
                gene=gene,
            )

        except (FileNotFoundError, ValueError, KeyError) as e_orig:
            pass
        except Exception as e_general_orig:
            print(
                f"Unexpected error processing original graph for {cell}-{gene}: {e_general_orig}. Skipping pair."
            )
            pass

        if original_graph_data is None:
            continue

        # --- Process Augmented Graph ---
        aug_path_base = f"{path}/{cell}/{cell}_{n_sectors}_{m_rings}_k{k_neighbor}_aug"
        nodes_file_aug = f"{aug_path_base}/{gene}_node_matrix.csv"
        adj_file_aug = f"{aug_path_base}/{gene}_adj_matrix.csv"
        augmented_graph_data = None

        try:
            # Step 1: Robustly read node data
            try:
                node_features_aug_df_raw = pd.read_csv(
                    nodes_file_aug,
                    usecols=["count", "nuclear_position", "is_edge", "is_virtual"],
                )
            except ValueError:
                try:
                    node_features_aug_df_raw = pd.read_csv(
                        nodes_file_aug, usecols=["count", "nuclear_position", "is_edge"]
                    )
                    node_features_aug_df_raw["is_virtual"] = 0
                except ValueError:
                    try:
                        node_features_aug_df_raw = pd.read_csv(
                            nodes_file_aug, usecols=["count", "nuclear_position"]
                        )
                        node_features_aug_df_raw["is_edge"] = 0
                        node_features_aug_df_raw["is_virtual"] = 0
                    except (FileNotFoundError, ValueError) as e:
                        print(
                            f"Augmented node_matrix file error for {nodes_file_aug}: {e}. Original graph will be kept if processed."
                        )
                        raise e from None

            if node_features_aug_df_raw.empty:
                print(
                    f"Augmented node_matrix {nodes_file_aug} is empty. Original graph will be kept."
                )
                raise FileNotFoundError

            node_features_aug_df = node_features_aug_df_raw.copy()

            # Step 2: Feature Engineering for augmented graph
            total_count_aug = node_features_aug_df["count"].sum()
            node_features_aug_df["count_ratio"] = (
                0.0
                if total_count_aug == 0
                else node_features_aug_df["count"] / total_count_aug
            )
            count_features_aug_list = (
                node_features_aug_df["count_ratio"]
                .apply(
                    lambda x: emb.get_sinusoidal_embedding_for_continuous_value(
                        x, dim=12
                    )
                )
                .tolist()
            )
            count_features_aug_embedded_df = pd.DataFrame(
                count_features_aug_list,
                columns=[f"dim_{i}" for i in range(12)],
                index=node_features_aug_df.index,
            )

            node_features_aug_df["nuclear_position_cat"] = pd.Categorical(
                node_features_aug_df["nuclear_position"], categories=expected_pos_cats
            )
            position_aug_one_hot_df = pd.get_dummies(
                node_features_aug_df["nuclear_position_cat"], prefix="pos", dtype=int
            )

            is_edge_aug_df = node_features_aug_df[["is_edge"]]

            final_node_features_aug_df = pd.concat(
                [
                    count_features_aug_embedded_df,
                    position_aug_one_hot_df,
                    is_edge_aug_df,
                ],
                axis=1,
            )

            # Virtual nodes keep their original features (no longer set to zero)
            node_features_aug_tensor = torch.tensor(
                final_node_features_aug_df.values, dtype=torch.float
            )

            # Step 3: Load Adjacency Matrix for augmented graph
            try:
                adj_matrix_aug_df = pd.read_csv(adj_file_aug)
            except FileNotFoundError:
                print(
                    f"Augmented adj_matrix file not found: {adj_file_aug}. Original graph will be kept."
                )
                raise

            if adj_matrix_aug_df.empty and node_features_aug_tensor.shape[0] > 0:
                edge_index_aug = torch.empty((2, 0), dtype=torch.long)
            elif node_features_aug_tensor.shape[0] == 0:
                edge_index_aug = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index_aug = torch.tensor(
                    np.array(adj_matrix_aug_df.values.nonzero()), dtype=torch.long
                )

            augmented_graph_data = Data(
                x=node_features_aug_tensor,
                edge_index=edge_index_aug,
                cell=cell,
                gene=gene,
            )

        except (FileNotFoundError, ValueError, KeyError) as e_aug:
            pass
        except Exception as e_general_aug:
            print(
                f"Unexpected error processing augmented graph for {cell}-{gene}: {e_general_aug}. Original graph will be kept if available."
            )
            pass

        # Add to lists if both original and augmented were processed successfully
        if original_graph_data and augmented_graph_data:
            graphs.append(original_graph_data)
            aug_graphs.append(augmented_graph_data)
        elif original_graph_data:
            print(
                f"Only original graph processed for {cell}-{gene}. Augmented failed or was missing."
            )
            pass

    return graphs, aug_graphs
