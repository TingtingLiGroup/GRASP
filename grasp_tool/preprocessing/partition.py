import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns
import pickle
from shapely.geometry import Polygon, Point
import warnings
from scipy.spatial import distance_matrix
from tqdm import tqdm
import os
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering
import multiprocessing as mp
import networkx as nx

# import ot
from matplotlib.patches import PathPatch
from matplotlib.path import Path

warnings.filterwarnings("ignore")


def classify_center_points_with_edge(
    center_points, nuclear_boundary_df_registered, is_edge, epsilon=0.1
):
    polygon_coords = list(
        zip(
            nuclear_boundary_df_registered["x_c_s"],
            nuclear_boundary_df_registered["y_c_s"],
        )
    )
    polygon = Polygon(polygon_coords)
    classifications = []
    for idx, point in enumerate(center_points):
        if is_edge[idx]:
            classifications.append("edge")
            continue
        point_geom = Point(point)
        if polygon.contains(point_geom):
            classifications.append("inside")
        elif polygon.touches(point_geom):
            classifications.append("boundary")
        else:
            distance_to_boundary = polygon.boundary.distance(point_geom)
            if distance_to_boundary <= epsilon:
                classifications.append("boundary")
            else:
                classifications.append("outside")
    return classifications


def save_node_data_to_csv_old(
    center_points, is_virtual, plot_dir, gene, node_counts, k, nuclear_positions
):
    node_data = []
    for idx, (x, y) in enumerate(center_points):
        node_data.append(
            {
                "node_id": idx,
                "x": x,
                "y": y,
                "is_virtual": 1 if is_virtual[idx] else 0,
                "count": node_counts[idx],
                "nuclear_position": nuclear_positions[idx],
            }
        )
    node_df = pd.DataFrame(node_data)
    node_df.to_csv(os.path.join(plot_dir, f"{gene}_node_matrix.csv"), index=False)
    num_nodes = len(center_points)
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                distance_matrix[i, j] = 0
            elif is_virtual[i] and is_virtual[j]:
                # distance_matrix[i, j] = np.inf
                distance_matrix[i, j] = 1e6
            elif is_virtual[i] or is_virtual[j]:
                # distance_matrix[i, j] = np.inf
                distance_matrix[i, j] = 1e6
            else:
                distance_matrix[i, j] = np.linalg.norm(
                    np.array(center_points[i]) - np.array(center_points[j])
                )
    distance_matrix = pd.DataFrame(distance_matrix)
    distance_matrix.to_csv(
        os.path.join(plot_dir, f"{gene}_dis_matrix.csv"), index=False
    )
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        if is_virtual[i]:
            continue
        nearest_indices = np.argsort(distance_matrix[i])[: k + 1]
        for idx in nearest_indices:
            if not is_virtual[idx]:
                adjacency_matrix[i, idx] = 1
    np.fill_diagonal(adjacency_matrix, 0)
    # Make adjacency symmetric.
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
    adjacency_matrix = pd.DataFrame(adjacency_matrix)
    adjacency_matrix.to_csv(
        os.path.join(plot_dir, f"{gene}_adj_matrix.csv"), index=False
    )


# Count points per sector/ring (shared centroid locations)
def count_points_in_areas_same(df, n_sectors, m_rings, r):
    df["theta"] = np.arctan2(df["y_c_s"], df["x_c_s"])
    df["radius"] = np.sqrt(df["x_c_s"] ** 2 + df["y_c_s"] ** 2)
    count_matrix = np.zeros((m_rings, n_sectors))
    theta_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    radius_edges = np.linspace(0, r, m_rings + 1)
    center_points = []
    point_counts = []
    is_virtual = []
    is_edge = []
    for i in range(m_rings):
        for j in range(n_sectors):
            points_in_ring = df[
                (df["radius"] > radius_edges[i]) & (df["radius"] <= radius_edges[i + 1])
            ]
            points_in_sector = points_in_ring[
                (points_in_ring["theta"] >= theta_edges[j])
                & (points_in_ring["theta"] < theta_edges[j + 1])
            ]
            count = len(points_in_sector)
            count_matrix[i, j] = count
            point_counts.append(count)
            theta_center = (theta_edges[j] + theta_edges[j + 1]) / 2
            radius_center = (radius_edges[i] + radius_edges[i + 1]) / 2
            x_center, y_center = (
                radius_center * np.cos(theta_center),
                radius_center * np.sin(theta_center),
            )
            weight = count if count > 0 else 1
            center_points.append((x_center, y_center))
            is_virtual.append(False if count > 0 else True)
            is_edge.append(True if i == m_rings - 1 or i == m_rings - 2 else False)
    return count_matrix, center_points, point_counts, is_virtual, is_edge


# Count points per sector/ring (centroid varies per sector)
def count_points_in_areas(df, n_sectors, m_rings, r):
    df["theta"] = np.arctan2(df["y_c_s"], df["x_c_s"])
    df["radius"] = np.sqrt(df["x_c_s"] ** 2 + df["y_c_s"] ** 2)
    count_matrix = np.zeros((m_rings, n_sectors))
    theta_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    radius_edges = np.linspace(0, r, m_rings + 1)
    center_points = []
    point_counts = []
    is_virtual = []
    for i in range(m_rings):
        for j in range(n_sectors):
            points_in_ring = df[
                (df["radius"] > radius_edges[i]) & (df["radius"] <= radius_edges[i + 1])
            ]
            points_in_sector = points_in_ring[
                (points_in_ring["theta"] >= theta_edges[j])
                & (points_in_ring["theta"] < theta_edges[j + 1])
            ]
            count = len(points_in_sector)
            count_matrix[i, j] = count
            point_counts.append(count)
            theta_center = (theta_edges[j] + theta_edges[j + 1]) / 2
            radius_center = (radius_edges[i] + radius_edges[i + 1]) / 2
            x_center, y_center = (
                radius_center * np.cos(theta_center),
                radius_center * np.sin(theta_center),
            )
            if count > 0:
                x_center = points_in_sector["x_c_s"].mean()
                y_center = points_in_sector["y_c_s"].mean()
                is_virtual.append(False)
            else:
                is_virtual.append(True)
            center_points.append((x_center, y_center))
    return count_matrix, center_points, point_counts, is_virtual


def build_graph_k_nearest(center_points, k):
    edges = []
    center_points = np.array(center_points)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(center_points)
    distances, indices = nbrs.kneighbors(center_points)
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            edges.append((center_points[i], center_points[j]))
    return edges


# Visualize partition, centers, and edges
def plot_cell_partition(
    cell,
    df,
    center_points,
    point_counts,
    edges,
    r,
    gene,
    is_virtual,
    n_sectors,
    m_rings,
    plot_dir,
    nuclear_boundary_df_registered,
):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect("equal")
    ax.axis("off")
    theta = np.linspace(0, 2 * np.pi, n_sectors + 1)
    radii = np.linspace(0, r, m_rings + 1)
    for rad in radii:
        circle = plt.Circle(
            (0, 0), rad, color="grey", fill=False, linestyle="--", linewidth=0.5
        )
        ax.add_artist(circle)
    for angle in theta:
        ax.plot(
            [0, r * np.cos(angle)],
            [0, r * np.sin(angle)],
            color="grey",
            linestyle="--",
            linewidth=0.5,
        )
    ax.scatter(df["x_c_s"], df["y_c_s"], s=1, color="blue", label="Gene Points")
    center_points = np.array(center_points)
    point_sizes = np.array(point_counts) * 0.2
    actual_centers = center_points[np.logical_not(is_virtual)]
    virtual_centers = center_points[is_virtual]
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        start_index = np.where((center_points == (x1, y1)).all(axis=1))[0][0]
        end_index = np.where((center_points == (x2, y2)).all(axis=1))[0][0]
        if is_virtual[start_index] or is_virtual[end_index]:
            line_style = "dashed"
            color = "gainsboro"  # color = 'orange'
        else:
            line_style = "solid"
            color = "green"
        ax.plot([x1, x2], [y1, y2], color=color, linestyle=line_style, linewidth=0.3)
    ax.scatter(
        virtual_centers[:, 0], virtual_centers[:, 1], color="gainsboro", s=2
    )  # , label="Virtual Region Centers"
    ax.scatter(
        actual_centers[:, 0],
        actual_centers[:, 1],
        color="red",
        s=point_sizes[np.logical_not(is_virtual)],
    )  # , label="Actual Region Centers"
    polygon_coords = list(
        zip(
            nuclear_boundary_df_registered["x_c_s"],
            nuclear_boundary_df_registered["y_c_s"],
        )
    )
    polygon = Polygon(polygon_coords)
    boundary_x, boundary_y = zip(*polygon_coords)
    ax.plot(boundary_x, boundary_y, color="blue", linewidth=1)
    # colors = {'inside': 'green', 'outside': 'red', 'boundary': 'orange'}
    # for point, classification in zip(center_points, classifications):
    #     ax.scatter(*point, color=colors[classification], label=f'{classification}', s=25, edgecolor='grey')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.title(f"Cell {cell} - Gene {gene}")
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(
        os.path.join(plot_dir, f"{gene}_partition_plot.png"),
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    plt.close()


def build_graph_with_networkx(center_points, edges, is_virtual):
    G = nx.Graph()
    for idx, (x, y) in enumerate(center_points):
        G.add_node(idx, pos=(x, y), is_virtual=is_virtual[idx])
    edges = [(tuple(edge[0]), tuple(edge[1])) for edge in edges]
    G.add_edges_from(edges)
    return G


def save_node_data_to_csv_nonposition(
    center_points, is_virtual, plot_dir, gene, node_counts, k
):
    node_data = []
    for idx, (x, y) in enumerate(center_points):
        node_data.append(
            {
                "node_id": idx,
                "x": x,
                "y": y,
                "is_virtual": 1 if is_virtual[idx] else 0,
                "count": node_counts[idx],
            }
        )
    node_df = pd.DataFrame(node_data)
    node_df.to_csv(os.path.join(plot_dir, f"{gene}_node_matrix.csv"), index=False)
    num_nodes = len(center_points)
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                distance_matrix[i, j] = 0
            elif is_virtual[i] and is_virtual[j]:
                # distance_matrix[i, j] = np.inf
                distance_matrix[i, j] = 1e6
            elif is_virtual[i] or is_virtual[j]:
                # distance_matrix[i, j] = np.inf
                distance_matrix[i, j] = 1e6
            else:
                distance_matrix[i, j] = np.linalg.norm(
                    np.array(center_points[i]) - np.array(center_points[j])
                )
    distance_matrix = pd.DataFrame(distance_matrix)
    distance_matrix.to_csv(
        os.path.join(plot_dir, f"{gene}_dis_matrix.csv"), index=False
    )
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        if is_virtual[i]:
            continue
        nearest_indices = np.argsort(distance_matrix[i])[: k + 1]
        for idx in nearest_indices:
            if not is_virtual[idx]:
                adjacency_matrix[i, idx] = 1
    np.fill_diagonal(adjacency_matrix, 0)
    adjacency_matrix = pd.DataFrame(adjacency_matrix)
    adjacency_matrix.to_csv(
        os.path.join(plot_dir, f"{gene}_adj_matrix.csv"), index=False
    )


def save_node_data_to_csv(
    center_points,
    is_virtual,
    is_edge,
    plot_dir,
    gene,
    node_counts,
    k,
    nuclear_positions,
):
    node_data = []
    for idx, (x, y) in enumerate(center_points):
        node_data.append(
            {
                "node_id": idx,
                "x": x,
                "y": y,
                "is_virtual": 1 if is_virtual[idx] else 0,
                "is_edge": 1 if is_edge[idx] else 0,  # Added is_edge column
                "count": node_counts[idx],
                "nuclear_position": nuclear_positions[idx],
            }
        )
    node_df = pd.DataFrame(node_data)
    node_df.to_csv(os.path.join(plot_dir, f"{gene}_node_matrix.csv"), index=False)
    num_nodes = len(center_points)
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                distance_matrix[i, j] = 0
            elif is_virtual[i] and is_virtual[j]:
                # distance_matrix[i, j] = np.inf
                distance_matrix[i, j] = 1e6
            elif is_virtual[i] or is_virtual[j]:
                # distance_matrix[i, j] = np.inf
                distance_matrix[i, j] = 1e6
            else:
                distance_matrix[i, j] = np.linalg.norm(
                    np.array(center_points[i]) - np.array(center_points[j])
                )
    distance_matrix = pd.DataFrame(distance_matrix)
    distance_matrix.to_csv(
        os.path.join(plot_dir, f"{gene}_dis_matrix.csv"), index=False
    )
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        if is_virtual[i]:
            continue
        nearest_indices = np.argsort(distance_matrix[i])[: k + 1]
        for idx in nearest_indices:
            if not is_virtual[idx]:
                adjacency_matrix[i, idx] = 1
    np.fill_diagonal(adjacency_matrix, 0)
    adjacency_matrix = pd.DataFrame(adjacency_matrix)
    adjacency_matrix.to_csv(
        os.path.join(plot_dir, f"{gene}_adj_matrix.csv"), index=False
    )


def plot_cell_partition_heatmap_noposition(
    cell, gene, point_counts, n_sectors, m_rings, r, plot_dir
):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect("equal")
    ax.axis("off")
    theta_edges = np.linspace(0, 2 * np.pi, n_sectors + 1)
    radius_edges = np.linspace(0, r, m_rings + 1)
    max_count = max(point_counts) if len(point_counts) > 0 else 1
    normalized_counts = np.array(point_counts) / max_count
    for sector_idx in range(n_sectors):
        for ring_idx in range(m_rings):
            theta_start = theta_edges[sector_idx]
            theta_end = theta_edges[sector_idx + 1]
            radius_start = radius_edges[ring_idx]
            radius_end = radius_edges[ring_idx + 1]
            index = ring_idx * n_sectors + sector_idx
            count = normalized_counts[index] if index < len(normalized_counts) else 0
            color = plt.cm.YlOrRd(count)
            path_data = [
                (
                    Path.MOVETO,
                    (
                        -radius_start * np.cos(theta_start),
                        -radius_start * np.sin(theta_start),
                    ),
                ),
                (
                    Path.LINETO,
                    (
                        -radius_end * np.cos(theta_start),
                        -radius_end * np.sin(theta_start),
                    ),
                ),
                (
                    Path.LINETO,
                    (-radius_end * np.cos(theta_end), -radius_end * np.sin(theta_end)),
                ),
                (
                    Path.LINETO,
                    (
                        -radius_start * np.cos(theta_end),
                        -radius_start * np.sin(theta_end),
                    ),
                ),
                (
                    Path.CLOSEPOLY,
                    (
                        -radius_start * np.cos(theta_start),
                        -radius_start * np.sin(theta_start),
                    ),
                ),
            ]
            path = Path([p[1] for p in path_data], [p[0] for p in path_data])
            patch = PathPatch(path, facecolor=color, edgecolor="grey", lw=0.5)
            ax.add_patch(patch)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.title(f"Cell {cell} - Gene {gene}")
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(
        os.path.join(plot_dir, f"{gene}_partition_heatmap.png"),
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    plt.close()


def plot_cell_partition_heatmap(
    cell,
    gene,
    point_counts,
    n_sectors,
    m_rings,
    r,
    plot_dir,
    nuclear_boundary_df_registered,
):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect("equal")
    ax.axis("off")
    theta_edges = np.linspace(0, 2 * np.pi, n_sectors + 1)
    radius_edges = np.linspace(0, r, m_rings + 1)
    max_count = max(point_counts) if len(point_counts) > 0 else 1
    normalized_counts = np.array(point_counts) / max_count
    for sector_idx in range(n_sectors):
        for ring_idx in range(m_rings):
            theta_start = theta_edges[sector_idx]
            theta_end = theta_edges[sector_idx + 1]
            radius_start = radius_edges[ring_idx]
            radius_end = radius_edges[ring_idx + 1]
            index = ring_idx * n_sectors + sector_idx
            count = normalized_counts[index] if index < len(normalized_counts) else 0
            color = plt.cm.YlOrRd(count)
            path_data = [
                (
                    Path.MOVETO,
                    (
                        -radius_start * np.cos(theta_start),
                        -radius_start * np.sin(theta_start),
                    ),
                ),
                (
                    Path.LINETO,
                    (
                        -radius_end * np.cos(theta_start),
                        -radius_end * np.sin(theta_start),
                    ),
                ),
                (
                    Path.LINETO,
                    (-radius_end * np.cos(theta_end), -radius_end * np.sin(theta_end)),
                ),
                (
                    Path.LINETO,
                    (
                        -radius_start * np.cos(theta_end),
                        -radius_start * np.sin(theta_end),
                    ),
                ),
                (
                    Path.CLOSEPOLY,
                    (
                        -radius_start * np.cos(theta_start),
                        -radius_start * np.sin(theta_start),
                    ),
                ),
            ]
            path = Path([p[1] for p in path_data], [p[0] for p in path_data])
            patch = PathPatch(path, facecolor=color, edgecolor="grey", lw=0.5)
            ax.add_patch(patch)

    # Optional: add centroid markers
    # center_points = np.array(center_points)
    # actual_centers = center_points[np.logical_not(is_virtual)]
    # virtual_centers = center_points[is_virtual]
    # ax.scatter(actual_centers[:, 0], actual_centers[:, 1], c='red', s=10, label='Actual Centers')
    # ax.scatter(virtual_centers[:, 0], virtual_centers[:, 1], c='grey', s=5, label='Virtual Centers')

    polygon_coords = list(
        zip(
            nuclear_boundary_df_registered["x_c_s"],
            nuclear_boundary_df_registered["y_c_s"],
        )
    )
    polygon = Polygon(polygon_coords)
    boundary_x, boundary_y = zip(*polygon_coords)
    ax.plot(boundary_x, boundary_y, color="blue", linewidth=1)
    # colors = {'inside': 'green', 'outside': 'red', 'boundary': 'orange'}
    # for point, classification in zip(center_points, classifications):
    #     ax.scatter(*point, color=colors[classification], label=f'{classification}', s=25, edgecolor='grey')
    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.title(f"Cell {cell} - Gene {gene}")
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(
        os.path.join(plot_dir, f"{gene}_partition_heatmap.png"),
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    plt.close()


def classify_nuclear_position(
    center_points, nuclear_boundary_df_registered, epsilon=0.1
):  # Renamed, removed is_edge
    polygon_coords = list(
        zip(
            nuclear_boundary_df_registered["x_c_s"],
            nuclear_boundary_df_registered["y_c_s"],
        )
    )
    polygon = Polygon(polygon_coords)
    classifications = []
    for point in center_points:  # Removed idx, no is_edge check here
        point_geom = Point(point)
        if polygon.contains(point_geom):
            classifications.append("inside")
        elif polygon.touches(point_geom):
            classifications.append("boundary")
        else:
            distance_to_boundary = polygon.boundary.distance(point_geom)
            if distance_to_boundary <= epsilon:
                classifications.append("boundary")
            else:
                classifications.append("outside")
    return classifications


def plot_partition_nuclear_position(
    center_points, nuclear_boundary_df_registered, classifications, cell, gene, plot_dir
):
    polygon_coords = list(
        zip(
            nuclear_boundary_df_registered["x_c_s"],
            nuclear_boundary_df_registered["y_c_s"],
        )
    )
    polygon = Polygon(polygon_coords)
    fig, ax = plt.subplots(figsize=(4, 4))
    boundary_x, boundary_y = zip(*polygon_coords)
    ax.plot(boundary_x, boundary_y, color="blue", linewidth=1)
    colors = {"inside": "green", "outside": "red", "boundary": "orange"}
    for point, classification in zip(center_points, classifications):
        ax.scatter(
            *point,
            color=colors[classification],
            label=f"{classification}",
            s=25,
            edgecolor="grey",
        )
    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.axis("off")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", "box")
    ax.set_title(f"Cell {cell} - Gene {gene}")
    # plt.show()
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(
        os.path.join(plot_dir, f"{gene}_partition_nuclear_position.png"),
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    plt.close()
