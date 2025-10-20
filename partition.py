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
import ot  # 导入最优传输库
from matplotlib.patches import PathPatch
from matplotlib.path import Path
warnings.filterwarnings("ignore")

def classify_center_points_with_edge(center_points, nuclear_boundary_df_registered, is_edge, epsilon=0.1):
    polygon_coords = list(zip(nuclear_boundary_df_registered['x_c_s'], nuclear_boundary_df_registered['y_c_s']))
    polygon = Polygon(polygon_coords)  # 创建多边形
    classifications = []
    for idx, point in enumerate(center_points):
        if is_edge[idx]:  # 优先判断是否为边缘点
            classifications.append("edge")
            continue
        point_geom = Point(point)  # 将中心点转为点对象
        if polygon.contains(point_geom):  # 判断点的位置
            classifications.append("inside")
        elif polygon.touches(point_geom):
            classifications.append("boundary")
        else:
            distance_to_boundary = polygon.boundary.distance(point_geom) # 检查点是否接近边界
            if distance_to_boundary <= epsilon:
                classifications.append("boundary")
            else:
                classifications.append("outside")
    return classifications


def save_node_data_to_csv_old(center_points, is_virtual, plot_dir, gene, node_counts, k, nuclear_positions):
    node_data = []
    for idx, (x, y) in enumerate(center_points):
        node_data.append({
            'node_id': idx,
            'x': x,
            'y': y,
            'is_virtual': 1 if is_virtual[idx] else 0,  # 真实节点为 1，虚拟节点为 0
            'count': node_counts[idx],  # 点数
            'nuclear_position': nuclear_positions[idx]  # 核分类信息
        })
    node_df = pd.DataFrame(node_data)
    node_df.to_csv(os.path.join(plot_dir, f"{gene}_node_matrix.csv"), index=False)
    num_nodes = len(center_points)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                distance_matrix[i, j] = 0  # 同一节点距离为0
            elif is_virtual[i] and is_virtual[j]:
                # distance_matrix[i, j] = np.inf  # 虚拟节点与虚拟节点距离为无穷大
                distance_matrix[i, j] = 1e6
            elif is_virtual[i] or is_virtual[j]:
                # distance_matrix[i, j] = np.inf  # 虚拟节点与真实节点距离为无穷大
                distance_matrix[i, j] = 1e6
            else:
                distance_matrix[i, j] = np.linalg.norm(np.array(center_points[i]) - np.array(center_points[j]))  # 真实节点之间的欧几里得距离
    distance_matrix = pd.DataFrame(distance_matrix) # 保存距离矩阵
    distance_matrix.to_csv(os.path.join(plot_dir, f"{gene}_dis_matrix.csv"), index=False)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)  # 计算邻接矩阵
    for i in range(num_nodes):
        if is_virtual[i]:
            continue  # 跳过虚拟节点
        nearest_indices = np.argsort(distance_matrix[i])[: k+1]  # # 找到与节点i距离最近的k个节点，包含自身，故+k
        for idx in nearest_indices:
            if not is_virtual[idx]:  # 只连接真实节点
                adjacency_matrix[i, idx] = 1  # 连接节点i和最近的k个节点
    np.fill_diagonal(adjacency_matrix, 0)  # 自连接为0
    # 使邻接矩阵对称
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
    adjacency_matrix = pd.DataFrame(adjacency_matrix)
    adjacency_matrix.to_csv(os.path.join(plot_dir, f"{gene}_adj_matrix.csv"), index=False)

# 统计每个区域的散点数并计算分区质心 (质心位置相同)
def count_points_in_areas_same(df, n_sectors, m_rings, r):  
    df['theta'] = np.arctan2(df['y_c_s'], df['x_c_s'])
    df['radius'] = np.sqrt(df['x_c_s']**2 + df['y_c_s']**2)
    count_matrix = np.zeros((m_rings, n_sectors))  # 初始化计数矩阵
    theta_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    radius_edges = np.linspace(0, r, m_rings + 1)
    center_points = []
    point_counts = []  # 保存每个区域的点数
    is_virtual = []    # 标记虚拟节点
    is_edge = []       # 标记边缘节点
    for i in range(m_rings):  # 遍历所有环
        for j in range(n_sectors):  # 遍历所有扇区
            points_in_ring = df[(df['radius'] > radius_edges[i]) & (df['radius'] <= radius_edges[i + 1])]
            points_in_sector = points_in_ring[(points_in_ring['theta'] >= theta_edges[j]) & (points_in_ring['theta'] < theta_edges[j + 1])]
            count = len(points_in_sector)
            count_matrix[i, j] = count
            point_counts.append(count)  # 添加点数
            theta_center = (theta_edges[j] + theta_edges[j + 1]) / 2  # 计算当前区域的几何中心
            radius_center = (radius_edges[i] + radius_edges[i + 1]) / 2
            x_center, y_center = radius_center * np.cos(theta_center), radius_center * np.sin(theta_center)
            weight = count if count > 0 else 1  # 如果区域内没有点，权重为1（虚拟节点）            
            center_points.append((x_center, y_center))  # 保存当前区域的中心点
            is_virtual.append(False if count > 0 else True)  # 有点的区域为实际节点，无点的为虚拟节点
            is_edge.append(True if i == m_rings - 1 or i == m_rings - 2 else False)  # 最外两层环标记为边缘点
    return count_matrix, center_points, point_counts, is_virtual, is_edge

# 统计每个区域的散点数并计算分区质心 (质心位置不相同)
def count_points_in_areas(df, n_sectors, m_rings, r):  # 统计每个区域的散点数量
    df['theta'] = np.arctan2(df['y_c_s'], df['x_c_s'])
    df['radius'] = np.sqrt(df['x_c_s']**2 + df['y_c_s']**2)
    count_matrix = np.zeros((m_rings, n_sectors))
    theta_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    radius_edges = np.linspace(0, r, m_rings + 1)
    center_points = []
    point_counts = []  # 保存每个中心点的点数
    is_virtual = []    # 标记虚拟节点
    for i in range(m_rings):  # 计算每个区域的中心点并统计数量
        for j in range(n_sectors):
            points_in_ring = df[(df['radius'] > radius_edges[i]) & (df['radius'] <= radius_edges[i+1])]
            points_in_sector = points_in_ring[(points_in_ring['theta'] >= theta_edges[j]) & (points_in_ring['theta'] < theta_edges[j+1])]
            count = len(points_in_sector)
            count_matrix[i, j] = count
            point_counts.append(count)  # 添加点数
            theta_center = (theta_edges[j] + theta_edges[j+1]) / 2  # 计算分区的中心点
            radius_center = (radius_edges[i] + radius_edges[i+1]) / 2
            x_center, y_center = radius_center * np.cos(theta_center), radius_center * np.sin(theta_center)
            if count > 0:  # 计算质心
                x_center = points_in_sector['x_c_s'].mean()
                y_center = points_in_sector['y_c_s'].mean()
                is_virtual.append(False)
            else:
                is_virtual.append(True)
            center_points.append((x_center, y_center))
    return count_matrix, center_points, point_counts, is_virtual

def build_graph_k_nearest(center_points, k):
    edges = []
    center_points = np.array(center_points)# 使用k近邻算法生成连接边
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(center_points)  # k+1 因为包括自己
    distances, indices = nbrs.kneighbors(center_points)
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # 跳过第一个邻居（自己）
            edges.append((center_points[i], center_points[j]))       
    return edges

# 可视化细胞分区、中心点和连接
def plot_cell_partition(cell, df, center_points, point_counts, edges, r, gene, is_virtual, n_sectors, m_rings, plot_dir, nuclear_boundary_df_registered):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    ax.axis('off')
    theta = np.linspace(0, 2 * np.pi, n_sectors + 1) # 绘制细胞的同心圆和扇形
    radii = np.linspace(0, r, m_rings + 1)
    for rad in radii:
        circle = plt.Circle((0, 0), rad, color='grey', fill=False, linestyle='--', linewidth=0.5)
        ax.add_artist(circle)
    for angle in theta:
        ax.plot([0, r * np.cos(angle)], [0, r * np.sin(angle)], color='grey', linestyle='--', linewidth=0.5)
    ax.scatter(df['x_c_s'], df['y_c_s'], s=1, color='blue', label="Gene Points")
    center_points = np.array(center_points)  # 绘制区域中心点，使用区域内点的数量作为大小
    point_sizes = np.array(point_counts) * 0.2  # 根据需要调整缩放因子
    actual_centers = center_points[np.logical_not(is_virtual)]
    virtual_centers = center_points[is_virtual]
    for edge in edges:  # 绘制连接的边
        (x1, y1), (x2, y2) = edge
        start_index = np.where((center_points == (x1, y1)).all(axis=1))[0][0]
        end_index = np.where((center_points == (x2, y2)).all(axis=1))[0][0]
        if is_virtual[start_index] or is_virtual[end_index]:
            line_style = 'dashed'  # 虚拟节点相关的连线使用虚线
            color = 'gainsboro' # color = 'orange'
        else:
            line_style = 'solid'   # 实际节点之间的连线使用实线
            color = 'green'  
        ax.plot([x1, x2], [y1, y2], color=color, linestyle=line_style, linewidth=0.3)
    # 虚拟节点：灰色，固定大小
    ax.scatter(virtual_centers[:, 0], virtual_centers[:, 1], color='gainsboro', s=2) #, label="Virtual Region Centers"
    # 实际节点：红色，大小根据点数
    ax.scatter(actual_centers[:, 0], actual_centers[:, 1], color='red', s=point_sizes[np.logical_not(is_virtual)]) #, label="Actual Region Centers"
    polygon_coords = list(zip(nuclear_boundary_df_registered['x_c_s'], nuclear_boundary_df_registered['y_c_s']))
    polygon = Polygon(polygon_coords)
    boundary_x, boundary_y = zip(*polygon_coords)
    ax.plot(boundary_x, boundary_y, color='blue', linewidth=1)
    # colors = {'inside': 'green', 'outside': 'red', 'boundary': 'orange'}
    # for point, classification in zip(center_points, classifications):
    #     ax.scatter(*point, color=colors[classification], label=f'{classification}', s=25, edgecolor='grey')
    # 移除边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.title(f"Cell {cell} - Gene {gene}")
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, f"{gene}_partition_plot.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
def build_graph_with_networkx(center_points, edges, is_virtual):
    G = nx.Graph()
    for idx, (x, y) in enumerate(center_points):  # 添加节点和属性
        G.add_node(idx, pos=(x, y), is_virtual=is_virtual[idx]) 
    edges = [(tuple(edge[0]), tuple(edge[1])) for edge in edges]  # 添加边
    G.add_edges_from(edges)
    return G

def save_node_data_to_csv_nonposition(center_points, is_virtual, plot_dir, gene, node_counts, k):
    node_data = []
    for idx, (x, y) in enumerate(center_points):
        node_data.append({
            'node_id': idx,
            'x': x,
            'y': y,
            'is_virtual': 1 if is_virtual[idx] else 0,  # 真实节点为1，虚拟节点为0
            'count': node_counts[idx]  # 新增计数列
        })
    node_df = pd.DataFrame(node_data)
    node_df.to_csv(os.path.join(plot_dir, f"{gene}_node_matrix.csv"), index=False)
    num_nodes = len(center_points)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                distance_matrix[i, j] = 0  # 同一节点距离为0
            elif is_virtual[i] and is_virtual[j]:
                # distance_matrix[i, j] = np.inf  # 虚拟节点与虚拟节点距离为无穷大
                distance_matrix[i, j] = 1e6
            elif is_virtual[i] or is_virtual[j]:
                # distance_matrix[i, j] = np.inf  # 虚拟节点与真实节点距离为无穷大
                distance_matrix[i, j] = 1e6
            else:
                distance_matrix[i, j] = np.linalg.norm(np.array(center_points[i]) - np.array(center_points[j]))  # 真实节点之间的欧几里得距离
    distance_matrix = pd.DataFrame(distance_matrix) # 保存距离矩阵
    distance_matrix.to_csv(os.path.join(plot_dir, f"{gene}_dis_matrix.csv"), index=False)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)  # 计算邻接矩阵
    for i in range(num_nodes):
        if is_virtual[i]:
            continue  # 跳过虚拟节点
        nearest_indices = np.argsort(distance_matrix[i])[: k+1]  # # 找到与节点i距离最近的k个节点，包含自身，故+k
        for idx in nearest_indices:
            if not is_virtual[idx]:  # 只连接真实节点
                adjacency_matrix[i, idx] = 1  # 连接节点i和最近的k个节点
    np.fill_diagonal(adjacency_matrix, 0)  # 自连接为0
    adjacency_matrix = pd.DataFrame(adjacency_matrix)
    adjacency_matrix.to_csv(os.path.join(plot_dir, f"{gene}_adj_matrix.csv"), index=False)

def save_node_data_to_csv(center_points, is_virtual, is_edge, plot_dir, gene, node_counts, k , nuclear_positions):
    node_data = []
    for idx, (x, y) in enumerate(center_points):
        node_data.append({
            'node_id': idx,
            'x': x,
            'y': y,
            'is_virtual': 1 if is_virtual[idx] else 0,
            'is_edge': 1 if is_edge[idx] else 0,  # Added is_edge column
            'count': node_counts[idx],
            'nuclear_position': nuclear_positions[idx]
        })
    node_df = pd.DataFrame(node_data)
    node_df.to_csv(os.path.join(plot_dir, f"{gene}_node_matrix.csv"), index=False)
    num_nodes = len(center_points)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                distance_matrix[i, j] = 0  # 同一节点距离为0
            elif is_virtual[i] and is_virtual[j]:
                # distance_matrix[i, j] = np.inf  # 虚拟节点与虚拟节点距离为无穷大
                distance_matrix[i, j] = 1e6
            elif is_virtual[i] or is_virtual[j]:
                # distance_matrix[i, j] = np.inf  # 虚拟节点与真实节点距离为无穷大
                distance_matrix[i, j] = 1e6
            else:
                distance_matrix[i, j] = np.linalg.norm(np.array(center_points[i]) - np.array(center_points[j]))  # 真实节点之间的欧几里得距离
    distance_matrix = pd.DataFrame(distance_matrix) # 保存距离矩阵
    distance_matrix.to_csv(os.path.join(plot_dir, f"{gene}_dis_matrix.csv"), index=False)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)  # 计算邻接矩阵
    for i in range(num_nodes):
        if is_virtual[i]:
            continue  # 跳过虚拟节点
        nearest_indices = np.argsort(distance_matrix[i])[: k+1]  # # 找到与节点i距离最近的k个节点，包含自身，故+k
        for idx in nearest_indices:
            if not is_virtual[idx]:  # 只连接真实节点
                adjacency_matrix[i, idx] = 1  # 连接节点i和最近的k个节点
    np.fill_diagonal(adjacency_matrix, 0)  # 自连接为0
    adjacency_matrix = pd.DataFrame(adjacency_matrix)
    adjacency_matrix.to_csv(os.path.join(plot_dir, f"{gene}_adj_matrix.csv"), index=False)

def plot_cell_partition_heatmap_noposition(cell, gene, point_counts, n_sectors, m_rings, r, plot_dir):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    ax.axis('off')
    theta_edges = np.linspace(0, 2 * np.pi, n_sectors + 1)  # 扇区的角度
    radius_edges = np.linspace(0, r, m_rings + 1)  # 环形的半径
    # 将点数归一化到颜色范围 [0, 1]
    max_count = max(point_counts) if len(point_counts) > 0 else 1
    normalized_counts = np.array(point_counts) / max_count
    for sector_idx in range(n_sectors):  # 绘制每个分区并用颜色表示归一化的点数
        for ring_idx in range(m_rings):
            theta_start = theta_edges[sector_idx]
            theta_end = theta_edges[sector_idx + 1]
            radius_start = radius_edges[ring_idx]
            radius_end = radius_edges[ring_idx + 1] 
            # 获取当前分区的归一化点数，用颜色映射
            index = ring_idx * n_sectors + sector_idx
            count = normalized_counts[index] if index < len(normalized_counts) else 0
            color = plt.cm.YlOrRd(count)  # 使用viridis颜色映射
            path_data = [   # 构造分区的路径并绘制
                (Path.MOVETO, (-radius_start * np.cos(theta_start), -radius_start * np.sin(theta_start))),
                (Path.LINETO, (-radius_end * np.cos(theta_start), -radius_end * np.sin(theta_start))),
                (Path.LINETO, (-radius_end * np.cos(theta_end), -radius_end * np.sin(theta_end))),
                (Path.LINETO, (-radius_start * np.cos(theta_end), -radius_start * np.sin(theta_end))),
                (Path.CLOSEPOLY, (-radius_start * np.cos(theta_start), -radius_start * np.sin(theta_start))),
            ]
            path = Path([p[1] for p in path_data], [p[0] for p in path_data])
            patch = PathPatch(path, facecolor=color, edgecolor='grey', lw=0.5)
            ax.add_patch(patch)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.title(f"Cell {cell} - Gene {gene}")
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, f"{gene}_partition_heatmap.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def plot_cell_partition_heatmap(cell, gene, point_counts, n_sectors, m_rings, r, plot_dir, nuclear_boundary_df_registered):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    ax.axis('off')
    theta_edges = np.linspace(0, 2 * np.pi, n_sectors + 1)  # 扇区的角度
    radius_edges = np.linspace(0, r, m_rings + 1)  # 环形的半径
    # 将点数归一化到颜色范围 [0, 1]
    max_count = max(point_counts) if len(point_counts) > 0 else 1
    normalized_counts = np.array(point_counts) / max_count
    for sector_idx in range(n_sectors):  # 绘制每个分区并用颜色表示归一化的点数
        for ring_idx in range(m_rings):
            theta_start = theta_edges[sector_idx]
            theta_end = theta_edges[sector_idx + 1]
            radius_start = radius_edges[ring_idx]
            radius_end = radius_edges[ring_idx + 1] 
            # 获取当前分区的归一化点数，用颜色映射
            index = ring_idx * n_sectors + sector_idx
            count = normalized_counts[index] if index < len(normalized_counts) else 0
            color = plt.cm.YlOrRd(count)  # 使用viridis颜色映射
            path_data = [   # 构造分区的路径并绘制
                (Path.MOVETO, (-radius_start * np.cos(theta_start), -radius_start * np.sin(theta_start))),
                (Path.LINETO, (-radius_end * np.cos(theta_start), -radius_end * np.sin(theta_start))),
                (Path.LINETO, (-radius_end * np.cos(theta_end), -radius_end * np.sin(theta_end))),
                (Path.LINETO, (-radius_start * np.cos(theta_end), -radius_start * np.sin(theta_end))),
                (Path.CLOSEPOLY, (-radius_start * np.cos(theta_start), -radius_start * np.sin(theta_start))),
            ]
            path = Path([p[1] for p in path_data], [p[0] for p in path_data])
            patch = PathPatch(path, facecolor=color, edgecolor='grey', lw=0.5)
            ax.add_patch(patch)
    
    # 添加实际分区的质心点
    # center_points = np.array(center_points)
    # actual_centers = center_points[np.logical_not(is_virtual)]
    # virtual_centers = center_points[is_virtual]
    # ax.scatter(actual_centers[:, 0], actual_centers[:, 1], c='red', s=10, label='Actual Centers')
    # ax.scatter(virtual_centers[:, 0], virtual_centers[:, 1], c='grey', s=5, label='Virtual Centers')
    
    polygon_coords = list(zip(nuclear_boundary_df_registered['x_c_s'], nuclear_boundary_df_registered['y_c_s']))
    polygon = Polygon(polygon_coords)
    boundary_x, boundary_y = zip(*polygon_coords)
    ax.plot(boundary_x, boundary_y, color='blue', linewidth=1)
    # colors = {'inside': 'green', 'outside': 'red', 'boundary': 'orange'}
    # for point, classification in zip(center_points, classifications):
    #     ax.scatter(*point, color=colors[classification], label=f'{classification}', s=25, edgecolor='grey')
    # 移除边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.title(f"Cell {cell} - Gene {gene}")
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, f"{gene}_partition_heatmap.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def classify_nuclear_position(center_points, nuclear_boundary_df_registered, epsilon=0.1): # Renamed, removed is_edge
    polygon_coords = list(zip(nuclear_boundary_df_registered['x_c_s'], nuclear_boundary_df_registered['y_c_s']))
    polygon = Polygon(polygon_coords)  # 创建多边形
    classifications = []
    for point in center_points: # Removed idx, no is_edge check here
        point_geom = Point(point)  # 将中心点转为点对象
        if polygon.contains(point_geom):  # 判断点的位置
            classifications.append("inside")
        elif polygon.touches(point_geom):
            classifications.append("boundary")
        else:
            distance_to_boundary = polygon.boundary.distance(point_geom) # 检查点是否接近边界
            if distance_to_boundary <= epsilon:
                classifications.append("boundary")
            else:
                classifications.append("outside")
    return classifications

def plot_partition_nuclear_position(center_points, nuclear_boundary_df_registered,classifications, cell, gene, plot_dir):
    polygon_coords = list(zip(nuclear_boundary_df_registered['x_c_s'], nuclear_boundary_df_registered['y_c_s']))
    polygon = Polygon(polygon_coords)
    fig, ax = plt.subplots(figsize=(4, 4))
    boundary_x, boundary_y = zip(*polygon_coords)
    ax.plot(boundary_x, boundary_y, color='blue', linewidth=1)
    colors = {'inside': 'green', 'outside': 'red', 'boundary': 'orange'}
    for point, classification in zip(center_points, classifications):
        ax.scatter(*point, color=colors[classification], label=f'{classification}', s=25, edgecolor='grey')
    # 移除边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axis('off') 
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', 'box')
    ax.set_title(f'Cell {cell} - Gene {gene}')
    # plt.show()
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, f"{gene}_partition_nuclear_position.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
