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
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering
import multiprocessing as mp
import networkx as nx
import ot  # 导入最优传输库
warnings.filterwarnings("ignore")

def read_data(dataset, target_cells, df_registered, n_sectors, m_rings, k_neighbor, if_same):
    unique_genes = df_registered['gene'].unique()
    target_genes = unique_genes.tolist()
    # print(target_genes)
    matching_paths = []
    for cell in target_cells:
        if if_same == "yes":
            cell_dir = os.path.join(f"/home/lixiangyu/DMG/bento/GOT_code/2_normalized_cell/4_{dataset}_partition_same/", cell)
            if not os.path.exists(cell_dir):
                print(f"Directory {cell_dir} does not exist.")
                continue
        else:
            cell_dir = os.path.join(f"/home/lixiangyu/DMG/bento/GOT_code/2_normalized_cell/4_{dataset}_partition/", cell)
            if not os.path.exists(cell_dir):
                print(f"Directory {cell_dir} does not exist.")
                continue

        for root, dirs, files in os.walk(cell_dir):
            if f"{n_sectors}_{m_rings}_k{k_neighbor}" in root:
                for file in files:
                    gene_in_file = any(gene in file for gene in target_genes)   # 检查文件是否包含目标基因名称
                    # if gene_in_file and file.endswith("adjacency_matrix.csv"):  # 检查文件是否符合格式条件
                    if gene_in_file and file.endswith("distance_matrix.csv"):
                        file_path = os.path.join(root, file)
                        # file_path = file_path.replace("_adjacency_matrix.csv", "")
                        file_path = file_path.replace("_distance_matrix.csv", "")
                        matching_paths.append(file_path)
    return matching_paths

def load_graph_data(paths):
    graphs = {}
    for path in paths:
        cell_name = path.split('/')[-3]  # 假设细胞名位于路径的倒数第三部分
        gene_name = path.split('/')[-1]  # 假设基因名位于路径的倒数第二部分
        graph_name = f"{cell_name}_{gene_name}"  # 创建图的唯一名称
        data = pd.read_csv(f"{path}_nodes.csv")
        features = data[['x', 'y', 'is_virtual']].to_numpy() # 提取节点特征
        adj_matrix = pd.read_csv(f"{path}_adjacency_matrix.csv")
        # adj_matrix = pd.read_csv(f"{path}_distance_matrix.csv")
        weights = data['count'].to_numpy() # 提取权重
        # name = path.split('/')[-1]
        graphs[graph_name] = (features, adj_matrix, weights) # 将数据存储为元组
    return graphs
