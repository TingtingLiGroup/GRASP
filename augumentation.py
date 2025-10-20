import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rotate_nodes(node_matrix, angle):   # 函数：对节点进行旋转
    theta = np.radians(angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x, y = node_matrix['x'], node_matrix['y']
    node_matrix['x'] = x * cos_theta - y * sin_theta
    node_matrix['y'] = x * sin_theta + y * cos_theta
    return node_matrix

def dropout_nodes(adj_matrix, node_matrix, dropout_ratio):  # 函数：节点和边的随机删除
    real_nodes = node_matrix[node_matrix['is_virtual'] == 0].index
    # n = len(node_matrix)
    num_real_nodes = len(real_nodes)
    num_drop = int(num_real_nodes * dropout_ratio)
    drop_nodes = np.random.choice(real_nodes, size=num_drop, replace=False)  # 随机选择节点进行删除
    node_matrix.loc[drop_nodes, 'is_virtual'] = 1  # 更新节点矩阵
    for node in drop_nodes:  # 更新邻接矩阵
        adj_matrix.iloc[node, :] = 0
        adj_matrix.iloc[:, node] = 0
    return adj_matrix, node_matrix

# def add_nodes(adj_matrix, node_matrix, add_ratio):
#     n = len(node_matrix)
#     num_drop = int(n * dropout_ratio)
#     add_nodes = np.random.choice(node_matrix.index, size=num_drop, replace=False)  # 随机选择节点进行删除
#     node_matrix.loc[add_nodes, 'is_virtual'] = 0  # 更新节点矩阵
#     for node in add_nodes:  # 更新邻接矩阵
#         for i in range(len(node_matrix)):
#             if node_matrix.loc[i, 'is_virtual'] == 0:  # 只有可见节点才会连接 
#                 adj_matrix.iloc[node, i] = 1
#                 adj_matrix.iloc[i, node] = 1
#     return adj_matrix, node_matrix

def plot_graph(adj_matrix_before, node_matrix_before, adj_matrix_after, node_matrix_after, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
    titles = [f"{title} (Original)", f"{title} (After Dropout)"]
    adj_matrices = [adj_matrix_before, adj_matrix_after]
    node_matrices = [node_matrix_before, node_matrix_after]
    for ax, adj_matrix, node_matrix, sub_title in zip(axes, adj_matrices, node_matrices, titles):
        ax.set_title(sub_title)
        for i in range(len(adj_matrix)):  # 绘制边
            for j in range(len(adj_matrix)):
                if adj_matrix.iloc[i, j] == 1:
                    x_coords = [node_matrix.loc[i, 'x'], node_matrix.loc[j, 'x']]
                    y_coords = [node_matrix.loc[i, 'y'], node_matrix.loc[j, 'y']]
                    ax.plot(x_coords, y_coords, 'gray', alpha=0.6, linewidth=0.5)
        virtual_nodes = node_matrix[node_matrix['is_virtual'] == 1]  # 绘制节点
        real_nodes = node_matrix[node_matrix['is_virtual'] == 0]
        ax.scatter(virtual_nodes['x'], virtual_nodes['y'], color='lightgray', label='Virtual Nodes', s=5)
        ax.scatter(real_nodes['x'], real_nodes['y'], color='red', label='Real Nodes', s=5)
        # ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    plt.tight_layout()
    plt.savefig(f"{save_path}/{title}_comparison.png")
    # plt.show()
    # plt.figure(figsize=(4, 4))
    # plt.title(f"{title}")
    # for i in range(len(adj_matrix)):   # 绘制边
    #     for j in range(len(adj_matrix)):
    #         if adj_matrix.iloc[i, j] == 1:
    #             x_coords = [node_matrix.loc[i, 'x'], node_matrix.loc[j, 'x']]
    #             y_coords = [node_matrix.loc[i, 'y'], node_matrix.loc[j, 'y']]
    #             plt.plot(x_coords, y_coords, 'gray', alpha=0.6, linewidth=0.5)
    # virtual_nodes = node_matrix[node_matrix['is_virtual'] == 1]   # 绘制节点
    # real_nodes = node_matrix[node_matrix['is_virtual'] == 0]
    # plt.scatter(virtual_nodes['x'], virtual_nodes['y'], color='lightgray', label='virtual nodes', s=5)
    # plt.scatter(real_nodes['x'], real_nodes['y'], color='red', label='real_nodes', s=5)
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # # plt.grid()
    # plt.savefig(f"{save_path}/{title}.png")
    # plt.show()
    
    

