import numpy as np
import matplotlib.pyplot as plt
import ot  # Python Optimal Transport library
import pandas as pd
import os
import seaborn as sns

# 随机生成均匀分布的散点
def generate_random_points(radius, num_points): # 在圆形区域内随机生成散点
    center = (0, 0) 
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = radius * np.sqrt(np.random.uniform(0, 1, num_points))
    # radii = np.random.uniform(0, radius, num_points)  # 改为线性分布

    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    points_df = pd.DataFrame({'x': x, 'y': y})
    return points_df, np.vstack([x, y]).T # 每一行代表一个散点的 (x, y) 坐标

def simulate_expected_distance(radius, num_points, num_simulations=10): 
# 两个随机生成的点集之间的 Wasserstein 距离
    distances = []
    for _ in range(num_simulations):
        # 在 [0, 1]² 区域中生成 N 个均匀分布的点
        _, points1 = generate_random_points(radius, num_points)
        _, points2 = generate_random_points(radius, num_points)
        a = np.ones(num_points) / num_points
        b = np.ones(num_points) / num_points
        cost_matrix = ot.dist(points1, points2, metric='euclidean') # 计算成本矩阵
        # 计算 Wasserstein 距离
        # distance = ot.sinkhorn2(a, b, cost_matrix, reg=0.1)  # 返回平方距离
        distance = ot.emd2(a, b, cost_matrix) # 计算两个点集之间的 Wasserstein 距离
        distances.append(distance)
    return np.mean(distances)  # 返回期望值

def theoretical_expected_distance(radius, num_points, alpha=0.5):  # 使用理论公式计算样本点数 N 的 Wasserstein 距离期望值。
    return radius * num_points ** (-alpha)   # 参数:N (int): 样本点数。  C (float): 公式中的常数。 alpha (float): 收敛速度的指数。

def calculate_adjusted_wasserstein_distance(original_points, target_points, num_points, expected_distance, epsilon=0.1):
    # 归一化点坐标
    # original_points = (original_points - np.min(original_points, axis=0)) / (np.max(original_points, axis=0) - np.min(original_points, axis=0))
    # target_points = (target_points - np.min(target_points, axis=0)) / (np.max(target_points, axis=0) - np.min(target_points, axis=0))
    # # 下采样点集
    # sampled_original = original_points[np.random.choice(len(original_points), num_points, replace=False)]
    # sampled_target = target_points[np.random.choice(len(target_points), num_points, replace=False)]
    # # 设置均匀权重
    a = np.ones(num_points) / num_points
    b = np.ones(num_points) / num_points
    # 计算成本矩阵
    cost_matrix = ot.dist(original_points, target_points, metric='euclidean')
    
    # cost_matrix = ot.dist(sampled_original, sampled_target, metric='euclidean') + 1e-9
    # 计算原始 Wasserstein 距离
    # wasserstein_distance = np.sqrt(ot.sinkhorn2(a, b, cost_matrix, reg=epsilon)) # 返回平方距离
    wasserstein_distance = ot.emd2(a, b, cost_matrix)
    adjusted_distance = abs(wasserstein_distance - expected_distance)  # 调整 Wasserstein 距离
    return wasserstein_distance, adjusted_distance # 计算实际的 Wasserstein 距离，并返回它与预期距离的差异

def plot_points(dataset, data1, data2, gene, cell, radius, wasserstein_distance, path): # 可视化原始基因表达点和生成的目标均匀分布点的差异
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    # 子图1：原始分布
    axes[0].scatter(data1['x_c_s'], data1['y_c_s'], s=3, alpha=0.6, color='blue', label=f'Gene: {gene}')
    circle1 = plt.Circle((0, 0), radius, color='black', fill=False, label='Cell Boundary')  # 画细胞边界
    axes[0].add_patch(circle1)
    axes[0].set_title(f'Original Points', fontsize=10)
    axes[0].set_aspect('equal')
    axes[0].axis('off')

    # 子图2：目标均匀分布
    axes[1].scatter(data2['x'], data2['y'], s=3, alpha=0.6, color='green', label=f'Target Points')
    circle2 = plt.Circle((0, 0), radius, color='black', fill=False, label='Cell Boundary')  # 画细胞边界
    axes[1].add_patch(circle2)
    axes[1].set_title(f'Target Points', fontsize=10)
    axes[1].set_aspect('equal')
    axes[1].axis('off')

    num = len(data1)
    fig.suptitle(f'Cell: {cell} - Gene: {gene}\nMean_adjusted_distances: {wasserstein_distance:.4f} - num: {num}', fontsize=12)
    save_dir=f'{path}/{gene}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{cell}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    # plt.tight_layout()
    # plt.show()
    
    
def statics_plot(dataset, gene, path):
    file = f"{path}/{gene}_distances.csv"
    if not os.path.exists(file):
        print(f"1. Skipping {gene} (file not found).")

    else:
        df = pd.read_csv(f'{path}/{gene}_distances.csv', index_col=0)
        mean_value = df["mean_adjusted_distances"].mean()  # 统计信息
        median_value = df["mean_adjusted_distances"].median()
        std_value = df["mean_adjusted_distances"].std()
        # print(f"Mean: {mean_value:.3f}, Median: {median_value:.3f}, Std Dev: {std_value:.3f}")
        plt.figure(figsize=(4, 3))
        sns.histplot(df["mean_wasserstein_distances"], kde=True, color="skyblue", edgecolor="grey", bins=20, label="mean_wasserstein_distances") # 每个细胞的 Wasserstein 距离均值，表示测量的真实 Wasserstein 距离
        sns.histplot(df["expected_distance"], kde=True, color="royalblue", edgecolor="grey", bins=20, label="expected_distance") # 每个细胞的理论 Wasserstein 距离均值，表示理论上的 Wasserstein 距离
        sns.histplot(df["mean_adjusted_distances"], kde=True, color="pink", edgecolor="grey", bins=20, label="mean_adjusted_distances") # 每个细胞的调整后的 Wasserstein 距离均值，表示调整后的 Wasserstein 距离

        # 添加均值和中位数线
        plt.axvline(mean_value, color="red", linestyle="--", label=f"Mean: {mean_value:.3f}")
        plt.axvline(median_value, color="green", linestyle="--", label=f"Medianm: {median_value:.3f}")
        plt.title(f"Distribution of Wasserstein Distances {gene}", fontsize=10)
        plt.xlabel("Wasserstein Distance", fontsize=8)
        plt.ylabel("Frequency", fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(prop={'size': 8})
        plt.grid(False)
        save_path = os.path.join(path, f'{gene}_plot.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        # plt.show()