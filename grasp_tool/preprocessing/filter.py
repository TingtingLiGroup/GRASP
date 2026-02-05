import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os
import seaborn as sns


def _lazy_import_ot():
    """Import POT (Python Optimal Transport) only when needed."""

    try:
        import ot  # type: ignore

        return ot
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "This feature requires POT (Python Optimal Transport, import name: `ot`).\n"
            "Install: pip install POT\n"
            "Or (extras): pip install grasp-tool[ot]"
        ) from e


def generate_random_points(radius, num_points):
    center = (0, 0)
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = radius * np.sqrt(np.random.uniform(0, 1, num_points))

    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    points_df = pd.DataFrame({"x": x, "y": y})
    return points_df, np.vstack([x, y]).T


def simulate_expected_distance(radius, num_points, num_simulations=10):
    ot = _lazy_import_ot()
    distances = []
    for _ in range(num_simulations):
        _, points1 = generate_random_points(radius, num_points)
        _, points2 = generate_random_points(radius, num_points)
        a = np.ones(num_points) / num_points
        b = np.ones(num_points) / num_points
        cost_matrix = ot.dist(points1, points2, metric="euclidean")
        distance = ot.emd2(a, b, cost_matrix)
        distances.append(distance)
    return np.mean(distances)


def theoretical_expected_distance(radius, num_points, alpha=0.5):
    return radius * num_points ** (-alpha)


def calculate_adjusted_wasserstein_distance(
    original_points, target_points, num_points, expected_distance, epsilon=0.1
):
    ot = _lazy_import_ot()
    a = np.ones(num_points) / num_points
    b = np.ones(num_points) / num_points
    cost_matrix = ot.dist(original_points, target_points, metric="euclidean")

    # cost_matrix = ot.dist(sampled_original, sampled_target, metric='euclidean') + 1e-9
    wasserstein_distance = ot.emd2(a, b, cost_matrix)
    adjusted_distance = abs(wasserstein_distance - expected_distance)
    return (
        wasserstein_distance,
        adjusted_distance,
    )


def plot_points(dataset, data1, data2, gene, cell, radius, wasserstein_distance, path):
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    axes[0].scatter(
        data1["x_c_s"],
        data1["y_c_s"],
        s=3,
        alpha=0.6,
        color="blue",
        label=f"Gene: {gene}",
    )
    circle1 = plt.Circle(
        (0, 0), radius, color="black", fill=False, label="Cell Boundary"
    )
    axes[0].add_patch(circle1)
    axes[0].set_title(f"Original Points", fontsize=10)
    axes[0].set_aspect("equal")
    axes[0].axis("off")

    axes[1].scatter(
        data2["x"], data2["y"], s=3, alpha=0.6, color="green", label=f"Target Points"
    )
    circle2 = plt.Circle(
        (0, 0), radius, color="black", fill=False, label="Cell Boundary"
    )
    axes[1].add_patch(circle2)
    axes[1].set_title(f"Target Points", fontsize=10)
    axes[1].set_aspect("equal")
    axes[1].axis("off")

    num = len(data1)
    fig.suptitle(
        f"Cell: {cell} - Gene: {gene}\nMean_adjusted_distances: {wasserstein_distance:.4f} - num: {num}",
        fontsize=12,
    )
    save_dir = f"{path}/{gene}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{cell}.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    # plt.tight_layout()
    # plt.show()


def statics_plot(dataset, gene, path):
    file = f"{path}/{gene}_distances.csv"
    if not os.path.exists(file):
        print(f"1. Skipping {gene} (file not found).")

    else:
        df = pd.read_csv(f"{path}/{gene}_distances.csv", index_col=0)
        mean_value = df["mean_adjusted_distances"].mean()
        median_value = df["mean_adjusted_distances"].median()
        std_value = df["mean_adjusted_distances"].std()
        # print(f"Mean: {mean_value:.3f}, Median: {median_value:.3f}, Std Dev: {std_value:.3f}")
        plt.figure(figsize=(4, 3))
        sns.histplot(
            df["mean_wasserstein_distances"],
            kde=True,
            color="skyblue",
            edgecolor="grey",
            bins=20,
            label="mean_wasserstein_distances",
        )
        sns.histplot(
            df["expected_distance"],
            kde=True,
            color="royalblue",
            edgecolor="grey",
            bins=20,
            label="expected_distance",
        )
        sns.histplot(
            df["mean_adjusted_distances"],
            kde=True,
            color="pink",
            edgecolor="grey",
            bins=20,
            label="mean_adjusted_distances",
        )

        # Add mean/median markers
        plt.axvline(
            mean_value, color="red", linestyle="--", label=f"Mean: {mean_value:.3f}"
        )
        plt.axvline(
            median_value,
            color="green",
            linestyle="--",
            label=f"Medianm: {median_value:.3f}",
        )
        plt.title(f"Distribution of Wasserstein Distances {gene}", fontsize=10)
        plt.xlabel("Wasserstein Distance", fontsize=8)
        plt.ylabel("Frequency", fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(prop={"size": 8})
        plt.grid(False)
        save_path = os.path.join(path, f"{gene}_plot.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        # plt.show()
