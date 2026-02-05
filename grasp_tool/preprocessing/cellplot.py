import math
import os
import matplotlib.pyplot as plt
from shapely.wkt import loads
from shapely.geometry import Polygon
import pandas as pd
from tqdm import tqdm
import seaborn as sns


def plot_raw_cell(dataset, cell_boundary, nuclear_boundary, path):
    save_dir = f"{path}/1_{dataset}_raw_cell_plot"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cells = list(cell_boundary.keys())
    num_cells = len(cells)
    for idx, cell in enumerate(cells):
        plt.figure(figsize=(4, 4))
        plt.plot(
            cell_boundary[cell]["x"],
            cell_boundary[cell]["y"],
            label="Cell Boundary",
            color="black",
        )
        if cell in nuclear_boundary:
            plt.plot(
                nuclear_boundary[cell]["x"],
                nuclear_boundary[cell]["y"],
                label="Nucleus Boundary",
                color="red",
            )
        plt.title(f"Cell: {cell}")
        save_path = os.path.join(save_dir, f"{cell}.png")
        plt.savefig(save_path)

        plt.close()
    print(f"All cell images have been saved to {save_dir}")


def plot_raw_gene_distribution(dataset, cell_boundary, nuclear_boundary, df, path):
    cells = list(cell_boundary.keys())
    num_cells = len(cells)
    # tqdm progress bar
    for idx, cell in tqdm(enumerate(cells), total=num_cells, desc="Processing cells"):
        cell_data = df[df["cell"] == cell]
        save_dir = f"{path}/{dataset}/raw_gene/cell_{cell}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for gene in cell_data["gene"].unique():
            plt.figure(figsize=(3, 3))
            plt.plot(
                cell_boundary[cell]["x"],
                cell_boundary[cell]["y"],
                label="Cell Boundary",
                color="black",
            )
            if cell in nuclear_boundary:
                plt.plot(
                    nuclear_boundary[cell]["x"],
                    nuclear_boundary[cell]["y"],
                    label="Nucleus Boundary",
                    color="red",
                )
            gene_data = cell_data[cell_data["gene"] == gene]
            plt.scatter(
                gene_data["x"],
                gene_data["y"],
                label=f"Gene: {gene}",
                s=3,
                alpha=0.5,
                color="blue",
            )
            if (
                dataset == "simulated1"
                or dataset == "simulated2"
                or dataset == "simulated3"
            ):
                plt.title(f"{cell} - {gene}")
            elif (
                dataset == "merscope_liver_data2"
                or dataset == "merscope_liver_data3"
                or dataset == "merscope_liver_data4"
            ):
                plt.title(f"Gene: {gene}")
            else:
                plt.title(f"Cell: {cell} - Gene: {gene}")
            # save_path = os.path.join(save_dir, f'{gene}.png')
            plt.axis("off")
            # plt.savefig(save_path)
            plt.savefig(
                f"{save_dir}/{gene}.png", format="png", dpi=300, bbox_inches="tight"
            )
            plt.savefig(f"{save_dir}/{gene}.pdf", format="pdf", bbox_inches="tight")
            plt.savefig(f"{save_dir}/{gene}.svg", format="svg", bbox_inches="tight")
            plt.close()
    print(f"All cell images have been saved to {save_dir}")


def plot_raw_gene_distribution_without_nuclear(
    dataset, cell_boundary, df_registered, path
):
    # for cell_name, cell_data in cell_boundary.items():
    for cell_name, cell_data in tqdm(
        cell_boundary.items(), desc="Processing cells", leave=True
    ):
        fig_path = f"{path}/{dataset}/raw_gene/{cell_name}"
        os.makedirs(fig_path, exist_ok=True)
        sub_df_registered = df_registered[df_registered["cell"] == cell_name]
        gene_list = sub_df_registered["gene"].unique()
        for gene in tqdm(gene_list, desc=f"Plotting for {cell_name}", leave=False):
            gene_data = sub_df_registered[sub_df_registered["gene"] == gene]
            plt.figure(figsize=(3, 3))
            cell_polygon = Polygon(cell_data[["x", "y"]])
            x, y = cell_polygon.exterior.xy
            plt.plot(x, y, linestyle="-", color="black", linewidth=1)
            plt.scatter(gene_data["x"], gene_data["y"], s=3, color="cornflowerblue")
            plt.title(f"{cell_name} - {gene}")
            plt.axis("off")
            plt.tight_layout()
            # plt.savefig(f'{path}/{gene}.png', dpi=300)
            plt.savefig(
                f"{fig_path}/{gene}.png", format="png", dpi=300, bbox_inches="tight"
            )
            plt.savefig(f"{fig_path}/{gene}.pdf", format="pdf", bbox_inches="tight")
            plt.savefig(f"{fig_path}/{gene}.svg", format="svg", bbox_inches="tight")
            # plt.show()
            plt.close()


def plot_gene_galleries_from_df(
    dataset_name,
    df_to_plot,
    cell_boundary_dict,
    nuclear_boundary_dict,
    output_base_path,
    plots_per_gallery=48,
    cols_per_gallery=6,
):
    """Create per-gene gallery figures (grid of cells) from a DataFrame."""

    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
        print(f"Created output directory: {output_base_path}")

    if not all(col in df_to_plot.columns for col in ["gene", "cell", "x", "y"]):
        print("ERROR: df_to_plot must contain columns: gene, cell, x, y")
        return

    unique_genes = df_to_plot["gene"].unique()
    print(f"Found {len(unique_genes)} genes")

    for gene_name in unique_genes:
        print(f"\nProcessing gene: {gene_name}...")
        gene_specific_df = df_to_plot[df_to_plot["gene"] == gene_name]

        # Cells that contain this gene
        cells_with_this_gene = gene_specific_df["cell"].unique()
        if len(cells_with_this_gene) == 0:
            print(f"No points found for gene {gene_name}; skip")
            continue

        gene_output_folder = os.path.join(output_base_path, dataset_name, gene_name)
        if not os.path.exists(gene_output_folder):
            os.makedirs(gene_output_folder)

        num_cells_for_this_gene = len(cells_with_this_gene)

        # Build gallery figures for this gene
        for i in range(0, num_cells_for_this_gene, plots_per_gallery):
            batch_cell_ids = cells_with_this_gene[i : i + plots_per_gallery]
            current_batch_size = len(batch_cell_ids)

            rows_this_gallery = math.ceil(current_batch_size / cols_per_gallery)

            fig, axes = plt.subplots(
                rows_this_gallery,
                cols_per_gallery,
                figsize=(
                    cols_per_gallery * 3.5,
                    rows_this_gallery * 3.5,
                ),
            )
            # Normalize axes to a 2D array for indexing
            if rows_this_gallery == 1 and cols_per_gallery == 1:
                axes = [[axes]]
            elif rows_this_gallery == 1:
                axes = [axes]
            elif cols_per_gallery == 1:
                axes = [[ax] for ax in axes]

            for plot_idx, cell_id in enumerate(batch_cell_ids):
                ax_row = plot_idx // cols_per_gallery
                ax_col = plot_idx % cols_per_gallery
                ax = axes[ax_row][ax_col]

                # 1) Cell boundary
                if cell_id in cell_boundary_dict:
                    cb = cell_boundary_dict[cell_id]
                    ax.plot(cb["x"], cb["y"], color="black", linewidth=0.8)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Missing cell boundary",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="red",
                    )

                # 2) Nuclear boundary (optional)
                if cell_id in nuclear_boundary_dict:
                    nb_data = nuclear_boundary_dict[cell_id]
                    # nb_data can be a dict or a DataFrame
                    if (
                        isinstance(nb_data, dict)
                        and "x" in nb_data
                        and hasattr(nb_data["x"], "__len__")
                        and len(nb_data["x"]) > 0
                        and "y" in nb_data
                        and hasattr(nb_data["y"], "__len__")
                        and len(nb_data["y"]) > 0
                    ):
                        ax.plot(
                            nb_data["x"],
                            nb_data["y"],
                            color="dimgray",
                            linestyle="--",
                            linewidth=0.7,
                        )
                    elif (
                        isinstance(nb_data, pd.DataFrame)
                        and not nb_data.empty
                        and "x" in nb_data.columns
                        and "y" in nb_data.columns
                        and len(nb_data["x"]) > 0
                        and len(nb_data["y"]) > 0
                    ):
                        ax.plot(
                            nb_data["x"],
                            nb_data["y"],
                            color="dimgray",
                            linestyle="--",
                            linewidth=0.7,
                        )

                # 3) Points
                points_in_cell_gene = gene_specific_df[
                    gene_specific_df["cell"] == cell_id
                ]
                ax.scatter(
                    points_in_cell_gene["x"],
                    points_in_cell_gene["y"],
                    s=5,
                    alpha=0.7,
                    color="blue",
                )  # s=3, alpha=0.5

                ax.set_title(f"Cell: {cell_id}", fontsize=7)
                ax.axis("off")
                ax.set_aspect("equal", adjustable="box")

            # Remove unused subplots
            for k in range(current_batch_size, rows_this_gallery * cols_per_gallery):
                ax_row = k // cols_per_gallery
                ax_col = k % cols_per_gallery
                fig.delaxes(axes[ax_row][ax_col])

            plt.tight_layout(pad=0.5)
            gallery_num = (i // plots_per_gallery) + 1
            save_path = os.path.join(
                gene_output_folder, f"{gene_name}_gallery_{gallery_num}.png"
            )

            try:
                plt.savefig(save_path, dpi=200)
                print(f"Saved: {save_path}")
            except Exception as e:
                print(f"Failed to save {save_path}: {e}")
            plt.close(fig)

    print("\nDone")


# Registered gene scatter per cell (with nuclear boundary)
def plot_register_gene_distribution(
    dataset, df_registered, path, nuclear_boundary_df_registered
):
    cells = df_registered["cell"].unique()
    for cell in tqdm(cells, desc="Plotting per cell"):
        save_dir = f"{path}/{dataset}/registered_gene/{cell}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cell_gene_data = df_registered[df_registered["cell"] == cell]

        # Nuclear boundary data (optional). Some datasets may not have nucleus
        # boundaries for every cell.
        nuclear_boundary_df = None
        try:
            candidate = nuclear_boundary_df_registered[
                nuclear_boundary_df_registered["cell"] == cell
            ]
            if (
                candidate is not None
                and hasattr(candidate, "empty")
                and not candidate.empty
                and "x_c_s" in candidate.columns
                and "y_c_s" in candidate.columns
            ):
                nuclear_boundary_df = candidate
        except Exception:
            nuclear_boundary_df = None
        if not cell_gene_data.empty:
            genes = cell_gene_data["gene"].unique()
            for gene in tqdm(genes, desc=f"Plotting for cell {cell}", leave=False):
                # print(f'Cell: {cell} - Gene: {gene}')
                plt.figure(figsize=(4, 4))
                radius = 1
                circle = plt.Circle(
                    (0, 0),
                    radius,
                    color="gray",
                    fill=False,
                    label="Cell Boundary",
                    linewidth=1,
                )
                plt.gca().add_patch(circle)
                gene_data = cell_gene_data[cell_gene_data["gene"] == gene]
                plt.scatter(
                    gene_data["x_c_s"],
                    gene_data["y_c_s"],
                    label=f"Gene: {gene}",
                    s=2,
                    color="cornflowerblue",
                )
                # Remove spines
                ax = plt.gca()
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

                if nuclear_boundary_df is not None:
                    polygon_coords = list(
                        zip(
                            nuclear_boundary_df["x_c_s"],
                            nuclear_boundary_df["y_c_s"],
                        )
                    )
                    if polygon_coords:
                        boundary_x, boundary_y = zip(*polygon_coords)
                        ax.plot(boundary_x, boundary_y, color="darkgray", linewidth=1)
                # Hide axes
                plt.axis("off")
                if (
                    dataset == "simulated1"
                    or dataset == "simulated2"
                    or dataset == "simulated3"
                ):
                    plt.title(f"{cell} - {gene}")
                else:
                    plt.title(f"Cell: {cell} - Gene: {gene}")
                # save_path = os.path.join(save_dir, f'{cell}_{gene}.png')
                # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.savefig(
                    f"{save_dir}/{cell}_{gene}.png",
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{save_dir}/{cell}_{gene}.pdf", format="pdf", bbox_inches="tight"
                )
                plt.savefig(
                    f"{save_dir}/{cell}_{gene}.svg", format="svg", bbox_inches="tight"
                )
                plt.close()
    # print(f"All cell and gene images have been saved to {save_dir}")


# Registered gene scatter per cell (without nuclear boundary)
def plot_register_gene_distribution_without_nuclear(
    dataset, df_registered, cell_radii, path
):
    cells = df_registered["cell"].unique()
    for cell in tqdm(cells, desc="Plotting per cell"):
        save_dir = f"{path}/{dataset}/registered_gene/{cell}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cell_gene_data = df_registered[df_registered["cell"] == cell]
        if not cell_gene_data.empty:
            genes = cell_gene_data["gene"].unique()
            for gene in tqdm(genes, desc=f"Plotting for cell {cell}", leave=False):
                # print(f'Cell: {cell} - Gene: {gene}')
                plt.figure(figsize=(4, 4))
                radius = 1
                circle = plt.Circle(
                    (0, 0),
                    radius,
                    color="gray",
                    fill=False,
                    label="Cell Boundary",
                    linewidth=1,
                )
                plt.gca().add_patch(circle)
                gene_data = cell_gene_data[cell_gene_data["gene"] == gene]
                plt.scatter(
                    gene_data["x_c_s"],
                    gene_data["y_c_s"],
                    label=f"Gene: {gene}",
                    s=2,
                    color="cornflowerblue",
                )
                # Remove spines
                ax = plt.gca()
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                plt.axis("off")
                plt.title(f"Cell: {cell} - Gene: {gene}")
                # save_path = os.path.join(save_dir, f'{cell}_{gene}.png')
                # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.savefig(
                    f"{save_dir}/{cell}_{gene}.png",
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{save_dir}/{cell}_{gene}.pdf", format="pdf", bbox_inches="tight"
                )
                plt.savefig(
                    f"{save_dir}/{cell}_{gene}.svg", format="svg", bbox_inches="tight"
                )
                plt.close()


def plot_each_batch(dataset, adata, batch, path):
    save_dir = f"{path}/{dataset}/each_batch"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    adata_sub = adata[adata.obs["batch"] == batch]
    points = adata_sub.uns["points"]
    df = pd.DataFrame(points)
    df = df[df["batch"] == int(batch)]
    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)
    df = df[df["gene"].isin(set(adata.var_names))]
    df = df[df["cell"].isin(set(adata.obs_names))]
    gene_all = df["gene"].value_counts()
    cell_all = df["cell"].value_counts()
    cell_shape = adata_sub.obs["cell_shape"].to_frame()
    nucleus_shape = adata_sub.obs["nucleus_shape"].to_frame()
    plt.figure(figsize=(6, 4))
    for index, row in cell_shape.iterrows():
        polygon = loads(row["cell_shape"])
        x, y = polygon.exterior.xy
        plt.plot(x, y, linestyle="-", color="grey", linewidth=1)
        centroid = polygon.centroid
        cx, cy = centroid.x, centroid.y
        plt.text(cx, cy, str(index), fontsize=10, ha="center", color="darkblue")
    for index, row in nucleus_shape.iterrows():
        polygon = loads(row["nucleus_shape"])
        x, y = polygon.exterior.xy
        plt.plot(x, y, linestyle="-", color="darkgray", linewidth=1)
    plt.title(f"Batch {batch}")
    plt.axis("off")
    plt.savefig(
        f"{save_dir}/batch{batch}_plot.png", format="png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(f"{save_dir}/batch{batch}_plot.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{save_dir}/batch{batch}_plot.svg", format="svg", bbox_inches="tight")
    # plt.savefig(f"{save_dir}/batch{batch}_plot.png", format='png', dpi=400)
    # plt.show()
