import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from scipy.stats import mode
import igraph as ig
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from leidenalg import find_partition
import igraph as ig
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import igraph as ig
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import os
import warnings
import time
from typing import List, Dict, Tuple, Optional, Union, Any

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _lazy_import_umap():
    """Import `umap` only when actually needed.

    Some environments may pull in TensorFlow when importing `umap`, which can
    emit noisy logs. Avoid importing it at module import time.
    """

    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import umap  # type: ignore

        return umap
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing optional dependency `umap-learn` (import name: `umap`).\n"
            "Install: pip install umap-learn"
        ) from e


########################################################
# Unified evaluation + visualization utilities.
#
# The main entrypoint is evaluate_and_visualize(...), which replaces older
# per-mode plotting helpers.
########################################################

# Preprocessing method names.
PREPROCESS_BASIC = "basic"  # Basic
PREPROCESS_SCALER = "scaler"  # StandardScaler
PREPROCESS_PCA = "pca"  # PCA
PREPROCESS_SELECT = "select"  # Feature selection

# All supported preprocessing methods.
ALL_PREPROCESS_METHODS = [
    PREPROCESS_BASIC,
    PREPROCESS_SCALER,
    PREPROCESS_PCA,
    PREPROCESS_SELECT,
]

# TODO: performance/memory improvements for TSNE/UMAP on large datasets.


def compute_metrics(true_labels, predicted_labels):
    """Compute clustering metrics.

    Returns ARI (adjusted rand index) and NMI (normalized mutual information).
    """
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return ari, nmi


def map_labels_with_hungarian(true_labels, predicted_labels):
    """Map predicted cluster labels to true labels via Hungarian matching."""
    # Ensure numpy arrays.
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)

    # Unique label values.
    true_classes = np.unique(true_labels)
    predicted_classes = np.unique(predicted_labels)

    # Build cost matrix.
    n_true = len(true_classes)
    n_pred = len(predicted_classes)
    max_size = max(n_true, n_pred)
    cost_matrix = np.zeros((max_size, max_size))

    # Fill cost matrix (negative overlap -> Hungarian solves min-cost).
    for i, t_class in enumerate(true_classes):
        for j, p_class in enumerate(predicted_classes):
            # Overlap between (t_class, p_class)
            match = np.sum((true_labels == t_class) & (predicted_labels == p_class))
            cost_matrix[i, j] = -match

    # Pad extra rows/cols with a large cost.
    if n_true < max_size or n_pred < max_size:
        large_val = abs(np.min(cost_matrix)) * 10
        for i in range(n_true, max_size):
            cost_matrix[i, :] = large_val
        for j in range(n_pred, max_size):
            cost_matrix[:, j] = large_val

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build mapping
    label_mapping = {}
    for row, col in zip(row_ind, col_ind):
        if row < n_true and col < n_pred:  # Valid match
            label_mapping[predicted_classes[col]] = true_classes[row]

    # Handle unmatched predicted labels (if any)
    unmatched_pred_labels = set(predicted_classes) - set(label_mapping.keys())
    if unmatched_pred_labels:
        # Find best-effort mapping for unmatched labels
        for unmatched_label in unmatched_pred_labels:
            # Overlap with each true class
            overlaps = []
            for t_class in true_classes:
                overlap = np.sum(
                    (predicted_labels == unmatched_label) & (true_labels == t_class)
                )
                overlaps.append((t_class, overlap))

            # Assign to the most overlapping true class
            best_class = max(overlaps, key=lambda x: x[1])[0]
            label_mapping[unmatched_label] = best_class

    # Mapped label array
    mapped_labels = np.array([label_mapping[label] for label in predicted_labels])

    return mapped_labels, label_mapping


def compute_metrics_with_classification_report(true_labels, predicted_labels):
    """Compute metrics from sklearn's classification_report.

    Returns: (accuracy, precision, recall, f1_score) using macro average.
    """
    report = classification_report(
        true_labels, predicted_labels, output_dict=True, zero_division="warn"
    )
    accuracy = report.get("accuracy", 0.0)

    macro_avg_report = report.get("macro avg", {})
    precision = macro_avg_report.get("precision", 0.0)
    recall = macro_avg_report.get("recall", 0.0)
    f1_score = macro_avg_report.get("f1-score", 0.0)
    return accuracy, precision, recall, f1_score


def load_location_data(df, dataset, graphs_number=None, specific_label_file=None):
    """
    Load location/label data and merge it into the input DataFrame.

    Args:
        df: Input DataFrame.
        dataset: Dataset name.
        graphs_number: Optional graph count suffix used in label file naming.
        specific_label_file: Optional label filename or an absolute path.

    Returns:
        DataFrame: DataFrame with an added/updated 'location' column.
    """
    # Priority order: label column names from high to low.
    priority_label_cols = [
        "groundtruth_wzx",
        "groundtruth",
        "label",
        "location",
        "cluster",
        "category",
        "type",
    ]

    # Warn if required columns are missing; do not early-return.
    if "cell" not in df.columns or "gene" not in df.columns:
        print(
            f"Warning: DataFrame missing required columns ('cell' and/or 'gene'), merge may fail"
        )

    # Candidate base paths for label files.
    base_paths = [
        "../1_input/label",
        "../1_input/label_annotation",
        "../../GCN_CL/1_input/label",
        "../../GCN_CL/1_input/label_annotation",
        "./1_input/label",
        "./1_input/label_annotation",
    ]

    # Try to locate a label file.
    label_file = None

    # Allow absolute paths.
    # 1) If user provided an absolute path and it exists, use it directly.
    if (
        specific_label_file
        and os.path.isabs(specific_label_file)
        and os.path.exists(specific_label_file)
    ):
        label_file = specific_label_file
        print(f"Using absolute label file path: {label_file}")
    # 2) If user provided a relative filename, search under base_paths.
    elif specific_label_file:
        for base_path in base_paths:
            path = f"{base_path}/{specific_label_file}"
            if os.path.exists(path):
                label_file = path
                print(f"Using specified label file: {label_file}")
                break

    # If no specific file was found, try common naming patterns.
    if label_file is None:
        for base_path in base_paths:
            possible_files = [
                f"{base_path}/{dataset}_label.csv",
                f"{base_path}/{dataset}_labeled.csv",
            ]
            if graphs_number:
                possible_files.extend(
                    [
                        f"{base_path}/{dataset}_graph{graphs_number}_label.csv",
                        f"{base_path}/{dataset}_graph{graphs_number}_labeled.csv",
                    ]
                )

            for file_path in possible_files:
                if os.path.exists(file_path):
                    label_file = file_path
                    print(f"Found label file: {label_file}")
                    break

            if label_file:
                break

    # If a label file is found, load and process it.
    if label_file:
        try:
            # Read label file.
            label_df = pd.read_csv(label_file)

            # Print label file columns for debugging.
            print(f"Label file columns: {label_df.columns.tolist()}")

            # Find label columns by priority order.
            found_label_cols = []
            for col in priority_label_cols:
                if col in label_df.columns:
                    found_label_cols.append(col)
                    print(
                        f"Found label column: '{col}' (priority {priority_label_cols.index(col) + 1})"
                    )

            if found_label_cols:
                primary_label_col = found_label_cols[0]  # highest-priority column
                print(f"Using '{primary_label_col}' as primary label column")
            else:
                print("Warning: No recognized label column found in label file")
                primary_label_col = None

            # Robustly infer label file format.
            # Long format: has 'cell' and 'gene' columns (+ one or more label columns).
            # Wide format: one row per cell, many gene columns.

            if "cell" in label_df.columns and "gene" in label_df.columns:
                print(f"Detected long format label file")

                # Keep only required columns.
                keep_cols = ["cell", "gene"] + found_label_cols
                # Only keep columns that exist.
                label_df = label_df[
                    [col for col in keep_cols if col in label_df.columns]
                ]

                # Merge into input dataframe.
                try:
                    merged_df = df.merge(label_df, on=["gene", "cell"], how="left")
                    print(f"Merged on both 'gene' and 'cell' columns")
                except Exception as e:
                    print(f"Error in gene+cell merge: {e}, trying cell-only merge")
                    try:
                        # Fallback to merging only on 'cell'.
                        merged_df = df.merge(label_df, on=["cell"], how="left")
                        print(f"Merged on 'cell' column only")
                    except Exception as e2:
                        print(f"All merge attempts failed: {e2}")
                        # Fallback to default location.
                        df["location"] = "unknown"
                        return df

                # Normalize the primary label column to 'location' (if needed).
                if primary_label_col and primary_label_col != "location":
                    if primary_label_col in merged_df.columns:
                        print(f"Renaming '{primary_label_col}' to 'location'")
                        merged_df["location"] = merged_df[primary_label_col]

                # Ensure 'location' exists.
                if "location" not in merged_df.columns:
                    print("Creating 'location' column from highest priority label")
                    if found_label_cols:
                        merged_df["location"] = merged_df[found_label_cols[0]]
                    else:
                        print("Warning: No label columns found, using 'unknown'")
                        merged_df["location"] = "unknown"

                merged_df["location"] = merged_df["location"].fillna("unknown")
                print(f"Merged data, final shape: {merged_df.shape}")

                # Print label distribution.
                label_counts = merged_df["location"].value_counts()
                print(f"Label distribution (top 10): \n{label_counts.head(10)}")

                return merged_df

            # Wide format: row is a cell, columns are genes (and possibly metadata).
            elif "cell" in label_df.columns and len(label_df.columns) > 2:
                print(
                    f"Detected wide format label file, columns: {label_df.columns.tolist()}"
                )

                # Confirm it looks like a typical wide format with gene columns.
                non_gene_cols = ["cell", "Cell"] + priority_label_cols
                potential_gene_cols = [
                    col for col in label_df.columns if col not in non_gene_cols
                ]

                # Heuristic: treat as wide format if there are many gene columns.
                if len(potential_gene_cols) > 5:  # assume at least 5 gene columns
                    print(
                        f"Converting wide format to long format with {len(potential_gene_cols)} gene columns"
                    )

                    # Preserve non-gene columns for later merge.
                    cell_info_cols = []
                    if primary_label_col:
                        cell_info_cols = [primary_label_col]
                    else:
                        # If no label column is found, keep all non-gene columns.
                        cell_info_cols = [
                            col
                            for col in non_gene_cols
                            if col in label_df.columns and col != "cell"
                        ]

                    cell_info_df = None
                    if cell_info_cols:
                        cell_info_df = label_df[["cell"] + cell_info_cols].copy()
                        print(f"Preserved cell information columns: {cell_info_cols}")

                    # Convert wide -> long.
                    long_df = pd.melt(
                        label_df,
                        id_vars="cell",
                        value_vars=potential_gene_cols,
                        var_name="gene",
                        value_name="gene_value",
                    )

                    # Merge into input dataframe.
                    try:
                        merged_df = df.merge(long_df, on=["gene", "cell"], how="left")
                        print(f"Merged gene-cell data")
                    except Exception as e:
                        print(f"Error in gene-cell merge for wide format: {e}")
                        # Fallback to default location.
                        df["location"] = "unknown"
                        return df

                    # Merge preserved cell info columns (if any).
                    if cell_info_df is not None:
                        try:
                            merged_df = merged_df.merge(
                                cell_info_df, on="cell", how="left"
                            )
                            print(f"Added cell information columns")
                        except Exception as e:
                            print(f"Error merging cell info: {e}")

                    # Set 'location' column.
                    if primary_label_col and primary_label_col in merged_df.columns:
                        merged_df["location"] = merged_df[primary_label_col]
                        print(f"Set 'location' from '{primary_label_col}'")
                    else:
                        # Fallback to 'gene_value'.
                        merged_df["location"] = merged_df["gene_value"]
                        print(f"Set 'location' from gene_value column")

                    # Ensure 'location' has values.
                    merged_df["location"] = merged_df["location"].fillna("unknown")
                    print(
                        f"Merged data from wide format, final shape: {merged_df.shape}"
                    )

                    # Print label distribution.
                    label_counts = merged_df["location"].value_counts()
                    print(f"Label distribution (top 10): \n{label_counts.head(10)}")

                    return merged_df
                else:
                    print(
                        f"Label file has 'cell' column but doesn't appear to be a typical wide format"
                    )

                    # Try merging based on 'cell'.
                    try:
                        # Use discovered label columns.
                        useful_cols = ["cell"]
                        if found_label_cols:
                            useful_cols.extend(found_label_cols)
                        else:
                            # If no label columns are found, keep all likely label cols.
                            useful_cols.extend(
                                [
                                    col
                                    for col in label_df.columns
                                    if col in priority_label_cols
                                    or col not in non_gene_cols
                                ]
                            )

                        # Only keep existing columns.
                        label_subset = label_df[
                            [col for col in useful_cols if col in label_df.columns]
                        ].copy()

                        # Merge.
                        merged_df = df.merge(label_subset, on="cell", how="left")

                        # Set 'location' column.
                        if primary_label_col and primary_label_col in merged_df.columns:
                            merged_df["location"] = merged_df[primary_label_col]
                            print(f"Set 'location' from '{primary_label_col}'")
                        else:
                            # If no primary label column is available, use default.
                            print("No primary label column available")
                            merged_df["location"] = "unknown"

                        merged_df["location"] = merged_df["location"].fillna("unknown")

                        print(
                            f"Merged data using cell-based join, final shape: {merged_df.shape}"
                        )

                        # Print label distribution.
                        label_counts = merged_df["location"].value_counts()
                        print(f"Label distribution (top 10): \n{label_counts.head(10)}")

                        return merged_df
                    except Exception as e:
                        print(f"Error in cell-based merge: {e}")

            # Unrecognized format.
            print(f"Unrecognized label file format, using default 'unknown' location")
            df["location"] = "unknown"
            return df

        except Exception as e:
            print(f"Error processing label file {label_file}: {e}")
            import traceback

            traceback.print_exc()

    # If all attempts fail, add a default 'location' column.
    print(
        "Could not find or process a suitable label file, using default 'unknown' location"
    )
    if "location" not in df.columns:
        df["location"] = "unknown"

    return df


def apply_clustering(features, n_clusters, clustering_methods=None):
    """
    Apply multiple clustering methods.

    Args:
        features: Feature matrix.
        n_clusters: Number of clusters.
        clustering_methods: Optional list of method names to run.

    Returns:
        clustering_methods: Dict of method name -> cluster labels.
    """
    all_methods = {
        "KMeans": KMeans(n_clusters=n_clusters, random_state=2025)
        .fit(features)
        .labels_,
        "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters).fit_predict(
            features
        ),
        "SpectralClustering": SpectralClustering(
            n_clusters=n_clusters, random_state=2025, affinity="nearest_neighbors"
        ).fit_predict(features),
        "GaussianMixture": GaussianMixture(n_components=n_clusters, random_state=2025)
        .fit(features)
        .predict(features),
    }

    # If methods are specified, return only those.
    if clustering_methods is not None:
        return {k: v for k, v in all_methods.items() if k in clustering_methods}

    # Otherwise return all methods.
    return all_methods


def evaluate_clustering(
    true_labels, clustering_methods, df, save_path, num_epochs, lr, suffix=""
):
    """
    Evaluate clustering results.

    Args:
        true_labels: Ground-truth labels.
        clustering_methods: Dict of method name -> predicted labels.
        df: DataFrame.
        save_path: Output directory.
        num_epochs: Current epoch.
        lr: Learning rate.
        suffix: Filename suffix.

    Returns:
        metrics: Dict of evaluation metrics.
    """
    # Check label distribution.
    unique_labels = np.unique(true_labels)
    unique_labels_count = len(unique_labels)
    print("\nLabel distribution:")
    print(f"Unique label count: {unique_labels_count}")
    print(
        f"Label values: {unique_labels[:10]}{'...' if len(unique_labels) > 10 else ''}"
    )

    # If all samples share one label, ARI/NMI become 0 by definition.
    if unique_labels_count == 1:
        print(f"Warning: all samples share the same label '{unique_labels[0]}'")
        print(
            "This makes ARI/NMI equal to 0 and can make accuracy-like metrics appear high, "
            "because there is no class variation to compare."
        )
        print("Please check the label file or load_location_data() processing.\n")
    elif unique_labels_count < 2:
        print("Error: too few unique labels for meaningful clustering evaluation")
        return {"error": "insufficient_labels"}

    metrics = {}
    for method, predicted_labels in clustering_methods.items():
        mapped_labels, label_mapping = map_labels_with_hungarian(
            true_labels, predicted_labels
        )
        accuracy, precision, recall, f1_score = (
            compute_metrics_with_classification_report(true_labels, mapped_labels)
        )
        print(f"{method} Label Mapping: {label_mapping}")

        ari, nmi = compute_metrics(true_labels, predicted_labels)
        metrics[method] = {
            "ARI": ari,
            "NMI": nmi,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score,
            "Label Mapping": label_mapping,
        }
        df[method + "_cluster"] = predicted_labels

    print(f"Epoch {num_epochs} lr={lr} suffix={suffix} clustering metrics:")
    for method, values in metrics.items():
        print(
            f"{method}: ARI = {values['ARI']:.4f}, NMI = {values['NMI']:.4f}, "
            f"Accuracy = {values['Accuracy']:.4f}, Precision = {values['Precision']:.4f}, "
            f"Recall = {values['Recall']:.4f}, F1-Score = {values['F1-Score']:.4f}"
        )

    # Save clustering outputs.
    if suffix:
        df.to_csv(
            f"{save_path}/epoch{num_epochs}_lr{lr}_clusters_{suffix}.csv", index=False
        )

        # Save metrics.
        with open(
            f"{save_path}/epoch{num_epochs}_lr{lr}_metrics_{suffix}.txt", "w"
        ) as f:
            for method, values in metrics.items():
                f.write(
                    f"{method}:\nARI = {values['ARI']:.4f}, NMI = {values['NMI']:.4f},\n "
                    f"Accuracy = {values['Accuracy']:.4f}, Precision = {values['Precision']:.4f},\n"
                    f"Recall = {values['Recall']:.4f}, F1-Score = {values['F1-Score']:.4f}\n"
                )
                f.write(f"{method} Label Mapping: {values['Label Mapping']}\n")
                f.write(f"\n")
    else:
        df.to_csv(f"{save_path}/epoch{num_epochs}_lr{lr}_clusters.csv", index=False)

        # Save metrics.
        with open(f"{save_path}/epoch{num_epochs}_lr{lr}_metrics.txt", "w") as f:
            for method, values in metrics.items():
                f.write(
                    f"{method}:\nARI = {values['ARI']:.4f}, NMI = {values['NMI']:.4f},\n "
                    f"Accuracy = {values['Accuracy']:.4f}, Precision = {values['Precision']:.4f},\n"
                    f"Recall = {values['Recall']:.4f}, F1-Score = {values['F1-Score']:.4f}\n"
                )
                f.write(f"{method} Label Mapping: {values['Label Mapping']}\n")
                f.write(f"\n")

    return metrics


def visualize_clustering(
    reduced_features_tsne,
    reduced_features_umap,
    true_labels,
    clustering_methods,
    num_epochs,
    lr,
    size=15,
):
    """
    Visualize clustering results.

    Args:
        reduced_features_tsne: t-SNE 2D features.
        reduced_features_umap: UMAP 2D features.
        true_labels: Ground-truth labels.
        clustering_methods: Dict of method name -> predicted labels.
        num_epochs: Current epoch.
        lr: Learning rate.
        size: Point size.

    Returns:
        fig: Matplotlib figure.
    """
    fig, axes = plt.subplots(
        len(clustering_methods) + 1, 2, figsize=(16, 4 * (len(clustering_methods) + 1))
    )
    fig.suptitle(f"Epoch {num_epochs} lr={lr}", fontsize=16)

    # True labels.
    sns.scatterplot(
        x=reduced_features_tsne[:, 0],
        y=reduced_features_tsne[:, 1],
        hue=true_labels,
        palette="Set1",
        ax=axes[0, 0],
        s=size,
        legend="full",
    )
    axes[0, 0].set_title("t-SNE with True Labels")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1, 1))

    sns.scatterplot(
        x=reduced_features_umap[:, 0],
        y=reduced_features_umap[:, 1],
        hue=true_labels,
        palette="Set1",
        ax=axes[0, 1],
        s=size,
        legend="full",
    )
    axes[0, 1].set_title("UMAP with True Labels")
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Predicted clusters.
    for i, (method, predicted_labels) in enumerate(clustering_methods.items(), start=1):
        sns.scatterplot(
            x=reduced_features_tsne[:, 0],
            y=reduced_features_tsne[:, 1],
            hue=predicted_labels,
            palette="Set2",
            ax=axes[i, 0],
            s=size,
            legend="full",
        )
        axes[i, 0].set_title(f"t-SNE with {method} Clusters")
        axes[i, 0].legend(loc="upper left", bbox_to_anchor=(1, 1))

        sns.scatterplot(
            x=reduced_features_umap[:, 0],
            y=reduced_features_umap[:, 1],
            hue=predicted_labels,
            palette="Set3",
            ax=axes[i, 1],
            s=size,
            legend="full",
        )
        axes[i, 1].set_title(f"UMAP with {method} Clusters")
        axes[i, 1].legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    return fig


def preprocess_features(df, preprocess_method="basic", true_labels=None):
    """
    Preprocess features according to the given method.

    Args:
        df: DataFrame.
        preprocess_method: One of 'basic', 'scaler', 'pca', 'select'.
        true_labels: Labels used by feature selection.

    Returns:
        features: Preprocessed feature matrix.
    """
    # Exclude known non-feature columns.
    non_feature_cols = [
        "cell",
        "gene",
        "graph_id",
        "original_graph_id",
        "augmented_graph_id",
        "location",
        "groundtruth",
        "groundtruth_wzx",
        "label",
        "cluster",
        "category",
        "type",
        "SCT_cluster",
        "cell_type",
        "celltype",  # additional common non-feature column name
        # Also exclude columns generated by clustering methods.
        "KMeans_cluster",
        "Agglomerative_cluster",
        "SpectralClustering_cluster",
        "GaussianMixture_cluster",
    ]

    # Candidate feature columns.
    candidate_feature_names = [
        col
        for col in df.columns
        if col not in non_feature_cols and not str(col).endswith("_cluster")
    ]

    if not candidate_feature_names:
        print(
            f"Warning: No candidate feature columns found after excluding non_feature_cols for method '{preprocess_method}'. Original columns: {df.columns.tolist()}"
        )
        return np.array([])

    # Work on a copy to safely attempt numeric conversion.
    df_candidate_features = df[candidate_feature_names].copy()

    actually_numeric_cols = []
    for col_name in df_candidate_features.columns:
        try:
            # Coerce column to numeric; non-convertible values become NaN.
            converted_series = pd.to_numeric(
                df_candidate_features[col_name], errors="coerce"
            )

            # Keep numeric columns that are not all-NaN after coercion.
            if (
                pd.api.types.is_numeric_dtype(converted_series)
                and not converted_series.isnull().all()
            ):
                # Heuristic: if coercion wipes most values, it's likely not a true numeric feature.
                original_non_na_count = (
                    df_candidate_features[col_name].dropna().shape[0]
                )
                converted_non_na_count = converted_series.dropna().shape[0]
                if original_non_na_count > 0 and (
                    converted_non_na_count / original_non_na_count < 0.5
                ):
                    print(
                        f"Warning: Column '{col_name}' lost significant data after numeric coercion ({original_non_na_count} -> {converted_non_na_count} non-NaNs). Excluding."
                    )
                else:
                    actually_numeric_cols.append(col_name)
                    # Replace with numeric series to help downstream .values.
                    df_candidate_features[col_name] = converted_series
            else:
                print(
                    f"Warning: Column '{col_name}' excluded. Not purely numeric or became all NaNs after coercion."
                )
        except Exception as e:
            print(
                f"Warning: Column '{col_name}' encountered an error during numeric conversion ('{e}') and will be excluded."
            )

    if not actually_numeric_cols:
        print(
            f"Error: No numeric feature columns remaining after filtering for method '{preprocess_method}'."
        )
        return np.array([])

    # Extract features from the converted dataframe.
    features = df_candidate_features[actually_numeric_cols].values

    # Ensure float dtype.
    if features.dtype == np.object_:
        print(
            f"Warning: Features array for method '{preprocess_method}' has dtype 'object' before scaling. Attempting astype(float)."
        )
        try:
            features = features.astype(float)
        except ValueError as e_astype:
            error_msg = (
                f"Failed to convert object-type features to float for method '{preprocess_method}'. "
                f"Problematic data likely still exists. Original error: {e_astype}"
            )
            print(f"Error: {error_msg}")
            # If this happens, upstream data likely contains non-numeric features.
            # return np.full(features.shape, np.nan)
            raise ValueError(error_msg) from e_astype

    # Apply preprocessing.
    if preprocess_method == "basic":
        pass  # no-op
    elif preprocess_method == "scaler":
        if features.size == 0:
            print(
                f"Warning: No features to scale for method '{preprocess_method}'. Skipping scaling."
            )
            return features  # avoid StandardScaler errors
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    elif preprocess_method == "pca":
        if features.size == 0:
            print(
                f"Warning: No features for PCA for method '{preprocess_method}'. Skipping PCA."
            )
            return features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        pca = PCA(n_components=0.95)  # keep 95% variance
        features = pca.fit_transform(features)
        print(f"  PCA applied. Features shape after PCA: {features.shape}")
    elif preprocess_method == "select":
        if features.size == 0:
            print(
                f"Warning: No features for selection for method '{preprocess_method}'. Skipping selection."
            )
            return features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        if true_labels is not None:
            selector = SelectKBest(score_func=f_classif, k=min(50, features.shape[1]))
            features = selector.fit_transform(features, true_labels)
    else:
        raise ValueError(f"Unsupported preprocess method: {preprocess_method}")

    return features


def evaluate_and_visualize(
    dataset,
    df,
    save_path,
    num_epochs,
    lr,
    n_clusters=8,
    size=15,
    vis_methods=None,
    visualize=True,
    graphs_number=None,
    clustering_methods=None,
    reduce_dims=True,
    specific_label_file=None,
    tsne_perplexity: float = 30.0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.2,
):
    """
    Evaluate clustering results and optionally create visualizations.

    Args:
        dataset: Dataset name.
        df: DataFrame containing embeddings.
        save_path: Output directory.
        num_epochs: Current epoch.
        lr: Learning rate.
        n_clusters: Number of clusters.
        size: Point size.
        vis_methods: List of visualization/preprocess methods.
        visualize: Whether to generate visualization images.
        graphs_number: Optional graph count suffix used in label file naming.
        clustering_methods: Optional list of clustering method names.
        reduce_dims: Whether to run dimensionality reduction.
        specific_label_file: Optional label filename or absolute path.
        tsne_perplexity: t-SNE perplexity.
        umap_n_neighbors: UMAP n_neighbors.
        umap_min_dist: UMAP min_dist.

    Returns:
        Tuple[Dict, Dict]: (metrics for all methods, figures for all methods)
    """
    # Timing.
    start_time = time.time()

    # Normalize visualization methods.
    if vis_methods is None:
        # Default: run all.
        vis_methods = ALL_PREPROCESS_METHODS
    elif isinstance(vis_methods, str):
        # Normalize a single string into a list.
        vis_methods = [vis_methods]

    # Validate visualization methods.
    for method in vis_methods:
        if method not in ALL_PREPROCESS_METHODS:
            raise ValueError(
                f"Unsupported visualization method: {method}. Available: {ALL_PREPROCESS_METHODS}"
            )

    # Load location/label data.
    df = load_location_data(df, dataset, graphs_number, specific_label_file)

    # Extract labels.
    true_labels = df["location"].astype(str).values

    # Initialize result dicts.
    all_metrics = {}
    all_figures = {}

    # If no methods are requested, return empty results.
    if not vis_methods:
        print("No visualization methods specified. Returning empty results.")
        return all_metrics, all_figures

    # Track total method count for progress printing.
    total_methods = len(vis_methods)

    # Process each method.
    for i, method in enumerate(vis_methods, 1):
        method_start = time.time()
        print(f"\n[{i}/{total_methods}] Start processing method: {method}")

        # Preprocess features.
        features = preprocess_features(df, method, true_labels)

        # Dimensionality reduction (only when requested).
        reduced_features_tsne = None
        reduced_features_umap = None

        if visualize and reduce_dims:
            dim_start = time.time()
            print("  Running dimensionality reduction...")
            try:
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=min(tsne_perplexity, len(df) - 1),
                )
                reduced_features_tsne = tsne.fit_transform(features)
            except Exception as e:
                print(f"  t-SNE reduction failed: {e}")
                reduced_features_tsne = np.zeros((len(df), 2))  # blank fallback

            try:
                umap_model = _lazy_import_umap().UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=min(umap_n_neighbors, len(df) - 1),
                    min_dist=umap_min_dist,
                )
                reduced_features_umap = umap_model.fit_transform(features)
            except Exception as e:
                print(f"  UMAP reduction failed: {e}")
                reduced_features_umap = np.zeros((len(df), 2))  # blank fallback

            dim_time = time.time() - dim_start
            print(f"  Dimensionality reduction done in {dim_time:.2f}s")

        # Clustering.
        cluster_start = time.time()
        clustering_methods_dict = apply_clustering(
            features, n_clusters, clustering_methods
        )
        cluster_time = time.time() - cluster_start
        print(f"  Clustering done in {cluster_time:.2f}s")

        # Evaluation.
        eval_start = time.time()
        metrics = evaluate_clustering(
            true_labels,
            clustering_methods_dict,
            df,
            save_path,
            num_epochs,
            lr,
            suffix=method if method != PREPROCESS_BASIC else "",
        )
        all_metrics[method] = metrics
        eval_time = time.time() - eval_start
        print(f"  Evaluation done in {eval_time:.2f}s")

        # Visualization.
        if visualize and reduce_dims:
            vis_start = time.time()
            print("  Creating visualization...")
            fig = visualize_clustering(
                reduced_features_tsne,
                reduced_features_umap,
                true_labels,
                clustering_methods_dict,
                num_epochs,
                lr,
                size,
            )

            fig_suffix = "" if method == PREPROCESS_BASIC else f"_{method}"
            fig_path = (
                f"{save_path}/epoch{num_epochs}_lr{lr}_visualization{fig_suffix}.png"
            )
            fig.savefig(fig_path)
            all_figures[method] = fig

            vis_time = time.time() - vis_start
            print(f"  Visualization saved to {fig_path} in {vis_time:.2f}s")

        method_time = time.time() - method_start
        print(f"Method {method} done in {method_time:.2f}s")

    total_time = time.time() - start_time
    print(f"\nAll methods completed in {total_time:.2f}s")

    return all_metrics, all_figures


def plot_scatter_mutilbatch_tensorboard(*args, **kwargs):
    """
    Deprecated: use evaluate_and_visualize(vis_methods=['basic']).
    """
    warnings.warn(
        "plot_scatter_mutilbatch_tensorboard is deprecated; use evaluate_and_visualize",
        DeprecationWarning,
    )
    metrics, figs = evaluate_and_visualize(*args, vis_methods=["basic"], **kwargs)
    return metrics.get("basic", {}), figs.get("basic")


def plot_scatter_mutilbatch_tensorboard(
    dataset,
    df,
    save_path,
    num_epochs,
    lr,
    n_clusters,
    size=15,
    visualize=True,
    graphs_number=None,
    clustering_methods=None,
):
    """
    Basic visualization using raw features.

    Args:
        dataset: Dataset name.
        df: DataFrame.
        save_path: Output directory.
        num_epochs: Current epoch.
        lr: Learning rate.
        n_clusters: Number of clusters.
        size: Point size.
        visualize: Whether to create visualization images.
        graphs_number: Optional graph count suffix used in label file naming.
        clustering_methods: Optional list of clustering method names.

    Returns:
        tuple: (metrics dict, figure if visualize=True)
    """
    # Load location labels.
    df = load_location_data(df, dataset, graphs_number)

    # Extract features and labels.
    features = df.drop(columns=["cell", "gene", "location"]).values
    true_labels = df["location"].astype(str).values

    # Dimensionality reduction (only when visualization is requested).
    reduced_features_tsne = None
    reduced_features_umap = None
    if visualize:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_features_tsne = tsne.fit_transform(features)

        umap_model = _lazy_import_umap().UMAP(
            n_components=2, random_state=42, n_neighbors=15, min_dist=0.2
        )
        reduced_features_umap = umap_model.fit_transform(features)

    # Clustering.
    clustering_methods_dict = apply_clustering(features, n_clusters, clustering_methods)

    # Evaluation.
    metrics = evaluate_clustering(
        true_labels, clustering_methods_dict, df, save_path, num_epochs, lr
    )

    # Visualization.
    if visualize:
        fig = visualize_clustering(
            reduced_features_tsne,
            reduced_features_umap,
            true_labels,
            clustering_methods_dict,
            num_epochs,
            lr,
            size,
        )
        fig_path = f"{save_path}/epoch{num_epochs}_lr{lr}_visualization.png"
        fig.savefig(fig_path)
        return metrics, fig
    else:
        return metrics, None


def plot_scatter_mutilbatch_scaler_tensorboard(
    dataset,
    df,
    save_path,
    num_epochs,
    lr,
    n_clusters,
    size=15,
    visualize=True,
    graphs_number=None,
    clustering_methods=None,
):
    """
    Visualization using standardized features.

    Args:
        Same as plot_scatter_mutilbatch_tensorboard.
    """
    # Load location labels.
    df = load_location_data(df, dataset, graphs_number)

    # Extract features and labels.
    features = df.drop(columns=["cell", "gene", "location"]).values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    true_labels = df["location"].astype(str).values

    # Dimensionality reduction (only when visualization is requested).
    reduced_features_tsne = None
    reduced_features_umap = None
    if visualize:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_features_tsne = tsne.fit_transform(features)

        umap_model = _lazy_import_umap().UMAP(
            n_components=2, random_state=42, n_neighbors=15, min_dist=0.2
        )
        reduced_features_umap = umap_model.fit_transform(features)

    # Clustering.
    clustering_methods_dict = apply_clustering(features, n_clusters, clustering_methods)

    # Evaluation.
    metrics = evaluate_clustering(
        true_labels, clustering_methods_dict, df, save_path, num_epochs, lr, "scaler"
    )

    # Visualization.
    if visualize:
        fig = visualize_clustering(
            reduced_features_tsne,
            reduced_features_umap,
            true_labels,
            clustering_methods_dict,
            num_epochs,
            lr,
            size,
        )
        fig_path = f"{save_path}/epoch{num_epochs}_lr{lr}_visualization_scaler.png"
        fig.savefig(fig_path)
        return metrics, fig
    else:
        return metrics, None


def plot_scatter_mutilbatch_pca_tensorboard(
    dataset,
    df,
    save_path,
    num_epochs,
    lr,
    n_clusters,
    size=15,
    visualize=True,
    graphs_number=None,
    clustering_methods=None,
):
    """
    Visualization with PCA-reduced features.

    Args:
        Same as plot_scatter_mutilbatch_tensorboard.
    """
    # Load location labels.
    df = load_location_data(df, dataset, graphs_number)

    # Extract features and labels.
    features = df.drop(columns=["cell", "gene", "location"]).values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    true_labels = df["location"].astype(str).values

    # PCA.
    pca = PCA(n_components=0.95)  # keep 95% variance
    features = pca.fit_transform(features)

    # Dimensionality reduction for plotting (only when requested).
    reduced_features_tsne = None
    reduced_features_umap = None
    if visualize:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_features_tsne = tsne.fit_transform(features)

        umap_model = _lazy_import_umap().UMAP(
            n_components=2, random_state=42, n_neighbors=15, min_dist=0.2
        )
        reduced_features_umap = umap_model.fit_transform(features)

    # Clustering.
    clustering_methods_dict = apply_clustering(features, n_clusters, clustering_methods)

    # Evaluation.
    metrics = evaluate_clustering(
        true_labels, clustering_methods_dict, df, save_path, num_epochs, lr, "pca"
    )

    # Visualization.
    if visualize:
        fig = visualize_clustering(
            reduced_features_tsne,
            reduced_features_umap,
            true_labels,
            clustering_methods_dict,
            num_epochs,
            lr,
            size,
        )
        fig_path = f"{save_path}/epoch{num_epochs}_lr{lr}_visualization_pca.png"
        fig.savefig(fig_path)
        return metrics, fig
    else:
        return metrics, None


def plot_scatter_mutilbatch_select_tensorboard(
    dataset,
    df,
    save_path,
    num_epochs,
    lr,
    n_clusters,
    size=15,
    visualize=True,
    graphs_number=None,
    clustering_methods=None,
):
    """
    Visualization using feature selection.

    Args:
        dataset: Dataset name.
        df: DataFrame.
        save_path: Output directory.
        num_epochs: Current epoch.
        lr: Learning rate.
        n_clusters: Number of clusters.
        size: Point size.
        visualize: Whether to create visualization images.
        graphs_number: Optional graph count suffix used in label file naming.
        clustering_methods: Optional list of clustering method names.

    Returns:
        tuple: (metrics dict, figure if visualize=True)
    """
    # Load location labels.
    df = load_location_data(df, dataset, graphs_number)

    # Extract features and labels.
    features = df.drop(columns=["cell", "gene", "location"]).values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    true_labels = df["location"].astype(str).values

    # Feature selection.
    selector = SelectKBest(score_func=f_classif, k=50)  # top-50 features
    features = selector.fit_transform(features, true_labels)

    # Dimensionality reduction for plotting (only when requested).
    reduced_features_tsne = None
    reduced_features_umap = None
    if visualize:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_features_tsne = tsne.fit_transform(features)

        umap_model = _lazy_import_umap().UMAP(
            n_components=2, random_state=42, n_neighbors=15, min_dist=0.2
        )
        reduced_features_umap = umap_model.fit_transform(features)

    # Clustering.
    clustering_methods_dict = apply_clustering(features, n_clusters, clustering_methods)

    # Evaluation.
    metrics = evaluate_clustering(
        true_labels, clustering_methods_dict, df, save_path, num_epochs, lr, "select"
    )

    # Visualization.
    if visualize:
        fig = visualize_clustering(
            reduced_features_tsne,
            reduced_features_umap,
            true_labels,
            clustering_methods_dict,
            num_epochs,
            lr,
            size,
        )
        fig_path = f"{save_path}/epoch{num_epochs}_lr{lr}_visualization_select.png"
        fig.savefig(fig_path)
        return metrics, fig
    else:
        return metrics, None


def plot_scatter_nolabel(
    dataset,
    df,
    save_path,
    num_epochs,
    lr,
    n_clusters,
    size=15,
    visualize=True,
    clustering_methods=None,
):
    """
    Clustering and visualization without ground-truth labels.

    Args:
        dataset: Dataset name.
        df: DataFrame.
        save_path: Output directory.
        num_epochs: Current epoch.
        lr: Learning rate.
        n_clusters: Number of clusters.
        size: Point size.
        visualize: Whether to create visualization images.
        clustering_methods: Optional list of clustering method names.
    """
    df["location"] = "other"

    features = df.drop(columns=["cell", "gene", "location"]).values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Cluster even if visualization is disabled.
    clustering_methods_dict = apply_clustering(features, n_clusters, clustering_methods)

    # Save clustering results.
    for method, labels in clustering_methods_dict.items():
        df[method + "_cluster"] = labels

    df.to_csv(f"{save_path}/epoch{num_epochs}_lr{lr}_clusters.csv", index=False)

    # Visualization (only when requested).
    if visualize:
        # Dimensionality reduction.
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_features_tsne = tsne.fit_transform(features)

        umap_model = _lazy_import_umap().UMAP(
            n_components=2, random_state=42, n_neighbors=15, min_dist=0.2
        )
        reduced_features_umap = umap_model.fit_transform(features)

        fig, axes = plt.subplots(
            len(clustering_methods_dict),
            2,
            figsize=(16, 4 * len(clustering_methods_dict)),
        )
        fig.suptitle(f"Epoch {num_epochs} lr={lr}", fontsize=16)

        for i, (method, predicted_labels) in enumerate(clustering_methods_dict.items()):
            sns.scatterplot(
                x=reduced_features_tsne[:, 0],
                y=reduced_features_tsne[:, 1],
                hue=predicted_labels,
                palette="Set2",
                ax=axes[i, 0],
                s=size,
                legend="full",
            )
            axes[i, 0].set_title(f"t-SNE with {method} Clusters")
            axes[i, 0].legend(loc="upper left", bbox_to_anchor=(1, 1))

            sns.scatterplot(
                x=reduced_features_umap[:, 0],
                y=reduced_features_umap[:, 1],
                hue=predicted_labels,
                palette="Set3",
                ax=axes[i, 1],
                s=size,
                legend="full",
            )
            axes[i, 1].set_title(f"UMAP with {method} Clusters")
            axes[i, 1].legend(loc="upper left", bbox_to_anchor=(1, 1))

        plt.tight_layout()
        fig.savefig(f"{save_path}/epoch{num_epochs}_lr{lr}_visualization_nolabel.png")
        return fig
    return None


def plot_embeddings_only(df, save_path, num_epochs, lr, visualize=False):
    """
    Save embeddings only (no clustering/evaluation).

    Args:
        df: DataFrame containing embeddings.
        save_path: Output directory.
        num_epochs: Current epoch.
        lr: Learning rate.
        visualize: Whether to create a quick embedding scatter plot.

    Returns:
        fig: Figure if visualize=True, else None.
    """
    # Save embeddings.
    df.to_csv(f"{save_path}/epoch{num_epochs}_lr{lr}_embedding.csv", index=False)
    print(f"Embeddings saved to {save_path}/epoch{num_epochs}_lr{lr}_embedding.csv")

    # Create a simple embedding visualization on demand.
    if visualize:
        # Extract features.
        features = df.drop(columns=["cell", "gene"]).values

        n_samples = features.shape[0]
        # Very small demos (e.g., smoke tests) can have only a handful of graphs.
        # t-SNE / UMAP can fail in these edge cases. Prefer skipping visualization
        # over crashing the whole training run.
        if n_samples < 3:
            print(f"Skip embedding visualization: too few samples (n={n_samples}).")
            return None

        # Standardize.
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Dimensionality reduction.
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples - 1))
        reduced_features_tsne = tsne.fit_transform(features)

        # UMAP's default init='spectral' can fail when n is extremely small (k >= n).
        # Use init='random' for small n to keep smoke tests robust.
        umap_init = "random" if n_samples < 4 else "spectral"
        umap_model = _lazy_import_umap().UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(15, n_samples - 1),
            min_dist=0.2,
            init=umap_init,
        )
        reduced_features_umap = umap_model.fit_transform(features)

        # Simple scatter plots.
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Color by cell.
        sns.scatterplot(
            x=reduced_features_tsne[:, 0],
            y=reduced_features_tsne[:, 1],
            hue=df["cell"],
            ax=axes[0],
            s=15,
            legend="auto",
        )
        axes[0].set_title("t-SNE Embedding by Cell Type")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Color by gene.
        sns.scatterplot(
            x=reduced_features_umap[:, 0],
            y=reduced_features_umap[:, 1],
            hue=df["gene"],
            ax=axes[1],
            s=15,
            legend="auto",
        )
        axes[1].set_title("UMAP Embedding by Gene Type")
        axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1))

        plt.tight_layout()
        fig.savefig(
            f"{save_path}/epoch{num_epochs}_lr{lr}_embeddings_visualization.png"
        )
        return fig

    return None


def plot_loss_weights(num_epochs, weights_a, weights_b, weights_c, save_path, lr):
    """
    Plot dynamic loss weights over epochs.

    Args:
        num_epochs: Total number of epochs.
        weights_a: Reconstruction loss weights.
        weights_b: Contrastive loss weights.
        weights_c: Clustering loss weights.
        save_path: Output directory.
        lr: Learning rate.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(
        range(num_epochs),
        weights_a,
        label="Reconstruction Loss Weight (a)",
        color="blue",
    )
    plt.plot(
        range(num_epochs),
        weights_b,
        label="Contrastive Loss Weight (b)",
        color="orange",
    )
    plt.plot(
        range(num_epochs), weights_c, label="Clustering Loss Weight (c)", color="green"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Weight")
    plt.title("Dynamic Loss Weights Over Epochs")
    plt.legend()
    plt.savefig(f"{save_path}/epoch{num_epochs}_lr{lr}_plot_learning_rate.png")
