########################################################
# Cell and nucleus boundary registration utilities.
#
# This module includes single-threaded and parallel variants used by the GRASP
# preprocessing pipeline.
########################################################

import numpy as np
import pandas as pd
import pickle
import timeit
import warnings
from typing import Any, cast
from tqdm import tqdm
import math
from math import pi
from multiprocessing import Pool, cpu_count
from functools import partial
import os
from datetime import datetime
from scipy.interpolate import interp1d, splprep, splev


def interpolate_boundary_points(
    cell_boundary_dict, target_points_per_cell=100, method="spline", smooth_factor=0
):
    """Interpolate boundary points to increase boundary resolution.

    Args:
        cell_boundary_dict: Mapping {cell_id: DataFrame with columns x,y}.
        target_points_per_cell: Target number of points after interpolation.
        method: "spline" or "linear".
        smooth_factor: Spline smoothing factor (0 means interpolate through all points).

    Returns:
        Mapping {cell_id: DataFrame with interpolated x,y}.
    """
    interpolated_boundary = {}

    print(f"Interpolating boundary points for {len(cell_boundary_dict)} cells...")

    for cell_id, boundary_df in tqdm(
        cell_boundary_dict.items(), desc="Boundary interpolation"
    ):
        try:
            cell_method = method

            # Original points
            x_points = boundary_df["x"].values
            y_points = boundary_df["y"].values

            x_new = None
            y_new = None

            # Need at least 3 points.
            if len(x_points) < 3:
                print(
                    f"WARNING: cell {cell_id} has too few boundary points "
                    f"({len(x_points)}); skip interpolation"
                )
                interpolated_boundary[cell_id] = boundary_df.copy()
                continue

            if len(x_points) >= target_points_per_cell:
                interpolated_boundary[cell_id] = boundary_df.copy()
                continue

            # Ensure a closed boundary.
            if not (
                np.isclose(x_points[0], x_points[-1])
                and np.isclose(y_points[0], y_points[-1])
            ):
                x_points = np.append(x_points, x_points[0])
                y_points = np.append(y_points, y_points[0])

            if cell_method == "spline":
                # Spline interpolation.
                distances = np.sqrt(np.diff(x_points) ** 2 + np.diff(y_points) ** 2)
                distances = np.insert(distances, 0, 0)
                cumulative_distance = np.cumsum(distances)

                # Avoid duplicated points.
                unique_indices = np.unique(cumulative_distance, return_index=True)[1]
                if len(unique_indices) < 3:
                    print(
                        f"WARNING: cell {cell_id} has too few unique points; "
                        "fall back to linear interpolation"
                    )
                    cell_method = "linear"
                else:
                    x_unique = x_points[unique_indices]
                    y_unique = y_points[unique_indices]
                    t_unique = cumulative_distance[unique_indices]

                    try:
                        tck, u = splprep(
                            [x_unique, y_unique], s=smooth_factor, per=True
                        )

                        u_new = np.linspace(
                            0, 1, target_points_per_cell, endpoint=False
                        )

                        x_new, y_new = splev(u_new, tck)

                    except Exception as e:
                        print(
                            f"WARNING: cell {cell_id} spline interpolation failed: {e}; "
                            "fall back to linear interpolation"
                        )
                        cell_method = "linear"

            if cell_method == "linear":
                # Linear interpolation.
                distances = np.sqrt(np.diff(x_points) ** 2 + np.diff(y_points) ** 2)
                distances = np.insert(distances, 0, 0)
                cumulative_distance = np.cumsum(distances)

                if cumulative_distance[-1] > 0:
                    normalized_distance = cumulative_distance / cumulative_distance[-1]
                else:
                    normalized_distance = cumulative_distance

                t_new = np.linspace(0, 1, target_points_per_cell, endpoint=False)

                try:
                    x_interp = interp1d(
                        normalized_distance,
                        x_points,
                        kind="linear",
                        assume_sorted=True,
                        bounds_error=False,
                        fill_value=cast(Any, "extrapolate"),
                    )
                    y_interp = interp1d(
                        normalized_distance,
                        y_points,
                        kind="linear",
                        assume_sorted=True,
                        bounds_error=False,
                        fill_value=cast(Any, "extrapolate"),
                    )

                    x_new = x_interp(t_new)
                    y_new = y_interp(t_new)

                except Exception as e:
                    print(
                        f"WARNING: cell {cell_id} linear interpolation failed: {e}; "
                        "keep original points"
                    )
                    interpolated_boundary[cell_id] = boundary_df.copy()
                    continue

            if x_new is None or y_new is None:
                interpolated_boundary[cell_id] = boundary_df.copy()
                continue

            interpolated_df = pd.DataFrame({"x": x_new, "y": y_new})

            interpolated_boundary[cell_id] = interpolated_df

        except Exception as e:
            print(f"ERROR: failed to process cell {cell_id}: {e}")
            interpolated_boundary[cell_id] = boundary_df.copy()

    original_points = sum(len(df) for df in cell_boundary_dict.values())
    new_points = sum(len(df) for df in interpolated_boundary.values())

    print("Interpolation complete:")
    print(f"  total_original_points: {original_points}")
    print(f"  total_interpolated_points: {new_points}")
    print(
        f"  mean_original_points_per_cell: {original_points / len(cell_boundary_dict):.1f}"
    )
    print(
        f"  mean_interpolated_points_per_cell: {new_points / len(interpolated_boundary):.1f}"
    )

    return interpolated_boundary


def enhance_boundary_resolution(
    cell_boundary_dict, min_points_per_cell=50, adaptive=True
):
    """Increase boundary resolution for each cell boundary.

    If adaptive=True, the target point count is derived from the boundary
    perimeter; otherwise, uses a fixed minimum.
    """
    enhanced_boundary = {}

    for cell_id, boundary_df in cell_boundary_dict.items():
        current_points = len(boundary_df)

        if adaptive:
            x_points = boundary_df["x"].values
            y_points = boundary_df["y"].values

            perimeter = 0
            for i in range(len(x_points)):
                next_i = (i + 1) % len(x_points)
                perimeter += np.sqrt(
                    (x_points[next_i] - x_points[i]) ** 2
                    + (y_points[next_i] - y_points[i]) ** 2
                )

            target_points = max(min_points_per_cell, int(perimeter / 2.5))
        else:
            target_points = min_points_per_cell

        if current_points >= target_points:
            enhanced_boundary[cell_id] = boundary_df.copy()
        else:
            # Interpolate to increase boundary resolution.
            temp_dict = {cell_id: boundary_df}
            interpolated = interpolate_boundary_points(
                temp_dict, target_points_per_cell=target_points, method="spline"
            )
            enhanced_boundary[cell_id] = interpolated[cell_id]

    return enhanced_boundary


def register_cells(
    data_df, cell_list_all, cell_mask_df, ntanbin_dict, epsilon=1e-10, nc_demo=None
):
    if nc_demo is None:
        nc_demo = len(cell_list_all)
    dict_registered = {}
    cell_radii = {}  # Per-cell maximum radius.
    df = data_df.copy()  # cp original data
    df_gbC = df.groupby("cell", observed=False)  # group by `cell`
    for ic, c in enumerate(tqdm(cell_list_all[:nc_demo], desc="Processing cells")):
        df_c = df_gbC.get_group(c).copy()  # df for cell c
        t = df_c.type.iloc[0]  # cell type for cell c
        mask_df_c = cell_mask_df[cell_mask_df.cell == c]  # get the mask df for cell c
        center_c = [
            int(df_c.centerX.iloc[0]),
            int(df_c.centerY.iloc[0]),
        ]  # nuclear center of cell c
        tanbin = np.linspace(0, pi / 2, ntanbin_dict[t] + 1)
        delta_tanbin = (2 * math.pi) / (ntanbin_dict[t] * 4)
        # add centered coord and ratio=y/x for df_c and mask_df_c
        df_c["x_c"] = df_c.x.copy() - center_c[0]
        df_c["y_c"] = df_c.y.copy() - center_c[1]
        df_c["d_c"] = (df_c.x_c.copy() ** 2 + df_c.y_c.copy() ** 2) ** 0.5
        df_c["arctan"] = np.absolute(np.arctan(df_c.y_c / (df_c.x_c + epsilon)))
        mask_df_c["x_c"] = mask_df_c.x.copy() - center_c[0]
        mask_df_c["y_c"] = mask_df_c.y.copy() - center_c[1]
        mask_df_c["d_c"] = (
            mask_df_c.x_c.copy() ** 2 + mask_df_c.y_c.copy() ** 2
        ) ** 0.5
        mask_df_c["arctan"] = np.absolute(
            np.arctan(mask_df_c.y_c / (mask_df_c.x_c + epsilon))
        )
        # in each quatrant, find dismax_c for each tanbin interval using mask_df_c
        mask_df_c_q_dict = {}
        mask_df_c_q_dict["0"] = mask_df_c[(mask_df_c.x_c >= 0) & (mask_df_c.y_c >= 0)]
        mask_df_c_q_dict["1"] = mask_df_c[(mask_df_c.x_c <= 0) & (mask_df_c.y_c >= 0)]
        mask_df_c_q_dict["2"] = mask_df_c[(mask_df_c.x_c <= 0) & (mask_df_c.y_c <= 0)]
        mask_df_c_q_dict["3"] = mask_df_c[(mask_df_c.x_c >= 0) & (mask_df_c.y_c <= 0)]
        # compute the dismax_c
        dismax_c_mat = np.zeros((ntanbin_dict[t], 4))
        for q in range(4):  # in each of the 4 quantrants
            mask_df_c_q = mask_df_c_q_dict[str(q)]
            mask_df_c_q["arctan_idx"] = (mask_df_c_q.arctan / delta_tanbin).astype(
                int
            )  # arctan_idx from 0 to self.ntanbin_dict[t]-1
            dismax_c_mat[
                mask_df_c_q.groupby("arctan_idx").max()["d_c"].index.to_numpy(), q
            ] = (
                mask_df_c_q.groupby("arctan_idx").max()["d_c"].values
            )  # automatically sorted by arctan_idx from 0 to self.ntanbin_dict[t]-1

        # for df_c, for arctan in each interval, find max dis using dismax_c
        df_c_q_dict = {}
        df_c_q_dict["0"] = df_c[(df_c.x_c >= 0) & (df_c.y_c >= 0)]
        df_c_q_dict["1"] = df_c[(df_c.x_c <= 0) & (df_c.y_c >= 0)]
        df_c_q_dict["2"] = df_c[(df_c.x_c <= 0) & (df_c.y_c <= 0)]
        df_c_q_dict["3"] = df_c[(df_c.x_c >= 0) & (df_c.y_c <= 0)]
        d_c_maxc_dict = {}
        for q in range(4):  # in each of the 4 quantrants
            df_c_q = df_c_q_dict[str(q)]
            d_c_maxc_q = np.zeros(len(df_c_q))
            df_c_q["arctan_idx"] = (df_c_q.arctan / delta_tanbin).astype(
                int
            )  # arctan_idx from 0 to self.ntanbin_dict[t]-1
            for ai in range(ntanbin_dict[t]):
                d_c_maxc_q[df_c_q.arctan_idx.values == ai] = dismax_c_mat[ai, q]
            d_c_maxc_dict[str(q)] = d_c_maxc_q
        d_c_maxc = np.zeros(len(df_c))
        d_c_maxc[(df_c.x_c >= 0) & (df_c.y_c >= 0)] = d_c_maxc_dict["0"]
        d_c_maxc[(df_c.x_c <= 0) & (df_c.y_c >= 0)] = d_c_maxc_dict["1"]
        d_c_maxc[(df_c.x_c <= 0) & (df_c.y_c <= 0)] = d_c_maxc_dict["2"]
        d_c_maxc[(df_c.x_c >= 0) & (df_c.y_c <= 0)] = d_c_maxc_dict["3"]
        df_c["d_c_maxc"] = d_c_maxc

        # scale centered x_c and y_c
        d_c_s = np.zeros(len(df_c))
        x_c_s = np.zeros(len(df_c))
        y_c_s = np.zeros(len(df_c))
        d_c_s = df_c.d_c / (df_c.d_c_maxc + epsilon)
        x_c_s = df_c.x_c * (d_c_s / (df_c.d_c + epsilon))
        y_c_s = df_c.y_c * (d_c_s / (df_c.d_c + epsilon))
        df_c["x_c_s"] = x_c_s
        df_c["y_c_s"] = y_c_s
        df_c["d_c_s"] = d_c_s

        # Store per-cell maximum radius.
        cell_radii[c] = np.max(df_c["d_c_maxc"])

        dict_registered[c] = df_c
        del df_c
    # concatenate to one df
    df_registered = pd.concat(list(dict_registered.values()))
    print(f"Number of cells registered {len(dict_registered)}")
    return df_registered, cell_radii


def specify_ntanbin(
    cell_list_dict,
    cell_mask_df,
    type_list,
    nc4ntanbin=10,
    high_res=200,
    max_ntanbin=25,
    input_ntanbin_dict=None,
    min_bp=5,
    min_ntanbin_error=3,
):
    ntanbin_dict = {}  # Initialize empty dict.
    if input_ntanbin_dict is not None:  # use customized ntanbin across cell types
        ntanbin_dict = input_ntanbin_dict

    if input_ntanbin_dict is None:  # compute ntanbin for each cell type:
        for t in type_list:
            # specify ntanbin_gen based on cell seg mask/boundary
            # random sample self.nc4ntanbin cells, allow replace
            cell_list_sampled = np.random.choice(
                cell_list_dict[t], nc4ntanbin, replace=True
            )
            cell_mask_df_sampled = cell_mask_df[
                cell_mask_df.cell.isin(cell_list_sampled)
            ]
            # compute                                                                                                                                                                                                                                                                                                                                                                                          the #x and #y unique coords of these sampled cells
            nxu_sampled = []
            nyu_sampled = []
            for c in cell_list_sampled:
                mask_c = cell_mask_df_sampled[cell_mask_df_sampled.cell == c]
                nxu_sampled.append(mask_c.x.nunique())
                nyu_sampled.append(mask_c.y.nunique())

            # specify ntanbin for pi/2 (a quantrant)
            # if resolution is super high
            if np.mean(nxu_sampled) > high_res and np.mean(nyu_sampled) > high_res:
                ntanbin = max_ntanbin
            # if resolution is not super high
            else:
                # require at least self.min_bp boundary points in each tanbin
                theta = 2 * np.arctan(min_bp / np.mean(nxu_sampled + nyu_sampled))
                ntanbin_ = (pi / 2) / theta
                ntanbin = np.ceil(ntanbin_)
                if ntanbin < min_ntanbin_error:
                    print(
                        f"Cell type {t} failed, resolution not high enougth to support the analysis"
                    )
                    ntanbin = 3
            # asign
            ntanbin_dict[t] = int(ntanbin)
    return ntanbin_dict


def process_chunk_cell(chunk, df_gbC, cell_mask_df, ntanbin_dict, epsilon):
    results = []
    for c in chunk:
        df_c = df_gbC.get_group(c).copy()  # Per-cell transcript table.
        t = df_c.type.iloc[0]  # Cell type.
        mask_df_c = cell_mask_df[
            cell_mask_df.cell == c
        ].copy()  # get the mask df for cell c
        center_c = [
            int(df_c.centerX.iloc[0]),
            int(df_c.centerY.iloc[0]),
        ]  # nuclear center of cell c
        tanbin = np.linspace(0, pi / 2, ntanbin_dict[t] + 1)  # Angle bins per quadrant.
        delta_tanbin = (2 * math.pi) / (
            ntanbin_dict[t] * 4
        )  # Full circle 2*pi is split into (ntanbin_dict[t] * 4) bins.
        mask_df_c["x_c"] = mask_df_c.x.copy() - center_c[0]
        mask_df_c["y_c"] = mask_df_c.y.copy() - center_c[1]
        mask_df_c["d_c"] = (
            mask_df_c.x_c.copy() ** 2 + mask_df_c.y_c.copy() ** 2
        ) ** 0.5
        mask_df_c["arctan"] = np.absolute(
            np.arctan(mask_df_c.y_c / (mask_df_c.x_c + epsilon))
        )
        # Split mask points into 4 quadrants.
        mask_df_c_q_dict = {
            "0": mask_df_c[(mask_df_c.x_c >= 0) & (mask_df_c.y_c >= 0)],
            "1": mask_df_c[(mask_df_c.x_c <= 0) & (mask_df_c.y_c >= 0)],
            "2": mask_df_c[(mask_df_c.x_c <= 0) & (mask_df_c.y_c <= 0)],
            "3": mask_df_c[(mask_df_c.x_c >= 0) & (mask_df_c.y_c <= 0)],
        }
        # compute the dismax_c
        dismax_c_mat = np.zeros((ntanbin_dict[t], 4))  # Shape: (n_bins, 4 quadrants).
        for q in range(4):  # in each of the 4 quantrants
            mask_df_c_q = mask_df_c_q_dict[str(q)].copy()  # Work on a copy.
            if len(mask_df_c_q) > 0:
                mask_df_c_q["arctan_idx"] = (
                    mask_df_c_q["arctan"] / delta_tanbin
                ).astype(int)  # arctan_idx from 0 to ntanbin_dict[t]-1.
                # Ensure arctan_idx stays in range.
                mask_df_c_q["arctan_idx"] = np.minimum(
                    mask_df_c_q["arctan_idx"], ntanbin_dict[t] - 1
                )
                max_distances = mask_df_c_q.groupby("arctan_idx").max()["d_c"]
                if not max_distances.empty:
                    dismax_c_mat[max_distances.index.to_numpy(), q] = (
                        max_distances.values
                    )

        # Fill missing max radius values.
        for q in range(4):
            for ai in range(ntanbin_dict[t]):
                if dismax_c_mat[ai, q] == 0:
                    # Use non-zero neighbors if available.
                    neighbors = [
                        i for i in range(ntanbin_dict[t]) if dismax_c_mat[i, q] > 0
                    ]
                    if neighbors:
                        dismax_c_mat[ai, q] = np.mean(
                            [dismax_c_mat[i, q] for i in neighbors]
                        )
                    else:
                        # Fall back to max radius within the quadrant.
                        max_q = np.max(dismax_c_mat[:, q])
                        if max_q > 0:
                            dismax_c_mat[ai, q] = max_q
                        else:
                            # If the whole quadrant is empty, use the global max.
                            max_all = np.max(dismax_c_mat)
                            if max_all > 0:
                                dismax_c_mat[ai, q] = max_all
                            else:
                                # If everything is empty, use a large default.
                                dismax_c_mat[ai, q] = 100

        # dismax_c_mat stores max radii for each (angle bin, quadrant).
        # add centered coord and ratio=y/x for df_c and mask_df_c
        df_c["x_c"] = df_c.x.copy() - center_c[0]  # Relative to cell center.
        df_c["y_c"] = df_c.y.copy() - center_c[1]
        df_c["d_c"] = (
            df_c.x_c.copy() ** 2 + df_c.y_c.copy() ** 2
        ) ** 0.5  # Distance to center.
        df_c["arctan"] = np.absolute(
            np.arctan(df_c.y_c / (df_c.x_c + epsilon))
        )  # Angle to x-axis.

        # Normalize coordinates.
        df_c_registered = normalize_dataset(
            df_c,
            dismax_c_mat,
            delta_tanbin,
            ntanbin_dict,
            t,
            epsilon,
            is_nucleus=False,
            clip_to_cell=True,
        )
        cell_radius = df_c_registered["d_c_maxc"].max()  # Per-cell maximum radius.
        results.append((df_c_registered, cell_radius))
    return results


def register_cells_parallel_chunked(
    data_df,
    cell_list_all,
    cell_mask_df,
    ntanbin_dict,
    epsilon=1e-10,
    nc_demo=None,
    chunk_size=5,
):
    if nc_demo is None:
        nc_demo = len(cell_list_all)
    df_gbC = data_df.groupby("cell", observed=False)  # Group by cell.
    chunks = list(chunk_list(cell_list_all[:nc_demo], chunk_size))  # Split into chunks.
    pool = Pool(processes=cpu_count() - 2)  # Leave some CPU for the system.
    process_chunk_partial = partial(
        process_chunk_cell,
        df_gbC=df_gbC,
        cell_mask_df=cell_mask_df,
        ntanbin_dict=ntanbin_dict,
        epsilon=epsilon,
    )
    results = list(
        tqdm(
            pool.imap(process_chunk_partial, chunks),
            total=len(chunks),
            desc="Processing chunks in parallel",
        )
    )  # Parallel processing.
    pool.close()  # Close pool.
    pool.join()  # Wait for all workers.
    all_cell_dfs = []  # Aggregate results.
    all_nuclear_dfs = []
    all_radii = {}
    for result_chunk in results:
        for df_c_registered, cell_radius in result_chunk:
            all_cell_dfs.append(df_c_registered)
            all_radii.update(
                {df_c_registered["cell"].iloc[0]: cell_radius}
            )  # Store per-cell radius in a dict.
    cell_df_registered = pd.concat(all_cell_dfs)
    return cell_df_registered, all_radii


def chunk_list(data_list, chunk_size):  # Split a list into chunks.
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i : i + chunk_size]


def process_chunk(
    chunk,
    df_gbC,
    cell_mask_df,
    nuclear_boundary,
    ntanbin_dict,
    epsilon,
    clip_to_cell=True,
    remove_outliers=False,
    verbose=False,
):
    """Process a chunk of cells and (optionally) their nucleus boundaries.

    Parameters
    ----------
    chunk:
        List of cell IDs to process.
    df_gbC:
        data_df grouped by cell (DataFrameGroupBy).
    cell_mask_df:
        Cell boundary mask points.
    nuclear_boundary:
        Mapping {cell_id: DataFrame with nucleus boundary points}.
    ntanbin_dict:
        Mapping {cell_type: number_of_angle_bins_per_quadrant}.
    epsilon:
        Small constant for numerical stability.
    clip_to_cell:
        If True, clip normalized distances (d_c_s) to <= 1.
    remove_outliers:
        If True, drop nucleus points that exceed the cell boundary.
    verbose:
        If True, print additional warnings and statistics.

    Returns
    -------
    list
        List of tuples: (df_c_registered, nuclear_boundary_c_registered, cell_radius).
    """
    results = []
    for c in chunk:
        try:
            df_c = df_gbC.get_group(c).copy()  # Per-cell transcript table.
        except KeyError:
            if verbose:
                print(f"Warning: Cell {c} not found in data_df")
            continue

        t = df_c.type.iloc[0]  # Cell type.

        # Try different lookup strategies (cell id types may differ).
        mask_df_c = cell_mask_df[cell_mask_df.cell == c].copy()
        if len(mask_df_c) == 0:
            # Try casting cell ids to match.
            if isinstance(c, str):
                if verbose:
                    print(f"Converting cell {c} to string")
                mask_df_c = cell_mask_df[cell_mask_df.cell.astype(str) == c].copy()
            elif isinstance(c, (int, np.integer)):
                if verbose:
                    print(f"Converting cell {c} to integer")
                mask_df_c = cell_mask_df[cell_mask_df.cell.astype(int) == c].copy()

        if len(mask_df_c) == 0:
            if verbose:
                print(f"Warning: No mask points found for cell {c}")
            continue

        try:
            nuclear_boundary_c = nuclear_boundary[
                c
            ].copy()  # Current cell nucleus boundary.
        except KeyError:
            if verbose:
                print(f"Warning: No nuclear boundary found for cell {c}")
            continue

        center_c = [
            int(df_c.centerX.iloc[0]),
            int(df_c.centerY.iloc[0]),
        ]  # nuclear center of cell c
        tanbin = np.linspace(0, pi / 2, ntanbin_dict[t] + 1)  # Angle bins per quadrant.
        delta_tanbin = (2 * math.pi) / (
            ntanbin_dict[t] * 4
        )  # Full circle 2*pi is split into (ntanbin_dict[t] * 4) bins.

        # Precompute mask coordinates relative to the cell center.
        mask_df_c["x_c"] = mask_df_c.x.copy() - center_c[0]
        mask_df_c["y_c"] = mask_df_c.y.copy() - center_c[1]
        mask_df_c["d_c"] = (
            mask_df_c.x_c.copy() ** 2 + mask_df_c.y_c.copy() ** 2
        ) ** 0.5
        mask_df_c["arctan"] = np.absolute(
            np.arctan(mask_df_c.y_c / (mask_df_c.x_c + epsilon))
        )

        # Split mask points into 4 quadrants.
        mask_df_c_q_dict = {
            "0": mask_df_c[(mask_df_c.x_c >= 0) & (mask_df_c.y_c >= 0)],
            "1": mask_df_c[(mask_df_c.x_c <= 0) & (mask_df_c.y_c >= 0)],
            "2": mask_df_c[(mask_df_c.x_c <= 0) & (mask_df_c.y_c <= 0)],
            "3": mask_df_c[(mask_df_c.x_c >= 0) & (mask_df_c.y_c <= 0)],
        }

        # For each angle bin (per quadrant), compute the maximum radius.
        dismax_c_mat = np.zeros((ntanbin_dict[t], 4))  # Shape: (n_bins, 4 quadrants).
        for q in range(4):  # in each of the 4 quantrants
            mask_df_c_q = mask_df_c_q_dict[str(q)].copy()  # Work on a copy.
            if len(mask_df_c_q) > 0:
                mask_df_c_q["arctan_idx"] = (
                    mask_df_c_q["arctan"] / delta_tanbin
                ).astype(int)  # arctan_idx from 0 to ntanbin_dict[t]-1.
                # Ensure arctan_idx stays in range.
                mask_df_c_q["arctan_idx"] = np.minimum(
                    mask_df_c_q["arctan_idx"], ntanbin_dict[t] - 1
                )
                max_distances = mask_df_c_q.groupby("arctan_idx").max()["d_c"]
                if not max_distances.empty:
                    dismax_c_mat[max_distances.index.to_numpy(), q] = (
                        max_distances.values
                    )  # automatically sorted by arctan_idx from 0 to self.ntanbin_dict[t]-1

        # Fill missing max radius values.
        fill_zero_indices = np.where(dismax_c_mat == 0)
        if len(fill_zero_indices[0]) > 0:
            for ai, q in zip(fill_zero_indices[0], fill_zero_indices[1]):
                # Find nearby non-zero values.
                neighbors = []
                for offset in range(1, ntanbin_dict[t]):
                    ai_before = (ai - offset) % ntanbin_dict[t]
                    ai_after = (ai + offset) % ntanbin_dict[t]
                    if dismax_c_mat[ai_before, q] > 0:
                        neighbors.append(dismax_c_mat[ai_before, q])
                    if dismax_c_mat[ai_after, q] > 0:
                        neighbors.append(dismax_c_mat[ai_after, q])
                    if neighbors:  # Stop once we find any non-zero neighbor.
                        break

                if neighbors:
                    dismax_c_mat[ai, q] = np.mean(neighbors)
                else:
                    # If no neighbors, use the mean of all non-zero values in this quadrant.
                    nonzero_in_q = dismax_c_mat[:, q][dismax_c_mat[:, q] > 0]
                    if len(nonzero_in_q) > 0:
                        dismax_c_mat[ai, q] = np.mean(nonzero_in_q)
                    else:
                        # If the quadrant is empty, use the mean of all non-zero values.
                        all_nonzero = dismax_c_mat[dismax_c_mat > 0]
                        if len(all_nonzero) > 0:
                            dismax_c_mat[ai, q] = np.mean(all_nonzero)
                        else:
                            # If everything is empty, fall back to a heuristic default.
                            dismax_c_mat[ai, q] = (
                                np.max(df_c["d_c"]) * 1.5
                            )  # Use 1.5x max gene-point distance.

        # dismax_c_mat stores max radii for each (angle bin, quadrant).
        # add centered coord and ratio=y/x for df_c and mask_df_c
        df_c["x_c"] = df_c.x.copy() - center_c[0]  # Relative to cell center.
        df_c["y_c"] = df_c.y.copy() - center_c[1]
        df_c["d_c"] = (
            df_c.x_c.copy() ** 2 + df_c.y_c.copy() ** 2
        ) ** 0.5  # Distance to center.
        df_c["arctan"] = np.absolute(
            np.arctan(df_c.y_c / (df_c.x_c + epsilon))
        )  # Angle to x-axis.
        # Nucleus boundary points relative to the cell center.
        nuclear_boundary_c["x_c"] = nuclear_boundary_c.x.copy() - center_c[0]
        nuclear_boundary_c["y_c"] = nuclear_boundary_c.y.copy() - center_c[1]
        nuclear_boundary_c["d_c"] = (
            nuclear_boundary_c.x_c**2 + nuclear_boundary_c.y_c**2
        ) ** 0.5
        nuclear_boundary_c["arctan"] = np.abs(
            np.arctan(nuclear_boundary_c.y_c / (nuclear_boundary_c.x_c + epsilon))
        )

        # Normalize cell and nucleus-boundary data.
        df_c_registered = normalize_dataset(
            df_c,
            dismax_c_mat,
            delta_tanbin,
            ntanbin_dict,
            t,
            epsilon,
            is_nucleus=False,
            clip_to_cell=True,
            remove_outliers=False,
        )
        nuclear_boundary_c_registered = normalize_dataset(
            nuclear_boundary_c,
            dismax_c_mat,
            delta_tanbin,
            ntanbin_dict,
            t,
            epsilon,
            is_nucleus=True,
            clip_to_cell=clip_to_cell,
            remove_outliers=remove_outliers,
        )
        nuclear_boundary_c_registered["cell"] = c

        # Compute the fraction of nucleus points exceeding the boundary.
        exceed_percent = 0
        if "exceeds_boundary" in nuclear_boundary_c_registered.columns:
            exceed_percent = (
                nuclear_boundary_c_registered["exceeds_boundary"].mean() * 100
            )
            if exceed_percent > 0 and verbose:
                print(
                    f"Cell {c}: {exceed_percent:.2f}% of nuclear boundary points exceed cell boundary"
                )

        cell_radius = df_c_registered["d_c_maxc"].max()  # Per-cell maximum radius.

        results.append((df_c_registered, nuclear_boundary_c_registered, cell_radius))

    return results


def register_cells_and_nuclei_parallel_chunked(
    data_df,
    cell_list_all,
    cell_mask_df,
    nuclear_boundary,
    ntanbin_dict,
    epsilon=1e-10,
    nc_demo=None,
    chunk_size=2,
    clip_to_cell=True,
    remove_outliers=False,
    verbose=False,
):
    """Register cells and nucleus boundaries in parallel (chunked).

    This function uses multiprocessing to process cells in chunks.

    Returns
    -------
    cell_df_registered:
        Registered transcript table for all processed cells.
    nuclear_boundary_df_registered:
        Registered nucleus boundary points for all processed cells.
    all_radii:
        Mapping {cell_id: cell_radius}.
    """
    if nc_demo is None:
        nc_demo = len(cell_list_all)
    df_gbC = data_df.groupby("cell", observed=False)  # Group by cell.
    chunks = list(chunk_list(cell_list_all[:nc_demo], chunk_size))  # Split into chunks.
    # pool = Pool(processes=cpu_count() - 2)  # Leave some CPU for the system.
    pool = Pool(processes=min(4, cpu_count() - 2))  # Cap worker count.
    process_chunk_partial = partial(
        process_chunk,
        df_gbC=df_gbC,
        cell_mask_df=cell_mask_df,
        nuclear_boundary=nuclear_boundary,
        ntanbin_dict=ntanbin_dict,
        epsilon=epsilon,
        clip_to_cell=clip_to_cell,
        remove_outliers=remove_outliers,
        verbose=verbose,
    )
    results = list(
        tqdm(
            pool.imap(process_chunk_partial, chunks),
            total=len(chunks),
            desc="Processing chunks in parallel",
        )
    )  # Parallel processing.
    pool.close()  # Close pool.
    pool.join()  # Wait for all workers.
    all_cell_dfs = []  # Aggregate results.
    all_nuclear_dfs = []
    all_radii = {}
    for result_chunk in results:
        for df_c_registered, nuclear_boundary_c_registered, cell_radius in result_chunk:
            all_cell_dfs.append(df_c_registered)
            all_nuclear_dfs.append(nuclear_boundary_c_registered)
            all_radii.update(
                {df_c_registered["cell"].iloc[0]: cell_radius}
            )  # Store per-cell radius in a dict.
    cell_df_registered = pd.concat(all_cell_dfs)
    nuclear_boundary_df_registered = pd.concat(all_nuclear_dfs)
    return cell_df_registered, nuclear_boundary_df_registered, all_radii


def register_cells_and_nuclei_parallel_chunked_constrained(
    data_df,
    cell_list_all,
    cell_mask_df,
    nuclear_boundary,
    ntanbin_dict,
    epsilon=1e-10,
    nc_demo=None,
    chunk_size=5,
    clip_to_cell=True,
    remove_outliers=False,
    verbose=True,
):
    """Chunked parallel registration with nucleus boundary constraint.

    Compared to register_cells_and_nuclei_parallel_chunked, this variant also
    returns per-cell statistics about nucleus points exceeding the cell boundary.
    """
    if nc_demo is None:
        nc_demo = len(cell_list_all)

    # Validate inputs first.
    missing_cells_mask = [
        c for c in cell_list_all[:nc_demo] if c not in cell_mask_df["cell"].unique()
    ]
    missing_cells_nuclear = [
        c for c in cell_list_all[:nc_demo] if c not in nuclear_boundary.keys()
    ]

    if missing_cells_mask or missing_cells_nuclear:
        print(f"Warning: Found {len(missing_cells_mask)} cells missing in mask_df")
        print(
            f"Warning: Found {len(missing_cells_nuclear)} cells missing in nuclear_boundary"
        )

        # Filter out cells with missing inputs.
        valid_cells = [
            c
            for c in cell_list_all[:nc_demo]
            if c in cell_mask_df["cell"].unique() and c in nuclear_boundary.keys()
        ]
        print(f"Proceeding with {len(valid_cells)} valid cells (originally {nc_demo})")
        cell_list_for_processing = valid_cells
    else:
        cell_list_for_processing = cell_list_all[:nc_demo]

    # Group input table and create processing chunks.
    df_gbC = data_df.groupby("cell", observed=False)
    chunks = list(chunk_list(cell_list_for_processing, chunk_size))

    # Create multiprocessing pool.
    pool = Pool(processes=min(4, cpu_count() - 2))
    process_chunk_partial = partial(
        process_chunk,
        df_gbC=df_gbC,
        cell_mask_df=cell_mask_df,
        nuclear_boundary=nuclear_boundary,
        ntanbin_dict=ntanbin_dict,
        epsilon=epsilon,
        clip_to_cell=clip_to_cell,
        remove_outliers=remove_outliers,
        verbose=verbose,
    )

    # Parallel processing.
    results = list(
        tqdm(
            pool.imap(process_chunk_partial, chunks),
            total=len(chunks),
            desc="Processing chunks in parallel",
        )
    )

    pool.close()
    pool.join()

    # Aggregate results.
    all_cell_dfs = []
    all_nuclear_dfs = []
    all_radii = {}
    all_nuclear_stats = []

    for result_chunk in results:
        for df_c_registered, nuclear_boundary_c_registered, cell_radius in result_chunk:
            all_cell_dfs.append(df_c_registered)
            all_nuclear_dfs.append(nuclear_boundary_c_registered)
            all_radii.update({df_c_registered["cell"].iloc[0]: cell_radius})
            # Per-cell nuclear boundary stats
            exceed_percent = 0.0
            exceed_count = 0
            num_points = int(len(nuclear_boundary_c_registered))
            if (
                num_points > 0
                and "exceeds_boundary" in nuclear_boundary_c_registered.columns
            ):
                exceed_series = nuclear_boundary_c_registered["exceeds_boundary"]
                exceed_percent = float(exceed_series.mean()) * 100.0
                exceed_count = int(exceed_series.sum())

            all_nuclear_stats.append(
                {
                    "cell": df_c_registered["cell"].iloc[0],
                    "exceed_percent": exceed_percent,
                    "exceed_count": exceed_count,
                    "num_nuclear_points": num_points,
                }
            )

    cell_df_registered = pd.concat(all_cell_dfs)
    nuclear_boundary_df_registered = pd.concat(all_nuclear_dfs)
    cell_nuclear_stats = pd.DataFrame(all_nuclear_stats)

    # Print summary stats.
    if verbose:
        cells_with_exceeding_nucleus = cell_nuclear_stats[
            cell_nuclear_stats["exceed_percent"] > 0
        ]
        if not cells_with_exceeding_nucleus.empty:
            mean_exceed = cells_with_exceeding_nucleus["exceed_percent"].mean()
            max_exceed = cells_with_exceeding_nucleus["exceed_percent"].max()
            print(
                f"\nFound {len(cells_with_exceeding_nucleus)} cells with nucleus exceeding cell boundary"
            )
            print(f"Average exceed percentage: {mean_exceed:.2f}%")
            print(f"Maximum exceed percentage: {max_exceed:.2f}%")
            print(f"After {'clipping' if clip_to_cell else 'leaving'} exceed points")

    return (
        cell_df_registered,
        nuclear_boundary_df_registered,
        all_radii,
        cell_nuclear_stats,
    )


def normalize_dataset(
    dataset,
    dismax_c_mat,
    delta_tanbin,
    ntanbin_dict,
    t,
    epsilon=1e-10,
    is_nucleus=False,
    clip_to_cell=True,
    remove_outliers=False,
):
    """Normalize points by the cell boundary (angle-binned max radius).

    Given per-angle-bin maximum radii (dismax_c_mat), compute normalized radius
    (d_c_s) and normalized coordinates (x_c_s, y_c_s).
    """
    dataset_normalized = dataset.assign(
        d_c_maxc=np.zeros(len(dataset)),
        d_c_s=np.zeros(len(dataset)),
        x_c_s=np.zeros(len(dataset)),
        y_c_s=np.zeros(len(dataset)),
    )

    # Track points exceeding the cell boundary.
    if is_nucleus:
        dataset_normalized["exceeds_boundary"] = False

    for q in range(4):
        dataset_q = dataset[
            (dataset.x_c >= 0) & (dataset.y_c >= 0)
            if q == 0
            else (dataset.x_c <= 0) & (dataset.y_c >= 0)
            if q == 1
            else (dataset.x_c <= 0) & (dataset.y_c <= 0)
            if q == 2
            else (dataset.x_c >= 0) & (dataset.y_c <= 0)
        ].copy()

        if len(dataset_q) > 0:
            dataset_q["arctan_idx"] = (dataset_q["arctan"] / delta_tanbin).astype(int)

            # Ensure arctan_idx stays in range.
            dataset_q["arctan_idx"] = np.minimum(
                dataset_q["arctan_idx"], ntanbin_dict[t] - 1
            )

            for ai in range(ntanbin_dict[t]):
                max_d = dismax_c_mat[ai, q]
                indices = dataset_q.index[dataset_q["arctan_idx"] == ai]
                dataset_normalized.loc[indices, "d_c_maxc"] = max_d

    # Normalized radial distance.
    dataset_normalized["d_c_s"] = dataset["d_c"] / (
        dataset_normalized["d_c_maxc"] + epsilon
    )

    # If nucleus points, mark those exceeding the boundary.
    if is_nucleus:
        dataset_normalized["exceeds_boundary"] = dataset_normalized["d_c_s"] > 1

    # Optionally remove out-of-bound nucleus points.
    if remove_outliers and is_nucleus:
        dataset_normalized = dataset_normalized[~dataset_normalized["exceeds_boundary"]]

    # Optionally clip to the cell boundary.
    if clip_to_cell:
        dataset_normalized["d_c_s"] = np.minimum(dataset_normalized["d_c_s"], 1.0)

    # Normalized coordinates.
    dataset_normalized["x_c_s"] = (
        dataset["x_c"] * dataset_normalized["d_c_s"] / (dataset["d_c"] + epsilon)
    )
    dataset_normalized["y_c_s"] = (
        dataset["y_c"] * dataset_normalized["d_c_s"] / (dataset["d_c"] + epsilon)
    )

    return dataset_normalized
