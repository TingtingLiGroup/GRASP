from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grasp-tool")
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    subparsers = parser.add_subparsers(dest="command")

    p_register = subparsers.add_parser(
        "register",
        help="Coordinate normalization (wraps grasp_tool.preprocessing.register)",
    )
    p_register.add_argument(
        "--pkl_file",
        required=True,
        help="Input raw data PKL (expects data_df/cell_mask_df/nuclear_boundary)",
    )
    p_register.add_argument(
        "--output_pkl",
        required=True,
        help="Output PKL path (will write df_registered etc)",
    )
    p_register.add_argument(
        "--nc_demo",
        type=int,
        default=None,
        help="Number of cells to process (default: all)",
    )
    p_register.add_argument(
        "--chunk_size",
        type=int,
        default=2,
        help="Multiprocessing chunk size (default: 2)",
    )
    p_register.add_argument(
        "--clip_to_cell",
        type=int,
        default=1,
        choices=[0, 1],
        help="Clip nucleus points to cell boundary (default: 1)",
    )
    p_register.add_argument(
        "--remove_outliers",
        type=int,
        default=0,
        choices=[0, 1],
        help="Remove nucleus points exceeding boundary (default: 0)",
    )
    p_register.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1],
        help="Verbose logging (default: 0)",
    )
    p_register.add_argument(
        "--epsilon",
        type=float,
        default=1e-10,
        help="Small constant to avoid division by zero (default: 1e-10)",
    )

    p_cellplot = subparsers.add_parser(
        "cellplot",
        help="Plot raw/registered transcript distributions (wraps grasp_tool.preprocessing.cellplot)",
    )
    p_cellplot.add_argument(
        "--pkl",
        "--pkl_file",
        dest="pkl",
        required=True,
        help="Input PKL path (raw or registered dict; required keys depend on --mode)",
    )
    p_cellplot.add_argument(
        "--output_dir",
        required=True,
        help="Output directory root for plots",
    )
    p_cellplot.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset tag used in output paths (default: inferred from PKL filename)",
    )
    p_cellplot.add_argument(
        "--mode",
        required=True,
        choices=["raw-cell", "registered-gene"],
        help="Plot mode",
    )
    p_cellplot.add_argument(
        "--cells",
        type=str,
        default=None,
        help="Comma-separated cell ids; default=all",
    )
    p_cellplot.add_argument(
        "--genes",
        type=str,
        default=None,
        help="Comma-separated gene names; default=all (registered-gene only)",
    )
    p_cellplot.add_argument(
        "--with_nuclear",
        type=int,
        default=1,
        choices=[0, 1],
        help="Plot nuclear boundary when available (registered-gene only; default: 1)",
    )

    # Pass-through wrappers for existing script-style CLIs
    # NOTE: add_help=False so that `grasp-tool portrait --help` is forwarded to the
    # underlying script-style CLI (instead of being consumed by argparse here).
    p_portrait = subparsers.add_parser(
        "portrait",
        add_help=False,
        help="Compute JS distances (wraps grasp_tool.preprocessing.portrait)",
    )
    p_portrait.add_argument("argv", nargs=argparse.REMAINDER)

    p_train = subparsers.add_parser(
        "train-moco",
        add_help=False,
        help="Run MoCo training (wraps grasp_tool.cli.train_moco)",
    )
    p_train.add_argument("argv", nargs=argparse.REMAINDER)

    # Drivers missing in the original repo: build graphs / augment / build train pkl
    p_partition = subparsers.add_parser(
        "partition-graphs",
        help="Build per-cell/per-gene node/adj CSVs from a df_registered PKL",
    )
    p_partition.add_argument(
        "--pkl", required=True, help="Input PKL containing df_registered"
    )
    p_partition.add_argument(
        "--graph_root",
        required=True,
        help="Output directory (will create <cell>/ and write <gene>_*.csv)",
    )
    p_partition.add_argument("--n_sectors", type=int, default=20)
    p_partition.add_argument("--m_rings", type=int, default=10)
    p_partition.add_argument("--k_neighbor", type=int, default=5)
    p_partition.add_argument("--epsilon", type=float, default=0.1)
    p_partition.add_argument(
        "--cells",
        type=str,
        default=None,
        help="Comma-separated cell ids; default=all",
    )
    p_partition.add_argument(
        "--genes",
        type=str,
        default=None,
        help="Comma-separated gene names; default=all",
    )

    p_aug = subparsers.add_parser(
        "augment-graphs",
        help="Create <cell>_aug/ by rotating + node dropout",
    )
    p_aug.add_argument("--graph_root", required=True)
    p_aug.add_argument("--dropout_ratio", type=float, default=0.1)
    p_aug.add_argument("--angle_min", type=float, default=0.0)
    p_aug.add_argument("--angle_max", type=float, default=360.0)
    p_aug.add_argument("--seed", type=int, default=2025)

    p_pkl = subparsers.add_parser(
        "build-train-pkl",
        help="Load graphs from graph_root and write a training PKL for train-moco",
    )
    p_pkl.add_argument("--pairs_csv", required=True, help="CSV with columns: cell,gene")
    p_pkl.add_argument("--graph_root", required=True)
    p_pkl.add_argument("--output_pkl", required=True)
    p_pkl.add_argument("--dataset", type=str, default="dataset")
    p_pkl.add_argument("--n_sectors", type=int, default=20)
    p_pkl.add_argument("--m_rings", type=int, default=10)
    p_pkl.add_argument("--k_neighbor", type=int, default=5)
    p_pkl.add_argument("--processes", type=int, default=8)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    argv_list = list(argv)

    # Pass-through commands: forward raw args to the underlying script-style CLI.
    if argv_list[:1] == ["portrait"]:
        return _run_portrait(argv_list[1:])
    if argv_list[:1] == ["train-moco"]:
        return _run_train_moco(argv_list[1:])

    parser = build_parser()
    args = parser.parse_args(argv_list)

    if args.version:
        from grasp_tool import __version__

        print(__version__)
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "register":
        return _run_register(
            pkl_file=args.pkl_file,
            output_pkl=args.output_pkl,
            nc_demo=args.nc_demo,
            chunk_size=args.chunk_size,
            clip_to_cell=bool(args.clip_to_cell),
            remove_outliers=bool(args.remove_outliers),
            verbose=bool(args.verbose),
            epsilon=args.epsilon,
        )

    if args.command == "cellplot":
        return _run_cellplot(
            pkl_path=args.pkl,
            output_dir=args.output_dir,
            dataset=args.dataset,
            mode=args.mode,
            cells_arg=args.cells,
            genes_arg=args.genes,
            with_nuclear=bool(args.with_nuclear),
        )

    if args.command == "partition-graphs":
        return _run_partition_graphs(
            pkl_path=args.pkl,
            graph_root=args.graph_root,
            n_sectors=args.n_sectors,
            m_rings=args.m_rings,
            k_neighbor=args.k_neighbor,
            epsilon=args.epsilon,
            cells_arg=args.cells,
            genes_arg=args.genes,
        )

    if args.command == "augment-graphs":
        return _run_augment_graphs(
            graph_root=args.graph_root,
            dropout_ratio=args.dropout_ratio,
            angle_min=args.angle_min,
            angle_max=args.angle_max,
            seed=args.seed,
        )

    if args.command == "build-train-pkl":
        return _run_build_train_pkl(
            dataset=args.dataset,
            pairs_csv=args.pairs_csv,
            graph_root=args.graph_root,
            output_pkl=args.output_pkl,
            n_sectors=args.n_sectors,
            m_rings=args.m_rings,
            k_neighbor=args.k_neighbor,
            processes=args.processes,
        )

    parser.print_help()
    return 2


def _run_portrait(argv_tail: Sequence[str]) -> int:
    import runpy

    sys_argv_prev = sys.argv[:]
    try:
        sys.argv = ["grasp-tool portrait"] + list(argv_tail)
        runpy.run_module("grasp_tool.preprocessing.portrait", run_name="__main__")
        return 0
    finally:
        sys.argv = sys_argv_prev


def _run_train_moco(argv_tail: Sequence[str]) -> int:
    # Import only when needed to keep the CLI lightweight.
    sys_argv_prev = sys.argv[:]
    try:
        sys.argv = ["grasp-tool train-moco"] + list(argv_tail)
        from grasp_tool.cli import train_moco

        try:
            rc = train_moco.main()
        except ModuleNotFoundError as e:
            # A last-resort guardrail. Most missing-deps errors are handled inside
            # grasp_tool.cli.train_moco.
            print(str(e))
            return 1

        return int(rc) if isinstance(rc, int) else 0
    finally:
        sys.argv = sys_argv_prev


def _load_registered_pkl(pkl_path: str):
    import pickle

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "df_registered" not in data:
        raise ValueError(
            f"PKL {pkl_path} must be a dict containing key 'df_registered'"
        )
    return data


def _load_raw_pkl(pkl_path: str) -> dict:
    import pickle

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"PKL {pkl_path} must contain a dict")
    return data


def _infer_dataset_name_from_pkl_path(pkl_path: str) -> str:
    name = Path(pkl_path).name
    if "_data_dict" in name:
        return name.split("_data_dict")[0]
    return Path(name).stem


def _run_register(
    *,
    pkl_file: str,
    output_pkl: str,
    nc_demo: Optional[int],
    chunk_size: int,
    clip_to_cell: bool,
    remove_outliers: bool,
    verbose: bool,
    epsilon: float,
) -> int:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        print(
            "Missing dependency: pandas. Install package dependencies first "
            "(e.g., pip install grasp-tool) or use the recommended conda env."
        )
        return 1

    from grasp_tool.preprocessing.register import (
        register_cells_and_nuclei_parallel_chunked_constrained,
        specify_ntanbin,
    )

    raw = _load_raw_pkl(pkl_file)

    data_df = raw.get("data_df")
    cell_mask_df = raw.get("cell_mask_df")
    nuclear_boundary = raw.get("nuclear_boundary")

    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("raw pkl must contain key 'data_df' as a pandas.DataFrame")
    if not isinstance(cell_mask_df, pd.DataFrame):
        raise ValueError(
            "raw pkl must contain key 'cell_mask_df' as a pandas.DataFrame"
        )
    if not isinstance(nuclear_boundary, dict):
        raise ValueError("raw pkl must contain key 'nuclear_boundary' as a dict")

    cell_list_all = raw.get("cell_list_all")
    if not isinstance(cell_list_all, list):
        if "cell" not in data_df.columns:
            raise ValueError("data_df must contain column 'cell'")
        cell_list_all = sorted(data_df["cell"].unique().tolist())

    ntanbin_dict = raw.get("ntanbin_dict")
    if not isinstance(ntanbin_dict, dict):
        cell_list_dict = raw.get("cell_list_dict")
        type_list = raw.get("type_list")

        if isinstance(cell_list_dict, dict) and isinstance(type_list, list):
            ntanbin_dict = specify_ntanbin(
                cell_list_dict=cell_list_dict,
                cell_mask_df=cell_mask_df,
                type_list=type_list,
            )
        else:
            # Fallback: infer from data_df
            if "type" not in data_df.columns:
                raise ValueError(
                    "ntanbin_dict missing and cannot infer: provide 'cell_list_dict'/'type_list' or ensure data_df has column 'type'"
                )
            inferred = {}
            for t, df_t in data_df.groupby("type", observed=False):
                inferred[str(t)] = sorted(df_t["cell"].unique().tolist())
            inferred_types = sorted(inferred.keys())
            ntanbin_dict = specify_ntanbin(
                cell_list_dict=inferred,
                cell_mask_df=cell_mask_df,
                type_list=inferred_types,
            )

    (
        cell_df_registered,
        nuclear_boundary_df_registered,
        all_radii,
        cell_nuclear_stats,
    ) = register_cells_and_nuclei_parallel_chunked_constrained(
        data_df=data_df,
        cell_list_all=cell_list_all,
        cell_mask_df=cell_mask_df,
        nuclear_boundary=nuclear_boundary,
        ntanbin_dict=ntanbin_dict,
        epsilon=epsilon,
        nc_demo=nc_demo,
        chunk_size=chunk_size,
        clip_to_cell=clip_to_cell,
        remove_outliers=remove_outliers,
        verbose=verbose,
    )

    payload = {
        "df_registered": cell_df_registered,
        "nuclear_boundary_df_registered": nuclear_boundary_df_registered,
        "cell_radii": all_radii,
        "cell_nuclear_stats": cell_nuclear_stats,
        "meta": {
            "input_pkl_file": pkl_file,
            "nc_demo": nc_demo,
            "chunk_size": chunk_size,
            "clip_to_cell": clip_to_cell,
            "remove_outliers": remove_outliers,
            "epsilon": epsilon,
        },
    }

    out_path = Path(output_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import pickle

    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Wrote registered PKL: {out_path}")
    return 0


def _run_cellplot(
    *,
    pkl_path: str,
    output_dir: str,
    dataset: Optional[str],
    mode: str,
    cells_arg: Optional[str],
    genes_arg: Optional[str],
    with_nuclear: bool,
) -> int:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        print(
            "Missing dependency: pandas. Install package dependencies first "
            "(e.g., pip install grasp-tool) or use the recommended conda env."
        )
        return 1

    from grasp_tool.preprocessing import cellplot

    data = _load_raw_pkl(pkl_path)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset or _infer_dataset_name_from_pkl_path(pkl_path)
    requested_cells = _parse_comma_list(cells_arg)
    requested_genes = _parse_comma_list(genes_arg)

    print(f"cellplot mode={mode} dataset={dataset_name}")
    print(f"Writing plots under: {output_root}")

    if mode == "raw-cell":
        cell_boundary = data.get("cell_boundary")
        if not isinstance(cell_boundary, dict):
            raise ValueError("PKL must contain key 'cell_boundary' as a dict")

        nuclear_boundary = data.get("nuclear_boundary")
        if nuclear_boundary is None:
            nuclear_boundary = {}
        if not isinstance(nuclear_boundary, dict):
            nuclear_boundary = {}

        if requested_cells:
            missing = [c for c in requested_cells if c not in cell_boundary]
            if missing:
                raise ValueError(
                    f"Requested cells not found in cell_boundary: {missing}"
                )

            cell_boundary = {c: cell_boundary[c] for c in requested_cells}
            nuclear_boundary = {
                c: nuclear_boundary[c] for c in requested_cells if c in nuclear_boundary
            }

        cellplot.plot_raw_cell(
            dataset=dataset_name,
            cell_boundary=cell_boundary,
            nuclear_boundary=nuclear_boundary,
            path=str(output_root),
        )
        return 0

    if mode == "registered-gene":
        df_registered = data.get("df_registered")
        if not isinstance(df_registered, pd.DataFrame):
            raise ValueError(
                "PKL must contain key 'df_registered' as a pandas.DataFrame"
            )

        df = df_registered
        if requested_cells:
            df = df[df["cell"].isin(requested_cells)]
        if requested_genes:
            df = df[df["gene"].isin(requested_genes)]
        if len(df) == 0:
            raise ValueError("No transcript records remain after filtering")

        nuclear_df = data.get("nuclear_boundary_df_registered")
        if with_nuclear and isinstance(nuclear_df, pd.DataFrame):
            if "cell" in nuclear_df.columns:
                nuclear_df = nuclear_df[nuclear_df["cell"].isin(df["cell"].unique())]
            if getattr(nuclear_df, "empty", True):
                nuclear_df = None
        else:
            nuclear_df = None

        if nuclear_df is not None:
            cellplot.plot_register_gene_distribution(
                dataset=dataset_name,
                df_registered=df,
                path=str(output_root),
                nuclear_boundary_df_registered=nuclear_df,
            )
            return 0

        cell_radii = data.get("cell_radii") or data.get("all_radii") or {}
        cellplot.plot_register_gene_distribution_without_nuclear(
            dataset=dataset_name,
            df_registered=df,
            cell_radii=cell_radii,
            path=str(output_root),
        )
        return 0

    raise ValueError(f"Unknown cellplot mode: {mode}")


def _parse_comma_list(value: Optional[str]) -> Optional[list]:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",")]
    items = [x for x in items if x]
    return items or None


def _run_partition_graphs(
    *,
    pkl_path: str,
    graph_root: str,
    n_sectors: int,
    m_rings: int,
    k_neighbor: int,
    epsilon: float,
    cells_arg: Optional[str],
    genes_arg: Optional[str],
) -> int:
    import math

    import pandas as pd

    from grasp_tool.preprocessing.partition import (
        classify_center_points_with_edge,
        count_points_in_areas_same,
        save_node_data_to_csv,
    )

    data = _load_registered_pkl(pkl_path)
    df_registered = data["df_registered"]
    if not isinstance(df_registered, pd.DataFrame):
        raise TypeError("data['df_registered'] must be a pandas.DataFrame")

    nuclear_boundary_df_registered = data.get("nuclear_boundary_df_registered")
    if not isinstance(nuclear_boundary_df_registered, pd.DataFrame):
        nuclear_boundary_df_registered = None
    cell_radii = data.get("cell_radii") or data.get("all_radii")

    requested_cells = _parse_comma_list(cells_arg)
    requested_genes = _parse_comma_list(genes_arg)

    cells_all = sorted(df_registered["cell"].unique().tolist())
    genes_all = sorted(df_registered["gene"].unique().tolist())

    cells = requested_cells or cells_all
    genes = requested_genes or genes_all

    root = Path(graph_root)
    root.mkdir(parents=True, exist_ok=True)

    required_cols = {"cell", "gene", "x_c_s", "y_c_s"}
    missing = required_cols - set(df_registered.columns)
    if missing:
        raise ValueError(f"df_registered missing required columns: {sorted(missing)}")

    for cell in cells:
        cell_dir = root / str(cell)
        cell_dir.mkdir(parents=True, exist_ok=True)

        df_cell = df_registered[df_registered["cell"] == cell]
        if len(df_cell) == 0:
            continue

        if isinstance(cell_radii, dict) and cell in cell_radii:
            r = float(cell_radii[cell])
        else:
            r_sq_max = float(((df_cell["x_c_s"] ** 2) + (df_cell["y_c_s"] ** 2)).max())
            r = math.sqrt(max(r_sq_max, 0.0))
            if r <= 0:
                r = 1.0

        nuclear_df_cell = None
        if (
            nuclear_boundary_df_registered is not None
            and "cell" in nuclear_boundary_df_registered.columns
        ):
            nuclear_df_cell = nuclear_boundary_df_registered[
                nuclear_boundary_df_registered["cell"] == cell
            ]

        for gene in genes:
            df_cell_gene = df_cell[df_cell["gene"] == gene]
            if len(df_cell_gene) == 0:
                continue

            _count_matrix, center_points, point_counts, is_virtual, is_edge = (
                count_points_in_areas_same(df_cell_gene.copy(), n_sectors, m_rings, r)
            )

            if nuclear_df_cell is not None and len(nuclear_df_cell) != 0:
                nuclear_positions = classify_center_points_with_edge(
                    center_points,
                    nuclear_df_cell,
                    is_edge,
                    epsilon=epsilon,
                )
            else:
                nuclear_positions = ["unknown"] * len(center_points)

            save_node_data_to_csv(
                center_points=center_points,
                is_virtual=is_virtual,
                is_edge=is_edge,
                plot_dir=str(cell_dir),
                gene=str(gene),
                node_counts=point_counts,
                k=k_neighbor,
                nuclear_positions=nuclear_positions,
            )

    return 0


def _run_augment_graphs(
    *,
    graph_root: str,
    dropout_ratio: float,
    angle_min: float,
    angle_max: float,
    seed: int,
) -> int:
    import numpy as np
    import pandas as pd

    from grasp_tool.preprocessing.augumentation import dropout_nodes, rotate_nodes

    if dropout_ratio < 0 or dropout_ratio >= 1:
        raise ValueError("--dropout_ratio must be in [0, 1)")
    if angle_max < angle_min:
        raise ValueError("--angle_max must be >= --angle_min")

    rng = np.random.default_rng(seed)

    root = Path(graph_root)
    if not root.exists():
        raise FileNotFoundError(f"graph_root not found: {graph_root}")

    for cell_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if cell_dir.name.endswith("_aug"):
            continue

        aug_dir = root / f"{cell_dir.name}_aug"
        aug_dir.mkdir(parents=True, exist_ok=True)

        for node_path in sorted(cell_dir.glob("*_node_matrix.csv")):
            gene = node_path.name[: -len("_node_matrix.csv")]
            adj_path = cell_dir / f"{gene}_adj_matrix.csv"
            if not adj_path.exists():
                continue

            node_df = pd.read_csv(node_path)
            adj_df = pd.read_csv(adj_path)

            angle = float(rng.uniform(angle_min, angle_max))
            node_df = rotate_nodes(node_df, angle)
            adj_df, node_df = dropout_nodes(adj_df, node_df, dropout_ratio)

            node_df.to_csv(aug_dir / node_path.name, index=False)
            adj_df.to_csv(aug_dir / adj_path.name, index=False)

    return 0


def _run_build_train_pkl(
    *,
    dataset: str,
    pairs_csv: str,
    graph_root: str,
    output_pkl: str,
    n_sectors: int,
    m_rings: int,
    k_neighbor: int,
    processes: int,
) -> int:
    import pickle

    import pandas as pd

    try:
        from grasp_tool.gnn.graphloader import generate_graph_data_target_parallel
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", "")
        if missing == "torch" or missing.startswith("torch_geometric"):
            print(
                "build-train-pkl requires torch and torch-geometric.\n"
                "Install them first (recommended: conda), then re-run build-train-pkl."
            )
            return 1
        raise

    df_pairs = pd.read_csv(pairs_csv)
    if not {"cell", "gene"}.issubset(df_pairs.columns):
        raise ValueError("pairs_csv must contain columns: cell,gene")

    original_graphs, augmented_graphs = generate_graph_data_target_parallel(
        dataset=dataset,
        df=df_pairs,
        path=graph_root,
        n_sectors=n_sectors,
        m_rings=m_rings,
        k_neighbor=k_neighbor,
        processes=processes,
    )

    gene_labels = [g.gene for g in original_graphs]
    cell_labels = [g.cell for g in original_graphs]

    payload = {
        "original_graphs": original_graphs,
        "augmented_graphs": augmented_graphs,
        "gene_labels": gene_labels,
        "cell_labels": cell_labels,
        "meta": {
            "dataset": dataset,
            "graph_root": graph_root,
            "n_sectors": n_sectors,
            "m_rings": m_rings,
            "k_neighbor": k_neighbor,
            "processes": processes,
            "pairs_csv": pairs_csv,
        },
    }

    out_path = Path(output_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    return 0
