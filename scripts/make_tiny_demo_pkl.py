#!/usr/bin/env python

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _parse_int_list(value: str) -> List[int]:
    items: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _parse_str_list(value: str) -> List[str]:
    items: List[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(part)
    return items


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Create a small (tiny) df_registered PKL + pairs.csv for smoke testing. "
            "Input PKL must contain a 'df_registered' DataFrame."
        )
    )
    p.add_argument(
        "--input_pkl",
        required=True,
        help="Input PKL path (must contain df_registered)",
    )
    p.add_argument(
        "--output_pkl",
        required=True,
        help="Output PKL path (will write tiny df_registered)",
    )
    p.add_argument(
        "--pairs_csv",
        required=True,
        help="Output pairs CSV path (columns: cell,gene)",
    )
    p.add_argument(
        "--max_cells",
        type=int,
        default=4,
        help="Number of cells to keep (default: 4)",
    )
    p.add_argument(
        "--max_genes",
        type=int,
        default=8,
        help="Number of genes to keep (default: 8)",
    )
    p.add_argument(
        "--min_transcripts_per_pair",
        type=int,
        default=2,
        help="Minimum transcripts for each (cell, gene) pair to keep a gene (default: 2)",
    )
    p.add_argument(
        "--cells",
        type=str,
        default=None,
        help="Optional comma-separated cell ids to use (overrides --max_cells)",
    )
    p.add_argument(
        "--genes",
        type=str,
        default=None,
        help="Optional comma-separated gene names to use (overrides --max_genes)",
    )
    return p


def _select_cells(df, *, max_cells: int, cells_arg: str | None) -> List[str]:
    cells_all = sorted(df["cell"].unique().tolist())
    if cells_arg:
        requested = _parse_str_list(cells_arg)
        missing = [c for c in requested if c not in set(cells_all)]
        if missing:
            raise ValueError(f"Requested cells not found in df_registered: {missing}")
        return requested
    return cells_all[:max_cells]


def _select_genes(
    df,
    *,
    cells: List[str],
    max_genes: int,
    min_transcripts_per_pair: int,
    genes_arg: str | None,
) -> List[str]:
    df_sub = df[df["cell"].isin(cells)]
    genes_all = sorted(df_sub["gene"].unique().tolist())
    if genes_arg:
        requested = _parse_str_list(genes_arg)
        missing = [g for g in requested if g not in set(genes_all)]
        if missing:
            raise ValueError(f"Requested genes not found in selected cells: {missing}")
        return requested

    # Choose genes that have enough transcripts for ALL selected cells.
    counts = df_sub.groupby(["cell", "gene"]).size().unstack(fill_value=0)
    gene_min = counts.min(axis=0)
    eligible = gene_min[gene_min >= min_transcripts_per_pair]
    if eligible.empty:
        raise ValueError(
            "No genes satisfy min_transcripts_per_pair across the selected cells. "
            "Try lowering --min_transcripts_per_pair or selecting different cells."
        )

    gene_total = counts.sum(axis=0).loc[eligible.index]
    top_genes = gene_total.sort_values(ascending=False).head(max_genes).index.tolist()
    return [str(g) for g in top_genes]


def _filter_optional_keys(payload: Dict, *, keep_cells: List[str]):
    # Keep the full payload by default, but shrink a few commonly-used fields.
    out = dict(payload)

    nuclear_df = out.get("nuclear_boundary_df_registered")
    if nuclear_df is not None:
        try:
            # pandas.DataFrame-like
            if hasattr(nuclear_df, "columns") and "cell" in nuclear_df.columns:
                out["nuclear_boundary_df_registered"] = nuclear_df[
                    nuclear_df["cell"].isin(keep_cells)
                ]
        except Exception:
            pass

    radii = out.get("cell_radii")
    if isinstance(radii, dict):
        out["cell_radii"] = {c: radii[c] for c in keep_cells if c in radii}

    return out


def main() -> int:
    args = build_parser().parse_args()

    import pandas as pd

    input_pkl = Path(args.input_pkl)
    with input_pkl.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise TypeError("Input PKL must contain a dict")
    if "df_registered" not in payload:
        raise ValueError("Input PKL must contain key 'df_registered'")

    df = payload["df_registered"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("payload['df_registered'] must be a pandas.DataFrame")
    for col in ["cell", "gene", "x_c_s", "y_c_s"]:
        if col not in df.columns:
            raise ValueError(f"df_registered missing required column: {col}")

    keep_cells = _select_cells(df, max_cells=args.max_cells, cells_arg=args.cells)
    keep_genes = _select_genes(
        df,
        cells=keep_cells,
        max_genes=args.max_genes,
        min_transcripts_per_pair=args.min_transcripts_per_pair,
        genes_arg=args.genes,
    )

    df_tiny = df[df["cell"].isin(keep_cells) & df["gene"].isin(keep_genes)].copy()
    if df_tiny.empty:
        raise ValueError("Tiny df_registered is empty after filtering")

    out_payload = _filter_optional_keys(payload, keep_cells=keep_cells)
    out_payload["df_registered"] = df_tiny
    out_payload.setdefault("meta", {})
    if isinstance(out_payload["meta"], dict):
        out_payload["meta"].update(
            {
                "tiny_demo": {
                    "input_pkl": str(input_pkl),
                    "max_cells": int(args.max_cells),
                    "max_genes": int(args.max_genes),
                    "min_transcripts_per_pair": int(args.min_transcripts_per_pair),
                    "selected_cells": keep_cells,
                    "selected_genes": keep_genes,
                    "n_transcripts": int(len(df_tiny)),
                    "n_cells": int(df_tiny["cell"].nunique()),
                    "n_genes": int(df_tiny["gene"].nunique()),
                }
            }
        )

    output_pkl = Path(args.output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pickle.dump(out_payload, f)

    pairs_csv = Path(args.pairs_csv)
    pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs = df_tiny[["cell", "gene"]].drop_duplicates().sort_values(["cell", "gene"])
    pairs.to_csv(pairs_csv, index=False)

    print("Wrote tiny PKL:", output_pkl)
    print("Wrote pairs CSV:", pairs_csv, "rows", len(pairs))
    print("Tiny df_registered shape:", df_tiny.shape)
    print("Cells:", keep_cells)
    print("Genes:", keep_genes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
