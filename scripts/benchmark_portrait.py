#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from grasp_tool.preprocessing.portrait import calculate_js_distances


def generate_sparse_registered_frame(pair_count: int, seed: int) -> pd.DataFrame:
    combination_count = math.ceil(pair_count / 0.03)
    cell_count = math.ceil(math.sqrt(combination_count))
    gene_count = math.ceil(combination_count / cell_count)
    cells = [f"cell_{index:04d}" for index in range(cell_count)]
    genes = [f"gene_{index:04d}" for index in range(gene_count)]

    rng = np.random.default_rng(seed)
    all_pairs = [(cell, gene) for gene in genes for cell in cells]
    selected_indices = rng.permutation(len(all_pairs))[:pair_count]

    rows = []
    for pair_index, selected_index in enumerate(selected_indices):
        cell, gene = all_pairs[int(selected_index)]
        transcript_count = 2 + pair_index % 4
        angles = np.linspace(0.0, 2.0 * np.pi, transcript_count, endpoint=False)
        radius = 0.05 + 0.01 * (pair_index % 7)
        rows.extend(
            {
                "cell": cell,
                "gene": gene,
                "x_c_s": radius * math.cos(angle),
                "y_c_s": radius * math.sin(angle),
            }
            for angle in angles
        )

    frame = pd.DataFrame(rows)
    frame["cell"] = pd.Categorical(frame["cell"], categories=cells)
    frame["gene"] = pd.Categorical(frame["gene"], categories=genes)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sparse portrait computation")
    parser.add_argument("--pair_count", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    frame = generate_sparse_registered_frame(args.pair_count, args.seed)
    input_path = args.output_dir / "registered.pkl"
    with input_path.open("wb") as handle:
        pickle.dump({"df_registered": frame}, handle)

    portrait_dir = args.output_dir / "portrait"
    start = time.perf_counter()
    distances = calculate_js_distances(
        pkl_file=str(input_path),
        output_dir=str(portrait_dir),
        max_count=10,
        transcript_window=30,
        bin_size=0.01,
        threshold=0.05,
        r_min=0.01,
        r_max=0.6,
        r_step=0.03,
        num_threads=args.num_threads,
        use_same_r=False,
        visualize_top_n=0,
        use_vectorized=True,
    )
    wall_seconds = time.perf_counter() - start

    summary = {
        "pair_count": args.pair_count,
        "cell_categories": int(len(frame["cell"].cat.categories)),
        "gene_categories": int(len(frame["gene"].cat.categories)),
        "transcript_count": int(len(frame)),
        "js_row_count": int(len(distances)),
        "num_threads": args.num_threads,
        "wall_seconds": wall_seconds,
        "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    with (args.output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
