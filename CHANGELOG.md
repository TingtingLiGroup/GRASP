# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

## [0.2.0] - 2026-09-24

Performance release. See `docs/OPTIMIZATION.md` for measurements, equivalence checks, and the
recommended configuration.

### Added

- `grasp-tool cellplot` CLI command for quick visualization.
- Repo-only tiny demo helpers under `scripts/` and `demo_pkl/`.
- `train-moco` throughput switches (all off by default): `--cache_train_batches`,
  `--pipeline_cached_h2d`, `--cached_h2d_prefetch_batches`, `--foreach_momentum`,
  `--prefetch_batches`, `--pin_prefetched_batches`, and `--matmul_precision`.
- `train-moco` research switches that change training semantics: `--fuse_positive_encoders` and
  `--reconstruction_negative_ratio`.
- `train-moco` evaluation controls: `--eval_freq`, `--eval_at_start`, and `--visualize`.
- `partition-graphs --write_distance_matrix` to restore the legacy distance-matrix CSV output.
- Per-epoch `training_history_lr<lr>.csv` with loss, learning rate, and training time.
- `scripts/train_multi_lr.py` for running independent learning rates on multiple GPUs.
- Benchmark and comparison scripts under `scripts/` and equivalence tests under `tests/`.

### Changed

- Vectorized partition graph construction (about 11x faster on 1000 graphs).
- Portrait now indexes only observed `(cell, gene)` pairs and schedules work per gene.
- JS positive generation uses a hash index instead of repeated DataFrame scans.
- Embeddings are computed with batched inference.
- `train-moco` now evaluates embeddings only at the final epoch and disables t-SNE/UMAP plots by
  default; use `--eval_at_start 1 --eval_freq 20 --visualize 1` for the previous behavior.
- `partition-graphs` no longer writes the unused `*_dis_matrix.csv` files by default.

## [0.1.3] - 2026-03-10

### Changed

- Update paper title: "for Analyzing Subcellular Localization Patterns" -> "to Analyze Subcellular Patterns".

## [0.1.2] - 2026-02-05

### Changed

- Clarify training dependency installation in `README.md` (including a pip CUDA wheel example for CUDA 12.1).
- Fix tutorial link in `README.md` to point to Read the Docs.
- Require ground-truth labels at (cell, gene) granularity for clustering evaluation; document the expected label CSV format.
- Make `grasp-tool train-moco --label_file` actually drive clustering evaluation, and simplify label loading accordingly.

## [0.1.1] - 2026-02-05

### Changed

- Update README so the PyPI long description matches the GitHub repo.
- Exclude `demo_pkl/` from source distributions (repo-only demo asset).

## [0.1.0] - 2026-02-05

### Added

- Initial PyPI release: `grasp-tool==0.1.0`.
- Packaged CLI entrypoints for the main pipeline (register/portrait/partition/augment/build-train-pkl/train-moco).
