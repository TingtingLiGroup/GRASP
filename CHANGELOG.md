# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added

- `grasp-tool cellplot` CLI command for quick visualization.
- Repo-only tiny demo helpers under `scripts/` and `demo_pkl/`.

## [0.1.1] - 2026-02-05

### Changed

- Update README so the PyPI long description matches the GitHub repo.
- Exclude `demo_pkl/` from source distributions (repo-only demo asset).

## [0.1.0] - 2026-02-05

### Added

- Initial PyPI release: `grasp-tool==0.1.0`.
- Packaged CLI entrypoints for the main pipeline (register/portrait/partition/augment/build-train-pkl/train-moco).
