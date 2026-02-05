# Releasing `grasp-tool`

This document describes the maintainer workflow for building and publishing `grasp-tool` to PyPI.

Key design choice:

- The PyPI package intentionally does NOT declare `torch` / `torch-geometric` as dependencies.
  Users should install the training stack via conda (recommended) following official instructions.

## Prerequisites

- Python 3.9
- Poetry (maintainers)
- Twine (for `twine check` and uploads)

Recommended (one-time):

```bash
python -m pip install --upgrade pip
python -m pip install --upgrade twine
```

## 0) Preflight checklist

- Confirm package metadata:
  - `pyproject.toml` name is `grasp-tool`
  - version is bumped (see below)
- Confirm local CLI works:

```bash
python -m grasp_tool --help
python -m grasp_tool train-moco --help
```

## 1) Bump version

Use SemVer (recommended). Examples:

```bash
poetry version patch
# or
poetry version minor
```

Note: `grasp-tool --version` reads the installed distribution version via `importlib.metadata`.

## 2) Build artifacts

```bash
rm -rf dist
poetry build
```

If your environment cannot fetch build requirements from PyPI (for example, SSL/network restrictions),
you can build without isolation as long as `poetry-core` is already installed:

```bash
rm -rf dist
mkdir -p dist
python -m build --no-isolation --outdir dist
```

Validate metadata and long description rendering:

```bash
python -m twine check dist/*
```

Optional: enforce "no CJK" gate for release artifacts (recommended for this repo):

```bash
python scripts/check_no_cjk.py
python scripts/check_no_cjk.py --scan-dist
```

## 3) Release smoke test (fresh environment)

We recommend testing in a clean conda env:

```bash
conda create -n grasp-release-check python=3.9 -y
conda activate grasp-release-check
```

Install the wheel you just built:

```bash
python -m pip install dist/*.whl
```

Acceptance checks:

```bash
grasp-tool --help
grasp-tool train-moco --help

# Demo (requires repo checkout / example data)
grasp-tool register \
  --pkl_file example_pkl/simulated1_data_dict.pkl \
  --output_pkl outputs/release_smoke_registered.pkl
```

Optional check (expected): running training without torch/pyg should show a clear error:

```bash
grasp-tool train-moco --dataset demo --pkl /tmp/not_exists.pkl
```

## 4) Publish to TestPyPI (recommended)

Use an API token (do NOT commit tokens to the repo):

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="pypi-..."
```

If you prefer a single command, use:

```bash
./scripts/publish_testpypi.sh
```

Upload:

```bash
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Then verify installing from TestPyPI:

```bash
python -m pip uninstall -y grasp-tool
python -m pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple grasp-tool==<VERSION>
grasp-tool --help
```

Or use the helper script:

```bash
./scripts/verify_testpypi.sh <VERSION>
```

## 5) Publish to PyPI

```bash
python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
```

## Notes

- Large assets and third-party reference code are excluded from sdists via `pyproject.toml` (`ELLA/`, `example_pkl/`, etc.).
- Training-related commands (`build-train-pkl`, `train-moco`) require `torch` + `torch-geometric` installed separately.
