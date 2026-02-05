# Contributing

Thanks for taking the time to contribute.

## Quick start (local)

Recommended (conda + pip):

```bash
conda create -n grasp python=3.9 -y
conda activate grasp
pip install -U pip
pip install -e .
```

Smoke checks:

```bash
python -m grasp_tool --help
python -m grasp_tool train-moco --help
```

## Tiny demo (repo checkout)

This repository includes a repo-only end-to-end smoke test script:

```bash
bash scripts/tiny_demo_example_pkl.sh
```

Notes:

- Training steps require `torch` and `torch-geometric` (not installed by default via `pip install grasp-tool`).
- `scripts/` and `example_pkl/` are repo-only; they are excluded from PyPI wheels.

## Release constraints (no CJK)

Release artifacts must not contain CJK (Chinese/Japanese/Korean) characters.

Before building, run:

```bash
python scripts/check_no_cjk.py
```

If you build distributions, also scan the built artifacts:

```bash
python -m build
python scripts/check_no_cjk.py --scan-dist
```

## Pull requests

- Keep changes minimal and focused.
- Avoid adding new heavy dependencies unless strictly necessary.
- Make sure `python scripts/check_no_cjk.py` passes.
