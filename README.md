# grasp-tool

- Distribution name: `grasp-tool`
- Import name: `grasp_tool`

This is the official PyTorch implementation of:

"GRASP: Modeling Transcript Spot Graph Representations for Analyzing Subcellular Localization Patterns and Cell Clustering in High-Resolution Spatial Transcriptomics".

Note: This codebase was uploaded along with the manuscript for peer review. The complete code will be released after acceptance.

## Usage

Check out the tutorial pages for demos and documentation:

- https://grasp-lilab.github.io/GRASP/

## Contact

If you have any questions, please don't hesitate to contact us.

- E-mail: litt@hsc.pku.edu.cn; huoyuying@bjmu.edu.cn; wuchaoxu@pku.edu.cn

## Requirements

- Python: 3.9+
- Training (optional): `build-train-pkl` and `train-moco` require PyTorch (`torch`) and PyTorch Geometric (`torch-geometric`).
  They are NOT installed by default via `pip install grasp-tool`.

## Repo layout

- `grasp_tool/`: installable package source
- `scripts/`: repo-only helper scripts (tiny demo, release tooling)
- `example_pkl/`: example raw input used by demos (repo-only; excluded from PyPI wheel)
- `demo_pkl/`: small pre-generated registered subset for smoke tests (repo-only)
- `envs/`: optional conda env templates
- `docs/`: maintainer docs

## Installation (recommended)

The recommended workflow is:

- Use conda/mamba to create a stable Python environment (and GPU toolchain if needed)
- Use pip to install the GRASP package from PyPI (`grasp-tool`)

### 1) Create a conda environment

```bash
conda create -n grasp python=3.9 -y
conda activate grasp
```

Or use the provided environment file (creates env `grasp` and installs `grasp-tool` via pip):

```bash
conda env create -f envs/grasp-base.yml
conda activate grasp
```

### 2) Install GRASP from PyPI

```bash
pip install grasp-tool
```

Smoke checks (should work without training deps):

```bash
grasp-tool --help
grasp-tool train-moco --help
```

If you plan to run training commands (`build-train-pkl`, `train-moco`), install the training stack first:

- PyTorch install selector: https://pytorch.org/get-started/locally/
- PyG install guide: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

If you're running from this repo checkout, you can also run a small demo:

```bash
grasp-tool register \
  --pkl_file example_pkl/simulated1_data_dict.pkl \
  --output_pkl outputs/simulated1_registered.pkl
```

## Tiny demo (end-to-end smoke test)

If you have a repo checkout, you can run a fast end-to-end smoke test on a small
subset of `example_pkl/simulated1_data_dict.pkl`.

This demo runs:

- `register` (with `--nc_demo 4`)
- create a tiny `df_registered` subset + `pairs.csv`
- `portrait` (JS distances)
- `partition-graphs`
- `augment-graphs`
- `build-train-pkl`
- `train-moco` for 1 epoch

Run:

```bash
# Default: uses conda env name "grasp"
bash scripts/tiny_demo_example_pkl.sh
```

Override the conda env name (useful if you have multiple envs):

```bash
GRASP_CONDA_ENV=<your_env_name> bash scripts/tiny_demo_example_pkl.sh
```

Outputs are written under:

- `outputs/tiny_demo_example_pkl_<timestamp>/`

Notes:

- The final training step requires `torch` and `torch-geometric`.
- `example_pkl/` is excluded from PyPI release artifacts; this demo is intended for
  repo checkouts.
- The `scripts/` directory is not part of the installed PyPI wheel. If you installed
  via `pip install grasp-tool`, you need to clone this repo to use this demo script.

Pre-generated demo artifact (repo-only):

- `demo_pkl/tiny_registered.pkl`: a small `df_registered` subset (suitable for `portrait`, `partition-graphs`, `cellplot`)
- `demo_pkl/pairs.csv`: pairs table for `build-train-pkl`

You can use it to skip `register` during manual smoke tests:

```bash
python -m grasp_tool portrait \
  --pkl_file demo_pkl/tiny_registered.pkl \
  --output_dir outputs/portrait_demo \
  --max_count 2 \
  --num_threads 1 \
  --visualize_top_n 0 \
  --use_same_r
```

### 3) Training dependencies (NOT installed by pip)

`pip install grasp-tool` intentionally does NOT pull in the training stack.

If you want to run training-related commands:

- `build-train-pkl` (needs `torch` + `torch-geometric`)
- `train-moco` (needs `torch` + `torch-geometric`)

Install them following the official PyTorch / PyTorch Geometric (PyG) instructions.
We recommend installing them via conda inside the same env.

Install PyTorch (pick one based on your setup):

- Official selector: https://pytorch.org/get-started/locally/

Examples:

```bash
# CPU-only
conda install -c pytorch pytorch torchvision torchaudio cpuonly

# CUDA (example: CUDA 12.1)
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
```

Install PyTorch Geometric (PyG):

- Official guide: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Example:

```bash
pip install torch-geometric
```

Make sure the installed PyG build matches your PyTorch and CUDA versions.

Verify your training stack:

```bash
python -c "import torch, torch_geometric; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('pyg', torch_geometric.__version__)"
```

If `cuda` is `False`, training will run on CPU.

## Development (maintainers)

Poetry is only needed for development and release.

```bash
poetry install
poetry run grasp-tool --help
```

## Environment (reproducibility baseline)

If you are new to the project, the simplest path is:

```bash
conda create -n grasp python=3.9 -y
conda activate grasp
pip install grasp-tool
```

If you prefer a single command, use:

```bash
conda env create -f envs/grasp-base.yml
conda activate grasp
```

Optional extras (only needed for non-core utilities):

- Optimal Transport utilities (`POT` / import name `ot`): `pip install grasp-tool[ot]`

## Full pipeline (from scratch)

The recommended example input is:

- `example_pkl/simulated1_data_dict.pkl`

This PKL contains the raw inputs required by `register` (and also already includes
`df_registered`; the steps below still show how to run the full pipeline end-to-end).

### 0) Register (coordinate normalization)

Input: a PKL dict containing at least:

- `data_df` (DataFrame; must include `cell`, `type`, `centerX`, `centerY`, `x`, `y`)
- `cell_mask_df` (DataFrame; columns: `cell`, `x`, `y`)
- `nuclear_boundary` (dict: cell -> DataFrame with columns `x`, `y`)

Run:

```bash
python -m grasp_tool register \
  --pkl_file example_pkl/simulated1_data_dict.pkl \
  --output_pkl outputs/simulated1_registered.pkl
```

Output: a PKL dict with keys:

- `df_registered`
- `nuclear_boundary_df_registered`
- `cell_radii`
- `cell_nuclear_stats`

### 0.5) (Optional) Cell/gene visualization (cellplot)

This is optional and mainly used for sanity-checking transcript spatial patterns.

Notes:

- This command can generate a large number of images if you do not restrict `--cells` and `--genes`.
- The input PKL must contain the required keys for the selected `--mode`.
  - `--mode raw-cell`: expects `cell_boundary` (and optionally `nuclear_boundary`) in the raw PKL.
    If your raw PKL does not have `cell_boundary`, use `--mode registered-gene` instead.

Registered-gene plots (recommended; uses `df_registered`):

```bash
python -m grasp_tool cellplot \
  --mode registered-gene \
  --pkl outputs/simulated1_registered.pkl \
  --output_dir outputs/cellplot \
  --dataset simulated1 \
  --cells cell_11 \
  --genes gene_6_2_1,gene_6_3_1 \
  --with_nuclear 1
```

Raw cell boundary plots (uses `cell_boundary` / `nuclear_boundary` in the raw PKL):

```bash
python -m grasp_tool cellplot \
  --mode raw-cell \
  --pkl example_pkl/simulated1_data_dict.pkl \
  --output_dir outputs/cellplot_raw \
  --dataset simulated1
```

### 1) (Optional) JS distance

```bash
python -m grasp_tool portrait \
  --pkl_file outputs/simulated1_registered.pkl \
  --output_dir outputs/portrait \
  --use_same_r \
  --visualize_top_n 0 \
  --auto_params
```

### 2) Partition + build per-cell graphs (node/adj CSV)

```bash
python -m grasp_tool partition-graphs \
  --pkl outputs/simulated1_registered.pkl \
  --graph_root outputs/graphs \
  --n_sectors 20 \
  --m_rings 10 \
  --k_neighbor 5
```

You can restrict scope for a quick smoke test:

```bash
python -m grasp_tool partition-graphs \
  --pkl outputs/simulated1_registered.pkl \
  --graph_root outputs/graphs_demo \
  --cells cell_11,cell_135 \
  --genes gene_0_0_0,gene_0_1_0
```

### 3) Graph augmentation

```bash
python -m grasp_tool augment-graphs \
  --graph_root outputs/graphs \
  --dropout_ratio 0.1 \
  --seed 2025
```

### 4) Build training PKL

You need a `pairs.csv` with columns: `cell,gene`.

This file defines which `(cell, gene)` pairs are included in the dataset.
`build-train-pkl` reads it, loads the corresponding graphs from `graph_root`, and writes a `train.pkl` used by `train-moco`.
You can use `pairs.csv` to subsample a large dataset for faster experiments.

Example (generate pairs from `df_registered`):

```bash
python -c "import pickle, pandas as pd; d=pickle.load(open('outputs/simulated1_registered.pkl','rb')); pairs=d['df_registered'][['cell','gene']].drop_duplicates(); pairs.to_csv('outputs/pairs.csv', index=False); print('wrote outputs/pairs.csv', len(pairs))"
```

Then:

```bash
python -m grasp_tool build-train-pkl \
  --pairs_csv outputs/pairs.csv \
  --graph_root outputs/graphs \
  --output_pkl outputs/train.pkl \
  --dataset simulated1
```

### 5) Train (MoCo)

Note: this stage requires `torch` and `torch-geometric`, which are NOT installed by
`pip install grasp-tool`. Install them via conda first.

```bash
python -m grasp_tool train-moco \
  --dataset simulated1 \
  --pkl outputs/train.pkl \
  --js 0 \
  --n 20 \
  --m 10 \
  --num_epoch 300 \
  --batch_size 64 \
  --cuda_device 0 \
  --output_dir outputs/embeddings
```

If you want to use JS for positive sampling:

```bash
python -m grasp_tool train-moco \
  --dataset simulated1 \
  --pkl outputs/train.pkl \
  --js 1 \
  --js_file outputs/portrait/js_distances_*.csv
```

## Outputs

This section summarizes what each stage writes to disk.

### register

Command:

```bash
python -m grasp_tool register --pkl_file <raw.pkl> --output_pkl <registered.pkl>
```

Output (`<registered.pkl>` is a dict with):

- `df_registered` (DataFrame): normalized transcript coordinates; contains at least `cell,gene,x_c_s,y_c_s` and also keeps original columns.
- `nuclear_boundary_df_registered` (DataFrame): normalized nucleus boundary points per cell (contains `cell,x_c_s,y_c_s` plus intermediate columns).
- `cell_radii` (dict): per-cell radius used by downstream partitioning.
- `cell_nuclear_stats` (DataFrame): per-cell nucleus exceed stats (`exceed_percent/exceed_count/num_nuclear_points`).
- `meta` (dict): run metadata.

### cellplot (optional)

Command:

```bash
python -m grasp_tool cellplot --mode <raw-cell|registered-gene> --pkl <input.pkl> --output_dir <dir>
```

Output (`<dir>`):

- Raw mode (`--mode raw-cell`): writes per-cell boundary plots under `1_<dataset>_raw_cell_plot/`.
- Registered mode (`--mode registered-gene`): writes per-cell/per-gene scatter plots under `<dataset>/registered_gene/<cell>/`.

### portrait (optional)

Command:

```bash
python -m grasp_tool portrait --pkl_file <registered.pkl> --output_dir <dir>
```

Output (`<dir>`):

- `js_distances_*.csv`: JS distance table used for positive sampling (when `train-moco --js 1`).

### partition-graphs

Command:

```bash
python -m grasp_tool partition-graphs --pkl <registered.pkl> --graph_root <graph_root>
```

Output directory layout (`<graph_root>`):

- `<graph_root>/<cell>/<gene>_node_matrix.csv`
- `<graph_root>/<cell>/<gene>_adj_matrix.csv`
- `<graph_root>/<cell>/<gene>_dis_matrix.csv`

These CSVs are the on-disk graph representation consumed by `build-train-pkl`.

### augment-graphs

Command:

```bash
python -m grasp_tool augment-graphs --graph_root <graph_root>
```

Output directory layout:

- `<graph_root>/<cell>_aug/<gene>_node_matrix.csv`
- `<graph_root>/<cell>_aug/<gene>_adj_matrix.csv`

### build-train-pkl

Command:

```bash
python -m grasp_tool build-train-pkl --pairs_csv <pairs.csv> --graph_root <graph_root> --output_pkl <train.pkl>
```

Output (`<train.pkl>` is a dict with):

- `original_graphs`: list of `torch_geometric.data.Data`
- `augmented_graphs`: list of `torch_geometric.data.Data`
- `gene_labels`, `cell_labels`: aligned labels for each graph
- `meta`: dataset tag + graph parameters + `pairs.csv` path

### train-moco

Command:

```bash
python -m grasp_tool train-moco --dataset <name> --pkl <train.pkl> --output_dir <out_root>
```

Output directory layout (`<out_root>`):

- `<out_root>/<run_id>/1_training_config.json`: the full resolved args snapshot
- `<out_root>/<run_id>/epoch{E}_lr{LR}_embedding.csv`: **main representation output**
  - columns: `feature_1..feature_d, cell, gene`
- checkpoints:
  - `<out_root>/<run_id>/epoch_{E}_lr_{LR}_checkpoint.pth`
  - (optional) `<out_root>/<run_id>/best_model_epoch_{E}_lr_{LR}.pth`
- best summary (only when clustering is enabled via `--num_clusters`):
  - `<out_root>/<run_id>/best_metrics_lr{LR}.json`
  - `<out_root>/<run_id>/best_{vis_method}_{cluster_method}_lr{LR}.png`
- evaluation / visualization (from `grasp_tool/gnn/plot_refined.py`):
  - `<out_root>/<run_id>/epoch{E}_lr{LR}_metrics*.txt`
  - `<out_root>/<run_id>/epoch{E}_lr{LR}_clusters*.csv`
  - `<out_root>/<run_id>/epoch{E}_lr{LR}_visualization*.png`
- `<out_root>/<run_id>/ALL_COMPLETED.txt`: written after all learning rates finish

## Key parameters (quick reference)

In general, you can always inspect the full list via:

```bash
python -m grasp_tool --help
python -m grasp_tool <command> --help
```

### register

- `--pkl_file`: input raw data dict PKL
- `--output_pkl`: output registered PKL (will contain `df_registered`)
- `--nc_demo`: process only first N cells (smoke test)
- `--chunk_size`: multiprocessing chunk size (speed/memory tradeoff)
- `--clip_to_cell`: `1` to clip nucleus to cell boundary; `0` to keep outside points
- `--remove_outliers`: `1` to drop nucleus points exceeding boundary
- `--epsilon`: numerical stability

### cellplot

- `--mode`: `raw-cell` or `registered-gene`
- `--pkl` / `--pkl_file`: input PKL path
- `--output_dir`: output directory root
- `--dataset`: dataset tag used in output paths (optional)
- `--cells`: restrict to a comma-separated subset of cells (recommended)
- `--genes`: restrict to a comma-separated subset of genes (registered-gene only; recommended)
- `--with_nuclear`: `1` to plot nucleus boundary if present, `0` to disable (registered-gene only)

### portrait

This command is a pass-through wrapper. Common knobs:

- `--auto_params`: auto-select `r_min/r_max/bin_size`
- `--use_same_r`: enforce the same `r` within each gene
- `--max_count`, `--transcript_window`: reduce compute for large datasets
- `--output_dir`: control where `js_distances_*.csv` is written

### partition-graphs

- `--pkl`: registered PKL (must contain `df_registered`)
- `--graph_root`: output root directory
- `--n_sectors`, `--m_rings`: partition resolution
- `--k_neighbor`: kNN graph connectivity
- `--cells`, `--genes`: restrict scope (smoke test)
- `--epsilon`: boundary classification tolerance

### augment-graphs

- `--graph_root`: directory created by `partition-graphs`
- `--dropout_ratio`: node dropout probability
- `--seed`: make augmentation deterministic
- `--angle_min`, `--angle_max`: rotation angle range (degrees)

### build-train-pkl

- `--pairs_csv`: CSV with columns `cell,gene`
- `--graph_root`: directory created by `partition-graphs` (and augmented by `augment-graphs`)
- `--output_pkl`: training PKL consumed by `train-moco`
- `--dataset`: dataset tag stored in metadata
- `--processes`: multiprocessing workers

### train-moco

This command runs the packaged training entrypoint (`grasp_tool.cli.train_moco`).

- `--pkl`: training PKL built by `build-train-pkl`
- `--output_dir`: output root directory
- `--lrs`: learning rate list (e.g. `--lrs 0.001` or `--lrs 0.001 0.002`)
- `--use_gradient_clipping`: `1` (default) to clip gradients, `0` to disable
- `--gradient_clip_norm`: max norm for gradient clipping
- `--js` + `--js_file`: use JS distance for positive sampling
- `--n`, `--m`: must match partition settings
- `--seed`: reproducibility
- `--num_epoch`, `--batch_size`: training schedule
- `--cuda_device`: GPU index
- `--num_clusters`: affects clustering evaluation (for very small datasets, set it <= num graphs)

## Reproducibility tips

- Always record: `n_sectors/m_rings/k_neighbor/dropout_ratio/seed` and the exact `pairs.csv`.
- Prefer writing all outputs under `outputs/` (or a dedicated run directory).
- For large runs, use tmux/screen; training can be slow due to evaluation + visualization.
