#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${GRASP_CONDA_ENV:-grasp}"

INPUT_PKL="${1:-$ROOT_DIR/example_pkl/simulated1_data_dict.pkl}"

if [[ ! -f "$INPUT_PKL" ]]; then
  echo "ERROR: input PKL not found: $INPUT_PKL" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/outputs/tiny_demo_example_pkl_${TS}"

REGISTERED_PKL="$OUT_DIR/registered.pkl"
TINY_PKL="$OUT_DIR/tiny_registered.pkl"
PAIRS_CSV="$OUT_DIR/pairs.csv"
PORTRAIT_DIR="$OUT_DIR/portrait"
GRAPH_ROOT="$OUT_DIR/graphs"
TRAIN_PKL="$OUT_DIR/train.pkl"
EMBED_ROOT="$OUT_DIR/embeddings"

mkdir -p "$OUT_DIR"

echo "== tiny demo: example_pkl =="
echo "env:        $ENV_NAME"
echo "input_pkl:  $INPUT_PKL"
echo "output_dir: $OUT_DIR"
echo

echo "[0/6] CLI smoke checks"
conda run -n "$ENV_NAME" python -m grasp_tool --version
conda run -n "$ENV_NAME" python -m grasp_tool --help >/dev/null
conda run -n "$ENV_NAME" python -m grasp_tool train-moco --help >/dev/null
echo "OK"
echo

echo "[1/6] register (nc_demo=4)"
conda run -n "$ENV_NAME" python -m grasp_tool register \
  --pkl_file "$INPUT_PKL" \
  --output_pkl "$REGISTERED_PKL" \
  --nc_demo 4
test -f "$REGISTERED_PKL"
echo "OK: $REGISTERED_PKL"
echo

echo "[2/6] make tiny df_registered + pairs.csv"
conda run -n "$ENV_NAME" python "$ROOT_DIR/scripts/make_tiny_demo_pkl.py" \
  --input_pkl "$REGISTERED_PKL" \
  --output_pkl "$TINY_PKL" \
  --pairs_csv "$PAIRS_CSV" \
  --max_cells 4 \
  --max_genes 8 \
  --min_transcripts_per_pair 10
test -f "$TINY_PKL"
test -f "$PAIRS_CSV"
echo "OK: $TINY_PKL"
echo "OK: $PAIRS_CSV"
echo

echo "[3/6] portrait (JS distances)"
mkdir -p "$PORTRAIT_DIR"
conda run -n "$ENV_NAME" python -m grasp_tool portrait \
  --pkl_file "$TINY_PKL" \
  --output_dir "$PORTRAIT_DIR" \
  --max_count 2 \
  --num_threads 1 \
  --visualize_top_n 0 \
  --use_same_r
echo "OK: $PORTRAIT_DIR"
echo

echo "[4/6] partition-graphs + augment-graphs"
mkdir -p "$GRAPH_ROOT"
conda run -n "$ENV_NAME" python -m grasp_tool partition-graphs \
  --pkl "$TINY_PKL" \
  --graph_root "$GRAPH_ROOT" \
  --n_sectors 4 \
  --m_rings 2 \
  --k_neighbor 3
conda run -n "$ENV_NAME" python -m grasp_tool augment-graphs \
  --graph_root "$GRAPH_ROOT" \
  --dropout_ratio 0.1 \
  --seed 2025
echo "OK: $GRAPH_ROOT"
echo

echo "[5/6] build-train-pkl + train-moco (1 epoch)"
conda run -n "$ENV_NAME" python -m grasp_tool build-train-pkl \
  --pairs_csv "$PAIRS_CSV" \
  --graph_root "$GRAPH_ROOT" \
  --output_pkl "$TRAIN_PKL" \
  --dataset tiny_demo \
  --n_sectors 4 \
  --m_rings 2 \
  --k_neighbor 3 \
  --processes 2
test -f "$TRAIN_PKL"

mkdir -p "$EMBED_ROOT"
conda run -n "$ENV_NAME" python -m grasp_tool train-moco \
  --dataset tiny_demo \
  --pkl "$TRAIN_PKL" \
  --js 0 \
  --n 4 \
  --m 2 \
  --num_epoch 1 \
  --batch_size 8 \
  --num_positive 2 \
  --k 64 \
  --lrs 0.001 \
  --cuda_device 0 \
  --output_dir "$EMBED_ROOT"

echo
echo "DONE"
echo "Outputs: $OUT_DIR"
