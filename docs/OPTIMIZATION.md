# GRASP Performance Optimization Summary

This document summarizes the performance optimizations introduced in `grasp-tool 0.2.0`,
including behavior changes, new parameters, measured results, and the recommended final
configuration.

## 1. Baseline

All comparisons use the following as the same "original version" baseline:

- GitHub repository: `https://github.com/TingtingLiGroup/GRASP.git`
- Commit: `d26686f723115cdeb20141b31c5f1733bd1f63d0`
- Released package: `grasp-tool 0.1.3`

The installed `grasp-tool 0.1.3` sources of the following files were verified file by file to be
identical to that commit:

- `grasp_tool/cli/main.py`
- `grasp_tool/cli/train_moco.py`
- `grasp_tool/gnn/gat_moco_final.py`
- `grasp_tool/gnn/plot_refined.py`
- `grasp_tool/preprocessing/partition.py`

Differences reported here are therefore the same whether measured against the GitHub commit or
against the released `0.1.3` package.

## 2. Overall results

This round of optimization covers:

1. Vectorized partition graph construction and leaner disk output.
2. Indexed Portrait access over observed `(cell, gene)` pairs with per-gene scheduling.
3. Indexed JS positive generation.
4. Optimized scheduling of training-time evaluation and visualization.
5. Batched embedding inference.
6. An immutable CPU training batch cache.
7. Pinned staging and an asynchronous H2D pipeline.
8. A foreach momentum-encoder EMA.
9. A multi-GPU launcher for independent learning rates.
10. Screening of wide-tolerance training candidates.
11. Full benchmarks, equivalence tests, and result-similarity gates.

Key measured results:

- Partition, 1000 graphs: `145.30 -> 12.91` s, about `11.3x`.
- Full pipeline, 1000 graphs: `260.18 -> 44.22` s, about `5.9x`.
- Portrait Stage 1 on a real CosMx subset: about `0.70 -> 0.06` s/gene, about `11.7x`.
- A full Portrait run over 3805 genes and 468,448 observed `(cell, gene)` pairs finished in about
  1 h 41 min and produced 4,643,052 JS records; the old version had not finished the same job after
  more than 22 h, so the real completion time is at least about `13x` shorter.
- JS positives for the full 111,589-graph dataset: previously about 2.8 h, now `1.35` s.
- CPU batch cache on the full 111,589-graph dataset: `67.967 -> 50.603` s/epoch, `25.5%` faster.
- Adding the H2D pipeline on top of the cache: `49.468 -> 45.988` s/epoch, a further `7.0%`.
- TF32 is about `3.5%` faster on the 10,000-graph matched benchmark and passes the three-seed,
  20-epoch result gate.
- An aggressive combination reaches `50.284 -> 26.549` s/epoch but fails the loss-curve gate and is
  not recommended for production.
- Optimized 200-epoch end-to-end training from existing PKLs: 20x10 takes
  `1.262 / 3.843 / 25.360` min on 800/2500/15000 graphs, and 30x15 takes
  `3.415 / 10.526 / 66.649` min.
- Full end-to-end time starting from registered data, with Portrait, partition, augmentation, and
  build-train-pkl timed separately: 20x10 takes `2.151 / 5.654 / 40.272` min, and 30x15 takes
  `5.033 / 14.825 / 94.916` min.
- Process-level peak GPU VRAM in the formal simulated benchmark: `3.5-5.5 GiB` for 20x10 and
  `14.6-19.6 GiB` for 30x15; all configurations stay below 24 GiB.
- Full MERFISH U2OS dataset (113,909 graphs): 20x10 full/train-only takes
  `373.666 / 158.760` min, and 30x15 takes `752.542 / 466.221` min; peak GPU VRAM is
  `3.61 / 17.88 GiB`, respectively.

## 3. Per-stage optimizations

### 3.1 Register, cellplot, and Portrait

The core algorithms of register and cellplot are unchanged. Portrait's r selection, NetworkX graph
construction, shortest paths, network portrait, and JS divergence numerics are also unchanged; only
data access and task scheduling were optimized.

Files involved:

- `grasp_tool/preprocessing/portrait.py`
- `scripts/benchmark_portrait.py`
- `tests/test_portrait_optimizations.py`

Problems in the original implementation:

- Categorical `groupby` materialized unobserved category combinations, so the diagnostic stage could
  produce a huge empty `(cell, gene)` Cartesian product.
- Stages 0/1 repeatedly filtered the full DataFrame inside loops and iterated over many
  non-existent `(cell, gene)` combinations.
- Distance matrices were submitted as one thread task per `(cell, gene)`, creating hundreds of
  thousands of futures on large data.

After optimization:

- Categorical `groupby` consistently uses `observed=True` and counts only combinations that exist.
- A `gene -> [(cell, row_indices)]` index is built once; later stages access the real rows directly.
- Stage 0 submits one task per gene and computes that gene's cell distance matrices in batch.
- Stages 1/2 reuse the same index instead of rescanning the full DataFrame.
- Separate timings for diagnostics, distance matrices, network portrait, and JS stages make real
  bottlenecks easy to locate.

Measurements:

- Synthetic 50/200/1000-pair data: about `1.002x / 1.030x / 1.010x`; these data are too small and
  fixed overhead dominates.
- On a real CosMx subset mimicking the sparse structure of 4114 cells and 3805 genes, Stage 1 drops
  from about `0.70` s/gene to `0.06` s/gene, about `11.7x`.
- The full 3805-gene job finishes in about 1 h 41 min and outputs 4,643,052 JS records. The old
  version had not finished the same job after more than 22 h, so the real completion time is at
  least about `13x` shorter; because the old version never finished, an exact end-to-end ratio
  cannot be given.
- A P1 candidate that replaced NetworkX with SciPy sparse matrices was evaluated. It was numerically
  equivalent but about `2.4%` slower on the 1000-pair benchmark, so it was reverted.

Effect on results:

- Network portraits produced by the indexed path and the old per-combination path in the same
  process are identical; the full 9-column CSV of the synthetic end-to-end test is also identical.
- Against an old CosMx CSV subset generated in a different process, row keys, integer columns,
  candidate order, and downstream positive indices are identical; floating-point columns agree
  within an absolute error of `1e-12`.
- Bitwise file-level identity across processes cannot be guaranteed for the last floating-point
  digits, because the original `js_divergence` uses `set.union`, whose iteration order varies
  between Python processes; the same jitter exists in the unoptimized code.
- No complete old output exists for the full 3805-gene job for row-by-row comparison. Correctness
  there rests on the unchanged numerical core, equivalence gates on synthetic and real subsets, and
  validation of finite values, value ranges, graph mapping, and positive generation on the full
  result.
- `observed=True` corrects the pair count in the diagnostic report from a spurious categorical
  Cartesian product to the real observed count; this statistic does not feed into JS results or
  training.

### 3.2 Partition-graphs

Files involved:

- `grasp_tool/preprocessing/partition.py`
- `grasp_tool/cli/main.py`

Changes:

- Per-sector, per-ring Pandas boolean filtering replaced with NumPy `searchsorted` and `bincount`.
- Node distances, kNN, and adjacency construction vectorized.
- Processing grouped by `(cell, gene)` to reduce repeated DataFrame filtering.
- Repeated statistics such as cell radius precomputed per group.
- The `*_dis_matrix.csv` files, which have no downstream consumer, are no longer written by default.
- The original node matrix and adjacency matrix formats are preserved.

Measurements:

- 50 graphs: `9.75 -> 2.22` s, about `4.4x`.
- 200 graphs: `30.47 -> 3.94` s, about `7.7x`.
- 1000 graphs: `145.30 -> 12.91` s, about `11.3x`.
- `count_points_in_areas_same` in the 50-graph profile: `10.19` s -> `0.016` s.
- 1000-graph output: from 3000 CSVs / 501.7 MB to 2000 CSVs / 92.1 MB.

Effect on results:

- Node counts and adjacency were compared element by element against the old algorithm.
- Radius, angular boundaries, distance ties, and all-virtual graphs are covered by tests.
- Default disk output no longer includes the distance-matrix CSVs.
- Distance matrices are still computed in memory for kNN; only writing them to disk is skipped by
  default.

### 3.3 Augment-graphs

The augmentation algorithm itself is unchanged:

- Node dropout behavior is unchanged.
- Rotation behavior is unchanged.
- Random-seed semantics are unchanged.

The indirect benefit is that partition no longer writes unused distance matrices, so graph
directories are smaller; no new algorithmic speedup is claimed for augmentation itself.

### 3.4 Build-train-pkl

The core algorithms of `graphloader.py` and build-train-pkl are unchanged.

This stage still reads only the node matrix and adjacency matrix, so partition no longer writing
distance matrices by default does not change the training PKL content.

This round's benchmarks include this stage in full-pipeline measurements, but it was not rewritten
and its data semantics are unchanged.

### 3.5 JS positive generation

Files involved:

- `grasp_tool/gnn/gat_moco_final.py`

For each query graph, the original implementation:

1. Scanned the whole JS DataFrame.
2. Selected candidate positives.
3. Linearly scanned all graphs again to find the index of each `(cell, gene)`.

After optimization:

- A `(cell, gene) -> graph index` mapping is built once.
- Rows are sorted once and grouped by query.
- Positive graph indices are obtained directly through a hash index.
- The original positive order, missing-value fallback, and first-augmented-positive semantics are
  preserved.

Measurements:

- 50 graphs: `0.059 -> 0.0021` s.
- 200 graphs: `0.255 -> 0.0022` s.
- 1000 graphs: `2.08 -> 0.0062` s.
- 2500 graphs: `8.05 -> 0.0134` s.
- 2500 real graphs: `34.44 -> 0.021` s.
- Full 111,589 graphs with 1,105,754 JS distances: `1.35` s.

Effect on results:

- Positive index lists match the old implementation item by item.
- Tests cover distance ties, missing-value fallback, and different graph counts.
- GW positive and random-window positive algorithms are unchanged.

## 4. Evaluation and visualization

Files involved:

- `grasp_tool/cli/train_moco.py`
- `grasp_tool/gnn/plot_refined.py`

Problems in the original version:

- Embedding evaluation ran at epoch 0, every 20 epochs, and at the final epoch.
- `visualize=True` was hard-coded, so t-SNE and UMAP ran repeatedly.
- Embeddings were computed with one GPU forward pass per graph and synchronized back to the CPU
  graph by graph.
- Embedding CSVs were written more than once.

After optimization:

- By default, embeddings are generated only at the final epoch.
- t-SNE/UMAP are disabled by default.
- Epoch-0 evaluation, intermediate evaluation, and plotting can be re-enabled explicitly.
- Embeddings use batched inference via `Batch.from_data_list`.
- Duplicate embedding CSV writes were removed.
- Matplotlib figures are closed promptly.
- A new `training_history_lr<lr>.csv` records per-epoch loss, learning rate, and training time.

Measurements:

- With 1000 graphs, the original two evaluations took `91.75` s, while the two actual training
  epochs took only `1.81` s.
- After optimization, 1000-graph training plus final evaluation dropped from `92.50` s to `8.70` s.
- On 50 real graphs, the maximum absolute difference between per-graph and batched embeddings is
  `4.32e-7`.

Effect on results:

- Training checkpoints and final embeddings are unchanged.
- Default outputs change: epoch-0 and intermediate embeddings and t-SNE/UMAP plots are no longer
  generated by default.
- The final embedding CSV is still generated.
- Checkpoints are still saved every 20 epochs, independent of `eval_freq`.

## 5. Strict-semantics training throughput

Strict-semantics optimizations preserve:

- Positive indices and graph order.
- Batch boundaries and tail-batch rules.
- The number of BN/Dropout calls.
- FIFO queue contents, sources, and update timing.
- The gradient-clipping scope.
- The Adam parameter set.
- The definition of the cross-graph dense `NxN` reconstruction loss.

### 5.1 Training batch boundaries and minimal cached objects

A unified batch-range generator was added that keeps the old rules:

- A remainder of 1 is merged into the previous batch.
- A remainder of 2 or more forms its own tail batch.
- A single-graph dataset does not form a training batch.

The cache keeps only what training needs:

- `x`
- `edge_index`
- `batch`
- `num_graphs`

The `cell` and `gene` string attributes, which training does not use, are not cached.

### 5.2 Immutable CPU batch cache

Switch: `--cache_train_batches 1`

Behavior:

- After positives are prepared, all query/positive PyG batch tensors are prebuilt.
- They are reused across epochs.
- They are also reused when several learning rates are trained sequentially in the same process.
- `.to(device)` returns a new object and does not modify the CPU cache.
- A RAM estimate is printed before the cache is built.
- The full cache stays pageable and does not lock all host memory.

Measurements:

- 1000 synthetic graphs: `1.621 -> 1.369` s, `15.5%` faster.
- 10,000 real graphs: `6.523 -> 4.714` s, `27.7%` faster.
- 111,589 real graphs: `67.967 -> 50.603` s, `25.5%` faster.
- The full cache takes about 9.42-9.45 GiB.
- Peak RSS rises from about 9.33 GiB to 18.86 GiB.

This feature is off by default because it needs extra RAM.

### 5.3 Bounded pinned staging and asynchronous H2D

Switches:

- `--pipeline_cached_h2d 1`
- `--cached_h2d_prefetch_batches 2`

Requirements:

- `--cache_train_batches 1` must also be enabled.
- Only effective for CUDA training.

Behavior:

- A small number of upcoming batches are taken from the pageable CPU cache.
- Only the bounded staging batches are pinned.
- The next batch is transferred ahead of time on a separate CUDA stream.
- The next H2D copy overlaps with the current GPU computation.
- The full cache of 9 GiB or more is not pinned.

Measurements:

- Strict baseline median on 111,589 graphs: `49.468` s/epoch.
- With the pipeline enabled: `45.988` s/epoch.
- About `7.0%` additional speedup.
- Throughput rises to about `2414.3 graphs/s`.
- Peak VRAM stays essentially unchanged at about 2.60 GiB.

### 5.4 Foreach momentum EMA

Switch: `--foreach_momentum 1`

Behavior:

- Per-parameter momentum-encoder EMA updates are merged into `torch._foreach_*` kernels.
- The original multiply-add order is preserved.

Measurements:

- 5000 independent EMA updates: `3.088 -> 0.782` s.
- 10,000 graphs with the H2D pipeline already enabled: `7.828 -> 7.282` s, about `7.0%`
  additional speedup.

Effect on results:

- CPU parameter updates are bitwise identical tensor by tensor.
- CUDA loss, parameters, BN buffers, queue, and queue pointer pass a `rtol=1e-6, atol=1e-6` gate.

### 5.5 GPU loss accumulation

Per-batch losses are no longer synchronized to the CPU with several immediate `.item()` calls;
instead they are accumulated on the GPU and converted to Python numbers at the end of the epoch.

This reduces CPU/GPU synchronization and does not change the loss definition.

### 5.6 Prefetch screening results

A bounded background PyG collation path was implemented:

- `--prefetch_batches`
- `--pin_prefetched_batches`

On real 10,000- and 111,589-graph jobs, however, background collation contended with main-thread
CUDA launches for CPU/GIL, and both pinned and unpinned prefetching were slower than the sequential
path.

Both switches are therefore kept for hardware-specific experiments, but default to `0` and are not
part of the recommended configuration.

## 6. Wide-tolerance training optimizations

### 6.1 Finally accepted: TF32

Switch:

```text
--matmul_precision high
```

TF32 keeps the same:

- GAT and projector architecture.
- Loss formula.
- 4 positives.
- Batch size and batch boundaries.
- Number of BN/Dropout calls.
- Queue and optimizer semantics.

It only changes the internal precision of float32 matrix multiplications on the GPU.

Measurements:

- About `3.5%` faster on the 10,000-graph matched benchmark.
- 111,589 graphs, seed 2025: `48.51 -> 47.18` s/epoch.
- 111,589 graphs, seed 2026: `50.88 -> 49.08` s/epoch.

Three-seed, 20-epoch gate:

- Linear CKA: `0.9988 / 0.9965 / 0.9908`
- kNN overlap: `0.9104 / 0.9167 / 0.8534`
- ARI: `0.7893 / 0.5447 / 0.5127`
- NMI: `0.7898 / 0.6116 / 0.6089`
- Loss correlation: `0.9995 / 0.9987 / 1.0000`

All values fall within the seed-to-seed variation of the original baseline.

This parameter still defaults to `highest`, because TF32 is not strictly numerically equivalent
parameter by parameter.

### 6.2 Kept as research switches but failed the final gate

#### Fused positive encoders

```text
--fuse_positive_encoders 1
```

Multiple positive batches are concatenated, run through the key encoder once, and split back into
representations in the original order.

It keeps positive contents and queue sources, but changes:

- The scope of key-encoder BatchNorm statistics.
- The structure of Dropout random-number calls.

#### Fewer positives

`--num_positive` is an existing parameter with default `4`. Setting it to `2` reduces key-encoder
computation and CPU cache size, but changes the contrastive objective.

#### Reconstruction negative sampling

```text
--reconstruction_negative_ratio 10
```

Behavior:

- All positive edges are kept.
- Negative pairs are sampled uniformly from the zero entries of the original dense target.
- Weighting by the total number of negatives makes the estimator match the original dense BCE in
  expectation.
- An independent generator is used so the global RNG used by GAT Dropout is not consumed.

`5 / 10 / 20` negatives per positive were screened; `10` gave the best throughput.

Aggressive combination:

```text
--matmul_precision high
--fuse_positive_encoders 1
--num_positive 2
--reconstruction_negative_ratio 10
```

Full 111,589 graphs:

- Strict baseline: `50.284` s/epoch, `2229.9 graphs/s`.
- Aggressive combination: `26.549` s/epoch, `4229.8 graphs/s`.
- Epoch time shortened by about `47.2%`.
- VRAM drops from about 2600.5 MiB to 686.2 MiB.
- CPU cache drops from about 9.45 GiB to about 5.64 GiB.

The embedding CKA, kNN, ARI, and NMI of this combination pass the three-seed gate, but loss-curve
correlation does not. After negative sampling and positive count were rolled back step by step, the
fused-positive path still failed the loss gate.

These options are therefore kept only as research switches and are not the final production
recommendation.

### 6.3 Reverted after screening

- Local `torch.compile` of reconstruction: `4.770 -> 5.193` s on the matched benchmark, i.e.
  slower.
- Fused Adam: `4.770 -> 4.659` s, only about `2.3%`, below the 3% retention threshold.
- BF16 encoder autocast: `4.770 -> 4.649` s, only about `2.5%`, below the threshold.
- Batch sizes 48/32/16: `4.942 / 5.879 / 12.156` s under the same cumulative setup, all slower than
  `3.699` s for batch size 64.
- `zero_grad(set_to_none=True)`: `68.52` s on 111,589 graphs, no better than `67.97` s for the
  original zero-fill.
- CPU queue-pointer mirror: `49.643` s, slower than the `49.468` s strict baseline.
- Reusing the dense adjacency buffer: still needs zeroing every batch; no reliable gain.
- Whole-step CUDA Graph: the five-view shapes of all 1744 real batches differ, so static capture is
  not suitable.

The compile, fused Adam, and BF16 production code has been reverted, and no `train-moco` user
switches are provided for them.

## 7. Multi-GPU learning-rate parallelism

New:

- `scripts/train_multi_lr.py`

Behavior:

- Launches an independent training process for each learning rate.
- Binds each process to one `cuda_device`.
- Assigns learning rates round-robin to the specified GPUs.
- Uses a separate output directory per process.
- This is not DDP and does not split a single model or a single epoch.

Benefits:

- Does not shorten the epoch of a single model.
- Brings the total wall-clock time of several learning-rate experiments close to that of the
  slowest single job.

Resource notes:

- Each process loads the PKL independently.
- Each process builds its own CPU batch cache.
- With 111,589 graphs and 4 processes, the caches alone may need about `4 x 9.45 GiB` of RAM.
- The RNG trajectories of independent processes differ from the original sequential training of
  multiple learning rates in one process.

## 8. New CLI parameters

### 8.1 Partition-graphs

#### `--write_distance_matrix`

- Default: `0`
- `0`: do not write the legacy distance-matrix CSVs, which have no downstream consumer.
- `1`: restore the old disk output layout.
- Does not change the node matrix or adjacency matrix.

### 8.2 Train-moco

#### `--visualize`

- Default: `0`
- `1`: generate t-SNE/UMAP whenever evaluation runs.

#### `--eval_freq`

- Default: `0`
- `0`: evaluate only at the final epoch.
- Greater than 0: evaluate every N epochs.
- Does not control checkpoint frequency.

#### `--eval_at_start`

- Default: `0`
- `1`: generate the epoch-0 embedding/evaluation before training.

#### `--prefetch_batches`

- Default: `0`
- Depth of the bounded queue for background CPU collation.
- Slower on current real jobs; not recommended by default.

#### `--pin_prefetched_batches`

- Default: `0`
- Pins only the bounded prefetched batches.
- Never pins the full PKL or the full cache.

#### `--cache_train_batches`

- Default: `0`
- `1`: prebuild the immutable CPU training batch cache.
- Recommended for large, multi-epoch jobs with enough host RAM.

#### `--pipeline_cached_h2d`

- Default: `0`
- `1`: enable pinned staging and asynchronous H2D.
- Requires `--cache_train_batches 1`.

#### `--cached_h2d_prefetch_batches`

- Default: `2`
- Controls the staging depth of the H2D pipeline.
- Used only when `pipeline_cached_h2d=1`.

#### `--foreach_momentum`

- Default: `0`
- `1`: update the momentum-encoder EMA with foreach kernels.

#### `--matmul_precision`

- Default: `highest`
- Choices: `highest / high / medium`
- Only the optional `high` was finally validated and recommended.
- `high` allows TF32 and introduces small numerical differences.

#### `--fuse_positive_encoders`

- Default: `0`
- `1`: fuse the key-encoder forward passes of multiple positives.
- Changes BN/Dropout behavior; for research experiments only.

#### `--reconstruction_negative_ratio`

- Default: `0`
- `0`: keep the original dense reconstruction BCE.
- Greater than 0: sample the given number of negative pairs per positive edge.
- Changes the reconstruction loss at every step; for research experiments only.

### 8.3 Related existing parameters

The following parameters are not new in this round but are closely related to the optimized
configuration:

- `--lrs`: when omitted, the four learning rates `0.001 / 0.002 / 0.005 / 0.01` are still trained
  sequentially; production commands should specify it explicitly.
- `--num_positive`: default `4`; reducing it changes the contrastive objective.
- `--batch_size`: default `64`; smaller batches were all slower in this round.
- `--num_epoch`: default `300`.
- `--cuda_device`: selects a single training GPU.

## 9. New helper scripts and tests

### `scripts/benchmark_pipeline.py`

Supports:

- Generating deterministic synthetic data.
- Measuring partition, augmentation, build-train-pkl, and training.
- Measuring steady-state training only from a real PKL.
- Warmup plus multi-epoch wall-clock-only measurement.
- Torch profiler, CUDA events, and RAM and VRAM recording.
- Mirroring candidate training-throughput parameters.
- Polling RSS of the full child-process tree with `psutil`, and filtering
  `nvidia-smi compute-apps` by process-tree PID, so that worker RAM is not missed and other users'
  GPU memory is not included.

### `scripts/benchmark_simulated_scalability.py`

Supports:

- Resumable single-run formal 200-epoch benchmarks of the 20x10 and 30x15 configurations on
  800/2500/15000-graph simulated data and the full 113,909-graph MERFISH U2OS dataset.
- Separate measurement of the full preprocessing pipeline from registered data and of direct
  training from existing step3 PKLs.
- Selecting target cell-gene pairs for partition exactly from the training-pairs CSV, so that
  low-count U2OS pairs not used for training do not enter the preprocessing workload.
- Dynamically checking for exclusive GPU use and rejecting contaminated results when foreign GPU
  processes are detected.
- Writing per-stage JSON, task-level CSV, full commands, hardware environment, and artifact
  acceptance results.

### `scripts/render_simulated_scalability.py`

Generates, from the formal CSVs, the English four-panel efficiency and memory figure covering the
three simulated datasets and the full U2OS dataset, as 600 DPI PNG, PDF, and SVG, together with a
full benchmark report.

### `scripts/compare_training_results.py`

Supports:

- Aligning baseline/candidate embeddings.
- Linear CKA.
- kNN overlap.
- ARI/NMI after KMeans.
- Loss correlation, normalized RMSE, and final-epoch loss difference.
- Using the baseline's seed-to-seed variation as the candidate gate.

### `scripts/train_multi_lr.py`

Runs multiple learning rates independently on multiple GPUs; not DDP.

### `scripts/benchmark_portrait.py`

Generates deterministic sparse registered data and measures Portrait wall time, output record
count, and peak RSS.

### `tests/test_portrait_optimizations.py`

Covers:

- Categorical diagnostics counting only observed pairs.
- Exact equivalence of indexed-path and old per-combination network portraits.
- Equivalence of the full 9-column JS CSV and downstream positive indices.

### `tests/test_performance_optimizations.py`

Covers:

- Partition boundaries and legacy equivalence.
- JS positive index equivalence.
- Batched embedding inference.
- Batch tail rules.
- Prefetch FIFO and RNG.
- Immutable cache and byte estimation.
- H2D pipeline.
- Foreach EMA.
- CLI defaults.
- Exact partition filtering by pairs CSV.
- Multi-GPU launcher.
- Wide-tolerance result comparison gate.

### `tests/test_simulated_scalability_benchmark.py`

Covers process-tree RAM, process-level GPU PID filtering, the U2OS label-free training command
contract, full -> train-only PKL reuse, and three-format export of the four-panel figure.

Current full automated validation: `43 passed`.

## 10. Recommended final configuration

### 10.1 Keep strict training semantics

```bash
python -m grasp_tool train-moco ... \
  --lrs 0.001 \
  --cache_train_batches 1 \
  --pipeline_cached_h2d 1 \
  --cached_h2d_prefetch_batches 2 \
  --foreach_momentum 1
```

Suitable when:

- Training is long and multi-epoch.
- Host RAM can afford about 9.45 GiB of additional cache.
- The original loss, BN/Dropout, queue, batch, and positive semantics should be preserved.

### 10.2 Allow small numerical differences

Add to the strict configuration:

```bash
--matmul_precision high
```

This is the only new wide-tolerance optimization in this round that passed the three-seed,
20-epoch final result gate.

### 10.3 Restore intermediate evaluation and plotting

```bash
--eval_at_start 1 \
--eval_freq 20 \
--visualize 1
```

### 10.4 Aggressive research configuration

```bash
--matmul_precision high \
--fuse_positive_encoders 1 \
--num_positive 2 \
--reconstruction_negative_ratio 10
```

This configuration gives close to 1.9x throughput but fails the loss-curve gate, and should not be
described as a result-equivalent optimization alongside the strict path.

## 11. Unchanged or unoptimized parts

- Core augmentation algorithm.
- Core build-train-pkl algorithm.
- Core graphloader data format.
- GW positive generation.
- Random-window positive generation.
- Portrait r selection, NetworkX shortest paths, network portrait, and JS divergence numerics.
- MoCo queue definition.
- The default path of the original dense reconstruction loss.
- Gradient-clipping scope.
- Original parameter ownership of the projector and optimizer.

## 12. Evidence

The detailed benchmark outputs below were generated locally and are not included in the
repository (`outputs/` is git-ignored). They can be regenerated with the scripts in `scripts/`.

- Full performance report: `outputs/perf_benchmark/PERFORMANCE_FINDINGS.md`
- Layered screening results: `outputs/perf_benchmark/wide_tolerance_layer_screening.json`
- TF32 three-seed gate:
  `outputs/perf_benchmark/wide_tolerance_validation/final_tf32_three_seed_epoch20_comparison.json`
- Simulated and MERFISH U2OS 200-epoch formal report:
  `outputs/perf_benchmark/simulated_200epoch_end_to_end/formal_200epoch_20260819/BENCHMARK_REPORT.md`
- Simulated and MERFISH U2OS task and stage data:
  `outputs/perf_benchmark/simulated_200epoch_end_to_end/formal_200epoch_20260819/tasks.csv`
  and `stages.csv`
- Simulated and MERFISH U2OS four-panel figure:
  `outputs/perf_benchmark/simulated_200epoch_end_to_end/formal_200epoch_20260819/optimized_grasp_simulated_scalability.{png,pdf,svg}`

Included in the repository:

- Portrait benchmark: `scripts/benchmark_portrait.py`
- Portrait equivalence tests: `tests/test_portrait_optimizations.py`
- Usage: `README.md`

## 13. Re-measured efficiency and memory on simulated data and the full U2OS dataset

The formal test uses a single learning rate `0.001`, 200 epochs, batch size 64, 4 positives, and
seed 2025, with the immutable batch cache, cached H2D pipeline, foreach momentum, and
`matmul_precision=high` enabled. Each task was run once, so results have no error bars.

Two timing scopes:

- Full pipeline: starting from an existing `df_registered`, runs Portrait, partition filtered by
  training pairs, augmentation, build-train-pkl, training initialization, 200 epochs, and final
  output as separate steps.
- Train only: starting from a matching, validated step3 PKL, includes PKL/JS loading, positive and
  cache construction, 200 epochs, and final output.

End-to-end time (Simulated 1 / 2 / 3):

- 20x10 full pipeline: `2.151 / 5.654 / 40.272` min.
- 20x10 train only: `1.262 / 3.843 / 25.360` min.
- 30x15 full pipeline: `5.033 / 14.825 / 94.916` min.
- 30x15 train only: `3.415 / 10.526 / 66.649` min.

Full MERFISH U2OS (989 cells, 135 genes, 113,909 graphs):

- 20x10 full pipeline: `373.666` min; train only: `158.760` min.
- 30x15 full pipeline: `752.542` min; train only: `466.221` min.
- In the full pipeline, Portrait takes `178.636 / 175.456` min and training takes
  `160.334 / 465.259` min for 20x10/30x15; the additional cost of the higher resolution comes
  mainly from partition, augmentation, and training.
- U2OS has no benchmark label file, so `clustering=false` is kept under the existing optimized
  training configuration and no final-epoch clustering evaluation is run.

Memory:

- Peak process-tree CPU RSS of the full pipeline is `7.0-8.2 GiB`, with the peak mainly coming from
  multiprocess build-train-pkl or large-scale training.
- Peak CPU RSS of train only is `1.8-11.4 GiB`, increasing with graph count and the 30x15 node
  count.
- Peak GPU VRAM is `3.5-5.5 GiB` for 20x10 and `14.6-19.6 GiB` for 30x15, never exceeding 24 GiB.
- Peak CPU RSS of the U2OS full pipeline is `71.1-71.2 GiB`, all from Portrait; train only needs
  `18.7 / 36.8 GiB` for 20x10/30x15.
- Peak GPU VRAM on U2OS is `3.61 / 17.88 GiB` for 20x10/30x15, still below the 24 GiB
  consumer-GPU reference line in the figure.

The GRASP runtime in the previous efficiency figure was mainly training-loop time, and its memory
values came from historical manual records. This section uses a new end-to-end scope that includes
initialization and writing final artifacts, so the absolute values of the two should not be treated
as like-for-like replacements.
