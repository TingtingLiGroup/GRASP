"""Network Portrait + Jensen-Shannon distance for transcript graphs.

This module computes pairwise similarity between per-cell/per-gene transcript
graphs using a network-portrait representation and Jensen-Shannon divergence.

Input
  A PKL that contains a DataFrame with transcript coordinates (typically the
  registered output with `df_registered`). The DataFrame is expected to contain:
    - cell, gene, x_c_s, y_c_s

Output
  A CSV of JS distances that can be used as an optional signal for selecting
  positive samples during training.
"""
## python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl/simulated_data1_data_dict.pkl --use_same_r --log_file /lustre/home/1910305118/data/GCN_CL/0_code/logs_simulated_data1/js_scipy_auto.log --visualize_top_n 0 --auto_params
## python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl/simulated_data1_data_dict.pkl --use_same_r --visualize_top_n 10 --log_file /lustre/home/1910305118/data/GCN_CL/0_code/logs_simulated_data1/js2.log
## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/seqfish_fibroblast_data_dict.pkl --use_same_r --log_file /lustre/home/1910305118/data/GCN_CL/0_code/logs_seqfish_fibroblast/js_scipy.log --visualize_top_n 0 --filter_pkl_file /lustre/home/1910305118/data/GCN_CL/5_graph_data/seqfish_fibroblast_cell171_gene2734_graph143789.pkl 2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/logs_seqfish_fibroblast/js_scipy_nohup.log &
## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/seqfish_cortex_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/logs_seqfish_fibroblast/js_scipy_cortex.log --visualize_top_n 0 2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/logs_seqfish_fibroblast/js_scipy_cortex_nohup.log &
## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/seqfish_cortex_data_dict_new.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/logs_seqfish_fibroblast/js_scipy_cortex2.log --visualize_top_n 0 --filter_pkl_file /lustre/home/1910305118/data/GCN_CL/5_graph_data/seqfish_cortex_new_cell708_gene93_graph708.pkl 2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/logs_seqfish_fibroblast/js_scipy_cortex_nohup2.log &
## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/seqfish_cortex_data_dict_new1.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/logs_seqfish_fibroblast/js_scipy_cortex_new1.log --output_dir /lustre/home/1910305118/data/GCN_CL/1_input/seqfish_cortex_sub19_portrait --visualize_top_n 0 --filter_pkl_file /lustre/home/1910305118/data/GCN_CL/5_graph_data/seqfish_cortex_new1_cell2405_gene92_graph2405.pkl 2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/logs_seqfish_fibroblast/js_scipy_cortex_new1.log &
## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/merscope_liver_data2_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/logs_merfish_liver/js_scipy_merfish_data3.log --output_dir /lustre/home/1910305118/data/GCN_CL/1_input/merfish_liver_data3_portrait --visualize_top_n 0 --filter_pkl_file /lustre/home/1910305118/data/GCN_CL/5_graph_data/merscope_liver_data3_cell281_gene139_graph13488.pkl 2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/logs_merfish_liver/js_scipy_merfish_data3.log &
## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/merscope_liver_data2_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/logs_merfish_liver/js_scipy_merfish_data4.log --output_dir /lustre/home/1910305118/data/GCN_CL/1_input/merfish_liver_data4_portrait --visualize_top_n 0 --filter_pkl_file /lustre/home/1910305118/data/GCN_CL/5_graph_data/merscope_liver_data4_cell919_gene176_graph43931.pkl 2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/logs_merfish_liver/js_scipy_merfish_data4.log &
## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/merscope_liver_data2_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_data4_central.log --output_dir /lustre/home/1910305118/data/GCN_CL/1_input/merfish_liver_data4_central_portrait --visualize_top_n 0 --filter_pkl_file /lustre/home/1910305118/data/GCN_CL/5_graph_data/merscope_liver_data4_central_cell182_gene106_graph9770.pkl 2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_data4_central.log &
## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/seqfish_cortex_data_dict_sub19.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/logs_seqfish_fibroblast/js_scipy_cortex_sub19.log --output_dir /lustre/home/1910305118/data/GCN_CL/1_input/seqfish_cortex_sub19_portrait --visualize_top_n 0  2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/logs_seqfish_fibroblast/js_scipy_cortex_sub19.log &
## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/seqfish_cortex_Astrocytes_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_seqfish_fibroblast/js_scipy_cortex_Astrocytes.log --filter_pkl_file seqfish_cortex_Astrocytes_cell528_gene22_graph528.pkl --output_dir /lustre/home/1910305118/data/GCN_CL/1_input/seqfish_cortex_Astrocytes_portrait --visualize_top_n 0  2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_seqfish_fibroblast/js_scipy_cortex_Astrocytes.log &
## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/seqfish_cortex_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_seqfish_fibroblast/js_scipy_seqfish_plus_all.log --output_dir /lustre/home/1910305118/data/GCN_CL/1_input/seqfish_plus_all_portrait --visualize_top_n 0  2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_seqfish_fibroblast/js_scipy_seqfish_plus_all.log &
## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/merscope_liver_data2_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_data4_protal.log --output_dir /lustre/home/1910305118/data/GCN_CL/1_input/merfish_liver_data4_protal_portrait --visualize_top_n 0 --filter_pkl_file /lustre/home/1910305118/data/GCN_CL/5_graph_data/merscope_liver_data4_protal_cell454_gene124_graph21222.pkl 2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_data4_protal.log &

## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/merscope_liver_data2_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_data4_central_bigger.log --output_dir /lustre/home/1910305118/data/GCN_CL/1_input/merfish_liver_data4_central_bigger_portrait --visualize_top_n 0 --filter_pkl_file /lustre/home/1910305118/data/GCN_CL/5_graph_data/merscope_liver_data_central_cell870_gene124_graph45025.pkl 2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_data4_central_bigger.log &
## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/merscope_liver_data2_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_data4_protal_bigger.log --output_dir /lustre/home/1910305118/data/GCN_CL/1_input/merfish_liver_data4_protal_bigger_portrait --visualize_top_n 0 --filter_pkl_file /lustre/home/1910305118/data/GCN_CL/5_graph_data/merscope_liver_data_protal_cell1713_gene143_graph79975.pkl 2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_data4_protal_bigger.log &

## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/merscope_intestine_Enterocyte_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_merfish_intestine/js_scipy_merfish_Enterocyte.log --output_dir /lustre/home/1910305118/data/GCN_CL/1_input/merfish_intestine_Enterocyte_portrait --visualize_top_n 0 --filter_pkl_file /lustre/home/1910305118/data/GCN_CL/5_graph_data/merscope_intestine_Enterocyte_cell419_gene17_graph905.pkl 2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_merfish_intestine/js_scipy_merfish_Enterocyte.log &


## nohup python portrait.py --pkl_file /lustre/home/1910305118/data/GCN_CL/1_input/pkl_data/simulated1_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_simulated1/js_scipy_simulated1.log --output_dir /lustre/home/1910305118/data/GCN_CL/6_analysis/js_portrait/simulated1_portrait --visualize_top_n 0 --filter_pkl_file /lustre/home/1910305118/data/GCN_CL/5.1_graph_data/simulated1_cell10_gene80_graph800_weight.pkl 2>&1 > /lustre/home/1910305118/data/GCN_CL/0_code/0.5_logs/logs_simulated1/js_scipy_simulated1.log &

## nohup python portrait.py --pkl_file /home/lixiangyu/hyy/GRASP/1_input/pkl_data/seqfish_fibroblast_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_seqfish_fibroblast/js_scipy.log --output_dir /home/lixiangyu/hyy/GRASP/6_analysis/js_portrait/seqfish_fibroblast_portrait --visualize_top_n 0 --filter_pkl_file /home/lixiangyu/hyy/GRASP/5.1_graph_data/seqfish_fibroblast_cell171_gene59_graph8068.pkl 2>&1 > /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_seqfish_fibroblast/js_scipy_nohup.log &

## nohup python portrait.py --pkl_file /home/lixiangyu/hyy/GRASP/1_input/pkl_data/simulated3_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_simulated/js_scipy_simulated3.log --output_dir /home/lixiangyu/hyy/GRASP/6_analysis/js_portrait/simulated3_portrait --visualize_top_n 0 --filter_pkl_file /home/lixiangyu/hyy/GRASP/5.1_graph_data/simulated3_original_cell50_gene400_graph15000.pkl 2>&1 > /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_simulated/js_scipy_simulated3.log &

## nohup python portrait.py --pkl_file /home/lixiangyu/hyy/GRASP/1_input/pkl_data/merscope_liver_data_region1_portal_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_region1_portal.log --output_dir /home/lixiangyu/hyy/GRASP/6_analysis/js_portrait/merfish_liver_region1_portal --visualize_top_n 0 --filter_pkl_file /home/lixiangyu/hyy/GRASP/5_graph_data/merscope_liver_data_region1_portal_cell1708_gene143_graph79172.pkl 2>&1 > /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_region1_portal.log &

## nohup python portrait.py --pkl_file /home/lixiangyu/hyy/GRASP/1_input/pkl_data/merscope_liver_data_region1_central_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_region1_central.log --output_dir /home/lixiangyu/hyy/GRASP/6_analysis/js_portrait/merfish_liver_region1_central --visualize_top_n 0 --filter_pkl_file /home/lixiangyu/hyy/GRASP/5_graph_data/merscope_liver_data_region1_central_cell1711_gene136_graph86074.pkl 2>&1 > /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_region1_central.log &

## nohup python portrait.py --pkl_file /home/lixiangyu/hyy/GRASP/1_input/pkl_data/merscope_liver_data_region2_central_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_region2_central.log --output_dir /home/lixiangyu/hyy/GRASP/6_analysis/js_portrait/merfish_liver_region2_central --visualize_top_n 0 --filter_pkl_file /home/lixiangyu/hyy/GRASP/5_graph_data/merscope_liver_data_region2_central_cell936_gene126_graph44910.pkl 2>&1 > /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_region2_central_nohup.log &

## nohup python portrait.py --pkl_file /home/lixiangyu/hyy/GRASP/1_input/pkl_data/merscope_liver_data_region2_portal_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_region2_portal.log --output_dir /home/lixiangyu/hyy/GRASP/6_analysis/js_portrait/merfish_liver_region2_portal --visualize_top_n 0 --filter_pkl_file /home/lixiangyu/hyy/GRASP/5_graph_data/merscope_liver_data_region2_portal_cell747_gene124_graph33523.pkl 2>&1 > /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_liver/js_scipy_merfish_region2_portal_nohup.log &


## nohup python portrait.py --pkl_file /home/lixiangyu/hyy/GRASP/1_input/pkl_data/merfish_intestine_Enterocyte_resegment_new_data_dict.pkl --use_same_r --max_count 30 --auto_params --log_file /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_intestine/js_scipy_enterocyte_resegment_new.log --output_dir /home/lixiangyu/hyy/GRASP/6_analysis/js_portrait/merfish_intestine_enterocyte_resegment_new_portrait --visualize_top_n 0 --filter_pkl_file /home/lixiangyu/hyy/GRASP/5_graph_data/merfish_intestine_Enterocyte_resegment_new_cell688_gene58_graph4331.pkl 2>&1 > /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_intestine/js_scipy_enterocyte_resegment_new_nohup.log &


# # ---------- Group 1 ----------
# nohup python portrait.py \
#   --pkl_file /home/lixiangyu/hyy/GRASP/1_input/pkl_data/merfish_u2os_data_dict.pkl \
#   --use_same_r --max_count 30 --auto_params \
#   --log_file /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_u2os/js_group1.log \
#   --output_dir /home/lixiangyu/hyy/GRASP/6_analysis/js_portrait/merfish_u2os_group1_portrait \
#   --visualize_top_n 0 \
#   --filter_pkl_file /home/lixiangyu/hyy/GRASP/5_graph_data/merfish_u2os_cell634_gene25_graph1000.pkl \
#   2>&1 > /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_u2os/js_group1_nohup.log &


# # ---------- Group 2 ----------
# nohup python portrait.py \
#   --pkl_file /home/lixiangyu/hyy/GRASP/1_input/pkl_data/merfish_u2os_data_dict.pkl \
#   --use_same_r --max_count 30 --auto_params \
#   --log_file /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_u2os/js_group2.log \
#   --output_dir /home/lixiangyu/hyy/GRASP/6_analysis/js_portrait/merfish_u2os_group2_portrait \
#   --visualize_top_n 0 \
#   --filter_pkl_file /home/lixiangyu/hyy/GRASP/5_graph_data/merfish_u2os_cell621_gene25_graph1000.pkl \
#   2>&1 > /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_u2os/js_group2_nohup.log &


# # ---------- Group 3 ----------
# nohup python portrait.py \
#   --pkl_file /home/lixiangyu/hyy/GRASP/1_input/pkl_data/merfish_u2os_data_dict.pkl \
#   --use_same_r --max_count 30 --auto_params \
#   --log_file /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_u2os/js_group3.log \
#   --output_dir /home/lixiangyu/hyy/GRASP/6_analysis/js_portrait/merfish_u2os_group3_portrait \
#   --visualize_top_n 0 \
#   --filter_pkl_file /home/lixiangyu/hyy/GRASP/5_graph_data/merfish_u2os_cell629_gene25_graph1000.pkl \
#   2>&1 > /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_u2os/js_group3_nohup.log &


# # ---------- Group 4 ----------
# nohup python portrait.py \
#   --pkl_file /home/lixiangyu/hyy/GRASP/1_input/pkl_data/merfish_u2os_data_dict.pkl \
#   --use_same_r --max_count 30 --auto_params \
#   --log_file /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_u2os/js_group4.log \
#   --output_dir /home/lixiangyu/hyy/GRASP/6_analysis/js_portrait/merfish_u2os_group4_portrait \
#   --visualize_top_n 0 \
#   --filter_pkl_file /home/lixiangyu/hyy/GRASP/5_graph_data/merfish_u2os_cell621_gene9_graph947.pkl \
#   2>&1 > /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_u2os/js_group4_nohup.log &


# # ---------- Group 5 ----------
# nohup python portrait.py \
#   --pkl_file /home/lixiangyu/hyy/GRASP/1_input/pkl_data/merfish_u2os_data_dict.pkl \
#   --use_same_r --max_count 30 --auto_params \
#   --log_file /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_u2os/js_group5.log \
#   --output_dir /home/lixiangyu/hyy/GRASP/6_analysis/js_portrait/merfish_u2os_group5_portrait \
#   --visualize_top_n 0 \
#   --filter_pkl_file /home/lixiangyu/hyy/GRASP/5_graph_data/merfish_u2os_cell989_gene25_graph23242.pkl \
#   2>&1 > /home/lixiangyu/hyy/GRASP/0_code/0.5_logs/logs_merfish_u2os/js_group5_nohup.log &

## Legacy note: derived from utils_code/portrait.py
import pandas as pd
import numpy as np
import networkx as nx

from scipy.spatial import distance_matrix
import time
from datetime import datetime
import os
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import pickle
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import KDTree
import random
import sys

# Matplotlib defaults (keep output deterministic and readable)
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("js_distance")

# Warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def find_r_for_isolated_threshold(
    df, threshold=0.05, r_min=0.01, r_max=0.6, step=0.03, verbose=False, dists=None
):
    """Find a connection radius that keeps isolated-node ratio under a threshold.

    This implementation uses a KDTree to approximate an appropriate radius based
    on nearest-neighbor distances.
    """
    positions = df[["x_c_s", "y_c_s"]].values
    N = len(df)

    # Edge cases
    if N <= 1:
        # Single point (or empty): return a small default radius.
        return min(r_min * 5, r_max * 0.1)

    # Compute nearest-neighbor distances via KDTree.
    tree = KDTree(positions)

    # Use k=2 because the first neighbor is the point itself.
    dists, _ = tree.query(positions, k=2)
    nn_dists = dists[:, 1]

    # Percentile: at most `threshold` fraction can be isolated.
    r = np.percentile(nn_dists, 100 * (1 - threshold))

    # Clamp to the configured range.
    r = max(r_min, min(r, r_max))

    if verbose:
        logger.debug(
            f"KDTree-based r={r:.4f} (percentile={100 * (1 - threshold):.1f} of NN distances)"
        )
        # Sanity-check the actual isolated ratio.
        isolated_count = np.sum(nn_dists > r)
        logger.debug(f"isolated_ratio={isolated_count / N:.4f} ({isolated_count}/{N})")

    return r


def build_weighted_graph(df, r, dists=None):
    """Build a weighted graph from transcript coordinates.

    Nodes are transcripts and edges connect pairs within radius `r`.
    Edge weights are Euclidean distances.
    """
    G = nx.Graph()
    positions = df[["x_c_s", "y_c_s"]].values

    # Add nodes.
    for i, pos in enumerate(positions):
        G.add_node(i, pos=pos)

    # Compute pairwise distances if not provided.
    if dists is None:
        dists = distance_matrix(positions, positions)

    # Add edges within radius.
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            dist = dists[i, j]
            if dist <= r:
                G.add_edge(i, j, weight=dist)

    return G


def get_network_portrait(G, bin_size=0.01, use_vectorized=True):
    """Compute a network portrait for a weighted graph."""

    # Node count
    n_nodes = len(G)
    if n_nodes <= 1:
        return {}, n_nodes

    # Implementation choice
    if use_vectorized:
        # Vectorized APSP via SciPy sparse shortest_path.
        rows, cols, weights = [], [], []
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1.0)
            rows.append(u)
            cols.append(v)
            weights.append(w)
            rows.append(v)
            cols.append(u)
            weights.append(w)

        A = csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))

        dist_mat = shortest_path(A, directed=False, unweighted=False, method="auto")

        degs = np.array([d for _, d in sorted(G.degree(), key=lambda x: x[0])])

        # 4) Exclude self-pairs and keep all i != j pairs.
        #    dist_mat is an n x n numpy array.
        i_idx, j_idx = np.nonzero(~np.eye(n_nodes, dtype=bool))
        dists = dist_mat[i_idx, j_idx]
        src_degs = degs[i_idx]

        # 5) Bin distances.
        bins = np.floor(dists / bin_size).astype(int)

        # 6) Vectorized counting.
        combined = np.column_stack([bins, src_degs])
        unique_ck, counts = np.unique(combined, axis=0, return_counts=True)

        # 7) Convert back to a dict.
        portrait = {
            (int(bin_l), int(deg)): int(cnt)
            for (bin_l, deg), cnt in zip(unique_ck, counts)
        }
    else:
        # Loop implementation: slower, but uses less memory.
        # All-pairs shortest path lengths.
        length_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        # Node degrees.
        degrees = dict(G.degree())

        portrait = {}

        for i in length_dict:
            for j in length_dict[i]:
                if i == j:
                    continue
                dist = length_dict[i][j]
                bin_l = int(dist // bin_size)
                deg = degrees[i]
                key = (bin_l, deg)
                portrait[key] = portrait.get(key, 0) + 1

    return portrait, n_nodes


def compute_weighted_distribution(portrait, N):
    """Convert a portrait count dict into a weighted probability distribution."""

    total_pairs = N * N
    dist = {}

    for (l, k), count in portrait.items():
        dist[(l, k)] = (k * count) / total_pairs

    return dist


def js_divergence(P, Q):
    """Compute Jensen-Shannon divergence between two distributions."""
    keys = set(P.keys()).union(Q.keys())
    p_vec = np.array([P.get(k, 0.0) for k in keys])
    q_vec = np.array([Q.get(k, 0.0) for k in keys])
    m_vec = 0.5 * (p_vec + q_vec)

    def safe_kl(p, q):
        mask = (p > 0) & (q > 0)
        return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

    return 0.5 * safe_kl(p_vec, m_vec) + 0.5 * safe_kl(q_vec, m_vec)


def plot_portrait(portrait, title="Network Portrait", save_path=None):
    """Plot a 2D heatmap for a network portrait."""
    if not portrait:
        logger.warning("Empty portrait; skip plotting")
        return

    df = pd.DataFrame([{"l": l, "k": k, "value": v} for (l, k), v in portrait.items()])

    pivot = df.pivot(index="l", columns="k", values="value").fillna(0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, cmap="viridis", annot=False)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Degree k", fontsize=12)
    plt.ylabel("Path length bin l", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Portrait saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_graph(G, title="Graph structure", save_path=None):
    """Plot the graph layout using stored node positions."""
    pos = nx.get_node_attributes(G, "pos")

    plt.figure(figsize=(8, 8))
    nx.draw(
        G,
        pos,
        node_size=30,
        node_color="skyblue",
        edge_color="gray",
        width=0.5,
        with_labels=False,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Graph saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def compare_graphs(
    df1, df2, r=None, bin_size=0.01, show_plots=True, save_dir=None, use_vectorized=True
):
    """Compare two transcript graphs and compute JS divergence."""

    # Auto-select radius if not provided.
    if r is None:
        r1 = find_r_for_isolated_threshold(df1, threshold=0.05)
        r2 = find_r_for_isolated_threshold(df2, threshold=0.05)
        r = max(r1, r2)
        logger.info(f"Auto-selected connection radius r = {r:.2f}")

    # Build graphs.
    G1 = build_weighted_graph(df1, r)
    G2 = build_weighted_graph(df2, r)

    # Compute network portraits.
    B1, N1 = get_network_portrait(G1, bin_size, use_vectorized)
    B2, N2 = get_network_portrait(G2, bin_size, use_vectorized)

    # Compute weighted distributions.
    P = compute_weighted_distribution(B1, N1)
    Q = compute_weighted_distribution(B2, N2)

    # Compute JS divergence.
    js = js_divergence(P, Q)
    logger.info(f"Jensen-Shannon divergence between two graphs: {js:.4f}")

    # Visualization.
    if show_plots or save_dir:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        if show_plots:
            plot_graph(G1, title="Graph 1 structure")
            plot_graph(G2, title="Graph 2 structure")
            plot_portrait(B1, title="Graph 1 network portrait")
            plot_portrait(B2, title="Graph 2 network portrait")

        if save_dir:
            plot_graph(
                G1,
                title="Graph 1 structure",
                save_path=f"{save_dir}/graph1_structure.png",
            )
            plot_graph(
                G2,
                title="Graph 2 structure",
                save_path=f"{save_dir}/graph2_structure.png",
            )
            plot_portrait(
                B1,
                title="Graph 1 network portrait",
                save_path=f"{save_dir}/graph1_portrait.png",
            )
            plot_portrait(
                B2,
                title="Graph 2 network portrait",
                save_path=f"{save_dir}/graph2_portrait.png",
            )

    return js


def find_gene_optimal_r(
    gene,
    df,
    cell_list,
    threshold=0.05,
    r_min=0.01,
    r_max=0.6,
    r_step=0.03,
    dist_dict=None,
):
    """
    Find an optimal r value for a gene (use the max r across cells).

    Args:
        gene: Gene ID.
        df: DataFrame containing transcripts.
        cell_list: List of cell IDs.
        threshold: Isolated-node ratio threshold.
        r_min, r_max, r_step: Search parameters for r.
        dist_dict: Optional precomputed distance matrices {(cell, gene): dist_matrix}.

    Returns:
        tuple: (optimal r for gene, dict {cell: dist_matrix})
    """
    logger.info(f"Computing optimal r for gene {gene}")

    # All transcripts for this gene.
    gene_df = df[df["gene"] == gene]

    cell_r_values = {}
    cell_dist_matrices = {}  # store distance matrix per cell
    transcript_counts = {}  # transcript count per cell

    # Compute r per cell.
    for cell in cell_list:
        cell_df = gene_df[gene_df["cell"] == cell]
        transcript_count = len(cell_df)
        transcript_counts[cell] = transcript_count

        # If there is only one transcript, use a small default radius.
        if transcript_count == 1:
            # Avoid connecting to far-away points.
            r_single = min(r_min * 5, r_max * 0.1)  # min(5*r_min, 0.1*r_max)
            cell_r_values[cell] = r_single
            # Single-point distance matrix.
            cell_dist_matrices[cell] = np.array([[0.0]])
            continue

        # Skip empty cells.
        if transcript_count == 0:
            continue

        try:
            # Prefer precomputed distance matrix.
            dists = dist_dict.get((cell, gene), None) if dist_dict else None

            # Compute if missing.
            if dists is None:
                positions = cell_df[["x_c_s", "y_c_s"]].values
                dists = distance_matrix(positions, positions)

            # Find optimal r for this cell.
            r = find_r_for_isolated_threshold(
                cell_df,
                threshold,
                r_min,
                r_max,
                r_step,
                verbose=False,
                dists=dists,
            )
            cell_r_values[cell] = r
            cell_dist_matrices[cell] = dists  # cache for reuse
        except Exception as e:
            logger.error(f"Failed to compute r for cell={cell}, gene={gene}: {e}")

    # Summary stats.
    total_cells = len(cell_list)
    valid_cells = len(cell_r_values)
    single_transcript_cells = sum(
        1 for count in transcript_counts.values() if count == 1
    )
    multi_transcript_cells = sum(1 for count in transcript_counts.values() if count > 1)
    zero_transcript_cells = sum(1 for count in transcript_counts.values() if count == 0)

    # If no r values are valid, return a safer default.
    if not cell_r_values:
        # Use a smaller default to avoid overly dense graphs.
        default_r = min(r_max * 0.3, r_min * 10)
        logger.warning(
            f"Gene {gene} has no valid r; using adjusted default {default_r:.4f} (original default: {r_max})"
        )
        logger.warning(
            f"Gene {gene} transcript stats: total_cells={total_cells}, zero={zero_transcript_cells}, one={single_transcript_cells}, multi={multi_transcript_cells}"
        )
        return default_r, {}

    # Return max r across cells and the distance-matrix cache.
    max_r = max(cell_r_values.values())
    min_r = min(cell_r_values.values())
    avg_r = np.mean(list(cell_r_values.values()))

    logger.info(
        f"Gene {gene} optimal r: {max_r:.2f} (from {valid_cells} cells, range: {min_r:.2f}-{max_r:.2f}, mean: {avg_r:.2f})"
    )
    logger.info(
        f"Gene {gene} transcript distribution: zero={zero_transcript_cells}, one={single_transcript_cells}, multi={multi_transcript_cells}"
    )

    return max_r, cell_dist_matrices


def precompute_portraits_for_gene(
    gene,
    df,
    cell_list,
    threshold=0.05,
    bin_size=0.01,
    r_min=0.01,
    r_max=0.6,
    r_step=0.03,
    use_same_r=True,
    use_vectorized=True,
    dist_dict=None,
):
    """
    Precompute network portraits for all cells of a gene.

    Args:
        gene: Gene identifier.
        df: Transcript DataFrame.
        cell_list: List of cell IDs.
        threshold: Isolated node ratio threshold.
        bin_size: Path length bin size.
        r_min, r_max, r_step: Radius search parameters.
        use_same_r: Whether to use a shared r for all cells within a gene.
        use_vectorized: Whether to use vectorized portrait computation.
        dist_dict: Optional precomputed distance matrices {(cell, gene): dist_matrix}.

    Returns:
        Dict: {(cell, gene): (weighted_distribution, node_count, r_value)}
    """
    distributions = {}

    logger.info(f"Start precomputing network portraits for gene {gene}")
    start_time = time.time()

    # Filter transcripts for this gene.
    gene_df = df[df["gene"] == gene]

    # Ensure there is data.
    if len(gene_df) == 0:
        logger.warning(f"Gene {gene} has no transcript records")
        return distributions

    # If using a shared r, compute gene-level r first (and reuse distance matrices).
    gene_r = None
    cell_dist_matrices = {}
    if use_same_r:
        gene_r, cell_dist_matrices = find_gene_optimal_r(
            gene, df, cell_list, threshold, r_min, r_max, r_step, dist_dict
        )

    # Process each cell.
    for cell in tqdm(
        cell_list,
        desc=f"Processing cells for gene {gene}",
        leave=False,
        disable=not sys.stdout.isatty(),
    ):
        # Filter transcripts for this cell.
        cell_df = gene_df[gene_df["cell"] == cell]
        transcript_count = len(cell_df)

        # Skip cells with no transcripts.
        if transcript_count == 0:
            logger.debug(f"Cell {cell}, gene {gene} has no transcripts; skipping")
            continue

        # Handle the single-transcript case.
        if transcript_count == 1:
            logger.debug(
                f"Cell {cell}, gene {gene} has 1 transcript; using a single-node portrait"
            )
            try:
                # Special handling for a single-node graph.
                # The portrait contains one node with degree=0 at path length=0.
                # In (l, k) bins: l=0 (self distance), k=0 (degree=0).
                single_portrait = {(0, 0): 1}  # One entry: (path_length=0, degree=0)
                weighted_dist = compute_weighted_distribution(single_portrait, 1)

                # Use a reasonable r.
                r = gene_r if use_same_r else min(r_min * 5, r_max * 0.1)

                distributions[(cell, gene)] = (weighted_dist, 1, r)
                continue
            except Exception as e:
                logger.error(
                    f"Failed single-transcript handling cell={cell}, gene={gene}: {e}"
                )
                continue

        # Multi-transcript case (original logic).
        try:
            # Prefer cell_dist_matrices, then fall back to global dist_dict.
            dists = cell_dist_matrices.get(cell, None)
            if dists is None and dist_dict:
                dists = dist_dict.get((cell, gene), None)

            # Compute distance matrix if missing.
            if dists is None:
                positions = cell_df[["x_c_s", "y_c_s"]].values
                dists = distance_matrix(positions, positions)

            # Use gene-level r, or find an r for this (cell, gene) pair.
            r = gene_r
            if not use_same_r:
                r = find_r_for_isolated_threshold(
                    cell_df,
                    threshold=threshold,
                    r_min=r_min,
                    r_max=r_max,
                    step=r_step,
                    dists=dists,
                )

            # Build graph (reuse the distance matrix).
            G = build_weighted_graph(cell_df, r, dists=dists)

            # Compute portrait and weighted distribution.
            portrait, N = get_network_portrait(G, bin_size, use_vectorized)
            weighted_dist = compute_weighted_distribution(portrait, N)

            distributions[(cell, gene)] = (weighted_dist, N, r)

        except Exception as e:
            logger.error(f"Failed processing cell={cell}, gene={gene}: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"Finished precomputing network portraits for gene {gene}: {len(distributions)} distributions, elapsed {elapsed:.2f}s"
    )

    return distributions


def find_js_distances_for_gene(
    gene, df, cell_list, portraits, bin_size=0.01, max_count=None, transcript_window=30
):
    """
    Compute JS divergence for cell pairs within a gene.

    Args:
        gene: Gene identifier.
        df: DataFrame containing transcripts.
        cell_list: List of cells.
        portraits: Precomputed portraits {(cell, gene): (weighted_distribution, node_count, r_value)}.
        bin_size: Path length bin size.
        max_count: Max comparisons per target cell.
        transcript_window: Candidate window on transcript count difference.

    Returns:
        List: JS divergence results.
    """
    logger.info(f"Computing JS divergences for gene {gene}")
    start_time = time.time()

    # Collect valid cells and transcript counts for this gene.
    valid_cells = []
    transcript_counts = {}
    r_values = {}

    for cell in cell_list:
        key = (cell, gene)
        if key in portraits:
            valid_cells.append(cell)
            _, N, r = portraits[key]
            transcript_counts[cell] = N
            r_values[cell] = r

    logger.info(f"Gene {gene}: {len(valid_cells)} valid cells")

    # Compute JS distances.
    all_distances = []
    processed_count = 0

    for i, target_cell in enumerate(valid_cells):
        target_transcript_count = transcript_counts[target_cell]
        target_r = r_values[target_cell]

        # Candidate cells with similar transcript counts.
        candidates = []
        for j, cell in enumerate(valid_cells):
            if cell != target_cell:
                transcript_diff = abs(transcript_counts[cell] - target_transcript_count)
                if transcript_diff <= transcript_window:
                    candidates.append((cell, transcript_diff))

        # Sort by transcript count difference.
        candidates.sort(key=lambda x: x[1])

        # Limit comparisons.
        if max_count is not None and len(candidates) > max_count:
            candidates = candidates[:max_count]

        # Compute JS divergence.
        for cell, transcript_diff in candidates:
            try:
                target_key = (target_cell, gene)
                other_key = (cell, gene)

                if target_key in portraits and other_key in portraits:
                    target_dist, _, _ = portraits[target_key]
                    other_dist, _, other_r = portraits[other_key]

                    js_distance = js_divergence(target_dist, other_dist)

                    all_distances.append(
                        (
                            target_cell,
                            gene,
                            cell,
                            gene,
                            transcript_counts[cell],
                            js_distance,
                            transcript_diff,
                            target_r,
                            other_r,
                        )
                    )

                    processed_count += 1
                    if processed_count % 100 == 0:
                        logger.debug(
                            f"Gene {gene}: computed {processed_count} JS distances"
                        )
            except Exception as e:
                logger.error(f"Failed JS divergence for {target_cell}-{cell}: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"Finished JS divergences for gene {gene}: {len(all_distances)} results, elapsed {elapsed:.2f}s"
    )

    return all_distances


def calculate_js_distances(
    pkl_file: str,
    output_dir: str = None,
    max_count: int = None,
    transcript_window: int = 30,
    bin_size: float = 0.01,
    threshold: float = 0.05,
    r_min: float = 0.01,
    r_max: float = 0.6,
    r_step: float = 0.03,
    num_threads: int = None,
    use_same_r: bool = True,
    visualize_top_n: int = 5,
    use_vectorized: bool = True,
    filter_pkl_file: str = None,
    auto_params: bool = False,
    n_bins: int = 50,
    min_percentile: float = 1.0,
    max_percentile: float = 99.0,
) -> pd.DataFrame:
    """
    Compute Jensen-Shannon divergence between transcript graphs, with optional auto r selection.

    Args:
        pkl_file: Input PKL path containing df_registered.
        output_dir: Output directory; if None, infer a default.
        max_count: Max comparisons per target cell.
        transcript_window: Candidate window on transcript count difference.
        bin_size: Path length bin size; use 'auto' to infer.
        threshold: Isolated-node ratio threshold.
        r_min: Minimum r for search; use 'auto' to infer.
        r_max: Maximum r for search; use 'auto' to infer.
        r_step: Step size for r search.
        num_threads: Thread pool size.
        use_same_r: Use a single r per gene (max across cells).
        visualize_top_n: Visualize top-N most similar pairs.
        use_vectorized: Use vectorized portrait computation.
        filter_pkl_file: Optional PKL to filter (cell, gene) pairs.
        auto_params: Auto-set r_min/r_max/bin_size based on distances.
        n_bins: Bin count for auto bin_size.
        min_percentile: Percentile used to infer r_min.
        max_percentile: Percentile used to infer r_max.

    Returns:
        pd.DataFrame: JS divergence results.
    """
    # Default thread count.
    if num_threads is None:
        import multiprocessing

        num_threads = min(multiprocessing.cpu_count() - 2, 20)

    # Auto parameter flags.
    auto_r_min = r_min == "auto" or auto_params
    auto_r_max = r_max == "auto" or auto_params
    auto_bin_size = bin_size == "auto" or auto_params

    # Use None placeholders for auto-calculated params.
    if auto_r_min:
        r_min = None
    else:
        r_min = float(r_min)

    if auto_r_max:
        r_max = None
    else:
        r_max = float(r_max)

    if auto_bin_size:
        bin_size = None
    else:
        bin_size = float(bin_size)

    logger.info(f"Computing JS divergence with {num_threads} threads")
    if auto_params or auto_r_min or auto_r_max or auto_bin_size:
        logger.info("Auto-calculating r_min, r_max, and bin_size")
    else:
        logger.info(
            f"r search params: threshold={threshold}, r_min={r_min}, r_max={r_max}, r_step={r_step}"
        )
        logger.info(f"Path length bin_size: {bin_size}")

    logger.info(
        f"{'Use a single r per gene' if use_same_r else 'Use per (cell, gene) r'}"
    )
    logger.info(
        f"{'Use vectorized portrait computation' if use_vectorized else 'Use loop-based portrait computation'}"
    )

    start_time = time.time()

    # Load data.
    with open(pkl_file, "rb") as f:
        data_dict = pickle.load(f)

    # Validate required key.
    if "df_registered" not in data_dict:
        raise ValueError(f"PKL file {pkl_file} does not contain df_registered")

    df = data_dict["df_registered"]
    logger.info(f"Loaded {len(df)} transcript records")

    # Validate required columns.
    required_cols = ["cell", "gene", "x_c_s", "y_c_s"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"df_registered is missing required columns: {missing_cols}")

    # Optionally filter (cell, gene) pairs.
    if filter_pkl_file and os.path.exists(filter_pkl_file):
        logger.info(f"Filtering (cell, gene) pairs using PKL file: {filter_pkl_file}")
        try:
            with open(filter_pkl_file, "rb") as f:
                filter_data = pickle.load(f)

            # Extract cell_labels and gene_labels.
            if "cell_labels" in filter_data and "gene_labels" in filter_data:
                # Require equal lengths to form (cell, gene) pairs.
                cell_labels = filter_data["cell_labels"]
                gene_labels = filter_data["gene_labels"]

                if len(cell_labels) == len(gene_labels):
                    # Build set of (cell, gene) pairs.
                    cell_gene_pairs = set(zip(cell_labels, gene_labels))
                    logger.info(
                        f"Extracted {len(cell_gene_pairs)} unique (cell, gene) pairs from filter file"
                    )

                    # Add temp (cell, gene) key for filtering.
                    df["cell_gene_pair"] = list(zip(df["cell"], df["gene"]))

                    # Filter df_registered to keep only pairs from the filter file.
                    original_len = len(df)
                    df = df[df["cell_gene_pair"].isin(cell_gene_pairs)]

                    # Drop temp column.
                    df = df.drop(columns=["cell_gene_pair"])

                    logger.info(
                        f"Filtered records: before={original_len}, after={len(df)}"
                    )

                    if len(df) == 0:
                        logger.warning(
                            "No records remain after filtering; check whether (cell, gene) pairs match"
                        )
                        return pd.DataFrame()
                else:
                    logger.warning(
                        f"Filter PKL has mismatched lengths: cell_labels={len(cell_labels)} vs gene_labels={len(gene_labels)}; cannot form exact pairs"
                    )
                    logger.info("Proceeding with all cells and genes")
            else:
                logger.warning(
                    f"Filter PKL file {filter_pkl_file} does not contain cell_labels or gene_labels"
                )
        except Exception as e:
            logger.error(f"Failed to read filter PKL file: {e}")
            logger.info("Proceeding with all cells and genes")

    # Unique cells and genes.
    cell_list = sorted(df["cell"].unique())
    gene_list = sorted(df["gene"].unique())

    logger.info(f"Dataset contains {len(cell_list)} cells and {len(gene_list)} genes")

    # Resolve output directory.
    if output_dir is None:
        # Try to infer the dataset name from the PKL filename.
        filename = os.path.basename(pkl_file)
        # Drop common suffixes, e.g. "_data_dict.pkl".
        dataset = filename.split("_data_dict")[0]
        if dataset == filename:  # No "_data_dict" suffix found.
            # Fall back to stripping the extension.
            dataset = os.path.splitext(filename)[0]

        logger.info(f"Derived dataset name from filename {filename}: {dataset}")

        # Determine data directory: search upward for a "GRASP" directory.
        pkl_abs_path = os.path.abspath(pkl_file)
        # Find the "GRASP" directory as project root.
        grasp_dir = None
        path_parts = pkl_abs_path.split(os.sep)
        for i, part in enumerate(path_parts):
            if part == "GRASP":
                grasp_dir = os.sep.join(path_parts[: i + 1])
                break

        if grasp_dir:
            # Under GRASP, search for a dataset directory (e.g. data1_simulated1).
            # Naming rule: directory name contains the dataset identifier.
            data_dir = None
            for item in os.listdir(grasp_dir):
                item_path = os.path.join(grasp_dir, item)
                if os.path.isdir(item_path) and dataset in item:
                    data_dir = item_path
                    break

            if data_dir:
                output_dir = os.path.join(data_dir, "step2_js")
                logger.info(f"Using data directory: {data_dir}")
            else:
                # If no matching data directory, use the PKL directory.
                pkl_dir = os.path.dirname(pkl_abs_path)
                output_dir = os.path.join(pkl_dir, "step2_js")
                logger.info(
                    f"No matching data directory found; using PKL directory: {pkl_dir}"
                )
        else:
            # If no GRASP directory found, use the PKL directory.
            pkl_dir = os.path.dirname(pkl_abs_path)
            output_dir = os.path.join(pkl_dir, "step2_js")
            logger.info(
                f"GRASP directory not found in path; using PKL directory: {pkl_dir}"
            )

    os.makedirs(output_dir, exist_ok=True)

    # Create visualization directory.
    vis_dir = f"{output_dir}/visualization"
    os.makedirs(vis_dir, exist_ok=True)

    # Analyze transcript distribution (optional diagnostics).
    logger.info("Analyzing transcript distribution (optional diagnostics)...")
    try:
        analyze_transcript_distribution(df, output_dir)
    except Exception as e:
        logger.warning(f"Transcript distribution analysis failed: {e}")

    # Stage 0: precompute all distance matrices globally.
    logger.info("Stage 0: precomputing all distance matrices")
    dist_dict = {}  # Global distance matrix dict {(cell, gene): dist_matrix}

    # Precompute all distance matrices using a thread pool.
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}

        # Submit all computation tasks.
        for gene in gene_list:
            gene_df = df[df["gene"] == gene]

            for cell in cell_list:
                cell_df = gene_df[gene_df["cell"] == cell]
                # Skip cells with insufficient transcripts.
                if len(cell_df) <= 1:
                    continue

                # Define a local function to compute distance matrices.
                def calc_dist_matrix(c_df):
                    positions = c_df[["x_c_s", "y_c_s"]].values
                    return distance_matrix(positions, positions)

                # Submit task.
                futures[(cell, gene)] = executor.submit(calc_dist_matrix, cell_df)

        # Collect results.
        for (cell, gene), future in tqdm(
            futures.items(),
            desc="Precomputing distance matrices",
            disable=not sys.stdout.isatty(),
        ):
            try:
                dist_dict[(cell, gene)] = future.result()
            except Exception as e:
                logger.error(
                    f"Failed to compute distance matrix for cell={cell}, gene={gene}: {e}"
                )

    logger.info(f"Distance matrix precompute done; {len(dist_dict)} (cell,gene) pairs")

    # Auto-select parameters based on precomputed distance matrices.
    if auto_r_min or auto_r_max or auto_bin_size:
        logger.info("Auto-selecting parameters from precomputed distance matrices")
        all_dists = []

        # Collect all non-zero distances (no sampling).
        for key, dmat in tqdm(
            dist_dict.items(),
            desc="Collecting distance samples",
            disable=not sys.stdout.isatty(),
        ):
            # Upper triangle (exclude self distances).
            triu_indices = np.triu_indices_from(dmat, k=1)
            dists = dmat[triu_indices]
            # Exclude zero and infinite distances.
            valid_dists = dists[(dists > 0) & (np.isfinite(dists))]
            if len(valid_dists) > 0:
                all_dists.append(valid_dists)

        if all_dists:
            all_dists = np.concatenate(all_dists)
            logger.info(f"Collected {len(all_dists)} valid distance values")

            # Percentile-based parameter estimation.
            if auto_r_min:
                r_min = float(np.percentile(all_dists, min_percentile))
                r_min = round(r_min, 2)  # keep two decimals
                logger.info(
                    f"Auto-set r_min = {r_min:.2f} ({min_percentile}% percentile)"
                )

            if auto_r_max:
                r_max = float(np.percentile(all_dists, max_percentile))
                r_max = round(r_max, 2)  # keep two decimals
                logger.info(
                    f"Auto-set r_max = {r_max:.2f} ({max_percentile}% percentile)"
                )

            if auto_bin_size:
                # Set bin_size = (r_max - r_min) / n_bins.
                bin_size = float((r_max - r_min) / n_bins)
                bin_size = round(bin_size, 2)  # keep two decimals
                bin_size = max(0.01, bin_size)  # ensure it is not too small
                logger.info(
                    f"Auto-set bin_size = {bin_size:.2f} (1/{n_bins} of distance range)"
                )
        else:
            logger.warning("Could not collect valid distance samples; using defaults")
            if auto_r_min:
                r_min = 0.01
            if auto_r_max:
                r_max = 0.6
            if auto_bin_size:
                bin_size = 0.01

    # Ensure parameters have reasonable defaults.
    if r_min is None:
        r_min = 0.01
    if r_max is None:
        r_max = 0.6
    if bin_size is None:
        bin_size = 0.01

    logger.info(
        f"Final params: r_min={r_min:.2f}, r_max={r_max:.2f}, bin_size={bin_size:.2f}"
    )

    # Stage 1: precompute all network portraits (including auto-selecting r).
    logger.info("Stage 1: precomputing network portraits")
    portraits = {}

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit gene-level precompute tasks.
        futures = {
            executor.submit(
                precompute_portraits_for_gene,
                gene,
                df,
                cell_list,
                threshold,
                bin_size,
                r_min,
                r_max,
                r_step,
                use_same_r,
                use_vectorized,
                dist_dict,  # pass the global distance matrix dict
            ): gene
            for gene in gene_list
        }

        # Collect results.
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Precomputing network portraits",
            disable=not sys.stdout.isatty(),
        ):
            gene = futures[future]
            try:
                gene_portraits = future.result()
                portraits.update(gene_portraits)
                logger.info(
                    f"Gene {gene} precompute done; {len(gene_portraits)} distributions"
                )
            except Exception as e:
                logger.error(f"Gene {gene} precompute failed: {e}")

    logger.info(f"Network portrait precompute done; {len(portraits)} (cell,gene) pairs")

    # Stage 2: compute JS divergence.
    logger.info("Stage 2: computing JS divergence")
    all_distances = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit gene-level JS divergence tasks.
        futures = {
            executor.submit(
                find_js_distances_for_gene,
                gene,
                df,
                cell_list,
                portraits,
                bin_size,
                max_count,
                transcript_window,
            ): gene
            for gene in gene_list
        }

        # Collect results.
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Computing JS divergence",
            disable=not sys.stdout.isatty(),
        ):
            gene = futures[future]
            try:
                gene_distances = future.result()
                all_distances.extend(gene_distances)
                logger.info(
                    f"Gene {gene} JS divergence done; {len(gene_distances)} results"
                )
            except Exception as e:
                logger.error(f"Gene {gene} JS divergence failed: {e}")

    # Clean up distance matrices to free memory.
    dist_dict.clear()

    # Save results to a DataFrame.
    if all_distances:
        distances_df = pd.DataFrame(
            all_distances,
            columns=[
                "target_cell",
                "target_gene",
                "cell",
                "gene",
                "num_transcripts",
                "js_distance",
                "transcript_diff",
                "target_r",
                "other_r",
            ],
        )

        # Save results.
        output_path = f"{output_dir}/js_distances_bin{bin_size:.4f}_count{max_count}_threshold{threshold}.csv"
        distances_df.to_csv(output_path, index=False)

        logger.info(f"\nDone. Results saved to: {output_path}")
        logger.info(f"Computed {len(distances_df)} JS divergence records")

        # Visualize top-N most similar pairs (smallest JS divergence).
        if visualize_top_n > 0:
            logger.info(
                f"Visualizing top {visualize_top_n} most similar pairs by JS divergence"
            )
            visualize_most_similar_pairs(
                df, distances_df, portraits, visualize_top_n, vis_dir, use_vectorized
            )

        # Total elapsed time.
        total_time = time.time() - start_time
        logger.info(f"Total time: {total_time:.2f}s")

        return distances_df
    else:
        logger.warning("WARNING: no JS divergences were computed")
        return pd.DataFrame()


def visualize_most_similar_pairs(
    df, distances_df, portraits, top_n=5, output_dir=None, use_vectorized=True
):
    """
    Visualize the top-N most similar cell pairs by JS divergence.

    Args:
        df: Transcript DataFrame.
        distances_df: JS divergence results DataFrame.
        portraits: Precomputed portrait dict.
        top_n: Number of pairs to visualize.
        output_dir: Output directory.
        use_vectorized: Whether to use vectorized portrait computation.
    """
    # Sort by JS divergence.
    sorted_df = distances_df.sort_values("js_distance").reset_index(drop=True)

    # Ensure output directory exists.
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Visualize the top-N pairs.
    for i in range(min(top_n, len(sorted_df))):
        row = sorted_df.iloc[i]

        target_cell = row["target_cell"]
        target_gene = row["target_gene"]
        other_cell = row["cell"]
        other_gene = row["gene"]
        js_dist = row["js_distance"]
        target_r = row["target_r"]
        other_r = row["other_r"]

        logger.info(
            f"Rank {i + 1} most similar pair: {target_cell}:{target_gene} - {other_cell}:{other_gene}, JS={js_dist:.4f}"
        )

        # Extract transcript data.
        target_df = df[(df["cell"] == target_cell) & (df["gene"] == target_gene)]
        other_df = df[(df["cell"] == other_cell) & (df["gene"] == other_gene)]

        # Skip if there are too few transcripts.
        if len(target_df) <= 1 or len(other_df) <= 1:
            logger.warning(
                f"Pair {target_cell}:{target_gene} - {other_cell}:{other_gene} has too few transcripts; skipping visualization"
            )
            continue

        # Build graphs.
        target_graph = build_weighted_graph(target_df, target_r)
        other_graph = build_weighted_graph(other_df, other_r)

        # Compute network portraits.
        target_portrait, _ = get_network_portrait(
            target_graph, bin_size=0.01, use_vectorized=use_vectorized
        )
        other_portrait, _ = get_network_portrait(
            other_graph, bin_size=0.01, use_vectorized=use_vectorized
        )

        # Create per-pair output directory.
        pair_dir = None
        if output_dir:
            pair_dir = f"{output_dir}/pair_{i + 1}_js{js_dist:.4f}"
            os.makedirs(pair_dir, exist_ok=True)

        # Visualize.
        pair_prefix = f"Rank {i + 1} JS={js_dist:.4f}: "

        if pair_dir:
            # Save graph structure.
            plot_graph(
                target_graph,
                title=f"{pair_prefix}{target_cell}:{target_gene} (r={target_r:.2f})",
                save_path=f"{pair_dir}/cell1_graph.png",
            )
            plot_graph(
                other_graph,
                title=f"{pair_prefix}{other_cell}:{other_gene} (r={other_r:.2f})",
                save_path=f"{pair_dir}/cell2_graph.png",
            )

            # Save portraits.
            plot_portrait(
                target_portrait,
                title=f"{pair_prefix}{target_cell}:{target_gene} Network portrait",
                save_path=f"{pair_dir}/cell1_portrait.png",
            )
            plot_portrait(
                other_portrait,
                title=f"{pair_prefix}{other_cell}:{other_gene} Network portrait",
                save_path=f"{pair_dir}/cell2_portrait.png",
            )

            # Save transcript scatter plots.
            plot_transcripts(
                target_df,
                target_r,
                title=f"{pair_prefix}{target_cell}:{target_gene} Transcripts",
                save_path=f"{pair_dir}/cell1_transcripts.png",
            )
            plot_transcripts(
                other_df,
                other_r,
                title=f"{pair_prefix}{other_cell}:{other_gene} Transcripts",
                save_path=f"{pair_dir}/cell2_transcripts.png",
            )

            logger.info(f"Saved pair visualization to: {pair_dir}")
        else:
            # Show graph structure.
            plot_graph(
                target_graph,
                title=f"{pair_prefix}{target_cell}:{target_gene} (r={target_r:.2f})",
            )
            plot_graph(
                other_graph,
                title=f"{pair_prefix}{other_cell}:{other_gene} (r={other_r:.2f})",
            )

            # Show portraits.
            plot_portrait(
                target_portrait,
                title=f"{pair_prefix}{target_cell}:{target_gene} Network portrait",
            )
            plot_portrait(
                other_portrait,
                title=f"{pair_prefix}{other_cell}:{other_gene} Network portrait",
            )

            # Show transcript scatter plots.
            plot_transcripts(
                target_df,
                target_r,
                title=f"{pair_prefix}{target_cell}:{target_gene} Transcripts",
            )
            plot_transcripts(
                other_df,
                other_r,
                title=f"{pair_prefix}{other_cell}:{other_gene} Transcripts",
            )


def plot_transcripts(df, r, title="Transcript distribution", save_path=None):
    """
    Plot transcript spatial distribution.

    Args:
        df: Transcript DataFrame.
        r: Connection radius.
        title: Plot title.
        save_path: Optional output file path.
    """
    plt.figure(figsize=(10, 8))

    # Plot transcript locations.
    plt.scatter(df["x_c_s"], df["y_c_s"], alpha=0.6, s=10)

    # Add a radius circle for each transcript.
    for _, row in df.iterrows():
        circle = plt.Circle(
            (row["x_c_s"], row["y_c_s"]),
            r,
            fill=False,
            color="gray",
            alpha=0.2,
            linestyle="--",
        )
        plt.gca().add_patch(circle)
        # plt.text(center_x, center_y + r + 0.1, f'r = {r:.3f}', ha='center', fontsize=10, color='red')  # center_x/center_y undefined

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def analyze_transcript_distribution(df, output_dir=None):
    """
    Analyze and report transcript distribution.

    Args:
        df: Transcript DataFrame.
        output_dir: Output directory. If None, only logs are emitted.

    Returns:
        Dict: Summary statistics.
    """
    logger.info("Analyzing transcript distribution...")

    # Basic stats.
    total_transcripts = len(df)
    unique_genes = df["gene"].nunique()
    unique_cells = df["cell"].nunique()

    # Transcripts per gene.
    gene_transcript_counts = df.groupby("gene").size()

    if not gene_transcript_counts.empty:
        gene_stats_values = {
            "transcript_per_gene_mean": float(gene_transcript_counts.mean()),
            "transcript_per_gene_median": float(gene_transcript_counts.median()),
            "transcript_per_gene_std": float(gene_transcript_counts.std())
            if not np.isnan(gene_transcript_counts.std())
            else None,
            "transcript_per_gene_min": int(gene_transcript_counts.min()),
            "transcript_per_gene_max": int(gene_transcript_counts.max()),
        }
    else:
        gene_stats_values = {
            "transcript_per_gene_mean": 0.0,
            "transcript_per_gene_median": 0.0,
            "transcript_per_gene_std": None,
            "transcript_per_gene_min": 0,
            "transcript_per_gene_max": 0,
        }
    gene_stats = {"total_genes": int(unique_genes), **gene_stats_values}

    # Transcripts per cell.
    cell_transcript_counts = df.groupby("cell").size()
    if not cell_transcript_counts.empty:
        cell_stats_values = {
            "transcript_per_cell_mean": float(cell_transcript_counts.mean()),
            "transcript_per_cell_median": float(cell_transcript_counts.median()),
            "transcript_per_cell_std": float(cell_transcript_counts.std())
            if not np.isnan(cell_transcript_counts.std())
            else None,
            "transcript_per_cell_min": int(cell_transcript_counts.min()),
            "transcript_per_cell_max": int(cell_transcript_counts.max()),
        }
    else:
        cell_stats_values = {
            "transcript_per_cell_mean": 0.0,
            "transcript_per_cell_median": 0.0,
            "transcript_per_cell_std": None,
            "transcript_per_cell_min": 0,
            "transcript_per_cell_max": 0,
        }
    cell_stats = {"total_cells": int(unique_cells), **cell_stats_values}

    # Transcripts per (cell, gene) pair.
    cell_gene_transcript_counts = df.groupby(["cell", "gene"]).size()
    if not cell_gene_transcript_counts.empty:
        pair_stats_values = {
            "transcript_per_pair_mean": float(cell_gene_transcript_counts.mean()),
            "transcript_per_pair_median": float(cell_gene_transcript_counts.median()),
            "transcript_per_pair_std": float(cell_gene_transcript_counts.std())
            if not np.isnan(cell_gene_transcript_counts.std())
            else None,
        }
    else:
        pair_stats_values = {
            "transcript_per_pair_mean": 0.0,
            "transcript_per_pair_median": 0.0,
            "transcript_per_pair_std": None,
        }
    pair_stats = {
        "total_cell_gene_pairs": int(
            len(cell_gene_transcript_counts)
        ),  # len() returns python int
        **pair_stats_values,
        "single_transcript_pairs": int((cell_gene_transcript_counts == 1).sum()),
        "multi_transcript_pairs": int((cell_gene_transcript_counts > 1).sum()),
    }

    # Count genes that are mostly single-transcript.
    genes_with_mostly_single_transcripts = 0
    if (
        unique_genes > 0 and not cell_gene_transcript_counts.empty
    ):  # Avoid processing if no genes or no pairs
        for gene in df["gene"].unique():
            gene_pairs = cell_gene_transcript_counts[
                cell_gene_transcript_counts.index.get_level_values("gene") == gene
            ]
            if not gene_pairs.empty:
                single_transcript_ratio = (gene_pairs == 1).sum() / len(gene_pairs)
                if single_transcript_ratio > 0.8:  # >80% pairs are single-transcript
                    genes_with_mostly_single_transcripts += 1

    problem_stats = {
        "genes_with_mostly_single_transcripts": int(
            genes_with_mostly_single_transcripts
        ),
        "problematic_gene_ratio": float(
            genes_with_mostly_single_transcripts / unique_genes
        )
        if unique_genes > 0
        else 0.0,
        "single_transcript_pair_ratio": float(
            pair_stats["single_transcript_pairs"] / pair_stats["total_cell_gene_pairs"]
        )
        if pair_stats["total_cell_gene_pairs"] > 0
        else 0.0,
    }

    # Summary.
    stats_summary = {
        "total_transcripts": int(total_transcripts),  # len() returns python int
        "gene_stats": gene_stats,
        "cell_stats": cell_stats,
        "pair_stats": pair_stats,
        "problem_stats": problem_stats,
    }

    # Report.
    logger.info("=" * 60)
    logger.info("Transcript distribution report")
    logger.info("=" * 60)
    logger.info(f"Total transcripts: {total_transcripts:,}")
    logger.info(f"Unique genes: {unique_genes:,}")
    logger.info(f"Unique cells: {unique_cells:,}")
    logger.info("")

    logger.info("Gene-level stats:")
    logger.info(
        f"  Mean transcripts per gene: {gene_stats['transcript_per_gene_mean']:.1f}"
    )
    logger.info(f"  Median: {gene_stats['transcript_per_gene_median']:.1f}")
    logger.info(
        f"  Range: {gene_stats['transcript_per_gene_min']}-{gene_stats['transcript_per_gene_max']}"
    )
    logger.info("")

    logger.info("Cell-level stats:")
    logger.info(
        f"  Mean transcripts per cell: {cell_stats['transcript_per_cell_mean']:.1f}"
    )
    logger.info(f"  Median: {cell_stats['transcript_per_cell_median']:.1f}")
    logger.info(
        f"  Range: {cell_stats['transcript_per_cell_min']}-{cell_stats['transcript_per_cell_max']}"
    )
    logger.info("")

    logger.info("(Cell, gene) pair-level stats:")
    logger.info(f"  Total (cell, gene) pairs: {pair_stats['total_cell_gene_pairs']:,}")
    logger.info(
        f"  Single-transcript pairs: {pair_stats['single_transcript_pairs']:,} ({pair_stats['single_transcript_pairs'] / pair_stats['total_cell_gene_pairs'] * 100:.1f}%)"
    )
    logger.info(
        f"  Multi-transcript pairs: {pair_stats['multi_transcript_pairs']:,} ({pair_stats['multi_transcript_pairs'] / pair_stats['total_cell_gene_pairs'] * 100:.1f}%)"
    )
    logger.info(
        f"  Mean transcripts per pair: {pair_stats['transcript_per_pair_mean']:.1f}"
    )
    logger.info("")

    logger.info("Potential issues:")
    logger.info(
        f"  Genes mostly single-transcript: {problem_stats['genes_with_mostly_single_transcripts']:,} ({problem_stats['problematic_gene_ratio'] * 100:.1f}%)"
    )
    logger.info(
        f"  Single-transcript pair ratio: {problem_stats['single_transcript_pair_ratio'] * 100:.1f}%"
    )

    if problem_stats["problematic_gene_ratio"] > 0.5:
        logger.warning(
            "WARNING: >50% of genes are mostly single-transcript; consider checking data quality or adjusting parameters"
        )
    elif problem_stats["single_transcript_pair_ratio"] > 0.7:
        logger.warning(
            "WARNING: >70% of (cell, gene) pairs have a single transcript; this may affect network portrait quality"
        )
    else:
        logger.info("Data quality looks OK for network portrait analysis")

    logger.info("=" * 60)

    # If output_dir is provided, write stats and plots.
    if output_dir:
        import json

        os.makedirs(output_dir, exist_ok=True)

        # Save stats.
        stats_file = os.path.join(output_dir, "transcript_distribution_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved detailed stats to: {stats_file}")

        # Save distribution plots.
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.hist(gene_transcript_counts, bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("Transcripts per gene")
        plt.ylabel("Number of genes")
        plt.title("Transcript count per gene")
        plt.yscale("log")

        plt.subplot(2, 2, 2)
        plt.hist(cell_transcript_counts, bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("Transcripts per cell")
        plt.ylabel("Number of cells")
        plt.title("Transcript count per cell")
        plt.yscale("log")

        plt.subplot(2, 2, 3)
        plt.hist(cell_gene_transcript_counts, bins=30, alpha=0.7, edgecolor="black")
        plt.xlabel("Transcripts per (cell, gene) pair")
        plt.ylabel("Number of pairs")
        plt.title("Transcript count per (cell, gene) pair")
        plt.yscale("log")

        plt.subplot(2, 2, 4)
        # Single vs multi transcript pairs.
        labels = ["Single-transcript pairs", "Multi-transcript pairs"]
        sizes = [
            pair_stats["single_transcript_pairs"],
            pair_stats["multi_transcript_pairs"],
        ]
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title("Single vs multi-transcript pairs")

        plt.tight_layout()
        dist_plot_file = os.path.join(output_dir, "transcript_distribution_plots.png")
        plt.savefig(dist_plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved distribution plots to: {dist_plot_file}")

    return stats_summary


if __name__ == "__main__":
    # CLI argument parsing.
    parser = argparse.ArgumentParser(
        description="Compute similarity between transcript graphs using JS divergence"
    )
    parser.add_argument(
        "--pkl_file",
        type=str,
        required=True,
        help="Path to a PKL containing df_registered",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory (optional)"
    )
    parser.add_argument(
        "--max_count", type=int, default=10, help="Max comparisons per target cell"
    )
    parser.add_argument(
        "--transcript_window",
        type=int,
        default=30,
        help="Transcript count difference window for candidate filtering",
    )
    parser.add_argument(
        "--bin_size",
        type=str,
        default="0.01",
        help='Path length bin size for JS divergence; set to "auto" to estimate',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Isolated node ratio threshold for selecting r",
    )
    parser.add_argument(
        "--r_min",
        type=str,
        default="0.01",
        help='Minimum r for search; set to "auto" to estimate',
    )
    parser.add_argument(
        "--r_max",
        type=str,
        default="0.6",
        help='Maximum r for search; set to "auto" to estimate',
    )
    parser.add_argument(
        "--r_step", type=float, default=0.03, help="Step size for r search"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="Number of threads (default: CPU count)",
    )
    parser.add_argument(
        "--use_same_r",
        action="store_true",
        help="Use a shared r for all cells within a gene (gene-level r)",
    )
    parser.add_argument(
        "--visualize_top_n",
        type=int,
        default=5,
        help="Visualize top-N most similar pairs by JS divergence (0 disables)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="js_distance_transcriptome.log",
        help="Log filename",
    )
    parser.add_argument(
        "--no_vectorized",
        action="store_true",
        help="Disable vectorized portrait computation (slower, lower memory)",
    )
    parser.add_argument(
        "--filter_pkl_file",
        type=str,
        default=None,
        help="Optional filter PKL path (contains cell_labels/gene_labels)",
    )
    parser.add_argument(
        "--auto_params", action="store_true", help="Auto-set r_min, r_max, and bin_size"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=50,
        help="Number of bins when auto-setting bin_size",
    )
    parser.add_argument(
        "--min_percentile",
        type=float,
        default=1.0,
        help="Percentile used when auto-setting r_min",
    )
    parser.add_argument(
        "--max_percentile",
        type=float,
        default=99.0,
        help="Percentile used when auto-setting r_max",
    )

    args = parser.parse_args()

    # Set log level.
    logger.setLevel(getattr(logging, args.log_level))

    # Add file handler.
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(file_handler)

    # Track program runtime.
    program_start_time = time.time()
    logger.info("=" * 80)
    logger.info("Program started")
    logger.info(f"CLI args: {vars(args)}")

    try:
        # Run main function.
        distances_df = calculate_js_distances(
            pkl_file=args.pkl_file,
            output_dir=args.output_dir,
            max_count=args.max_count,
            transcript_window=args.transcript_window,
            bin_size=args.bin_size,
            threshold=args.threshold,
            r_min=args.r_min,
            r_max=args.r_max,
            r_step=args.r_step,
            num_threads=args.num_threads,
            use_same_r=args.use_same_r,
            visualize_top_n=args.visualize_top_n,
            use_vectorized=(not args.no_vectorized),
            filter_pkl_file=args.filter_pkl_file,
            auto_params=args.auto_params,
            n_bins=args.n_bins,
            min_percentile=args.min_percentile,
            max_percentile=args.max_percentile,
        )

        # Print result stats.
        if not distances_df.empty:
            logger.info("\nResult stats:")
            logger.info(f"Total distances: {len(distances_df)}")
            logger.info(
                f"JS range: [{distances_df['js_distance'].min():.4f}, {distances_df['js_distance'].max():.4f}]"
            )
            logger.info(f"Mean JS: {distances_df['js_distance'].mean():.4f}")
            logger.info(f"Median JS: {distances_df['js_distance'].median():.4f}")
            logger.info(
                f"r range: [{distances_df['target_r'].min():.2f}, {distances_df['target_r'].max():.2f}]"
            )
            logger.info(f"Mean r: {distances_df['target_r'].mean():.2f}")

            # Preview first few rows.
            logger.info("\nResult preview:")
            logger.info(distances_df.head().to_string())

    except Exception as e:
        logger.error(f"Program failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise

    # Compute and report total runtime.
    program_end_time = time.time()
    total_program_time = program_end_time - program_start_time
    hours, remainder = divmod(total_program_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("=" * 80)
    logger.info(f"Program total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(
        f"Start time: {datetime.fromtimestamp(program_start_time).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(
        f"End time: {datetime.fromtimestamp(program_end_time).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info("=" * 80)

print("=" * 50)
