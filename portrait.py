"""
Network Portrait and Jensen-Shannon Divergence Calculator for Molecular Graph Similarity

本程序实现基于网络肖像(Network Portrait)和Jensen-Shannon散度的分子图相似性计算，
作为Gromov-Wasserstein距离的替代方案。

## 计算流程说明

### 1. 输入数据与预处理
- 读取包含转录本空间坐标的PKL文件，数据帧必须包含以下列:
  - 'cell': 细胞ID
  - 'gene': 基因ID 
  - 'x_c_s': 转录本X坐标
  - 'y_c_s': 转录本Y坐标

### 2. 预计算网络肖像(Network Portrait)
- **连接半径选择**:
  - 细胞级别: 为每个细胞搜索最小半径r，使孤立节点比例≤阈值(默认5%)
  - 基因级别(可选): 取同一基因所有细胞r值的最大值，确保可比性
  
- **加权图构建**:
  - 转录本作为节点，每对距离≤r的转录本之间添加边
  - 边权重为转录本间的欧氏距离
  
- **网络肖像计算**:
  - 计算所有节点对(i→j)的最短加权路径长度
  - 对每个源节点i，记录它的度k和到j的路径长度分箱ℓ
  - portrait[(ℓ,k)] = 满足此条件的节点对数量
  
- **加权分布转换**:
  - P(ℓ,k) = (k * portrait[(ℓ,k)]) / N²
  - 乘以k：体现度大的节点对网络贡献更大
  - 除以N²：归一化为概率分布

### 3. Jensen-Shannon散度计算
- 对同一基因下的两个细胞A、B:
  - 获取它们的网络肖像分布P_A和P_B
  - 构造中间分布M = 0.5(P_A + P_B)
  - 计算JS散度 = 0.5*KL(P_A||M) + 0.5*KL(P_B||M)
  - 散度值越小，两个网络越相似

### 4. 结果可视化
- **图结构可视化**: 展示转录本节点和连接边
- **热图可视化**: 展示网络肖像的二维分布

## 热图解读
- **横轴(k)**: 源节点的度
- **纵轴(ℓ)**: 路径长度分箱编号
- **颜色深浅**: 满足条件(ℓ,k)的节点对数量
- 热图展示了网络的局部属性(度分布)和全局连通性(路径长度分布)的组合特征
- 通过热图可直观比较不同细胞中同一基因空间表达模式的拓扑差异

## 使用示例
```bash
python portrait.py --pkl_file /path/to/data_dict.pkl --use_same_r --visualize_top_n 10
```

## 参数说明
- pkl_file: 包含转录本坐标的数据文件
- use_same_r: 对同一基因的所有细胞使用统一的r值
- r_min, r_max, r_step: r值搜索范围和步长(默认0.01-0.6,步长0.03)
- bin_size: 路径长度分箱大小(默认0.01)
- threshold: 孤立节点比例阈值(默认0.05)
- visualize_top_n: 可视化JS散度最小的前N个细胞对
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

## utils_code/portrait.py
## todo： 对每个基因（或每个 batch），分别收集它所有子图的最短路径长度 → 算出该组的 bin_size → 分别用它来做 Portrait 和 JS。
## done： 一次计算，复用距离矩阵，把"计算距离矩阵"提到外面来做, 用kd tree搜索r距离
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

# 设置matplotlib全局字体为Arial
plt.rcParams['font.family'] = ['SimHei', 'Arial']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("js_distance")
    
# 忽略警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def find_r_for_isolated_threshold(df, threshold=0.05, r_min=0.01, r_max=0.6, step=0.03, verbose=False, dists=None):
    """
    寻找最优连接半径，使孤立节点比例低于阈值
    使用KDTree优化计算，直接找出最近邻距离的百分位数
        
        Args:
        df: 包含坐标的DataFrame，必须包含x_c_s和y_c_s列
        threshold: 孤立节点比例阈值
        r_min, r_max: 半径范围
        step: 半径增加步长 (使用KDTree优化后此参数不再使用，但保留以维持接口兼容)
        verbose: 是否打印详细信息
        dists: 预计算的距离矩阵，如果为None则重新计算 (使用KDTree优化后此参数不再使用，但保留以维持接口兼容)
        
    Returns:
        float: 最优半径值
    """
    positions = df[['x_c_s', 'y_c_s']].values
    N = len(df)
    
    # 处理边界情况
    if N <= 1:
        # 对于单点或空数据，返回一个小的默认半径
        return min(r_min * 5, r_max * 0.1)  # 使用r_min的5倍或r_max的10%，取较小值
    
    # 使用KDTree直接计算最近邻距离
    tree = KDTree(positions)
    
    # k=2 因为第一个结果是点自身，第二个才是真正的最近邻
    # 如果只有一个点，query会失败，但我们已经在上面处理了这种情况
    dists, _ = tree.query(positions, k=2)
    nn_dists = dists[:, 1]  # 取每个点的最近邻距离
    
    # 计算百分位数 - 至多threshold比例的点可以是孤立的
    # 即对应(1-threshold)百分位的距离
    r = np.percentile(nn_dists, 100 * (1 - threshold))
    
    # 确保r在指定范围内
    r = max(r_min, min(r, r_max))
    
    if verbose:
        logger.debug(f"通过KDTree计算的最优r值: {r:.4f}, 对应第{100*(1-threshold):.1f}百分位的最近邻距离")
        # 计算实际孤立节点比例进行校验
        isolated_count = np.sum(nn_dists > r)
        logger.debug(f"实际孤立节点比例: {isolated_count/N:.4f} ({isolated_count}/{N})")
    
    return r

def build_weighted_graph(df, r, dists=None):
    """
    基于坐标和连接半径构建加权图
    
    Args:
        df: 包含坐标的DataFrame，必须包含x_c_s和y_c_s列
        r: 连接半径
        dists: 预计算的距离矩阵，如果为None则重新计算
        
    Returns:
        G: NetworkX图对象
    """
    G = nx.Graph()
    positions = df[['x_c_s', 'y_c_s']].values
    
    # 添加节点
    for i, pos in enumerate(positions):
        G.add_node(i, pos=pos)
    
    # 如果没有提供距离矩阵，则计算它
    if dists is None:
        dists = distance_matrix(positions, positions)
    
    # 添加边
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            dist = dists[i, j]
            if dist <= r:
                G.add_edge(i, j, weight=dist)
    
    return G

def get_network_portrait(G, bin_size=0.01, use_vectorized=True):
    """
    计算图的网络肖像
        
    Args:
        G: NetworkX图对象
        bin_size: 路径长度分箱大小
        use_vectorized: 是否使用向量化实现(默认True)
        
        Returns:
        Tuple: (portrait, node_count)
    """
    # 节点数量
    n_nodes = len(G)
    if n_nodes <= 1:  # 处理边界情况
        return {}, n_nodes
    
    # 选择实现方式
    if use_vectorized:
        # SciPy稀疏图加速实现 - 更高效的向量化方案
        # 1) 构建稀疏邻接矩阵
        rows, cols, weights = [], [], []
        for u, v, data in G.edges(data=True):
            w = data.get('weight', 1.0)
            rows.append(u); cols.append(v); weights.append(w)
            rows.append(v); cols.append(u); weights.append(w)  # 无向图需要添加对称边

        A = csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))

        # 2) SciPy 一次性算 APSP
        #    directed=False 保证无向， unweighted=False 用权重
        dist_mat = shortest_path(A, directed=False, unweighted=False, method='auto')

        # 3) 度数向量
        degs = np.array([d for _, d in sorted(G.degree(), key=lambda x: x[0])])

        # 4) 排除自环，提取所有 i≠j 对
        #    dist_mat 是 n×n 的 numpy 数组
        i_idx, j_idx = np.nonzero(~np.eye(n_nodes, dtype=bool))
        dists = dist_mat[i_idx, j_idx]
        src_degs = degs[i_idx]

        # 5) 分箱
        bins = np.floor(dists / bin_size).astype(int)

        # 6) 批量统计
        combined = np.column_stack([bins, src_degs])
        unique_ck, counts = np.unique(combined, axis=0, return_counts=True)

        # 7) 转回字典
        portrait = {
            (int(bin_l), int(deg)): int(cnt)
            for (bin_l, deg), cnt in zip(unique_ck, counts)
        }
    else:
        # 原始循环实现 - 可能较慢但内存占用较小
        # 获取所有节点对之间的最短路径长度
        length_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        # 获取节点度数
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
    """
    计算加权概率分布
    
    Args:
        portrait: 网络肖像
        N: 节点数量
        
    Returns:
        Dict: 加权概率分布
    """
    total_pairs = N * N  # 与您的代码保持一致
    dist = {}
    
    for (l, k), count in portrait.items():
        dist[(l, k)] = (k * count) / total_pairs
    
    return dist

def js_divergence(P, Q):
    """
    计算两个概率分布间的Jensen-Shannon散度
    
    Args:
        P, Q: 概率分布字典
        
    Returns:
        float: JS散度值
    """
    keys = set(P.keys()).union(Q.keys())
    p_vec = np.array([P.get(k, 0.0) for k in keys])
    q_vec = np.array([Q.get(k, 0.0) for k in keys])
    m_vec = 0.5 * (p_vec + q_vec)

    def safe_kl(p, q):
        mask = (p > 0) & (q > 0)
        return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

    return 0.5 * safe_kl(p_vec, m_vec) + 0.5 * safe_kl(q_vec, m_vec)

# 可视化函数
def plot_portrait(portrait, title="网络肖像", save_path=None):
    """
    可视化网络肖像
    
    Args:
        portrait: 网络肖像字典
        title: 图表标题
        save_path: 保存路径，None表示不保存
    """
    if not portrait:
        logger.warning("空的网络肖像，无法绘图")
        return
        
    df = pd.DataFrame([
        {"l": l, "k": k, "value": v}
        for (l, k), v in portrait.items()
    ])
    
    pivot = df.pivot(index="l", columns="k", values="value").fillna(0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, cmap='viridis', annot=False)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("度数 k", fontsize=12)
    plt.ylabel("路径长度分箱 l", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"网络肖像图已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_graph(G, title="图结构", save_path=None):
    """
    可视化图结构
    
    Args:
        G: NetworkX图对象
        title: 图表标题
        save_path: 保存路径，None表示不保存
    """
    pos = nx.get_node_attributes(G, 'pos')
    
    plt.figure(figsize=(8, 8))
    nx.draw(
        G, pos, node_size=30, node_color='skyblue',
        edge_color='gray', width=0.5, with_labels=False
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"图结构已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()

def compare_graphs(df1, df2, r=None, bin_size=0.01, show_plots=True, save_dir=None, use_vectorized=True):
    """
    比较两个图的相似性，计算JS散度
    
    Args:
        df1, df2: 包含坐标的DataFrame
        r: 连接半径，None表示自动计算
        bin_size: 路径长度分箱大小
        show_plots: 是否显示图形
        save_dir: 保存图形的目录，None表示不保存
        use_vectorized: 是否使用向量化实现计算网络肖像
        
    Returns:
        float: JS散度值
    """
    # 自动计算r值
    if r is None:
        r1 = find_r_for_isolated_threshold(df1, threshold=0.05)
        r2 = find_r_for_isolated_threshold(df2, threshold=0.05)
        r = max(r1, r2)
        logger.info(f"自动计算的连接半径 r = {r:.2f}")
    
    # 构建图
    G1 = build_weighted_graph(df1, r)
    G2 = build_weighted_graph(df2, r)
    
    # 计算网络肖像
    B1, N1 = get_network_portrait(G1, bin_size, use_vectorized)
    B2, N2 = get_network_portrait(G2, bin_size, use_vectorized)
    
    # 计算加权分布
    P = compute_weighted_distribution(B1, N1)
    Q = compute_weighted_distribution(B2, N2)
    
    # 计算JS散度
    js = js_divergence(P, Q)
    logger.info(f"两个图之间的Jensen-Shannon散度: {js:.4f}")
    
    # 可视化
    if show_plots or save_dir:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        if show_plots:
            plot_graph(G1, title="图1结构")
            plot_graph(G2, title="图2结构")
            plot_portrait(B1, title="图1网络肖像")
            plot_portrait(B2, title="图2网络肖像")
        
        if save_dir:
            plot_graph(G1, title="图1结构", 
                      save_path=f"{save_dir}/graph1_structure.png")
            plot_graph(G2, title="图2结构", 
                      save_path=f"{save_dir}/graph2_structure.png")
            plot_portrait(B1, title="图1网络肖像", 
                         save_path=f"{save_dir}/graph1_portrait.png")
            plot_portrait(B2, title="图2网络肖像", 
                         save_path=f"{save_dir}/graph2_portrait.png")
    
    return js

def find_gene_optimal_r(gene, df, cell_list, threshold=0.05, r_min=0.01, r_max=0.6, r_step=0.03, dist_dict=None):
    """
    为一个基因寻找最优r值(所有细胞的最大r值)
    
    Args:
        gene: 基因ID
        df: 包含转录本信息的DataFrame
        cell_list: 细胞ID列表
        threshold: 孤立节点比例阈值
        r_min, r_max, r_step: r值搜索参数
        dist_dict: 预计算的距离矩阵字典 {(cell, gene): 距离矩阵}
        
    Returns:
        tuple: (float: 该基因的最优r值, dict: {cell: (r值, 距离矩阵)})
    """
    logger.info(f"为基因 {gene} 计算最优r值")
    
    # 获取该基因的所有转录本
    gene_df = df[df['gene'] == gene]
    
    cell_r_values = {}
    cell_dist_matrices = {}  # 存储每个细胞的距离矩阵
    transcript_counts = {}  # 存储每个细胞的转录本数量统计
    
    # 为每个细胞计算r值
    for cell in cell_list:
        cell_df = gene_df[gene_df['cell'] == cell]
        transcript_count = len(cell_df)
        transcript_counts[cell] = transcript_count
        
        # 对于只有1个转录本的情况，使用一个小的默认半径
        if transcript_count == 1:
            # 单个转录本使用较小的半径，避免连接到过远的其他点
            r_single = min(r_min * 5, r_max * 0.1)  # 使用r_min的5倍或r_max的10%，取较小值
            cell_r_values[cell] = r_single
            # 单点距离矩阵
            cell_dist_matrices[cell] = np.array([[0.0]])
            continue
        
        # 如果该细胞的转录本数量为0，跳过
        if transcript_count == 0:
            continue
            
        try:
            # 优先使用预计算的距离矩阵
            dists = dist_dict.get((cell, gene), None) if dist_dict else None
            
            # 如果没有预计算的距离矩阵，则计算它
            if dists is None:
                positions = cell_df[['x_c_s', 'y_c_s']].values
                dists = distance_matrix(positions, positions)
            
            # 计算该细胞的最优r值
            r = find_r_for_isolated_threshold(
                cell_df, threshold, r_min, r_max, r_step, 
                verbose=False, dists=dists  # 传入预计算的距离矩阵
            )
            cell_r_values[cell] = r
            cell_dist_matrices[cell] = dists  # 保存距离矩阵以便复用
        except Exception as e:
            logger.error(f"计算细胞 {cell} 基因 {gene} 的r值失败: {e}")
    
    # 统计信息
    total_cells = len(cell_list)
    valid_cells = len(cell_r_values)
    single_transcript_cells = sum(1 for count in transcript_counts.values() if count == 1)
    multi_transcript_cells = sum(1 for count in transcript_counts.values() if count > 1)
    zero_transcript_cells = sum(1 for count in transcript_counts.values() if count == 0)
    
    # 如果没有有效的r值，返回更合理的默认值
    if not cell_r_values:
        # 使用较小的默认值，避免图过于稠密
        default_r = min(r_max * 0.3, r_min * 10)  # 使用r_max的30%或r_min的10倍，取较小值
        logger.warning(f"基因 {gene} 没有有效的r值，使用调整后的默认值 {default_r:.4f} (原默认值: {r_max})")
        logger.warning(f"基因 {gene} 转录本统计: 总细胞{total_cells}, 0转录本细胞{zero_transcript_cells}, 1转录本细胞{single_transcript_cells}, 多转录本细胞{multi_transcript_cells}")
        return default_r, {}
    
    # 返回所有细胞r值的最大值，以及距离矩阵字典
    max_r = max(cell_r_values.values())
    min_r = min(cell_r_values.values())
    avg_r = np.mean(list(cell_r_values.values()))
    
    logger.info(f"基因 {gene} 的最优r值: {max_r:.2f} (来自 {valid_cells} 个细胞, 范围: {min_r:.2f}-{max_r:.2f}, 平均: {avg_r:.2f})")
    logger.info(f"基因 {gene} 转录本分布: 0个转录本{zero_transcript_cells}细胞, 1个转录本{single_transcript_cells}细胞, >1个转录本{multi_transcript_cells}细胞")
    
    return max_r, cell_dist_matrices

def precompute_portraits_for_gene(gene, df, cell_list, threshold=0.05, bin_size=0.01, r_min=0.01, r_max=0.6, r_step=0.03, use_same_r=True, use_vectorized=True, dist_dict=None):
    """
    为一个基因的所有细胞预计算网络肖像
    
    Args:
        gene: 基因标识符
        df: 包含转录本信息的DataFrame
        cell_list: 细胞列表
        threshold: 孤立节点比例阈值
        bin_size: 路径长度分箱大小
        r_min, r_max, r_step: 半径搜索参数
        use_same_r: 是否对一个基因的所有细胞使用相同的r值
        use_vectorized: 是否使用向量化实现计算网络肖像
        dist_dict: 预计算的距离矩阵字典 {(cell, gene): 距离矩阵}
        
    Returns:
        Dict: {(cell, gene): (weighted_distribution, node_count, r_value)}
    """
    distributions = {}
    
    logger.info(f"预计算基因 {gene} 的网络肖像开始")
    start_time = time.time()
    
    # 过滤该基因的所有转录本
    gene_df = df[df['gene'] == gene]
    
    # 确保有数据
    if len(gene_df) == 0:
        logger.warning(f"基因 {gene} 没有转录本数据")
        return distributions
    
    # 如果使用相同的r值，先计算基因的最优r值，并获取预计算的距离矩阵
    gene_r = None
    cell_dist_matrices = {}
    if use_same_r:
        gene_r, cell_dist_matrices = find_gene_optimal_r(gene, df, cell_list, threshold, r_min, r_max, r_step, dist_dict)
    
    # 处理每个细胞
    for cell in tqdm(cell_list, desc=f"处理基因 {gene} 的细胞", leave=False, disable=not sys.stdout.isatty()):
        # 过滤该细胞的转录本
        cell_df = gene_df[gene_df['cell'] == cell]
        transcript_count = len(cell_df)
        
        # 跳过没有转录本的细胞
        if transcript_count == 0:
            logger.debug(f"细胞 {cell} 基因 {gene} 没有转录本，跳过")
            continue
        
        # 处理单转录本情况
        if transcript_count == 1:
            logger.debug(f"细胞 {cell} 基因 {gene} 只有1个转录本，创建单点网络肖像")
            try:
                # 单点图的特殊处理
                # 对于单点，网络肖像只包含度为0的节点
                # 路径长度分箱 (l, k) 中，l=0(自身距离), k=0(度为0)
                single_portrait = {(0, 0): 1}  # 只有一个(路径长度=0, 度=0)的条目
                weighted_dist = compute_weighted_distribution(single_portrait, 1)
                
                # 使用合适的r值
                r = gene_r if use_same_r else min(r_min * 5, r_max * 0.1)
                
                distributions[(cell, gene)] = (weighted_dist, 1, r)
                continue
            except Exception as e:
                logger.error(f"处理单转录本 cell={cell}, gene={gene} 失败: {e}")
                continue
        
        # 处理多转录本情况（原有逻辑）
        try:
            # 优先检查cell_dist_matrices，其次检查全局dist_dict
            dists = cell_dist_matrices.get(cell, None)
            if dists is None and dist_dict:
                dists = dist_dict.get((cell, gene), None)
            
            # 如果还没有距离矩阵，则计算
            if dists is None:
                positions = cell_df[['x_c_s', 'y_c_s']].values
                dists = distance_matrix(positions, positions)
            
            # 使用基因级别的r值，或为该细胞-基因对找到最优r值
            r = gene_r
            if not use_same_r:
                r = find_r_for_isolated_threshold(
                    cell_df, threshold=threshold, 
                    r_min=r_min, r_max=r_max, step=r_step,
                    dists=dists  # 传入预计算的距离矩阵
                )
            
            # 构建图，重用距离矩阵
            G = build_weighted_graph(cell_df, r, dists=dists)
            
            # 计算网络肖像和加权分布
            portrait, N = get_network_portrait(G, bin_size, use_vectorized)
            weighted_dist = compute_weighted_distribution(portrait, N)
            
            distributions[(cell, gene)] = (weighted_dist, N, r)
            
        except Exception as e:
            logger.error(f"处理 cell={cell}, gene={gene} 失败: {e}")

    elapsed = time.time() - start_time
    logger.info(f"预计算基因 {gene} 的网络肖像完成，得到 {len(distributions)} 个分布，耗时 {elapsed:.2f}秒")

    return distributions

def find_js_distances_for_gene(gene, df, cell_list, portraits, bin_size=0.01, max_count=None, transcript_window=30):
    """
    计算一个基因内所有细胞对之间的JS散度
    
    Args:
        gene: 基因标识符
        df: 包含转录本信息的DataFrame
        cell_list: 细胞列表
        portraits: 预计算的肖像字典 {(cell, gene): (weighted_distribution, node_count, r_value)}
        bin_size: 路径长度分箱大小
        max_count: 每个细胞计算的最大对数
        transcript_window: 转录本数量差异窗口
        
    Returns:
        List: JS散度结果
    """
    logger.info(f"计算基因 {gene} 的细胞对JS散度")
    start_time = time.time()
    
    # 获取该基因的所有有效细胞和转录本数量
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
    
    logger.info(f"基因 {gene} 有 {len(valid_cells)} 个有效细胞")
    
    # 计算所有可能的细胞对距离
    all_distances = []
    processed_count = 0
    
    for i, target_cell in enumerate(valid_cells):
        target_transcript_count = transcript_counts[target_cell]
        target_r = r_values[target_cell]
        
        # 筛选候选细胞（转录本数量相似的其他细胞）
        candidates = []
        for j, cell in enumerate(valid_cells):
            if cell != target_cell:
                transcript_diff = abs(transcript_counts[cell] - target_transcript_count)
                if transcript_diff <= transcript_window:
                    candidates.append((cell, transcript_diff))
        
        # 按转录本数量差异排序
        candidates.sort(key=lambda x: x[1])
        
        # 限制计算数量
        if max_count is not None and len(candidates) > max_count:
            candidates = candidates[:max_count]
        
        # 计算JS散度
        for cell, transcript_diff in candidates:
            try:
                target_key = (target_cell, gene)
                other_key = (cell, gene)
                
                if target_key in portraits and other_key in portraits:
                    target_dist, _, _ = portraits[target_key]
                    other_dist, _, other_r = portraits[other_key]
                    
                    js_distance = js_divergence(target_dist, other_dist)
                    
                    all_distances.append((
                        target_cell, gene,
                        cell, gene,
                        transcript_counts[cell], js_distance,
                        transcript_diff, target_r, other_r
                    ))
                    
                    processed_count += 1
                    if processed_count % 100 == 0:
                        logger.debug(f"基因 {gene}: 已计算 {processed_count} 个JS散度")
            except Exception as e:
                logger.error(f"计算 {target_cell}-{cell} 的JS散度失败: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"基因 {gene} 的JS散度计算完成，共 {len(all_distances)} 个结果，耗时 {elapsed:.2f}秒")
    
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
    max_percentile: float = 99.0
) -> pd.DataFrame:
    """
    计算基于转录本坐标的分子图之间的JS散度，包括自动选择最优r值
    
    Args:
        pkl_file: 包含df_registered的PKL文件路径
        output_dir: 输出目录，None表示使用默认路径
        max_count: 每个细胞计算的最大距离数
        transcript_window: 转录本数量差异窗口
        bin_size: 路径长度分箱大小，设为'auto'则自动计算
        threshold: 孤立节点比例阈值
        r_min: r值搜索最小值，设为'auto'则自动计算
        r_max: r值搜索最大值，设为'auto'则自动计算
        r_step: r值搜索步长
        num_threads: 线程数
        use_same_r: 是否对一个基因的所有细胞使用相同的r值
        visualize_top_n: 可视化JS散度最小的前N个细胞对
        use_vectorized: 是否使用向量化实现计算网络肖像(默认True)
        filter_pkl_file: 5_graph_data目录下的PKL文件路径，用于筛选细胞-基因对
        auto_params: 是否自动设置r_min、r_max和bin_size参数
        n_bins: 自动设置bin_size时使用的分箱数量
        min_percentile: 自动设置r_min时使用的百分位数
        max_percentile: 自动设置r_max时使用的百分位数
        
    Returns:
        pd.DataFrame: JS散度结果
    """
    # 设置默认线程数
    if num_threads is None:
        import multiprocessing
        num_threads = min(multiprocessing.cpu_count()-2, 20)
    
    # 根据参数，判断是否需要自动设置参数
    auto_r_min = r_min == 'auto' or auto_params
    auto_r_max = r_max == 'auto' or auto_params
    auto_bin_size = bin_size == 'auto' or auto_params
    
    # 如果需要自动计算参数，初始值设为None
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
    
    logger.info(f"使用 {num_threads} 个线程计算JS散度")
    if auto_params or auto_r_min or auto_r_max or auto_bin_size:
        logger.info("将自动计算r_min, r_max和bin_size")
    else:
        logger.info(f"最优r值搜索参数: threshold={threshold}, r_min={r_min}, r_max={r_max}, r_step={r_step}")
        logger.info(f"路径长度分箱大小: {bin_size}")
    
    logger.info(f"{'对同一基因使用相同r值' if use_same_r else '对每个细胞-基因对使用独立r值'}")
    logger.info(f"{'使用向量化实现计算网络肖像' if use_vectorized else '使用循环实现计算网络肖像'}")
        
    start_time = time.time()
    
    # 加载数据
    with open(pkl_file, 'rb') as f:
        data_dict = pickle.load(f)
    
    # 检查是否包含df_registered
    if 'df_registered' not in data_dict:
        raise ValueError(f"PKL文件 {pkl_file} 中不包含df_registered")
    
    df = data_dict['df_registered']
    logger.info(f"加载了 {len(df)} 条转录本记录")
    
    # 检查必要的列
    required_cols = ['cell', 'gene', 'x_c_s', 'y_c_s']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"df_registered缺少必要的列: {missing_cols}")
    
    # 如果提供了过滤用的PKL文件，则筛选细胞和基因
    if filter_pkl_file and os.path.exists(filter_pkl_file):
        logger.info(f"使用PKL文件 {filter_pkl_file} 筛选细胞-基因对")
        try:
            with open(filter_pkl_file, 'rb') as f:
                filter_data = pickle.load(f)
            
            # 提取cell_labels和gene_labels
            if 'cell_labels' in filter_data and 'gene_labels' in filter_data:
                # 确保两个列表长度相同，这样才能一一对应形成细胞-基因对
                cell_labels = filter_data['cell_labels']
                gene_labels = filter_data['gene_labels']
                
                if len(cell_labels) == len(gene_labels):
                    # 创建cell-gene对集合
                    cell_gene_pairs = set(zip(cell_labels, gene_labels))
                    logger.info(f"从过滤文件中提取了 {len(cell_gene_pairs)} 个唯一细胞-基因对")
                    
                    # 创建df的cell-gene对
                    df['cell_gene_pair'] = list(zip(df['cell'], df['gene']))
                    
                    # 过滤df_registered，只保留存在于PKL文件中的细胞-基因对
                    original_len = len(df)
                    df = df[df['cell_gene_pair'].isin(cell_gene_pairs)]
                    
                    # 删除临时列
                    df = df.drop(columns=['cell_gene_pair'])
                    
                    logger.info(f"过滤前: {original_len} 条记录，过滤后: {len(df)} 条记录")
                    
                    if len(df) == 0:
                        logger.warning("过滤后没有剩余记录，请检查细胞-基因对是否匹配")
                        return pd.DataFrame()
                else:
                    logger.warning(f"PKL文件中cell_labels({len(cell_labels)})和gene_labels({len(gene_labels)})长度不一致，无法形成精确的细胞-基因对")
                    logger.info("将使用所有细胞和基因继续计算")
            else:
                logger.warning(f"PKL文件 {filter_pkl_file} 中不包含cell_labels或gene_labels")
        except Exception as e:
            logger.error(f"读取过滤PKL文件失败: {e}")
            logger.info("将使用所有细胞和基因继续计算")
    
    # 获取唯一的细胞和基因
    cell_list = sorted(df['cell'].unique())
    gene_list = sorted(df['gene'].unique())
    
    logger.info(f"数据集包含 {len(cell_list)} 个细胞和 {len(gene_list)} 个基因")
    
    # 设置输出目录
    if output_dir is None:
        # 尝试从PKL文件路径获取完整的数据集名称
        filename = os.path.basename(pkl_file)
        # 去掉可能的后缀，如"_data_dict.pkl"
        dataset = filename.split('_data_dict')[0]
        if dataset == filename:  # 如果没有找到_data_dict后缀
            # 尝试其他可能的格式，比如直接去掉.pkl
            dataset = os.path.splitext(filename)[0]
            
        logger.info(f"从文件名 {filename} 中提取的数据集名称: {dataset}")
        output_dir = f"/lustre/home/1910305118/data/GCN_CL/1_input/{dataset}_portrait"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建可视化目录
    vis_dir = f"{output_dir}/visualization"
    os.makedirs(vis_dir, exist_ok=True)
    
    # 进行转录本数据分布分析
    logger.info("开始分析转录本数据分布特征...")
    try:
        analyze_transcript_distribution(df, output_dir)
    except Exception as e:
        logger.warning(f"转录本数据分析失败: {e}")
    
    # 阶段0: 全局预计算所有距离矩阵
    logger.info("阶段0: 全局预计算所有距离矩阵")
    dist_dict = {}  # 全局距离矩阵字典 {(cell, gene): 距离矩阵}
    
    # 使用线程池预计算所有距离矩阵
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        
        # 提交所有计算任务
        for gene in gene_list:
            gene_df = df[df['gene'] == gene]
            
            for cell in cell_list:
                cell_df = gene_df[gene_df['cell'] == cell]
                # 跳过没有足够数据的细胞
                if len(cell_df) <= 1:
                    continue
                
                # 定义本地函数计算距离矩阵
                def calc_dist_matrix(c_df):
                    positions = c_df[['x_c_s', 'y_c_s']].values
                    return distance_matrix(positions, positions)
                
                # 提交计算任务
                futures[(cell, gene)] = executor.submit(calc_dist_matrix, cell_df)
        
        # 收集计算结果
        for (cell, gene), future in tqdm(futures.items(), desc="预计算距离矩阵", disable=not sys.stdout.isatty()):
            try:
                dist_dict[(cell, gene)] = future.result()
            except Exception as e:
                logger.error(f"计算 cell={cell}, gene={gene} 的距离矩阵失败: {e}")
    
    logger.info(f"距离矩阵预计算完成，共 {len(dist_dict)} 个(cell,gene)对")
    
    # 如果需要自动设置参数，使用预计算的距离矩阵来计算
    if auto_r_min or auto_r_max or auto_bin_size:
        logger.info("基于预计算的距离矩阵自动设置参数")
        all_dists = []
        
        # 从预计算的距离矩阵中收集所有非零距离（使用全部数据，不限制采样数量）
        for key, dmat in tqdm(dist_dict.items(), desc="收集距离样本", disable=not sys.stdout.isatty()):
            # 获取上三角部分（排除自身距离）
            triu_indices = np.triu_indices_from(dmat, k=1)
            dists = dmat[triu_indices]
            # 排除0距离和无穷大距离
            valid_dists = dists[(dists > 0) & (np.isfinite(dists))]
            if len(valid_dists) > 0:
                all_dists.append(valid_dists)
        
        if all_dists:
            all_dists = np.concatenate(all_dists)
            logger.info(f"共收集了 {len(all_dists)} 个有效距离值")
            
            # 计算距离的分位数
            if auto_r_min:
                r_min = float(np.percentile(all_dists, min_percentile))
                r_min = round(r_min, 2)  # 保留两位小数
                logger.info(f"自动设置r_min = {r_min:.2f}（{min_percentile}%分位数）")
            
            if auto_r_max:
                r_max = float(np.percentile(all_dists, max_percentile))
                r_max = round(r_max, 2)  # 保留两位小数
                logger.info(f"自动设置r_max = {r_max:.2f}（{max_percentile}%分位数）")
            
            if auto_bin_size:
                # 设置bin_size为(r_max - r_min) / n_bins
                bin_size = float((r_max - r_min) / n_bins)
                bin_size = round(bin_size, 2)  # 保留两位小数
                bin_size = max(0.01, bin_size)  # 确保bin_size不会太小
                logger.info(f"自动设置bin_size = {bin_size:.2f}（距离范围的1/{n_bins}）")
        else:
            logger.warning("无法收集有效的距离样本，使用默认参数")
            if auto_r_min:
                r_min = 0.01
            if auto_r_max:
                r_max = 0.6
            if auto_bin_size:
                bin_size = 0.01
    
    # 确保参数有合理的默认值
    if r_min is None:
        r_min = 0.01
    if r_max is None:
        r_max = 0.6
    if bin_size is None:
        bin_size = 0.01
    
    logger.info(f"最终参数: r_min={r_min:.2f}, r_max={r_max:.2f}, bin_size={bin_size:.2f}")
    
    # 阶段1: 预计算所有图的网络肖像，包括自动选择最优r值
    logger.info("阶段1: 预计算网络肖像，包括自动选择最优r值")
    portraits = {}
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交基因级别的预计算任务
        futures = {
            executor.submit(
                precompute_portraits_for_gene, 
                gene, df, cell_list, threshold, bin_size, r_min, r_max, r_step, use_same_r, use_vectorized,
                dist_dict  # 传入全局距离矩阵字典
            ): gene for gene in gene_list
        }
        
        # 收集结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="预计算网络肖像", disable=not sys.stdout.isatty()):
            gene = futures[future]
            try:
                gene_portraits = future.result()
                portraits.update(gene_portraits)
                logger.info(f"基因 {gene} 预计算完成，共 {len(gene_portraits)} 个分布")
            except Exception as e:
                logger.error(f"基因 {gene} 预计算失败: {e}")
    
    logger.info(f"网络肖像预计算完成，共 {len(portraits)} 个细胞-基因对")
    
    # 阶段2: 计算JS散度
    logger.info("阶段2: 计算JS散度")
    all_distances = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交基因级别的JS散度计算任务
        futures = {
            executor.submit(
                find_js_distances_for_gene,
                gene, df, cell_list, portraits, 
                bin_size, max_count, transcript_window
            ): gene for gene in gene_list
        }
        
        # 收集结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="计算JS散度", disable=not sys.stdout.isatty()):
            gene = futures[future]
            try:
                gene_distances = future.result()
                all_distances.extend(gene_distances)
                logger.info(f"基因 {gene} JS散度计算完成，共 {len(gene_distances)} 个结果")
            except Exception as e:
                logger.error(f"基因 {gene} JS散度计算失败: {e}")
    
    # 清理不再需要的距离矩阵，释放内存
    dist_dict.clear()
    
    # 将结果保存到DataFrame
    if all_distances:
        distances_df = pd.DataFrame(
            all_distances, 
            columns=['target_cell', 'target_gene', 'cell', 'gene', 
                    'num_transcripts', 'js_distance', 'transcript_diff',
                    'target_r', 'other_r']
        )
        
        # 保存结果
        output_path = f"{output_dir}/js_distances_bin{bin_size:.4f}_count{max_count}_threshold{threshold}.csv"
        distances_df.to_csv(output_path, index=False)
        
        logger.info(f"\n计算完成，结果已保存至: {output_path}")
        logger.info(f"共计算出 {len(distances_df)} 条JS散度记录")
        
        # 可视化JS散度最小的前N个细胞对
        if visualize_top_n > 0:
            logger.info(f"开始可视化JS散度最小的前{visualize_top_n}个细胞对")
            visualize_most_similar_pairs(df, distances_df, portraits, visualize_top_n, vis_dir, use_vectorized)
        
        # 计算总耗时
        total_time = time.time() - start_time
        logger.info(f"总耗时: {total_time:.2f}秒")
        
        return distances_df
    else:
        logger.warning("警告：没有计算出任何JS散度")
        return pd.DataFrame()

def visualize_most_similar_pairs(df, distances_df, portraits, top_n=5, output_dir=None, use_vectorized=True):
    """
    可视化JS散度最小的前N个细胞对
    
    Args:
        df: 包含转录本信息的DataFrame
        distances_df: JS散度结果DataFrame
        portraits: 预计算的肖像字典
        top_n: 可视化的细胞对数量
        output_dir: 输出目录
        use_vectorized: 是否使用向量化实现计算网络肖像
    """
    # 按JS散度排序
    sorted_df = distances_df.sort_values('js_distance').reset_index(drop=True)
    
    # 确保输出目录存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 可视化前N个细胞对
    for i in range(min(top_n, len(sorted_df))):
        row = sorted_df.iloc[i]
        
        target_cell = row['target_cell']
        target_gene = row['target_gene']
        other_cell = row['cell']
        other_gene = row['gene']
        js_dist = row['js_distance']
        target_r = row['target_r']
        other_r = row['other_r']
        
        logger.info(f"第{i+1}相似的细胞对: {target_cell}:{target_gene} - {other_cell}:{other_gene}, JS散度: {js_dist:.4f}")
        
        # 提取转录本数据
        target_df = df[(df['cell'] == target_cell) & (df['gene'] == target_gene)]
        other_df = df[(df['cell'] == other_cell) & (df['gene'] == other_gene)]
        
        # 跳过没有足够转录本的情况
        if len(target_df) <= 1 or len(other_df) <= 1:
            logger.warning(f"细胞对 {target_cell}:{target_gene} - {other_cell}:{other_gene} 转录本数量不足，跳过可视化")
            continue
        
        # 构建图对象
        target_graph = build_weighted_graph(target_df, target_r)
        other_graph = build_weighted_graph(other_df, other_r)
        
        # 计算网络肖像
        target_portrait, _ = get_network_portrait(target_graph, bin_size=0.01, use_vectorized=use_vectorized)
        other_portrait, _ = get_network_portrait(other_graph, bin_size=0.01, use_vectorized=use_vectorized)
        
        # 创建细胞对子目录
        pair_dir = None
        if output_dir:
            pair_dir = f"{output_dir}/pair_{i+1}_js{js_dist:.4f}"
            os.makedirs(pair_dir, exist_ok=True)
        
        # 可视化图结构
        pair_prefix = f"第{i+1}名 JS={js_dist:.4f}: "
        
        if pair_dir:
            # 保存图结构
            plot_graph(target_graph, 
                      title=f"{pair_prefix}{target_cell}:{target_gene} (r={target_r:.2f})", 
                      save_path=f"{pair_dir}/cell1_graph.png")
            plot_graph(other_graph, 
                      title=f"{pair_prefix}{other_cell}:{other_gene} (r={other_r:.2f})", 
                      save_path=f"{pair_dir}/cell2_graph.png")
            
            # 保存网络肖像
            plot_portrait(target_portrait, 
                         title=f"{pair_prefix}{target_cell}:{target_gene} 网络肖像", 
                         save_path=f"{pair_dir}/cell1_portrait.png")
            plot_portrait(other_portrait, 
                         title=f"{pair_prefix}{other_cell}:{other_gene} 网络肖像", 
                         save_path=f"{pair_dir}/cell2_portrait.png")
            
            # 保存转录本坐标可视化
            plot_transcripts(target_df, target_r, 
                          title=f"{pair_prefix}{target_cell}:{target_gene} 转录本",
                          save_path=f"{pair_dir}/cell1_transcripts.png")
            plot_transcripts(other_df, other_r, 
                          title=f"{pair_prefix}{other_cell}:{other_gene} 转录本",
                          save_path=f"{pair_dir}/cell2_transcripts.png")
            
            logger.info(f"细胞对可视化已保存至: {pair_dir}")
        else:
            # 显示图结构
            plot_graph(target_graph, title=f"{pair_prefix}{target_cell}:{target_gene} (r={target_r:.2f})")
            plot_graph(other_graph, title=f"{pair_prefix}{other_cell}:{other_gene} (r={other_r:.2f})")
            
            # 显示网络肖像
            plot_portrait(target_portrait, title=f"{pair_prefix}{target_cell}:{target_gene} 网络肖像")
            plot_portrait(other_portrait, title=f"{pair_prefix}{other_cell}:{other_gene} 网络肖像")
            
            # 显示转录本坐标可视化
            plot_transcripts(target_df, target_r, title=f"{pair_prefix}{target_cell}:{target_gene} 转录本")
            plot_transcripts(other_df, other_r, title=f"{pair_prefix}{other_cell}:{other_gene} 转录本")

def plot_transcripts(df, r, title="转录本分布", save_path=None):
    """
    可视化转录本在空间中的分布
    
    Args:
        df: 包含转录本信息的DataFrame
        r: 连接半径
        title: 图标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制转录本位置
    plt.scatter(df['x_c_s'], df['y_c_s'], alpha=0.6, s=10)
    
    # 为每个转录本添加连接半径的圆
    for _, row in df.iterrows():
        circle = plt.Circle((row['x_c_s'], row['y_c_s']), r, fill=False, 
                           color='gray', alpha=0.2, linestyle='--')
        plt.gca().add_patch(circle)
        # plt.text(center_x, center_y + r + 0.1, f'r = {r:.3f}', ha='center', fontsize=10, color='red') # center_x, center_y 未定义
    
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_transcript_distribution(df, output_dir=None):
    """
    分析和报告转录本数据的分布情况
    
    Args:
        df: 包含转录本信息的DataFrame
        output_dir: 输出目录，如果为None则打印到日志
        
    Returns:
        Dict: 统计信息字典
    """
    logger.info("开始分析转录本数据分布...")
    
    # 基础统计
    total_transcripts = len(df)
    unique_genes = df['gene'].nunique()
    unique_cells = df['cell'].nunique()
    
    # 每个基因的转录本数量统计
    gene_transcript_counts = df.groupby('gene').size()
    
    if not gene_transcript_counts.empty:
        gene_stats_values = {
            'transcript_per_gene_mean': float(gene_transcript_counts.mean()),
            'transcript_per_gene_median': float(gene_transcript_counts.median()),
            'transcript_per_gene_std': float(gene_transcript_counts.std()) if not np.isnan(gene_transcript_counts.std()) else None,
            'transcript_per_gene_min': int(gene_transcript_counts.min()),
            'transcript_per_gene_max': int(gene_transcript_counts.max())
        }
    else:
        gene_stats_values = {
            'transcript_per_gene_mean': 0.0,
            'transcript_per_gene_median': 0.0,
            'transcript_per_gene_std': None,
            'transcript_per_gene_min': 0,
            'transcript_per_gene_max': 0
        }
    gene_stats = {'total_genes': int(unique_genes), **gene_stats_values}
    
    # 每个细胞的转录本数量统计
    cell_transcript_counts = df.groupby('cell').size()
    if not cell_transcript_counts.empty:
        cell_stats_values = {
            'transcript_per_cell_mean': float(cell_transcript_counts.mean()),
            'transcript_per_cell_median': float(cell_transcript_counts.median()),
            'transcript_per_cell_std': float(cell_transcript_counts.std()) if not np.isnan(cell_transcript_counts.std()) else None,
            'transcript_per_cell_min': int(cell_transcript_counts.min()),
            'transcript_per_cell_max': int(cell_transcript_counts.max())
        }
    else:
        cell_stats_values = {
            'transcript_per_cell_mean': 0.0,
            'transcript_per_cell_median': 0.0,
            'transcript_per_cell_std': None,
            'transcript_per_cell_min': 0,
            'transcript_per_cell_max': 0
        }
    cell_stats = {'total_cells': int(unique_cells), **cell_stats_values}
    
    # 每个(细胞,基因)对的转录本数量统计
    cell_gene_transcript_counts = df.groupby(['cell', 'gene']).size()
    if not cell_gene_transcript_counts.empty:
        pair_stats_values = {
            'transcript_per_pair_mean': float(cell_gene_transcript_counts.mean()),
            'transcript_per_pair_median': float(cell_gene_transcript_counts.median()),
            'transcript_per_pair_std': float(cell_gene_transcript_counts.std()) if not np.isnan(cell_gene_transcript_counts.std()) else None,
        }
    else:
        pair_stats_values = {
            'transcript_per_pair_mean': 0.0,
            'transcript_per_pair_median': 0.0,
            'transcript_per_pair_std': None,
        }
    pair_stats = {
        'total_cell_gene_pairs': int(len(cell_gene_transcript_counts)), # len() returns python int
        **pair_stats_values,
        'single_transcript_pairs': int((cell_gene_transcript_counts == 1).sum()),
        'multi_transcript_pairs': int((cell_gene_transcript_counts > 1).sum())
    }
    
    # 计算问题基因的比例
    genes_with_mostly_single_transcripts = 0
    if unique_genes > 0 and not cell_gene_transcript_counts.empty: # Avoid processing if no genes or no pairs
        for gene in df['gene'].unique():
            gene_pairs = cell_gene_transcript_counts[cell_gene_transcript_counts.index.get_level_values('gene') == gene]
            if not gene_pairs.empty:
                single_transcript_ratio = (gene_pairs == 1).sum() / len(gene_pairs)
                if single_transcript_ratio > 0.8:  # 80%以上的细胞只有1个转录本
                    genes_with_mostly_single_transcripts += 1
    
    problem_stats = {
        'genes_with_mostly_single_transcripts': int(genes_with_mostly_single_transcripts),
        'problematic_gene_ratio': float(genes_with_mostly_single_transcripts / unique_genes) if unique_genes > 0 else 0.0,
        'single_transcript_pair_ratio': float(pair_stats['single_transcript_pairs'] / pair_stats['total_cell_gene_pairs']) if pair_stats['total_cell_gene_pairs'] > 0 else 0.0
    }
    
    # 汇总统计信息
    stats_summary = {
        'total_transcripts': int(total_transcripts), # len() returns python int
        'gene_stats': gene_stats,
        'cell_stats': cell_stats,
        'pair_stats': pair_stats,
        'problem_stats': problem_stats
    }
    
    # 输出报告
    logger.info("=" * 60)
    logger.info("转录本数据分布分析报告")
    logger.info("=" * 60)
    logger.info(f"总转录本数量: {total_transcripts:,}")
    logger.info(f"唯一基因数量: {unique_genes:,}")
    logger.info(f"唯一细胞数量: {unique_cells:,}")
    logger.info("")
    
    logger.info("基因层面统计:")
    logger.info(f"  平均每个基因的转录本数: {gene_stats['transcript_per_gene_mean']:.1f}")
    logger.info(f"  中位数: {gene_stats['transcript_per_gene_median']:.1f}")
    logger.info(f"  范围: {gene_stats['transcript_per_gene_min']}-{gene_stats['transcript_per_gene_max']}")
    logger.info("")
    
    logger.info("细胞层面统计:")
    logger.info(f"  平均每个细胞的转录本数: {cell_stats['transcript_per_cell_mean']:.1f}")
    logger.info(f"  中位数: {cell_stats['transcript_per_cell_median']:.1f}")
    logger.info(f"  范围: {cell_stats['transcript_per_cell_min']}-{cell_stats['transcript_per_cell_max']}")
    logger.info("")
    
    logger.info("(细胞,基因)对层面统计:")
    logger.info(f"  总(细胞,基因)对数: {pair_stats['total_cell_gene_pairs']:,}")
    logger.info(f"  单转录本对数: {pair_stats['single_transcript_pairs']:,} ({pair_stats['single_transcript_pairs']/pair_stats['total_cell_gene_pairs']*100:.1f}%)")
    logger.info(f"  多转录本对数: {pair_stats['multi_transcript_pairs']:,} ({pair_stats['multi_transcript_pairs']/pair_stats['total_cell_gene_pairs']*100:.1f}%)")
    logger.info(f"  平均每对的转录本数: {pair_stats['transcript_per_pair_mean']:.1f}")
    logger.info("")
    
    logger.info("潜在问题分析:")
    logger.info(f"  主要为单转录本的基因数: {problem_stats['genes_with_mostly_single_transcripts']:,} ({problem_stats['problematic_gene_ratio']*100:.1f}%)")
    logger.info(f"  单转录本对占比: {problem_stats['single_transcript_pair_ratio']*100:.1f}%")
    
    if problem_stats['problematic_gene_ratio'] > 0.5:
        logger.warning("⚠️  超过50%的基因主要为单转录本，建议检查数据质量或调整分析参数")
    elif problem_stats['single_transcript_pair_ratio'] > 0.7:
        logger.warning("⚠️  超过70%的(细胞,基因)对只有单个转录本，这可能影响网络肖像的质量")
    else:
        logger.info("✓ 数据质量看起来不错，适合进行网络肖像分析")
    
    logger.info("=" * 60)
    
    # 如果指定了输出目录，保存详细统计到文件
    if output_dir:
        import json
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, "transcript_distribution_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"详细统计信息已保存到: {stats_file}")
        
        # 保存基因转录本分布图
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.hist(gene_transcript_counts, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('每个基因的转录本数量')
        plt.ylabel('基因数量')
        plt.title('基因转录本数量分布')
        plt.yscale('log')
        
        plt.subplot(2, 2, 2)
        plt.hist(cell_transcript_counts, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('每个细胞的转录本数量')
        plt.ylabel('细胞数量')
        plt.title('细胞转录本数量分布')
        plt.yscale('log')
        
        plt.subplot(2, 2, 3)
        plt.hist(cell_gene_transcript_counts, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('每个(细胞,基因)对的转录本数量')
        plt.ylabel('对数量')
        plt.title('(细胞,基因)对转录本数量分布')
        plt.yscale('log')
        
        plt.subplot(2, 2, 4)
        # 显示单转录本vs多转录本的比例
        labels = ['单转录本对', '多转录本对']
        sizes = [pair_stats['single_transcript_pairs'], pair_stats['multi_transcript_pairs']]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('单转录本对 vs 多转录本对')
        
        plt.tight_layout()
        dist_plot_file = os.path.join(output_dir, "transcript_distribution_plots.png")
        plt.savefig(dist_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"分布图已保存到: {dist_plot_file}")
    
    return stats_summary


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='使用JS散度计算转录本图之间的相似度')
    parser.add_argument('--pkl_file', type=str, required=True, 
                        help='包含df_registered的PKL文件路径')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='输出目录，不指定则使用默认路径')
    parser.add_argument('--max_count', type=int, default=10, 
                        help='每个细胞计算的最大距离数')
    parser.add_argument('--transcript_window', type=int, default=30, 
                        help='转录本数量差异窗口，用于初步筛选')
    parser.add_argument('--bin_size', type=str, default='0.01', 
                        help='路径长度分箱大小，用于JS散度计算，可设置为"auto"进行自动计算')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='孤立节点比例阈值，用于选择最优r值')
    parser.add_argument('--r_min', type=str, default='0.01',
                        help='r值搜索最小值，可设置为"auto"进行自动计算')
    parser.add_argument('--r_max', type=str, default='0.6',
                        help='r值搜索最大值，可设置为"auto"进行自动计算')
    parser.add_argument('--r_step', type=float, default=0.03,
                        help='r值搜索步长')
    parser.add_argument('--num_threads', type=int, default=None,
                        help='使用的线程数，默认为CPU核心数')
    parser.add_argument('--use_same_r', action='store_true',
                        help='对同一基因的所有细胞使用相同的r值(基因内的最大r值)')
    parser.add_argument('--visualize_top_n', type=int, default=5,
                        help='可视化JS散度最小的前N个细胞对，0表示不可视化')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别')
    parser.add_argument('--log_file', type=str, default='js_distance_transcriptome.log',
                        help='日志文件名')
    parser.add_argument('--no_vectorized', action='store_true', 
                        help='不使用向量化实现计算网络肖像(更慢但占用内存更少)')
    parser.add_argument('--filter_pkl_file', type=str, default=None,
                        help='用于筛选细胞-基因对的PKL文件路径，来自5_graph_data目录')
    parser.add_argument('--auto_params', action='store_true',
                        help='自动设置r_min, r_max和bin_size参数')
    parser.add_argument('--n_bins', type=int, default=50,
                        help='自动设置bin_size时使用的分箱数量')
    parser.add_argument('--min_percentile', type=float, default=1.0,
                        help='自动设置r_min时使用的百分位数')
    parser.add_argument('--max_percentile', type=float, default=99.0,
                        help='自动设置r_max时使用的百分位数')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logger.setLevel(getattr(logging, args.log_level))
    
    # 添加文件处理器
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)
    
    # 记录程序开始时间
    program_start_time = time.time()
    logger.info("=" * 80)
    logger.info("程序开始执行")
    logger.info(f"命令行参数: {vars(args)}")
    
    try:
        # 运行主函数
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
            max_percentile=args.max_percentile
        )
        
        # 输出结果统计
        if not distances_df.empty:
            logger.info("\n计算结果统计:")
            logger.info(f"总距离数: {len(distances_df)}")
            logger.info(f"距离范围: [{distances_df['js_distance'].min():.4f}, {distances_df['js_distance'].max():.4f}]")
            logger.info(f"平均距离: {distances_df['js_distance'].mean():.4f}")
            logger.info(f"中位数距离: {distances_df['js_distance'].median():.4f}")
            logger.info(f"r值范围: [{distances_df['target_r'].min():.2f}, {distances_df['target_r'].max():.2f}]")
            logger.info(f"平均r值: {distances_df['target_r'].mean():.2f}")
    
    # 输出前几行结果
            logger.info("\n计算结果预览:")
            logger.info(distances_df.head().to_string())
    
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # 计算并记录程序总运行时间
    program_end_time = time.time()
    total_program_time = program_end_time - program_start_time
    hours, remainder = divmod(total_program_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("=" * 80)
    logger.info(f"程序总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    logger.info(f"开始时间: {datetime.fromtimestamp(program_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结束时间: {datetime.fromtimestamp(program_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

print("=" * 50)