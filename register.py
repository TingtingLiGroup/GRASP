########################################################
# 细胞和核边界并行注册函数
# chunk_list - 辅助函数，用于将大列表分成小块，供并行处理函数使用
# normalize_dataset - 核心函数，用于将数据标准化，被所有处理函数调用
# specify_ntanbin - 用于确定每种细胞类型的角度分箱数量
# interpolate_boundary_points - 插值函数，用于增加边界点密度

# 处理单个细胞数据的函数：
# register_cells - 单线程版本，直接处理单个细胞
# register_cells_parallel_chunked + process_chunk_cell - 并行版本，只处理细胞数据
# process_chunk_cell 处理每个细胞块
# register_cells_parallel_chunked 管理并行处理

# 处理细胞和细胞核数据的函数：
# register_cells_and_nuclei_parallel_chunked + process_chunk - 基本版并行处理
# process_chunk 处理每个包含细胞和核的块
# register_cells_and_nuclei_parallel_chunked 管理并行处理

# 增强版处理函数：
# register_cells_and_nuclei_parallel_chunked_constrained + process_chunk - 增强版，提供更多选项
# 使用相同的 process_chunk 函数，但提供额外参数
# 增加了数据验证和统计功能

########################################################

import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import pickle
import timeit
import warnings
from tqdm import tqdm
import math
from math import pi
from multiprocessing import Pool, cpu_count
from functools import partial
import os 
from datetime import datetime
from scipy.interpolate import interp1d, splprep, splev

def interpolate_boundary_points(cell_boundary_dict, target_points_per_cell=100, method='spline', smooth_factor=0):
    """
    为细胞边界点进行插值，增加边界点密度
    
    Parameters:
    -----------
    cell_boundary_dict : dict
        细胞边界数据字典，格式：{cell_id: DataFrame with 'x', 'y' columns}
    target_points_per_cell : int, optional
        每个细胞目标边界点数量，默认100
    method : str, optional
        插值方法，'spline'（样条插值）或 'linear'（线性插值），默认'spline'
    smooth_factor : float, optional
        样条插值平滑因子，0表示通过所有点，值越大越平滑，默认0
        
    Returns:
    --------
    dict
        插值后的细胞边界数据字典
    """
    interpolated_boundary = {}
    
    print(f"开始对 {len(cell_boundary_dict)} 个细胞进行边界点插值...")
    
    for cell_id, boundary_df in tqdm(cell_boundary_dict.items(), desc="边界点插值"):
        try:
            # 获取原始边界点
            x_points = boundary_df['x'].values
            y_points = boundary_df['y'].values
            
            # 检查是否有足够的点进行插值
            if len(x_points) < 3:
                print(f"警告: 细胞 {cell_id} 边界点太少 ({len(x_points)} 个)，跳过插值")
                interpolated_boundary[cell_id] = boundary_df.copy()
                continue
            
            # 如果已经有足够的点，则减少目标点数
            if len(x_points) >= target_points_per_cell:
                interpolated_boundary[cell_id] = boundary_df.copy()
                continue
            
            # 确保边界是闭合的（首尾相接）
            if not (np.isclose(x_points[0], x_points[-1]) and np.isclose(y_points[0], y_points[-1])):
                x_points = np.append(x_points, x_points[0])
                y_points = np.append(y_points, y_points[0])
            
            if method == 'spline':
                # 使用样条插值
                # 计算累积弧长作为参数
                distances = np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2)
                distances = np.insert(distances, 0, 0)
                cumulative_distance = np.cumsum(distances)
                
                # 避免重复点导致的问题
                unique_indices = np.unique(cumulative_distance, return_index=True)[1]
                if len(unique_indices) < 3:
                    print(f"警告: 细胞 {cell_id} 唯一点太少，使用线性插值")
                    method = 'linear'
                else:
                    x_unique = x_points[unique_indices]
                    y_unique = y_points[unique_indices]
                    t_unique = cumulative_distance[unique_indices]
                    
                    # 参数化样条插值
                    try:
                        tck, u = splprep([x_unique, y_unique], s=smooth_factor, per=True)
                        
                        # 生成新的参数值
                        u_new = np.linspace(0, 1, target_points_per_cell, endpoint=False)
                        
                        # 计算插值点
                        x_new, y_new = splev(u_new, tck)
                        
                    except Exception as e:
                        print(f"警告: 细胞 {cell_id} 样条插值失败: {e}，改用线性插值")
                        method = 'linear'
            
            if method == 'linear':
                # 使用线性插值
                # 计算累积弧长
                distances = np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2)
                distances = np.insert(distances, 0, 0)
                cumulative_distance = np.cumsum(distances)
                
                # 归一化到 [0, 1]
                if cumulative_distance[-1] > 0:
                    normalized_distance = cumulative_distance / cumulative_distance[-1]
                else:
                    normalized_distance = cumulative_distance
                
                # 生成新的参数值
                t_new = np.linspace(0, 1, target_points_per_cell, endpoint=False)
                
                # 线性插值
                try:
                    x_interp = interp1d(normalized_distance, x_points, kind='linear', 
                                       assume_sorted=True, bounds_error=False, fill_value='extrapolate')
                    y_interp = interp1d(normalized_distance, y_points, kind='linear', 
                                       assume_sorted=True, bounds_error=False, fill_value='extrapolate')
                    
                    x_new = x_interp(t_new)
                    y_new = y_interp(t_new)
                    
                except Exception as e:
                    print(f"警告: 细胞 {cell_id} 线性插值失败: {e}，保持原始点")
                    interpolated_boundary[cell_id] = boundary_df.copy()
                    continue
            
            # 创建插值后的DataFrame
            interpolated_df = pd.DataFrame({
                'x': x_new,
                'y': y_new
            })
            
            interpolated_boundary[cell_id] = interpolated_df
            
        except Exception as e:
            print(f"错误: 处理细胞 {cell_id} 时发生异常: {e}")
            # 出错时保持原始数据
            interpolated_boundary[cell_id] = boundary_df.copy()
    
    # 统计信息
    original_points = sum(len(df) for df in cell_boundary_dict.values())
    new_points = sum(len(df) for df in interpolated_boundary.values())
    
    print(f"插值完成:")
    print(f"  原始边界点总数: {original_points}")
    print(f"  插值后边界点总数: {new_points}")
    print(f"  平均每个细胞原始点数: {original_points/len(cell_boundary_dict):.1f}")
    print(f"  平均每个细胞插值后点数: {new_points/len(interpolated_boundary):.1f}")
    
    return interpolated_boundary

def enhance_boundary_resolution(cell_boundary_dict, min_points_per_cell=50, adaptive=True):
    """
    自适应增强边界分辨率
    
    Parameters:
    -----------
    cell_boundary_dict : dict
        细胞边界数据字典
    min_points_per_cell : int, optional
        每个细胞最少边界点数量，默认50
    adaptive : bool, optional
        是否使用自适应策略，根据细胞大小调整目标点数，默认True
        
    Returns:
    --------
    dict
        增强后的细胞边界数据字典
    """
    enhanced_boundary = {}
    
    for cell_id, boundary_df in cell_boundary_dict.items():
        current_points = len(boundary_df)
        
        if adaptive:
            # 根据细胞周长估算合适的点数
            x_points = boundary_df['x'].values
            y_points = boundary_df['y'].values
            
            # 计算周长
            perimeter = 0
            for i in range(len(x_points)):
                next_i = (i + 1) % len(x_points)
                perimeter += np.sqrt((x_points[next_i] - x_points[i])**2 + 
                                   (y_points[next_i] - y_points[i])**2)
            
            # 根据周长确定目标点数 (每2-3个像素一个点)
            target_points = max(min_points_per_cell, int(perimeter / 2.5))
        else:
            target_points = min_points_per_cell
        
        # 如果当前点数足够，不进行插值
        if current_points >= target_points:
            enhanced_boundary[cell_id] = boundary_df.copy()
        else:
            # 进行插值
            temp_dict = {cell_id: boundary_df}
            interpolated = interpolate_boundary_points(
                temp_dict, 
                target_points_per_cell=target_points,
                method='spline'
            )
            enhanced_boundary[cell_id] = interpolated[cell_id]
    
    return enhanced_boundary

def register_cells(data_df, cell_list_all, cell_mask_df, ntanbin_dict, epsilon = 1e-10, nc_demo=None):
    if nc_demo is None:
        nc_demo = len(cell_list_all)
    dict_registered = {}
    cell_radii = {}  # 用来存储每个细胞的半径
    df = data_df.copy()  # cp original data
    df_gbC = df.groupby('cell', observed=False) # group by `cell`
    for ic, c in enumerate(tqdm(cell_list_all[:nc_demo], desc="Processing cells")):
        df_c = df_gbC.get_group(c).copy() # df for cell c
        t = df_c.type.iloc[0] # cell type for cell c
        mask_df_c = cell_mask_df[cell_mask_df.cell == c] # get the mask df for cell c
        center_c = [int(df_c.centerX.iloc[0]), int(df_c.centerY.iloc[0])] # nuclear center of cell c
        tanbin = np.linspace(0, pi/2, ntanbin_dict[t]+1)
        delta_tanbin = (2*math.pi)/(ntanbin_dict[t]*4)
        # add centered coord and ratio=y/x for df_c and mask_df_c
        df_c['x_c'] = df_c.x.copy() - center_c[0]
        df_c['y_c'] = df_c.y.copy() - center_c[1]
        df_c['d_c'] = (df_c.x_c.copy()**2+df_c.y_c.copy()**2)**0.5
        df_c['arctan'] = np.absolute(np.arctan(df_c.y_c / (df_c.x_c+epsilon)))
        mask_df_c['x_c'] = mask_df_c.x.copy() - center_c[0]
        mask_df_c['y_c'] = mask_df_c.y.copy() - center_c[1]
        mask_df_c['d_c'] = (mask_df_c.x_c.copy()**2+mask_df_c.y_c.copy()**2)**0.5
        mask_df_c['arctan'] = np.absolute(np.arctan(mask_df_c.y_c / (mask_df_c.x_c+epsilon)))
        # in each quatrant, find dismax_c for each tanbin interval using mask_df_c
        mask_df_c_q_dict = {}
        mask_df_c_q_dict['0'] = mask_df_c[(mask_df_c.x_c>=0) & (mask_df_c.y_c>=0)]
        mask_df_c_q_dict['1'] = mask_df_c[(mask_df_c.x_c<=0) & (mask_df_c.y_c>=0)]
        mask_df_c_q_dict['2'] = mask_df_c[(mask_df_c.x_c<=0) & (mask_df_c.y_c<=0)]
        mask_df_c_q_dict['3'] = mask_df_c[(mask_df_c.x_c>=0) & (mask_df_c.y_c<=0)]
        # compute the dismax_c
        dismax_c_mat = np.zeros((ntanbin_dict[t], 4))
        for q in range(4): # in each of the 4 quantrants
            mask_df_c_q = mask_df_c_q_dict[str(q)]
            mask_df_c_q['arctan_idx'] = (mask_df_c_q.arctan/delta_tanbin).astype(int) # arctan_idx from 0 to self.ntanbin_dict[t]-1
            dismax_c_mat[mask_df_c_q.groupby('arctan_idx').max()['d_c'].index.to_numpy(),q] = mask_df_c_q.groupby('arctan_idx').max()['d_c'].values # automatically sorted by arctan_idx from 0 to self.ntanbin_dict[t]-1

        # for df_c, for arctan in each interval, find max dis using dismax_c
        df_c_q_dict = {}
        df_c_q_dict['0'] = df_c[(df_c.x_c>=0) & (df_c.y_c>=0)]
        df_c_q_dict['1'] = df_c[(df_c.x_c<=0) & (df_c.y_c>=0)]
        df_c_q_dict['2'] = df_c[(df_c.x_c<=0) & (df_c.y_c<=0)]
        df_c_q_dict['3'] = df_c[(df_c.x_c>=0) & (df_c.y_c<=0)]
        d_c_maxc_dict = {}
        for q in range(4): # in each of the 4 quantrants
            df_c_q = df_c_q_dict[str(q)]
            d_c_maxc_q = np.zeros(len(df_c_q))
            df_c_q['arctan_idx'] = (df_c_q.arctan/delta_tanbin).astype(int) # arctan_idx from 0 to self.ntanbin_dict[t]-1
            for ai in range(ntanbin_dict[t]):
                d_c_maxc_q[df_c_q.arctan_idx.values==ai] = dismax_c_mat[ai,q]
            d_c_maxc_dict[str(q)] = d_c_maxc_q
        d_c_maxc = np.zeros(len(df_c))
        d_c_maxc[(df_c.x_c>=0) & (df_c.y_c>=0)] = d_c_maxc_dict['0']
        d_c_maxc[(df_c.x_c<=0) & (df_c.y_c>=0)] = d_c_maxc_dict['1']
        d_c_maxc[(df_c.x_c<=0) & (df_c.y_c<=0)] = d_c_maxc_dict['2']
        d_c_maxc[(df_c.x_c>=0) & (df_c.y_c<=0)] = d_c_maxc_dict['3']
        df_c['d_c_maxc'] = d_c_maxc

        # scale centered x_c and y_c 
        d_c_s = np.zeros(len(df_c))
        x_c_s = np.zeros(len(df_c))
        y_c_s = np.zeros(len(df_c))
        d_c_s = df_c.d_c/(df_c.d_c_maxc+epsilon)
        x_c_s = df_c.x_c*(d_c_s/(df_c.d_c+epsilon))
        y_c_s = df_c.y_c*(d_c_s/(df_c.d_c+epsilon))
        df_c['x_c_s'] = x_c_s
        df_c['y_c_s'] = y_c_s
        df_c['d_c_s'] = d_c_s

        # 保存当前细胞的最大半径
        cell_radii[c] = np.max(df_c['d_c_maxc'])

        dict_registered[c] = df_c
        del df_c
    # concatenate to one df
    df_registered = pd.concat(list(dict_registered.values()))
    print(f'Number of cells registered {len(dict_registered)}')
    return df_registered, cell_radii

def specify_ntanbin(cell_list_dict, cell_mask_df, type_list, nc4ntanbin=10, high_res = 200, max_ntanbin=25, input_ntanbin_dict=None, min_bp=5, min_ntanbin_error=3):
    ntanbin_dict = {}  # 初始化 ntanbin_dict 空字典
    if input_ntanbin_dict is not None: # use customized ntanbin across cell types
        ntanbin_dict = input_ntanbin_dict 

    if input_ntanbin_dict is None: # compute ntanbin for each cell type:
        for t in type_list:
            # specify ntanbin_gen based on cell seg mask/boundary
            # random sample self.nc4ntanbin cells, allow replace
            cell_list_sampled = np.random.choice(cell_list_dict[t], nc4ntanbin, replace=True)
            cell_mask_df_sampled = cell_mask_df[cell_mask_df.cell.isin(cell_list_sampled)]
            # compute                                                                                                                                                                                                                                                                                                                                                                                          the #x and #y unique coords of these sampled cells
            nxu_sampled = []
            nyu_sampled = []
            for c in cell_list_sampled:
                mask_c = cell_mask_df_sampled[cell_mask_df_sampled.cell==c]
                nxu_sampled.append(mask_c.x.nunique())
                nyu_sampled.append(mask_c.y.nunique())

            # specify ntanbin for pi/2 (a quantrant)
            # if resolution is super high
            if np.mean(nxu_sampled)>high_res and np.mean(nyu_sampled)>high_res:
                ntanbin=max_ntanbin
            # if resolution is not super high
            else:
                # require at least self.min_bp boundary points in each tanbin
                theta = 2*np.arctan(min_bp/np.mean(nxu_sampled+nyu_sampled))
                ntanbin_ = (pi/2)/theta
                ntanbin = np.ceil(ntanbin_)
                if ntanbin < min_ntanbin_error:
                    print(f'Cell type {t} failed, resolution not high enougth to support the analysis')
                    ntanbin = 3
            # asign
            ntanbin_dict[t] = int(ntanbin)
    return ntanbin_dict

def process_chunk_cell(chunk, df_gbC, cell_mask_df, ntanbin_dict, epsilon):
    results = []
    for c in chunk:
        df_c = df_gbC.get_group(c).copy() # df for cell c 获取细胞 c 的数据并复制出来
        t = df_c.type.iloc[0] # cell type for cell c 获取细胞类型 t
        mask_df_c = cell_mask_df[cell_mask_df.cell == c].copy() # get the mask df for cell c
        center_c = [int(df_c.centerX.iloc[0]), int(df_c.centerY.iloc[0])] # nuclear center of cell c
        tanbin = np.linspace(0, pi/2, ntanbin_dict[t]+1) # 将每个象限划分为多个扇区
        delta_tanbin = (2*math.pi)/(ntanbin_dict[t]*4) # 整个圆周 2π 被划分为 ntanbin_dict[t] * 4 个扇区
        mask_df_c['x_c'] = mask_df_c.x.copy() - center_c[0]
        mask_df_c['y_c'] = mask_df_c.y.copy() - center_c[1]
        mask_df_c['d_c'] = (mask_df_c.x_c.copy()**2+mask_df_c.y_c.copy()**2)**0.5
        mask_df_c['arctan'] = np.absolute(np.arctan(mask_df_c.y_c / (mask_df_c.x_c+epsilon)))
        mask_df_c_q_dict = {  # 将掩码数据划分为 4 个象限（四分之一圆）
            '0': mask_df_c[(mask_df_c.x_c >= 0) & (mask_df_c.y_c >= 0)],
            '1': mask_df_c[(mask_df_c.x_c <= 0) & (mask_df_c.y_c >= 0)],
            '2': mask_df_c[(mask_df_c.x_c <= 0) & (mask_df_c.y_c <= 0)],
            '3': mask_df_c[(mask_df_c.x_c >= 0) & (mask_df_c.y_c <= 0)]}    
        # compute the dismax_c
        dismax_c_mat = np.zeros((ntanbin_dict[t], 4)) # ntanbin_dict[t] 行和 4 列
        for q in range(4): # in each of the 4 quantrants
            mask_df_c_q = mask_df_c_q_dict[str(q)].copy()  # 创建副本，避免修改原始数据
            if len(mask_df_c_q) > 0:
                mask_df_c_q['arctan_idx'] = (mask_df_c_q['arctan'] / delta_tanbin).astype(int) # arctan_idx from 0 to self.ntanbin_dict[t]-1 # 是从 0 到 ntanbin_dict[t] - 1 的整数，表示每个点在当前象限内的角度区间
                # 确保arctan_idx不超出范围
                mask_df_c_q['arctan_idx'] = np.minimum(mask_df_c_q['arctan_idx'], ntanbin_dict[t]-1)
                max_distances = mask_df_c_q.groupby('arctan_idx').max()['d_c']
                if not max_distances.empty:
                    dismax_c_mat[max_distances.index.to_numpy(), q] = max_distances.values
        
        # 填充缺失的最大距离值
        for q in range(4):
            for ai in range(ntanbin_dict[t]):
                if dismax_c_mat[ai,q] == 0:
                    # 找相邻非零值进行填充
                    neighbors = [i for i in range(ntanbin_dict[t]) 
                               if dismax_c_mat[i,q] > 0]
                    if neighbors:
                        dismax_c_mat[ai,q] = np.mean([dismax_c_mat[i,q] 
                                                     for i in neighbors])
                    else:
                        # 如果没有非零邻居，使用该象限的最大距离
                        max_q = np.max(dismax_c_mat[:,q])
                        if max_q > 0:
                            dismax_c_mat[ai,q] = max_q
                        else:
                            # 如果整个象限都没有值，使用所有象限的最大值
                            max_all = np.max(dismax_c_mat)
                            if max_all > 0:
                                dismax_c_mat[ai,q] = max_all
                            else:
                                # 如果所有象限都没有值，使用一个默认值
                                dismax_c_mat[ai,q] = 100  # 设置一个较大的默认值
        
        # 一个包含每个角度区间最大距离的 Series，索引是 arctan_idx，值是对应的最大距离 行表示角度区间（由 arctan_idx 表示）。
        # 列表示象限编号 q（从 0 到 3）
        # 行表示分箱编号（arctan_idx），列表示象限编号
        # 最终结果：dismax_c_mat 存储了当前细胞核的所有分箱最大距离
        # add centered coord and ratio=y/x for df_c and mask_df_c
        df_c['x_c'] = df_c.x.copy() - center_c[0] # 相对于细胞中心的 x 和 y 坐标
        df_c['y_c'] = df_c.y.copy() - center_c[1] 
        df_c['d_c'] = (df_c.x_c.copy()**2+df_c.y_c.copy()**2)**0.5 # 点到中心的距离
        df_c['arctan'] = np.absolute(np.arctan(df_c.y_c / (df_c.x_c+epsilon)))  # 计算点与 x 轴的夹角
            
        # 标准化细胞和核边界数据
        df_c_registered = normalize_dataset(df_c, dismax_c_mat, delta_tanbin, ntanbin_dict, t, epsilon, is_nucleus=False, clip_to_cell=True)
        cell_radius = df_c_registered['d_c_maxc'].max()  # 计算细胞的最大半径
        results.append((df_c_registered, cell_radius))
    return results

def register_cells_parallel_chunked(data_df, cell_list_all, cell_mask_df, ntanbin_dict, epsilon=1e-10, nc_demo=None, chunk_size=5):
    if nc_demo is None:
        nc_demo = len(cell_list_all)
    df_gbC = data_df.groupby('cell', observed=False)  # 分组数据
    chunks = list(chunk_list(cell_list_all[:nc_demo], chunk_size))  # 将 cell_list_all 按块大小分组
    pool = Pool(processes=cpu_count() - 2)  # # 设置多进程池,留出部分 CPU 给系统
    process_chunk_partial = partial(process_chunk_cell, df_gbC=df_gbC, cell_mask_df=cell_mask_df,ntanbin_dict=ntanbin_dict,epsilon=epsilon)
    results = list(tqdm(pool.imap(process_chunk_partial, chunks), total=len(chunks), desc="Processing chunks in parallel"))   # 并行处理
    pool.close() # 关闭池
    pool.join() # 等待所有任务完成
    all_cell_dfs = [] # 聚合结果
    all_nuclear_dfs = []
    all_radii = {}
    for result_chunk in results:
        for df_c_registered, cell_radius in result_chunk:
            all_cell_dfs.append(df_c_registered)
            all_radii.update({df_c_registered['cell'].iloc[0]: cell_radius})# 将 cell_radius 转换为字典
    cell_df_registered = pd.concat(all_cell_dfs)
    return cell_df_registered, all_radii


def chunk_list(data_list, chunk_size):  # 将列表分块
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i + chunk_size]

def process_chunk(chunk, df_gbC, cell_mask_df, nuclear_boundary, ntanbin_dict, epsilon, clip_to_cell=True, remove_outliers=False, verbose=False):
    """
    处理一组细胞及其核边界
    
    Parameters:
    -----------
    chunk : list
        要处理的细胞ID列表
    df_gbC : pd.core.groupby.DataFrameGroupBy
        按细胞分组的DataFrame
    cell_mask_df : pd.DataFrame
        细胞掩码数据
    nuclear_boundary : dict
        核边界数据
    ntanbin_dict : dict
        每种细胞类型的角度分箱数量
    epsilon : float
        防止除以零的小常数
    clip_to_cell : bool, optional
        是否将核的d_c_s限制在1以内，默认True
    remove_outliers : bool, optional
        是否移除超出细胞边界的核点，默认False
    verbose : bool, optional
        是否输出详细信息，默认False
        
    Returns:
    --------
    list
        包含处理结果的列表，每个元素为 (df_c_registered, nuclear_boundary_c_registered, cell_radius)
    """
    results = []
    for c in chunk:
        try:
            df_c = df_gbC.get_group(c).copy() # df for cell c 获取细胞 c 的数据并复制出来
        except KeyError:
            if verbose:
                print(f"Warning: Cell {c} not found in data_df")
            continue
            
        t = df_c.type.iloc[0] # cell type for cell c 获取细胞类型 t
        
        # 尝试不同的查询方式
        mask_df_c = cell_mask_df[cell_mask_df.cell == c].copy()
        if len(mask_df_c) == 0:
            # 尝试转换为相同类型
            if isinstance(c, str):
                if verbose:
                    print(f"Converting cell {c} to string")
                mask_df_c = cell_mask_df[cell_mask_df.cell.astype(str) == c].copy()
            elif isinstance(c, (int, np.integer)):
                if verbose:
                    print(f"Converting cell {c} to integer")
                mask_df_c = cell_mask_df[cell_mask_df.cell.astype(int) == c].copy()
            
        if len(mask_df_c) == 0:
            if verbose:
                print(f"Warning: No mask points found for cell {c}")
            continue
        
        try:
            nuclear_boundary_c = nuclear_boundary[c].copy()  # 获取当前细胞核边界点
        except KeyError:
            if verbose:
                print(f"Warning: No nuclear boundary found for cell {c}")
            continue
            
        center_c = [int(df_c.centerX.iloc[0]), int(df_c.centerY.iloc[0])] # nuclear center of cell c
        tanbin = np.linspace(0, pi/2, ntanbin_dict[t]+1) # 将每个象限划分为多个扇区
        delta_tanbin = (2*math.pi)/(ntanbin_dict[t]*4) # 整个圆周 2π 被划分为 ntanbin_dict[t] * 4 个扇区
        
        # 处理掩码数据
        mask_df_c['x_c'] = mask_df_c.x.copy() - center_c[0]
        mask_df_c['y_c'] = mask_df_c.y.copy() - center_c[1]
        mask_df_c['d_c'] = (mask_df_c.x_c.copy()**2+mask_df_c.y_c.copy()**2)**0.5
        mask_df_c['arctan'] = np.absolute(np.arctan(mask_df_c.y_c / (mask_df_c.x_c+epsilon)))
        
        # 分象限
        mask_df_c_q_dict = {  # 将掩码数据划分为 4 个象限（四分之一圆）
            '0': mask_df_c[(mask_df_c.x_c >= 0) & (mask_df_c.y_c >= 0)],
            '1': mask_df_c[(mask_df_c.x_c <= 0) & (mask_df_c.y_c >= 0)],
            '2': mask_df_c[(mask_df_c.x_c <= 0) & (mask_df_c.y_c <= 0)],
            '3': mask_df_c[(mask_df_c.x_c >= 0) & (mask_df_c.y_c <= 0)]}
            
        # 计算每个角度区间的最大距离
        dismax_c_mat = np.zeros((ntanbin_dict[t], 4)) # ntanbin_dict[t] 行和 4 列
        for q in range(4): # in each of the 4 quantrants
            mask_df_c_q = mask_df_c_q_dict[str(q)].copy()  # 创建副本，避免修改原始数据
            if len(mask_df_c_q) > 0:
                mask_df_c_q['arctan_idx'] = (mask_df_c_q['arctan'] / delta_tanbin).astype(int) # arctan_idx from 0 to self.ntanbin_dict[t]-1 # 是从 0 到 ntanbin_dict[t] - 1 的整数，表示每个点在当前象限内的角度区间
                # 确保arctan_idx不超出范围
                mask_df_c_q['arctan_idx'] = np.minimum(mask_df_c_q['arctan_idx'], ntanbin_dict[t]-1)
                max_distances = mask_df_c_q.groupby('arctan_idx').max()['d_c']
                if not max_distances.empty:
                    dismax_c_mat[max_distances.index.to_numpy(), q] = max_distances.values  # automatically sorted by arctan_idx from 0 to self.ntanbin_dict[t]-1
        
        # 填充缺失的最大距离值
        fill_zero_indices = np.where(dismax_c_mat == 0)
        if len(fill_zero_indices[0]) > 0:
            for ai, q in zip(fill_zero_indices[0], fill_zero_indices[1]):
                # 找相邻非零值进行填充
                neighbors = []
                for offset in range(1, ntanbin_dict[t]):
                    ai_before = (ai - offset) % ntanbin_dict[t]
                    ai_after = (ai + offset) % ntanbin_dict[t]
                    if dismax_c_mat[ai_before, q] > 0:
                        neighbors.append(dismax_c_mat[ai_before, q])
                    if dismax_c_mat[ai_after, q] > 0:
                        neighbors.append(dismax_c_mat[ai_after, q])
                    if neighbors:  # 一旦找到非零邻居就停止
                        break
                        
                if neighbors:
                    dismax_c_mat[ai, q] = np.mean(neighbors)
                else:
                    # 如果没有非零邻居，使用该象限的所有非零值的平均值
                    nonzero_in_q = dismax_c_mat[:, q][dismax_c_mat[:, q] > 0]
                    if len(nonzero_in_q) > 0:
                        dismax_c_mat[ai, q] = np.mean(nonzero_in_q)
                    else:
                        # 如果整个象限都没有值，使用所有象限的非零值的平均值
                        all_nonzero = dismax_c_mat[dismax_c_mat > 0]
                        if len(all_nonzero) > 0:
                            dismax_c_mat[ai, q] = np.mean(all_nonzero)
                        else:
                            # 如果所有方向都没有边界点，使用一个默认值
                            dismax_c_mat[ai, q] = np.max(df_c['d_c']) * 1.5  # 使用基因点距离的1.5倍作为默认值
        
        # 一个包含每个角度区间最大距离的 Series，索引是 arctan_idx，值是对应的最大距离 行表示角度区间（由 arctan_idx 表示）。
        # 列表示象限编号 q（从 0 到 3）
        # 行表示分箱编号（arctan_idx），列表示象限编号
        # 最终结果：dismax_c_mat 存储了当前细胞核的所有分箱最大距离
        # add centered coord and ratio=y/x for df_c and mask_df_c
        df_c['x_c'] = df_c.x.copy() - center_c[0] # 相对于细胞中心的 x 和 y 坐标
        df_c['y_c'] = df_c.y.copy() - center_c[1] 
        df_c['d_c'] = (df_c.x_c.copy()**2+df_c.y_c.copy()**2)**0.5 # 点到中心的距离
        df_c['arctan'] = np.absolute(np.arctan(df_c.y_c / (df_c.x_c+epsilon)))  # 计算点与 x 轴的夹角
        # 核边界点的相对坐标和距离
        nuclear_boundary_c['x_c'] = nuclear_boundary_c.x.copy() - center_c[0]
        nuclear_boundary_c['y_c'] = nuclear_boundary_c.y.copy() - center_c[1]
        nuclear_boundary_c['d_c'] = (nuclear_boundary_c.x_c**2 + nuclear_boundary_c.y_c**2)**0.5
        nuclear_boundary_c['arctan'] = np.abs(np.arctan(nuclear_boundary_c.y_c / (nuclear_boundary_c.x_c + epsilon)))
        
        # 标准化细胞和核边界数据
        df_c_registered = normalize_dataset(df_c, dismax_c_mat, delta_tanbin, ntanbin_dict, t, epsilon, is_nucleus=False, clip_to_cell=True, remove_outliers=False)
        nuclear_boundary_c_registered = normalize_dataset(nuclear_boundary_c, dismax_c_mat, delta_tanbin, ntanbin_dict, t, epsilon, is_nucleus=True, clip_to_cell=clip_to_cell, remove_outliers=remove_outliers)
        nuclear_boundary_c_registered['cell'] = c
        
        # 计算超出边界的核点比例
        exceed_percent = 0
        if 'exceeds_boundary' in nuclear_boundary_c_registered.columns:
            exceed_percent = nuclear_boundary_c_registered['exceeds_boundary'].mean() * 100
            if exceed_percent > 0 and verbose:
                print(f"Cell {c}: {exceed_percent:.2f}% of nuclear boundary points exceed cell boundary")
        
        cell_radius = df_c_registered['d_c_maxc'].max()  # 计算细胞的最大半径
        
        results.append((df_c_registered, nuclear_boundary_c_registered, cell_radius))
    
    return results

def register_cells_and_nuclei_parallel_chunked(data_df, cell_list_all, cell_mask_df, nuclear_boundary, ntanbin_dict, epsilon=1e-10, nc_demo=None, chunk_size=2, clip_to_cell=True, remove_outliers=False,verbose=False):
    """
    并行处理细胞和核边界的函数
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        包含细胞数据的DataFrame
    cell_list_all : list
        所有细胞ID的列表
    cell_mask_df : pd.DataFrame
        细胞掩码数据
    nuclear_boundary : dict
        核边界数据
    ntanbin_dict : dict
        每种细胞类型的角度分箱数量
    epsilon : float, optional
        防止除以零的小常数，默认1e-10
    nc_demo : int, optional
        要处理的细胞数量，默认处理所有细胞
    chunk_size : int, optional
        每个处理块的大小，默认2
    clip_to_cell : bool, optional
        是否将核的d_c_s限制在1以内，默认True
    remove_outliers : bool, optional
        是否移除超出细胞边界的核点，默认False
        
    Returns:
    --------
    cell_df_registered : pd.DataFrame
        注册后的细胞数据
    nuclear_boundary_df_registered : pd.DataFrame
        注册后的核边界数据
    all_radii : dict
        每个细胞的最大半径
    """
    if nc_demo is None:
        nc_demo = len(cell_list_all)
    df_gbC = data_df.groupby('cell', observed=False)  # 分组数据
    chunks = list(chunk_list(cell_list_all[:nc_demo], chunk_size))  # 将 cell_list_all 按块大小分组
    # pool = Pool(processes=cpu_count() - 2)  # # 设置多进程池,留出部分 CPU 给系统
    pool = Pool(processes=min(4, cpu_count() - 2))  # 限制最多 4 个进程
    process_chunk_partial = partial(process_chunk, 
                                   df_gbC=df_gbC, 
                                   cell_mask_df=cell_mask_df,
                                   nuclear_boundary=nuclear_boundary,
                                   ntanbin_dict=ntanbin_dict,
                                   epsilon=epsilon,
                                   clip_to_cell=clip_to_cell,
                                   remove_outliers=remove_outliers,
                                   verbose=verbose)
    results = list(tqdm(pool.imap(process_chunk_partial, chunks), total=len(chunks), desc="Processing chunks in parallel"))   # 并行处理
    pool.close() # 关闭池
    pool.join() # 等待所有任务完成
    all_cell_dfs = [] # 聚合结果
    all_nuclear_dfs = []
    all_radii = {}
    for result_chunk in results:
        for df_c_registered, nuclear_boundary_c_registered, cell_radius in result_chunk:
            all_cell_dfs.append(df_c_registered)
            all_nuclear_dfs.append(nuclear_boundary_c_registered)
            all_radii.update({df_c_registered['cell'].iloc[0]: cell_radius})# 将 cell_radius 转换为字典
    cell_df_registered = pd.concat(all_cell_dfs)
    nuclear_boundary_df_registered = pd.concat(all_nuclear_dfs)
    return cell_df_registered, nuclear_boundary_df_registered, all_radii

def register_cells_and_nuclei_parallel_chunked_constrained(data_df, cell_list_all, cell_mask_df, nuclear_boundary, ntanbin_dict, epsilon=1e-10, nc_demo=None, chunk_size=5, clip_to_cell=True, remove_outliers=False, verbose=True):
    """
    限制细胞核边界在细胞边界内的并行处理函数
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        包含细胞数据的DataFrame
    cell_list_all : list
        所有细胞ID的列表
    cell_mask_df : pd.DataFrame
        细胞掩码数据
    nuclear_boundary : dict
        核边界数据
    ntanbin_dict : dict
        每种细胞类型的角度分箱数量
    epsilon : float, optional
        防止除以零的小常数，默认1e-10
    nc_demo : int, optional
        要处理的细胞数量，默认处理所有细胞
    chunk_size : int, optional
        每个处理块的大小，默认5
    clip_to_cell : bool, optional
        是否将核的d_c_s限制在1以内，默认True
    remove_outliers : bool, optional
        是否移除超出细胞边界的核点，默认False
    verbose : bool, optional
        是否输出详细信息，默认True
        
    Returns:
    --------
    cell_df_registered : pd.DataFrame
        注册后的细胞数据
    nuclear_boundary_df_registered : pd.DataFrame
        注册后的核边界数据
    all_radii : dict
        每个细胞的最大半径
    cell_nuclear_stats : pd.DataFrame
        每个细胞的核边界超出情况统计
    """
    if nc_demo is None:
        nc_demo = len(cell_list_all)
    
    # 首先验证数据完整性
    missing_cells_mask = [c for c in cell_list_all[:nc_demo] if c not in cell_mask_df['cell'].unique()]
    missing_cells_nuclear = [c for c in cell_list_all[:nc_demo] if c not in nuclear_boundary.keys()]
    
    if missing_cells_mask or missing_cells_nuclear:
        print(f"Warning: Found {len(missing_cells_mask)} cells missing in mask_df")
        print(f"Warning: Found {len(missing_cells_nuclear)} cells missing in nuclear_boundary")
        
        # 过滤掉缺失数据的细胞
        valid_cells = [c for c in cell_list_all[:nc_demo] 
                      if c in cell_mask_df['cell'].unique() and c in nuclear_boundary.keys()]
        print(f"Proceeding with {len(valid_cells)} valid cells (originally {nc_demo})")
        cell_list_for_processing = valid_cells
    else:
        cell_list_for_processing = cell_list_all[:nc_demo]
    
    # 分组数据并创建处理块
    df_gbC = data_df.groupby('cell', observed=False)
    chunks = list(chunk_list(cell_list_for_processing, chunk_size))
    
    # 创建多进程池
    pool = Pool(processes=min(4, cpu_count() - 2))
    process_chunk_partial = partial(process_chunk, 
                                   df_gbC=df_gbC, 
                                   cell_mask_df=cell_mask_df,
                                   nuclear_boundary=nuclear_boundary,
                                   ntanbin_dict=ntanbin_dict,
                                   epsilon=epsilon,
                                   clip_to_cell=clip_to_cell,
                                   remove_outliers=remove_outliers,
                                   verbose=verbose)
    
    # 并行处理
    results = list(tqdm(pool.imap(process_chunk_partial, chunks), 
                        total=len(chunks), 
                        desc="Processing chunks in parallel"))
    
    pool.close()
    pool.join()
    
    # 聚合结果
    all_cell_dfs = []
    all_nuclear_dfs = []
    all_radii = {}
    all_nuclear_stats = []
    
    for result_chunk in results:
        for df_c_registered, nuclear_boundary_c_registered, cell_radius in result_chunk:
            all_cell_dfs.append(df_c_registered)
            all_nuclear_dfs.append(nuclear_boundary_c_registered)
            all_radii.update({df_c_registered['cell'].iloc[0]: cell_radius})
            all_nuclear_stats.append(nuclear_stats)
    
    cell_df_registered = pd.concat(all_cell_dfs)
    nuclear_boundary_df_registered = pd.concat(all_nuclear_dfs)
    cell_nuclear_stats = pd.DataFrame(all_nuclear_stats)
    
    # 打印统计信息
    if verbose:
        cells_with_exceeding_nucleus = cell_nuclear_stats[cell_nuclear_stats['exceed_percent'] > 0]
        if not cells_with_exceeding_nucleus.empty:
            mean_exceed = cells_with_exceeding_nucleus['exceed_percent'].mean()
            max_exceed = cells_with_exceeding_nucleus['exceed_percent'].max()
            print(f"\nFound {len(cells_with_exceeding_nucleus)} cells with nucleus exceeding cell boundary")
            print(f"Average exceed percentage: {mean_exceed:.2f}%")
            print(f"Maximum exceed percentage: {max_exceed:.2f}%")
            print(f"After {'clipping' if clip_to_cell else 'leaving'} exceed points")
    
    return cell_df_registered, nuclear_boundary_df_registered, all_radii, cell_nuclear_stats

def normalize_dataset(dataset, dismax_c_mat, delta_tanbin, ntanbin_dict, t, epsilon=1e-10, is_nucleus=False, clip_to_cell=True, remove_outliers=False):
    """
    标准化数据集，将点相对于中心的距离标准化
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        要标准化的数据集
    dismax_c_mat : np.ndarray
        每个角度区间的最大距离矩阵
    delta_tanbin : float
        角度步长
    ntanbin_dict : dict
        每种细胞类型的角度分箱数量
    t : str
        细胞类型
    epsilon : float
        防止除以零的小常数
    is_nucleus : bool, optional
        是否是核数据，默认False
    clip_to_cell : bool, optional
        是否将d_c_s限制在1以内，默认True
    remove_outliers : bool, optional
        是否移除超出细胞边界的点，默认False
                
    Returns:
    --------
    pd.DataFrame
        标准化后的数据集
    """
    dataset_normalized = dataset.assign(
        d_c_maxc=np.zeros(len(dataset)),
        d_c_s=np.zeros(len(dataset)),
        x_c_s=np.zeros(len(dataset)),
        y_c_s=np.zeros(len(dataset))
    )
    
    # 记录超出边界的点
    if is_nucleus:
        dataset_normalized['exceeds_boundary'] = False
        
    for q in range(4):
        dataset_q = dataset[
            (dataset.x_c >= 0) & (dataset.y_c >= 0) if q == 0 else
            (dataset.x_c <= 0) & (dataset.y_c >= 0) if q == 1 else
            (dataset.x_c <= 0) & (dataset.y_c <= 0) if q == 2 else
            (dataset.x_c >= 0) & (dataset.y_c <= 0)
        ].copy()
        
        if len(dataset_q) > 0:
            dataset_q['arctan_idx'] = (dataset_q['arctan'] / delta_tanbin).astype(int)
            
            # 确保arctan_idx不超出范围
            dataset_q['arctan_idx'] = np.minimum(dataset_q['arctan_idx'], ntanbin_dict[t]-1)
            
            for ai in range(ntanbin_dict[t]):
                max_d = dismax_c_mat[ai, q]
                indices = dataset_q.index[dataset_q['arctan_idx'] == ai]
                dataset_normalized.loc[indices, 'd_c_maxc'] = max_d
    
    # 计算标准化距离
    dataset_normalized['d_c_s'] = dataset['d_c'] / (dataset_normalized['d_c_maxc'] + epsilon)
    
    # 如果是核数据，标记超出边界的点
    if is_nucleus:
        dataset_normalized['exceeds_boundary'] = dataset_normalized['d_c_s'] > 1 # 更新超出边界的标记
        
    # 如果需要移除超出边界的点且是核数据
    if remove_outliers and is_nucleus:
        dataset_normalized = dataset_normalized[~dataset_normalized['exceeds_boundary']]
        
    # 如果需要将d_c_s限制在1以内
    if clip_to_cell:
        # 限制d_c_s最大值为1
        dataset_normalized['d_c_s'] = np.minimum(dataset_normalized['d_c_s'], 1.0)
    
    # 计算标准化坐标
    dataset_normalized['x_c_s'] = dataset['x_c'] * dataset_normalized['d_c_s'] / (dataset['d_c'] + epsilon)
    dataset_normalized['y_c_s'] = dataset['y_c'] * dataset_normalized['d_c_s'] / (dataset['d_c'] + epsilon)
    
    return dataset_normalized
