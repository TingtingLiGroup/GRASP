import math
import os
import matplotlib.pyplot as plt
from shapely.wkt import loads
from shapely.geometry import Polygon
import pandas as pd
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from PIL import Image

## 绘制每个细胞归一化前的细胞核和细胞膜边界
def plot_raw_cell(dataset, cell_boundary, nuclear_boundary, path):
    save_dir= f'{path}/1_{dataset}_raw_cell_plot'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cells = list(cell_boundary.keys())
    num_cells = len(cells)
    for idx, cell in enumerate(cells):
        plt.figure(figsize=(4, 4))
        plt.plot(cell_boundary[cell]['x'], cell_boundary[cell]['y'], label='Cell Boundary', color='black')
        if cell in nuclear_boundary:   # 绘制细胞核边界
            plt.plot(nuclear_boundary[cell]['x'], nuclear_boundary[cell]['y'], label='Nucleus Boundary', color='red')
        plt.title(f'Cell: {cell}')
        save_path = os.path.join(save_dir, f'{cell}.png')
        plt.savefig(save_path)
        
        plt.close()  # 关闭当前绘图，节省内存
    print(f"All cell images have been saved to {save_dir}")

## 绘制每个细胞归一化前的转录本的分布（包括细胞核边界） 
def plot_raw_gene_distribution(dataset, cell_boundary, nuclear_boundary, df, path):
    cells = list(cell_boundary.keys())
    num_cells = len(cells)
    # 使用 tqdm 包装循环，添加进度条
    for idx, cell in tqdm(enumerate(cells), total=num_cells, desc="Processing cells"):
        cell_data = df[df['cell'] == cell]
        save_dir = f'{path}/{dataset}/raw_gene/cell_{cell}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for gene in cell_data['gene'].unique():
            plt.figure(figsize=(3, 3))
            plt.plot(cell_boundary[cell]['x'], cell_boundary[cell]['y'], label='Cell Boundary', color='black')
            if cell in nuclear_boundary:   # 绘制细胞核边界
                plt.plot(nuclear_boundary[cell]['x'], nuclear_boundary[cell]['y'], label='Nucleus Boundary', color='red')
            gene_data = cell_data[cell_data['gene'] == gene]
            plt.scatter(gene_data['x'], gene_data['y'], label=f'Gene: {gene}', s=3, alpha=0.5, color='blue')
            if dataset == 'simulated1' or dataset == 'simulated2'or dataset == 'simulated3':
                plt.title(f'{cell} - {gene}')
            elif dataset == 'merscope_liver_data2' or dataset == 'merscope_liver_data3' or dataset == 'merscope_liver_data4' :
                plt.title(f'Gene: {gene}')
            else:
                plt.title(f'Cell: {cell} - Gene: {gene}')
            # save_path = os.path.join(save_dir, f'{gene}.png')
            plt.axis('off')
            # plt.savefig(save_path)
            plt.savefig(f'{save_dir}/{gene}.png', format='png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{save_dir}/{gene}.pdf', format='pdf', bbox_inches='tight')
            plt.savefig(f'{save_dir}/{gene}.svg', format='svg', bbox_inches='tight')
            plt.close()  # 关闭当前绘图，节省内存
    print(f"All cell images have been saved to {save_dir}")

## 绘制每个细胞归一化前的转录本的分布（不包括细胞核边界）
def plot_raw_gene_distribution_without_nuclear(dataset, cell_boundary, df_registered, path):    
    # for cell_name, cell_data in cell_boundary.items():
    for cell_name, cell_data in tqdm(cell_boundary.items(), desc="Processing cells", leave=True):
        fig_path = f"{path}/{dataset}/raw_gene/{cell_name}"
        os.makedirs(fig_path, exist_ok=True)
        sub_df_registered = df_registered[df_registered['cell'] == cell_name]
        gene_list = sub_df_registered['gene'].unique()
        for gene in tqdm(gene_list, desc=f"Plotting for {cell_name}", leave=False):
            gene_data = sub_df_registered[sub_df_registered['gene'] == gene]
            plt.figure(figsize=(3, 3))
            cell_polygon = Polygon(cell_data[['x', 'y']])
            x, y = cell_polygon.exterior.xy
            plt.plot(x, y, linestyle='-', color='black', linewidth=1)
            plt.scatter(gene_data['x'], gene_data['y'], s=3, color='cornflowerblue')
            plt.title(f'{cell_name} - {gene}')
            plt.axis('off')
            plt.tight_layout() 
            # plt.savefig(f'{path}/{gene}.png', dpi=300)
            plt.savefig(f'{fig_path}/{gene}.png', format='png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{fig_path}/{gene}.pdf', format='pdf', bbox_inches='tight')
            plt.savefig(f'{fig_path}/{gene}.svg', format='svg', bbox_inches='tight')
            # plt.show()
            plt.close()

## 绘制每个基因的分布
def plot_gene_galleries_from_df(
    dataset_name,
    df_to_plot, # 这将是未经过滤的 df_registered
    cell_boundary_dict,
    nuclear_boundary_dict,
    output_base_path, # 例如: "../3_output_gene_galleries"
    plots_per_gallery=48, # 每张大图放多少个小图
    cols_per_gallery=6    # 每张大图的列数
    ):
    """
    为df_to_plot中的每个基因创建一个文件夹，并在其中生成一张或多张大的网格图。
    每张网格图包含该基因在不同细胞中的可视化。

    参数:
    - dataset_name (str): 数据集名称，主要用于信息展示或路径。
    - df_to_plot (pd.DataFrame): 包含基因表达点的数据框，
                                 需要有 'gene', 'cell', 'x', 'y' 列。
    - cell_boundary_dict (dict): 字典，键是细胞ID，值是包含细胞边界x, y坐标的字典
                                 (例如, {'cell1': {'x': [...], 'y': [...]}, ...})。
    - nuclear_boundary_dict (dict): 类似cell_boundary_dict，但用于核边界。
                                    如果某个细胞没有核边界，可以不存在对应的键。
    - output_base_path (str): 保存所有基因文件夹的根目录。
    - plots_per_gallery (int): 每张大的网格图最多包含的小图数量。
    - cols_per_gallery (int): 网格图中每行的图像数量。
    """

    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
        print(f"创建输出根目录: {output_base_path}")

    if not all(col in df_to_plot.columns for col in ['gene', 'cell', 'x', 'y']):
        print("错误: df_to_plot 必须包含 'gene', 'cell', 'x', 'y' 列。")
        return

    unique_genes = df_to_plot['gene'].unique()
    print(f"找到 {len(unique_genes)} 个独立基因进行处理。")

    for gene_name in unique_genes:
        print(f"\n正在处理基因: {gene_name}...")
        gene_specific_df = df_to_plot[df_to_plot['gene'] == gene_name]
        
        # 获取这个基因出现过的所有细胞
        cells_with_this_gene = gene_specific_df['cell'].unique()
        if len(cells_with_this_gene) == 0:
            print(f"基因 {gene_name} 没有在任何细胞中找到数据点。已跳过。")
            continue

        gene_output_folder = os.path.join(output_base_path, dataset_name, gene_name)
        if not os.path.exists(gene_output_folder):
            os.makedirs(gene_output_folder)

        num_cells_for_this_gene = len(cells_with_this_gene)
        
        # --- 开始为当前基因生成网格图 ---
        for i in range(0, num_cells_for_this_gene, plots_per_gallery):
            batch_cell_ids = cells_with_this_gene[i : i + plots_per_gallery]
            current_batch_size = len(batch_cell_ids)
            
            rows_this_gallery = math.ceil(current_batch_size / cols_per_gallery)
            
            fig, axes = plt.subplots(
                rows_this_gallery, 
                cols_per_gallery, 
                figsize=(cols_per_gallery * 3.5, rows_this_gallery * 3.5) # 每个小图大约3x3英寸
            )
            # 确保axes是2D的，方便索引
            if rows_this_gallery == 1 and cols_per_gallery == 1:
                axes = [[axes]]
            elif rows_this_gallery == 1:
                axes = [axes]
            elif cols_per_gallery == 1:
                axes = [[ax] for ax in axes]


            for plot_idx, cell_id in enumerate(batch_cell_ids):
                ax_row = plot_idx // cols_per_gallery
                ax_col = plot_idx % cols_per_gallery
                ax = axes[ax_row][ax_col]

                # 1. 绘制细胞边界
                if cell_id in cell_boundary_dict:
                    cb = cell_boundary_dict[cell_id]
                    ax.plot(cb['x'], cb['y'], color='black', linewidth=0.8)
                else:
                    ax.text(0.5, 0.5, "细胞边界缺失", ha='center', va='center', fontsize=8, color='red')
                
                # 2. 绘制核边界 (如果存在)
                if cell_id in nuclear_boundary_dict:
                    nb_data = nuclear_boundary_dict[cell_id]
                    # 检查 nb_data 是否是一个有效的字典，并且包含非空的 'x' 和 'y' 坐标
                    if isinstance(nb_data, dict) and \
                       'x' in nb_data and hasattr(nb_data['x'], '__len__') and len(nb_data['x']) > 0 and \
                       'y' in nb_data and hasattr(nb_data['y'], '__len__') and len(nb_data['y']) > 0:
                        ax.plot(nb_data['x'], nb_data['y'], color='dimgray', linestyle='--', linewidth=0.7)
                    elif isinstance(nb_data, pd.DataFrame) and not nb_data.empty and \
                         'x' in nb_data.columns and 'y' in nb_data.columns and \
                         len(nb_data['x']) > 0 and len(nb_data['y']) > 0:
                        ax.plot(nb_data['x'], nb_data['y'], color='dimgray', linestyle='--', linewidth=0.7)
                    # 如果 nb_data 可能是 None 或者空的字典/DataFrame，这里的条件会处理掉

                # 3. 绘制该基因在该细胞中的点
                points_in_cell_gene = gene_specific_df[gene_specific_df['cell'] == cell_id]
                ax.scatter(points_in_cell_gene['x'], points_in_cell_gene['y'], s=5, alpha=0.7, color='blue') # s=3, alpha=0.5

                ax.set_title(f"Cell: {cell_id}", fontsize=7)
                ax.axis('off') # 关闭坐标轴和刻度
                ax.set_aspect('equal', adjustable='box') # 保持比例

            # 关闭多余的子图
            for k in range(current_batch_size, rows_this_gallery * cols_per_gallery):
                ax_row = k // cols_per_gallery
                ax_col = k % cols_per_gallery
                fig.delaxes(axes[ax_row][ax_col])

            plt.tight_layout(pad=0.5)
            gallery_num = (i // plots_per_gallery) + 1
            save_path = os.path.join(gene_output_folder, f"{gene_name}_gallery_{gallery_num}.png")
            
            try:
                plt.savefig(save_path, dpi=200) # dpi可调整
                print(f"已保存网格图: {save_path}")
            except Exception as e:
                print(f"保存图像失败 {save_path}: {e}")
            plt.close(fig) # 关闭图像，释放内存

    print("\n已完成所有基因的处理。")

## 绘制每个细胞中每个基因归一化后的散点图（包括细胞核边界） 
def plot_register_gene_distribution(dataset, df_registered, path, nuclear_boundary_df_registered):
    cells = df_registered['cell'].unique()
    for cell in tqdm(cells, desc="Plotting per cell"):
        save_dir=f'{path}/{dataset}/registered_gene/{cell}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cell_gene_data = df_registered[df_registered['cell'] == cell]
        ## 提取核边界数据
        nuclear_boundary_df = nuclear_boundary_df_registered[nuclear_boundary_df_registered['cell'] == cell]
        if not cell_gene_data.empty:
            genes = cell_gene_data['gene'].unique()
            for gene in tqdm(genes, desc=f"Plotting for cell {cell}", leave=False):
                # print(f'Cell: {cell} - Gene: {gene}')
                plt.figure(figsize=(4, 4))
                radius = 1
                circle = plt.Circle((0, 0), radius, color='gray', fill=False, label='Cell Boundary', linewidth=1)  # 画细胞边界
                plt.gca().add_patch(circle)
                # 在 (0, 0) 处绘制红色十字标记
                # plt.axhline(0, color='red', linestyle='--', linewidth=0.1)  # 水平线
                # plt.axvline(0, color='red', linestyle='--', linewidth=0.1)  # 垂直线
                # 获取该基因的数据
                gene_data = cell_gene_data[cell_gene_data['gene'] == gene]
                # 绘制基因的散点图
                plt.scatter(gene_data['x_c_s'], gene_data['y_c_s'], label=f'Gene: {gene}', s=2, color='cornflowerblue')
                # 去掉边框
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                polygon_coords = list(zip(nuclear_boundary_df['x_c_s'], nuclear_boundary_df['y_c_s']))
                polygon = Polygon(polygon_coords)
                boundary_x, boundary_y = zip(*polygon_coords)
                ax.plot(boundary_x, boundary_y, color='darkgray', linewidth=1)
                # 去掉坐标轴和刻度
                plt.axis('off')
                if dataset == 'simulated1' or dataset == 'simulated2'or dataset == 'simulated3':
                    plt.title(f'{cell} - {gene}')
                else:
                    plt.title(f'Cell: {cell} - Gene: {gene}')
                # save_path = os.path.join(save_dir, f'{cell}_{gene}.png')
                # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.savefig(f'{save_dir}/{cell}_{gene}.png', format='png', dpi=300, bbox_inches='tight')
                plt.savefig(f'{save_dir}/{cell}_{gene}.pdf', format='pdf', bbox_inches='tight')
                plt.savefig(f'{save_dir}/{cell}_{gene}.svg', format='svg', bbox_inches='tight')
                plt.close()  # 关闭当前绘图，节省内存
    # print(f"All cell and gene images have been saved to {save_dir}")
    
 ## 绘制每个细胞中每个基因归一化后的散点图（不包括细胞核边界） 
def plot_register_gene_distribution_without_nuclear(dataset, df_registered, cell_radii, path):
    cells = df_registered['cell'].unique()
    for cell in tqdm(cells, desc="Plotting per cell"):
        save_dir=f'{path}/{dataset}/registered_gene/{cell}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cell_gene_data = df_registered[df_registered['cell'] == cell]
        if not cell_gene_data.empty:
            genes = cell_gene_data['gene'].unique()
            for gene in tqdm(genes, desc=f"Plotting for cell {cell}", leave=False):
                # print(f'Cell: {cell} - Gene: {gene}')
                plt.figure(figsize=(4, 4))
                radius = 1
                circle = plt.Circle((0, 0), radius, color='gray', fill=False, label='Cell Boundary', linewidth=1)  # 画细胞边界
                plt.gca().add_patch(circle)
                # 获取该基因的数据
                gene_data = cell_gene_data[cell_gene_data['gene'] == gene]
                # 绘制基因的散点图
                plt.scatter(gene_data['x_c_s'], gene_data['y_c_s'], label=f'Gene: {gene}', s=2, color='cornflowerblue')
                # 去掉边框
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                plt.axis('off')
                # 设置标题和图例
                plt.title(f'Cell: {cell} - Gene: {gene}')
                # save_path = os.path.join(save_dir, f'{cell}_{gene}.png')
                # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.savefig(f'{save_dir}/{cell}_{gene}.png', format='png', dpi=300, bbox_inches='tight')
                plt.savefig(f'{save_dir}/{cell}_{gene}.pdf', format='pdf', bbox_inches='tight')
                plt.savefig(f'{save_dir}/{cell}_{gene}.svg', format='svg', bbox_inches='tight')
                plt.close() 
               
def plot_each_batch(dataset, adata, batch, path):
    save_dir= f'{path}/{dataset}/each_batch'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    adata_sub = adata[adata.obs['batch'] == batch]
    points = adata_sub.uns['points']
    df = pd.DataFrame(points)
    df = df[df['batch'] == int(batch)]
    df['cell'] = df['cell'].astype(str)
    df['gene'] = df['gene'].astype(str)
    df = df[df['gene'].isin(set(adata.var_names))]
    df = df[df['cell'].isin(set(adata.obs_names))]
    gene_all = df['gene'].value_counts()
    cell_all = df['cell'].value_counts()
    cell_shape = adata_sub.obs["cell_shape"].to_frame()
    nucleus_shape = adata_sub.obs["nucleus_shape"].to_frame()
    plt.figure(figsize=(6, 4))
    for index, row in cell_shape.iterrows():
        polygon = loads(row['cell_shape'])
        x, y = polygon.exterior.xy # 提取多边形的外边界坐标
        plt.plot(x, y, linestyle='-', color='grey', linewidth=1)
        centroid = polygon.centroid  # 获取细胞的质心位置（多边形的中心）
        cx, cy = centroid.x, centroid.y
        plt.text(cx, cy, str(index), fontsize=10, ha='center', color='darkblue') # 在质心位置标记细胞编号
    for index, row in nucleus_shape.iterrows():
        polygon = loads(row['nucleus_shape'])
        x, y = polygon.exterior.xy # 提取多边形的外边界坐标
        plt.plot(x, y, linestyle='-', color='darkgray', linewidth=1)
    plt.title(f'Batch {batch}')
    plt.axis('off')
    plt.savefig(f'{save_dir}/batch{batch}_plot.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/batch{batch}_plot.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{save_dir}/batch{batch}_plot.svg', format='svg', bbox_inches='tight')
    # plt.savefig(f"{save_dir}/batch{batch}_plot.png", format='png', dpi=400)
    # plt.show()

# def plot_register_gene_distribution_simulated(dataset, points, cell_boundary, nuclear_boundary, path):
#     save_dir= f'{path}/{dataset}/raw_gene'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     cells = list(cell_boundary.keys())
#     # for idx, cell in enumerate(cells):
#     for idx, cell in tqdm(enumerate(cells), total=len(cells), desc="Processing cells", unit="cell"):
#         df = points[points['cell'] == cell]
#         gene_names = df['gene'].unique() 
#         # for gene_idx, gene_name in enumerate(gene_names):  # 遍历每个基因
#         for gene_idx, gene_name in tqdm(enumerate(gene_names), total=len(gene_names), desc=f"Processing genes for cell {cell}", leave=False, unit="gene"):
#             plt.figure(figsize=(3, 2))
#             plt.plot(cell_boundary[cell]['x'], cell_boundary[cell]['y'], label='Cell Boundary', color='black')
#             if cell in nuclear_boundary:   # 绘制细胞核边界
#                 plt.plot(nuclear_boundary[cell]['x'], nuclear_boundary[cell]['y'], label='Nucleus Boundary', color='red')
#             gene_points = df[df['gene'] == gene_name][['x', 'y']].values # 获取基因的空间点数据
#             if len(gene_points) > 0:
#                 x_coords = gene_points[:, 0]
#                 y_coords = gene_points[:, 1]
#                 plt.scatter(x_coords, y_coords, s=2, color="cornflowerblue", label=f"Gene: {gene_name}")
#             plt.axis('off')
#             # 保存图像
#             cell_save_dir = os.path.join(save_dir, f"cell_{cell}")
#             if not os.path.exists(cell_save_dir):
#                 os.makedirs(cell_save_dir)
#             plt.title(f"{cell} - {gene_name}", fontsize=8)
#             plt.savefig(f'{cell_save_dir}/{gene_name}.png', format='png', dpi=300, bbox_inches='tight')
#             plt.savefig(f'{cell_save_dir}/{gene_name}.pdf', format='pdf', bbox_inches='tight')
#             plt.savefig(f'{cell_save_dir}/{gene_name}.svg', format='svg', bbox_inches='tight')
#             plt.close() 
            
# def plot_register_gene_distribution_kde_simulated(dataset, points, cell_boundary, nuclear_boundary, path):
#     save_dir = f'{path}/2_{dataset}_raw_gene_kde'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     cells = list(cell_boundary.keys())
#     # 使用 tqdm 包装外层循环，为细胞添加进度条
#     for idx, cell in tqdm(enumerate(cells), total=len(cells), desc="Processing cells", unit="cell"):
#         df = points[points['cell'] == cell]
#         gene_names = df['gene'].unique()
#         # 使用 tqdm 包装内层循环，为基因添加进度条
#         for gene_idx, gene_name in tqdm(enumerate(gene_names), total=len(gene_names), desc=f"Processing genes for cell {cell}", leave=False, unit="gene"):
#             plt.figure(figsize=(3, 2))
#             # 绘制细胞边界
#             plt.plot(cell_boundary[cell]['x'], cell_boundary[cell]['y'], label='Cell Boundary', color='black')
#             # 如果有细胞核边界，则绘制细胞核边界
#             if cell in nuclear_boundary:
#                 plt.plot(nuclear_boundary[cell]['x'], nuclear_boundary[cell]['y'], label='Nucleus Boundary', color='red')
#             # 获取基因的空间点数据
#             gene_points = df[df['gene'] == gene_name][['x', 'y']].values
#             if len(gene_points) > 0:
#                 x_coords = gene_points[:, 0]
#                 y_coords = gene_points[:, 1]
#                 # 使用 seaborn 绘制密度图
#                 sns.kdeplot(x=x_coords, y=y_coords, fill=True, cmap="Blues", alpha=0.5, levels=5, thresh=0.1)
#             # 设置标题和关闭坐标轴
#             plt.title(f"{cell} - {gene_name}", fontsize=8)
#             plt.axis('off')
#             # 保存图像
#             cell_save_dir = os.path.join(save_dir, f"cell_{cell}")
#             if not os.path.exists(cell_save_dir):
#                 os.makedirs(cell_save_dir)
#             plt.savefig(f"{cell_save_dir}/{gene_name}.png", format='png', dpi=300)
#             plt.close()
   
   
# # def plot_raw_gene_distribution(dataset, df, type="dot"): # dot/kde
#     num_cells = len(df) # 计算子图数量
#     cols = 8 # 每行显示5个图像
#     rows = (num_cells // cols) + (num_cells % cols > 0)  # 计算行数
#     fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 2)) # 创建子图
#     axes = axes.flatten()  # 将二维的axes展平成一维，便于索引
#     if type == "dot":
#         for i, (gene, cell) in enumerate(zip(df["gene"], df["cell"])): # 遍历每个细胞并加载图像
#             if dataset == "merfish_u2os" or dataset == "seqfish":
#                 image_path = os.path.join(f"/home/lixiangyu/hyy/GCN_CL/2_scaled_cell/2_{dataset}_raw_gene/cell_{cell}/{gene}.png")
#             elif dataset == "simulated":
#                 image_path = os.path.join(f"/home/lixiangyu/hyy/GCN_CL/2_scaled_cell/2_{dataset}_raw_gene/cell_{cell}/{gene}.png")       
#             else:
#                 image_path = os.path.join(f"/home/lixiangyu/hyy/GCN_CL/2_scaled_cell/2_{dataset}_raw_gene/{cell}/{gene}.png")       
            
#             if os.path.exists(image_path):
#                 img = Image.open(image_path) 
#                 axes[i].imshow(img)
#                 axes[i].set_title(f"Cell: {cell} - {gene}")
#                 axes[i].axis("off")  # 关闭坐标轴
#             else:
#                 axes[i].axis("off")
#                 axes[i].set_title(f"Cell: {cell}\nImage not found")
#         for j in range(len(df), len(axes)):
#             axes[j].axis("off")  # 关闭多余的子图坐标轴

#         plt.tight_layout()
#         plt.show()
#     else:
#         for i, (gene, cell) in enumerate(zip(df["gene"], df["cell"])): # 遍历每个细胞并加载图像
#             if dataset == "merfish_u2os" or dataset == "seqfish":
#                 image_path = os.path.join(f"/home/lixiangyu/hyy/GCN_CL/2_scaled_cell/2_{dataset}_raw_gene_kde/cell_{cell}/{gene}.png")
#             elif dataset == "simulated":
#                 image_path = os.path.join(f"/home/lixiangyu/hyy/GCN_CL/2_scaled_cell/2_{dataset}_raw_gene_kde/cell_{cell}/{gene}.png")      
#             else:
#                 image_path = os.path.join(f"/home/lixiangyu/hyy/GCN_CL/2_scaled_cell/2_{dataset}_raw_gene_kde/{cell}/{gene}.png")       
            
#             if os.path.exists(image_path):
#                 img = Image.open(image_path) 
#                 axes[i].imshow(img)
#                 axes[i].set_title(f"Cell: {cell} - {gene}")
#                 axes[i].axis("off")  # 关闭坐标轴
#             else:
#                 axes[i].axis("off")
#                 axes[i].set_title(f"Cell: {cell}\nImage not found")
#         for j in range(len(df), len(axes)):
#             axes[j].axis("off")  # 关闭多余的子图坐标轴

#         plt.tight_layout()
#         plt.show()