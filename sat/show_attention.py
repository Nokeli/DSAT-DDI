# import torch
# from torch_geometric.data import Data
# import seaborn as sns
# import matplotlib.pyplot as plt
# class CustomData(Data):
#     def __inc__(self, key, value,*args):
#         if key == 'rel_index':  # rel_index 是边类型，不进行偏移
#             return 0
#         return super().__inc__(key, value)
# # 加载数据
# data = torch.load('drug2_data.pt')  # 假设是 CustomDataBatch 对象
# attn_weights = torch.load('att_drug2.pt')  # 形状 [128, 2, 277, 277]
# # print(data)
# # print(attn_weights)
#
#
# def extract_single_sample(data, attn_weights, sample_idx=0):
#     """提取指定样本的子图和对应注意力矩阵"""
#     # 获取节点范围
#     start = data.ptr[sample_idx].item()
#     end = data.ptr[sample_idx + 1].item()
#     num_nodes = end - start
#
#     # 提取子图数据
#     node_mask = (data.batch == sample_idx)
#     edge_mask = node_mask[data.edge_index[0]]
#     edge_index = data.edge_index[:, edge_mask] - start  # 调整边索引为局部编号
#
#     # 提取注意力子矩阵 [num_heads, num_nodes, num_nodes]
#     sample_attn = attn_weights[sample_idx, :, :num_nodes, :num_nodes]
#
#     return {
#         "nodes": data.x[node_mask],  # 节点特征
#         "edge_index": edge_index,  # 边索引（局部编号）
#         "attention": sample_attn,  # 注意力矩阵
#         "num_nodes": num_nodes
#     }
#
#
# # 示例：提取第0个样本
# sample_data = extract_single_sample(data, attn_weights, sample_idx=0)
# attn = sample_data['attention']
# attn = (attn - attn.min()) / (attn.max() - attn.min())
# sample_data['attention'] = attn
# print(sample_data['attention'].shape)
# print(f"Sample 0: {sample_data['num_nodes']} nodes, {sample_data['edge_index'].shape[1]} edges")
#
# def plot_attention_heatmap(attention, head_idx=0):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(attention[head_idx].cpu().numpy(),
#                 cmap="viridis",
#                 xticklabels=False,
#                 yticklabels=False)
#     plt.title(f"Attention Head {head_idx}")
#     plt.xlabel("Key Nodes")
#     plt.ylabel("Query Nodes")
#     plt.show()
#
# # 可视化第0个头的注意力
# plot_attention_heatmap(sample_data["attention"], head_idx=1)
# import networkx as nx
#
#
# # def plot_graph_with_attention(edge_index, attention, head_idx=0, top_k=30):
# #     # 确保所有张量在CPU上（可视化不需要GPU）
# #     edge_index = edge_index.cpu()
# #     attention = attention.cpu()
# #
# #     # 生成三角矩阵索引（确保在CPU）
# #     num_nodes = attention.size(1)
# #     triu_indices = torch.triu_indices(num_nodes, num_nodes, 1)
# #
# #     # 获取注意力最高的top-k边
# #     attn_matrix = attention[head_idx]
# #     top_values, top_indices = torch.topk(attn_matrix[triu_indices[0], triu_indices[1]], k=top_k)
# #     top_edges = triu_indices[:, top_indices]  # 现在两者都在CPU上
# #
# #     # 绘制图（后续代码不变）
# #     G = nx.Graph()
# #     G.add_edges_from(edge_index.t().tolist())
# #     pos = nx.spring_layout(G, seed=42)
# #
# #     nx.draw(G, pos, with_labels=True, node_size=200, font_size=8, edge_color="gray", alpha=0.5)
# #     nx.draw_networkx_edges(
# #         G, pos, edgelist=top_edges.t().tolist(),
# #         edge_color="red", width=2.0, alpha=0.8
# #     )
# #     plt.title(f"Top-{top_k} Attention Edges (Head {head_idx})")
# #     plt.show()
# #
# import networkx as nx
# import matplotlib.pyplot as plt
# import torch
# import numpy as np
# from matplotlib.colors import LinearSegmentedColormap
#
#
# # def plot_graph_with_attention(edge_index, attention, head_idx=0, top_k=30):
# #     # 数据准备
# #     edge_index = edge_index.cpu()
# #     attention = attention.cpu()
# #     num_nodes = attention.size(1)
# #
# #     # 计算节点重要性（被关注度的均值）
# #     node_importance = attention[head_idx].mean(dim=1)  # [num_nodes]
# #
# #     # 创建图对象
# #     G = nx.Graph()
# #     G.add_edges_from(edge_index.t().tolist())
# #
# #     # 布局优化（Kamada-Kawai算法保持拓扑结构）
# #     pos = nx.kamada_kawai_layout(G, weight=None)
# #
# #     # ---- 可视化设置 ----
# #     plt.figure(figsize=(12, 10))
# #
# #     # 1. 绘制所有边（灰色细线）
# #     nx.draw_networkx_edges(
# #         G, pos,
# #         edge_color="lightgray",
# #         width=0.8,
# #         alpha=0.3
# #     )
# #
# #     # 2. 创建颜色映射（黄->橙->红）
# #     cmap = LinearSegmentedColormap.from_list('attn', ['#FFFF99', '#FFA500', '#FF0000'])
# #
# #     # 3. 绘制节点（颜色和大小映射重要性）
# #     nodes = nx.draw_networkx_nodes(
# #         G, pos,
# #         node_size=100 + 1000 * node_importance,  # 大小正比于重要性
# #         node_color=node_importance,
# #         cmap=cmap,
# #         vmin=0,
# #         vmax=node_importance.max(),
# #         edgecolors='black',
# #         linewidths=1
# #     )
# #
# #     # 4. 标注原子编号（仅标注重要节点）
# #     important_nodes = [i for i in range(num_nodes) if node_importance[i] > node_importance.mean()]
# #     nx.draw_networkx_labels(
# #         G, pos,
# #         labels={i: str(i) for i in important_nodes},
# #         font_size=10,
# #         font_weight='bold'
# #     )
# #
# #     # 5. 添加颜色条
# #     cbar = plt.colorbar(nodes, shrink=0.8)
# #     cbar.set_label('Attention Weight', fontsize=12)
# #
# #     plt.title(f"Molecular Graph with Node Attention (Head {head_idx})", fontsize=14)
# #     plt.axis('off')
# #     plt.tight_layout()
# #     plt.savefig('molecule_attention_nodes.png', dpi=300, bbox_inches='tight')
# #     plt.show()
# def plot_graph_with_attention(edge_index, attention, head_idx=0, top_k=30):
#     # 数据准备
#     edge_index = edge_index.cpu()
#     attention = attention.cpu()
#     num_nodes = attention.size(1)
#
#     # 计算节点重要性（被关注度的均值）
#     node_importance = attention[head_idx].mean(dim=1)  # [num_nodes]
#
#     # 创建图对象
#     G = nx.Graph()
#     G.add_edges_from(edge_index.t().tolist())
#
#     # 布局优化（Kamada-Kawai算法保持拓扑结构）
#     pos = nx.kamada_kawai_layout(G, weight=None)
#
#     # ---- 可视化设置 ----
#     plt.figure(figsize=(12, 10))
#
#     # 1. 绘制所有边（灰色细线） - 修改了alpha和width参数
#     nx.draw_networkx_edges(
#         G, pos,
#         edge_color="gray",  # 改为更明显的颜色
#         width=1.5,          # 加粗线条
#         alpha=0.8           # 提高不透明度
#     )
#
#     # 2. 创建颜色映射（黄->橙->红）
#     cmap = LinearSegmentedColormap.from_list('attn', ['#FFFF99', '#FFA500', '#FF0000'])
#
#     # 3. 绘制节点（颜色和大小映射重要性）
#     nodes = nx.draw_networkx_nodes(
#         G, pos,
#         node_size=100 + 1000 * node_importance,
#         node_color=node_importance,
#         cmap=cmap,
#         vmin=0,
#         vmax=node_importance.max(),
#         edgecolors='black',
#         linewidths=1
#     )
#
#     # 4. 标注原子编号（仅标注重要节点）
#     important_nodes = [i for i in range(num_nodes) if node_importance[i] > node_importance.mean()]
#     nx.draw_networkx_labels(
#         G, pos,
#         labels={i: str(i) for i in important_nodes},
#         font_size=10,
#         font_weight='bold'
#     )
#
#     # 5. 添加颜色条
#     cbar = plt.colorbar(nodes, shrink=0.8)
#     cbar.set_label('Attention Weight', fontsize=12)
#
#     plt.title(f"Molecular Graph with Node Attention (Head {head_idx})", fontsize=14)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('molecule_attention_nodes.png', dpi=300, bbox_inches='tight')
#     plt.show()
# # 示例：可视化第1个头的注意力边
# plot_graph_with_attention(sample_data["edge_index"], sample_data["attention"], head_idx=1)
#
# from rdkit import Chem
# from rdkit.Chem import Draw, AllChem
# import matplotlib.pyplot as plt
#
# # 转换为分子对象
# smiles = 'COC(=O)C1=CC=CC(=C1)C2=C(C=CC=C2)C(=O)N3CCN(CC3)C4CCCCC4'
# mol = Chem.MolFromSmiles(smiles)
# AllChem.Compute2DCoords(mol)  # 生成2D坐标
#
# # 绘制化学结构图
# Draw.MolToImage(mol)
#
#
import os
import sys
from tqdm.auto import tqdm
import numpy as np
import os
import torch
from torch.nn import functional as F
import pandas as pd
from PIL import Image
import io
np.seterr(divide='ignore', invalid='ignore')
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw  import SimilarityMaps
from rdkit.Chem.Draw import rdMolDraw2D
def visualize_attention(smiles, atom_weights, file_name):
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(mol)
    # print(mol.GetNumAtoms())
    # print(atom_weights.shape)
    Chem.rdDepictor.Compute2DCoords(mol)

    mean = np.mean(atom_weights)

    atom_weights = (mean-atom_weights).tolist()
    atom_weights = [i*2 if i<0 else i*5 for i in atom_weights]

    negative_indices = [i for i, w in enumerate(atom_weights) if w < 0]
    positive_indices = sorted(
        [i for i, w in enumerate(atom_weights) if w > 0],
        key=lambda i: atom_weights[i],
        reverse=True
    )[:5]
    selected_indices = set(negative_indices + positive_indices)
    filtered_weights = [
        atom_weights[i] if i in selected_indices else 0
        for i in range(len(atom_weights))
    ]

    drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
    N = mol.GetNumAtoms()
    # print(N)
    # top_indices = sorted(range(len(atom_weights)), key=lambda i: atom_weights[i], reverse=True)[:5]
    # top_indices = sorted(range(len(atom_weights)), key=lambda i: atom_weights[i], reverse=True)[-3:]

    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        contourLines=10,
        weights=filtered_weights,
        alpha=0.8,
        draw2d=drawer
    )
    drawer.FinishDrawing()
    # with open(file_name, "wb") as f:
    #     f.write(drawer.GetDrawingText())
    # 获取绘制的图像数据
    png = drawer.GetDrawingText()

    # 将 PNG 数据转换为 PIL 图像对象
    image = Image.open(io.BytesIO(png))

    # 保存为 TIFF 格式
    image.save(file_name, format='png')
#读取SMILES序列
SMILES = []
with open('../data/new_smiles.txt','r') as f:
    lines = f.readlines()
    for i in lines:
        i = i.strip()
        SMILES.append(i)
for i in [133,183]:
    file_path = 'false_drug2_att_{}.pt'.format(i)
    if os.path.exists(file_path):
        print("文件存在")
        attn = torch.load(file_path)
        #attn = attn.squeeze(0)  # 形状变为 [2, 23, 23]
        combined_attn = attn.mean(dim=2)  # 形状 [23, 23]

        # 步骤 3: 聚合原子权重（列求和 = 关注度聚合值）
        atom_weights = combined_attn.sum(dim=0)  # 形状 [23]

        # 转换到 numpy
        atom_weights = atom_weights.cpu().numpy()
        print(atom_weights)# 或 .cpu().numpy() 如果是GPU张量
        visualize_attention(SMILES[i], atom_weights, str(i))
    else:
        continue
    #attn = attn.squeeze(0)  # 形状变为 [2, 23, 23]

    # 步骤 2: 处理多头（取平均）
