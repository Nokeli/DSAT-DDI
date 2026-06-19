import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
class CustomData(Data):
    def __inc__(self, key, value,*args):
        if key == 'rel_index':  # rel_index 是边类型，不进行偏移
            return 0
        return super().__inc__(key, value)
kg = torch.load('drug_kg.pt')
edge_index = kg.edge_index.cpu()
rel = kg.rel_index.cpu().numpy()
attention = torch.load('drug1kg_133.pt')
edges = edge_index.t().tolist()
center_node = 4
# 创建有向图
src, tgt = edge_index
neighbors = torch.cat([tgt[src == center_node], src[tgt == center_node]]).unique().tolist()

# 获取 attention 分数（第 0 个 head）
attn_matrix = attention[0, 1]  # shape: [228, 228]
attn_weights = [(center_node, n, attn_matrix[center_node, n].item()) for n in neighbors]

# 归一化处理
raw_weights = [w for _, _, w in attn_weights]
nonzero_weights = [w for w in raw_weights if w > 0]

if nonzero_weights:
    min_w, max_w = min(nonzero_weights), max(nonzero_weights)
    def normalize(w):
        return 0 if w == 0 else 0.1 + 0.9 * (w - min_w) / (max_w - min_w)
    norm_weights = [normalize(w) for w in raw_weights]
else:
    norm_weights = [0.1 for _ in raw_weights]

# 构建图并使用归一化分数作为边权和标签
G = nx.DiGraph()
for (u, v, _), norm_w in zip(attn_weights, norm_weights):
    G.add_edge(u, v, weight=norm_w)

# 可视化
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(10, 6))
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight'] * 6 for u, v in G.edges()], alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=10)

# ✅ 使用归一化分数作为标签
edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Attention from Node 1 to Neighbors (Normalized Labels)", fontsize=13)
plt.axis('off')
plt.tight_layout()
plt.savefig("node1_attention_normalized_labels.pdf", dpi=600)
plt.show()