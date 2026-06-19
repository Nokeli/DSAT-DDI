# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
from utils import pad_batch, unpad_batch
from gnn_layers import get_simple_gnn_layer, EDGE_GNN_TYPES
import torch.nn.functional as F


class Attention(gnn.MessagePassing):
    """Multi-head Structure-Aware attention using PyG interface
    accept Batch data given by PyG

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    num_heads (int):        number of attention heads (default: 8)
    dropout (float):        dropout value (default: 0.0)
    bias (bool):            whether layers have an additive bias (default: False)
    symmetric (bool):       whether K=Q in dot-product attention (default: False)
    gnn_type (str):         GNN type to use in structure extractor. (see gnn_layers.py for options)
    se (str):               type of structure extractor ("gnn", "khopgnn")
    k_hop (int):            number of base GNN layers or the K hop size for khopgnn structure extractor (default=2).
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False,
        symmetric=False, gnn_type="gcn", se="gnn", k_hop=2, **kwargs):

        super().__init__(node_dim=0, aggr='add')
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.se = se

        self.gnn_type = gnn_type
        if self.se == "khopgnn":
            self.khop_structure_extractor = KHopStructureExtractor(embed_dim, gnn_type=gnn_type,
                                                          num_layers=k_hop, **kwargs)
        else:
            self.structure_extractor = StructureExtractor(embed_dim, gnn_type=gnn_type,
                                                          num_layers=k_hop, **kwargs)
        self.attend = nn.Softmax(dim=-1)

        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 关系感知偏置：把边的关系嵌入投影成每个头一个标量，加到注意力分数上。
        # 用于 KG 通道(dense within-subgraph attention)，对标 TIGER 的 relation-aware attention。
        self.rel_proj = nn.Linear(embed_dim, num_heads)

        self._reset_parameters()

        self.attn_sum = None

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
            x,
            edge_index,
            complete_edge_index,
            subgraph_node_index=None,
            subgraph_edge_index=None,
            subgraph_indicator_index=None,
            subgraph_edge_attr=None,
            edge_attr=None,
            ptr=None,
            return_attn=False):
        """
        Compute attention layer.

        Args:
        ----------
        x:                          input node features
        edge_index:                 edge index from the graph
        complete_edge_index:        edge index from fully connected graph
        subgraph_node_index:        documents the node index in the k-hop subgraphs
        subgraph_edge_index:        edge index of the extracted subgraphs
        subgraph_indicator_index:   indices to indicate to which subgraph corresponds to which node
        subgraph_edge_attr:         edge attributes of the extracted k-hop subgraphs
        edge_attr:                  edge attributes
        return_attn:                return attention (default: False)

        """
        # Compute value matrix

        v = self.to_v(x)

        # Compute structure-aware node embeddings
        if self.se == 'khopgnn': # k-subgraph SAT
            x_struct = self.khop_structure_extractor(
                x=x,
                edge_index=edge_index,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_attr=subgraph_edge_attr,
            )
        else: # k-subtree SAT
            x_struct = self.structure_extractor(x, edge_index, edge_attr)


        # Compute query and key matrices
        if self.symmetric:
            qk = self.to_qk(x_struct)
            qk = (qk, qk)
        else:
            qk = self.to_qk(x_struct).chunk(2, dim=-1)

        # Compute complete self-attention
        attn = None

        if complete_edge_index is not None:
            out = self.propagate(complete_edge_index, v=v, qk=qk, edge_attr=None, size=None,
                                 return_attn=return_attn)
            if return_attn:
                attn = self._attn
                self._attn = None
                attn = torch.sparse_coo_tensor(
                    complete_edge_index,
                    attn,
                ).to_dense().transpose(0, 1)

            out = rearrange(out, 'n h d -> n (h d)')
        else:
            out, attn = self.self_attn(qk, v, ptr, edge_index=edge_index,
                                       edge_attr=edge_attr, return_attn=return_attn)
        return self.out_proj(out), attn

    def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn):
        """Self-attention operation compute the dot-product attention """

        qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
        qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
        v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)
        attn = (qk_i * qk_j).sum(-1) * self.scale
        if edge_attr is not None:
            attn = attn + edge_attr
        attn = utils.softmax(attn, index, ptr, size_i)
        if return_attn:
            self._attn = attn
        attn = self.attn_dropout(attn)

        return v_j * attn.unsqueeze(-1)

    def self_attn(self, qk, v, ptr, edge_index=None, edge_attr=None, return_attn=False):
        """ Self attention which can return the attn """

        qk, mask = pad_batch(qk, ptr, return_mask=True)
        k, q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, N, N]

        # 关系感知偏置：把稀疏边的关系嵌入按 (graph, local_src, local_dst) 写入 dense 偏置矩阵，
        # 加到注意力分数上(只在真实边的节点对上有偏置)。多关系平行边用 accumulate 求和。
        if edge_attr is not None and edge_index is not None:
            bsz = len(ptr) - 1
            max_num_nodes = q.shape[-2]
            num_nodes = ptr[1:] - ptr[:-1]
            rel_bias = self.rel_proj(edge_attr)                       # [E, H]
            src, dst = edge_index[0], edge_index[1]                   # 全局节点索引
            node_batch = torch.repeat_interleave(
                torch.arange(bsz, device=ptr.device), num_nodes)     # 每个节点所属子图
            g = node_batch[src]                                      # [E]
            local_src = src - ptr[g]
            local_dst = dst - ptr[g]
            bias = dots.new_zeros(bsz, max_num_nodes, max_num_nodes, self.num_heads)
            bias.index_put_((g, local_src, local_dst), rel_bias, accumulate=True)
            dots = dots + bias.permute(0, 3, 1, 2)                    # [B, H, N, N]

        dots = dots.masked_fill(
            mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )

        dots = self.attend(dots)
        dots = self.attn_dropout(dots)

        v = pad_batch(v, ptr)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = unpad_batch(out, ptr)

        if return_attn:
            return out, dots
        return out, None


class StructureExtractor(nn.Module):
    r""" K-subtree structure extractor. Computes the structure-aware node embeddings using the
    k-hop subtree centered around each node.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers
    batch_norm (bool):      apply batch normalization or not
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree
    """

    def __init__(self, embed_dim, gnn_type="gcn", num_layers=3,
                 batch_norm=True, concat=False, khopgnn=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn
        self.concat = concat
        self.gnn_type = gnn_type
        layers = []
        for _ in range(num_layers):
            layers.append(get_simple_gnn_layer(gnn_type, embed_dim, **kwargs))
        self.gcn = nn.ModuleList(layers)

        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        inner_dim = (num_layers + 1) * embed_dim if concat else embed_dim

        if batch_norm:
            self.bn = nn.BatchNorm1d(inner_dim)

        self.out_proj = nn.Linear(inner_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None,
            subgraph_indicator_index=None, agg="sum"):
        x_cat = [x]
        for gcn_layer in self.gcn:
            # if self.gnn_type == "attn":
            #     x = gcn_layer(x, edge_index, None, edge_attr=edge_attr)
            if self.gnn_type in EDGE_GNN_TYPES:
                if edge_attr is None:
                    x = self.relu(gcn_layer(x, edge_index))
                else:
                    x = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
            else:
                x = self.relu(gcn_layer(x, edge_index))

            if self.concat:
                x_cat.append(x)

        if self.concat:
            x = torch.cat(x_cat, dim=-1)

        if self.khopgnn:
            if agg == "sum":
                x = scatter_add(x, subgraph_indicator_index, dim=0)
            elif agg == "mean":
                x = scatter_mean(x, subgraph_indicator_index, dim=0)
            return x

        if self.num_layers > 0 and self.batch_norm:
            x = self.bn(x)

        x = self.out_proj(x)
        return x


class KHopStructureExtractor(nn.Module):
    r""" K-subgraph structure extractor. Extracts a k-hop subgraph centered around
    each node and uses a GNN on each subgraph to compute updated structure-aware
    embeddings.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree (True)
    """
    def __init__(self, embed_dim, gnn_type="gcn", num_layers=3, batch_norm=True,
            concat=True, khopgnn=True, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn

        self.batch_norm = batch_norm

        self.structure_extractor = StructureExtractor(
            embed_dim,
            gnn_type=gnn_type,
            num_layers=num_layers,
            concat=False,
            khopgnn=True,
            **kwargs
        )

        if batch_norm:
            self.bn = nn.BatchNorm1d(2 * embed_dim)

        self.out_proj = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, x, edge_index, subgraph_edge_index, edge_attr=None,
            subgraph_indicator_index=None, subgraph_node_index=None,
            subgraph_edge_attr=None):

        x_struct = self.structure_extractor(
            x=x[subgraph_node_index],
            edge_index=subgraph_edge_index,
            edge_attr=subgraph_edge_attr,
            subgraph_indicator_index=subgraph_indicator_index,
            agg="sum",
        )
        x_struct = torch.cat([x, x_struct], dim=-1)
        if self.batch_norm:
            x_struct = self.bn(x_struct)
        x_struct = self.out_proj(x_struct)

        return x_struct


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    r"""Structure-Aware Transformer layer, made up of structure-aware self-attention and feed-forward network.

    Args:
    ----------
        d_model (int):      the number of expected features in the input (required).
        nhead (int):        the number of heads in the multiheadattention models (default=8).
        dim_feedforward (int): the dimension of the feedforward network model (default=512).
        dropout:            the dropout value (default=0.1).
        activation:         the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable (default: relu).
        batch_norm:         use batch normalization instead of layer normalization (default: True).
        pre_norm:           pre-normalization or post-normalization (default=False).
        gnn_type:           base GNN model to extract subgraph representations.
                            One can implememnt customized GNN in gnn_layers.py (default: gcn).
        se:                 structure extractor to use, either gnn or khopgnn (default: gnn).
        k_hop:              the number of base GNN layers or the K hop size for khopgnn structure extractor (default=2).
    """
    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                activation="relu", batch_norm=True, pre_norm=False,
                gnn_type="gcn", se="gnn", k_hop=2, **kwargs):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.self_attn = Attention(d_model, nhead, dropout=dropout,
            bias=False, gnn_type=gnn_type, se=se, k_hop=k_hop, **kwargs)
        self.batch_norm = batch_norm
        self.pre_norm = pre_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x, edge_index, complete_edge_index,
            subgraph_node_index=None, subgraph_edge_index=None,
            subgraph_edge_attr=None,
            subgraph_indicator_index=None,
            edge_attr=None, degree=None, ptr=None,
            return_attn=True,
        ):

        if self.pre_norm:
            x = self.norm1(x)

        x2, attn = self.self_attn(
            x,
            edge_index,
            complete_edge_index,
            edge_attr=edge_attr,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=ptr,
            return_attn=return_attn
        )

        if degree is not None:
            x2 = degree.unsqueeze(-1) * x2
        x = x + self.dropout1(x2)
        if self.pre_norm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)

        if not self.pre_norm:
            x = self.norm2(x)
        return x, attn
# -*- coding: utf-8 -*-
# import torch
# import numpy as np
# from torch import nn
# from torch_scatter import scatter_add, scatter_mean, scatter_max
# import torch_geometric.nn as gnn
# import torch_geometric.utils as utils
# from einops import rearrange
# from utils import pad_batch, unpad_batch
# from gnn_layers import get_simple_gnn_layer, EDGE_GNN_TYPES
# import torch.nn.functional as F
#
#
# class AdaptiveStructureExtractor(nn.Module):
#     r"""自适应结构提取器 - 动态确定每个节点的最佳邻域范围
#
#     Args:
#     ----------
#     embed_dim (int):        嵌入维度
#     gnn_type (str):        结构提取器中使用的GNN类型 (gcn, gin, pna等)
#     num_layers (int):      最大GNN层数/最大k值
#     batch_norm (bool):     是否应用批量归一化
#     concat (bool):         是否连接初始边特征
#     khopgnn (bool):        是否使用子图而非子树
#     """
#
#     def __init__(self, embed_dim, gnn_type="gcn", num_layers=3,
#                  batch_norm=True, concat=True, khopgnn=False, **kwargs):
#         super().__init__()
#         self.num_layers = num_layers
#         self.khopgnn = khopgnn
#         self.concat = concat
#         self.gnn_type = gnn_type
#
#         # 节点重要性预测网络
#         self.importance_net = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, 1),
#             nn.Sigmoid()
#         )
#
#         # 动态k值预测网络
#         self.k_predictor = nn.Sequential(
#             nn.Linear(embed_dim * 2, embed_dim),  # 输入节点自身特征和邻域聚合特征
#             nn.ReLU(),
#             nn.Linear(embed_dim, 1),
#             nn.Softplus()  # 保证k值为正
#         )
#
#         # 基础GNN层用于特征提取
#         layers = []
#         for _ in range(num_layers):
#             layers.append(get_simple_gnn_layer(gnn_type, embed_dim, **kwargs))
#         self.gcn = nn.ModuleList(layers)
#
#         self.relu = nn.ReLU()
#         self.batch_norm = batch_norm
#         inner_dim = (num_layers + 1) * embed_dim if concat else embed_dim
#
#         if batch_norm:
#             self.bn = nn.BatchNorm1d(inner_dim)
#
#         self.out_proj = nn.Linear(inner_dim, embed_dim)
#
#     def forward(self, x, edge_index, edge_attr=None,
#                 subgraph_indicator_index=None, agg="sum"):
#         # 1. 计算节点重要性分数
#         importance_scores = self.importance_net(x).squeeze(-1)
#
#         # 2. 动态k值预测
#         # 首先获取邻域聚合特征
#         neighbor_agg = scatter_mean(x[edge_index[0]], edge_index[1], dim=0, dim_size=x.size(0))
#         k_input = torch.cat([x, neighbor_agg], dim=-1)
#         k_values = self.k_predictor(k_input).squeeze(-1) + 1  # 确保k≥1
#         k_values = torch.clamp(k_values, max=self.num_layers)  # 限制最大k值
#
#         # 3. 自适应特征提取
#         x_cat = [x]
#         current_x = x
#
#         # 为每个节点存储其有效层数
#         valid_layers = torch.zeros_like(k_values, dtype=torch.int)
#
#         # 逐层处理
#         for layer_idx in range(self.num_layers):
#             # 只对k值大于当前层数的节点应用该层
#             active_mask = k_values > layer_idx
#
#             # 如果没有节点需要此层，跳过
#             if not torch.any(active_mask):
#                 continue
#
#             # 应用GNN层
#             gcn_layer = self.gcn[layer_idx]
#             if edge_attr is None:
#                 next_x = gcn_layer(current_x, edge_index)
#             else:
#                 next_x = gcn_layer(current_x, edge_index, edge_attr=edge_attr)
#
#             # 应用ReLU激活
#             next_x = self.relu(next_x)
#
#             # 更新当前特征：只更新需要此层的节点
#             current_x = torch.where(active_mask.unsqueeze(-1), next_x, current_x)
#
#             # 记录有效层数
#             valid_layers[active_mask] = layer_idx + 1
#
#             # 如果使用concat，存储所有层特征
#             if self.concat:
#                 x_cat.append(current_x)
#
#         # 应用重要性加权
#         weighted_x = current_x * importance_scores.unsqueeze(-1)
#
#         # 对于k-hop子图模式，进行聚合
#         if self.khopgnn:
#             if agg == "sum":
#                 weighted_x = scatter_add(weighted_x, subgraph_indicator_index, dim=0)
#             elif agg == "mean":
#                 weighted_x = scatter_mean(weighted_x, subgraph_indicator_index, dim=0)
#
#         if self.concat:
#             weighted_x = torch.cat(x_cat, dim=-1)
#
#         if self.num_layers > 0 and self.batch_norm:
#             weighted_x = self.bn(weighted_x)
#
#         return self.out_proj(weighted_x), k_values, importance_scores
#
#
# class AdaptiveAttention(gnn.MessagePassing):
#     """自适应结构感知注意力机制
#
#     Args:
#     ----------
#     embed_dim (int):        嵌入维度
#     num_heads (int):        注意力头数 (默认: 8)
#     dropout (float):        丢弃率 (默认: 0.0)
#     bias (bool):            是否使用偏置 (默认: False)
#     symmetric (bool):       是否对称 (Q=K) (默认: False)
#     gnn_type (str):         结构提取器中使用的GNN类型 (gcn, gin, pna等)
#     se (str):               结构提取器类型 ("gnn", "khopgnn")
#     k_hop (int):            基础GNN层数或k-hop子图的大小 (默认: 2)
#     """
#
#     def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False,
#                  symmetric=False, gnn_type="gcn", se="gnn", k_hop=2, **kwargs):
#
#         super().__init__(node_dim=0, aggr='add')
#         self.embed_dim = embed_dim
#         self.bias = bias
#         head_dim = embed_dim // num_heads
#         assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
#
#         self.num_heads = num_heads
#         self.scale = head_dim ** -0.5
#
#         self.se = se
#         self.gnn_type = gnn_type
#         self.k_hop = k_hop
#
#         # 初始化自适应结构提取器
#         self.structure_extractor = AdaptiveStructureExtractor(
#             embed_dim=embed_dim,
#             gnn_type=gnn_type,
#             num_layers=k_hop,
#             **kwargs
#         )
#
#         self.attend = nn.Softmax(dim=-1)
#
#         self.symmetric = symmetric
#         if symmetric:
#             self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
#         else:
#             self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
#         self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
#
#         self.attn_dropout = nn.Dropout(dropout)
#
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#
#         self._reset_parameters()
#
#         self.attn_sum = None
#         self.k_values = None
#         self.importance_scores = None
#
#     def _reset_parameters(self):
#         nn.init.xavier_uniform_(self.to_qk.weight)
#         nn.init.xavier_uniform_(self.to_v.weight)
#
#         if self.bias:
#             nn.init.constant_(self.to_qk.bias, 0.)
#             nn.init.constant_(self.to_v.bias, 0.)
#
#     def forward(self,
#                 x,
#                 edge_index,
#                 complete_edge_index,
#                 subgraph_node_index=None,
#                 subgraph_edge_index=None,
#                 subgraph_indicator_index=None,
#                 subgraph_edge_attr=None,
#                 edge_attr=None,
#                 ptr=None,
#                 return_attn=False):
#         """
#         计算注意力层
#
#         Args:
#         ----------
#         x:                          输入节点特征
#         edge_index:                 图的边索引
#         complete_edge_index:        全连接图的边索引
#         subgraph_node_index:        子图中节点的索引
#         subgraph_edge_index:        提取子图的边索引
#         subgraph_indicator_index:   指示每个节点属于哪个子图的索引
#         subgraph_edge_attr:         提取的k-hop子图的边属性
#         edge_attr:                  边属性
#         return_attn:                是否返回注意力权重 (默认: False)
#
#         Returns:
#         ----------
#         out: 输出节点特征
#         attn: 注意力权重 (当return_attn=True)
#         k_values: 动态k值
#         importance_scores: 节点重要性分数
#         """
#         # 计算值矩阵
#         v = self.to_v(x)
#
#         # 计算结构感知节点嵌入
#         if self.se == 'khopgnn':  # k-subgraph SAT
#             x_struct, k_values, importance_scores = self.structure_extractor(
#                 x=x,
#                 edge_index=edge_index,
#                 edge_attr=edge_attr,
#                 subgraph_indicator_index=subgraph_indicator_index,
#                 agg="sum"
#             )
#         else:  # k-subtree SAT
#             x_struct, k_values, importance_scores = self.structure_extractor(
#                 x=x,
#                 edge_index=edge_index,
#                 edge_attr=edge_attr
#             )
#
#         # 存储动态值
#         self.k_values = k_values
#         self.importance_scores = importance_scores
#
#         # 计算查询和键矩阵
#         if self.symmetric:
#             qk = self.to_qk(x_struct)
#             qk = (qk, qk)
#         else:
#             qk = self.to_qk(x_struct).chunk(2, dim=-1)
#
#         # 计算完整的自注意力
#         attn = None
#
#         if complete_edge_index is not None:
#             # 创建基于k值的注意力掩码
#             src, dst = complete_edge_index
#             k_mask = self.create_k_mask(k_values, src, dst)
#
#             # 传播带掩码的注意力
#             out = self.propagate(complete_edge_index, v=v, qk=qk, edge_attr=k_mask, size=None,
#                                  return_attn=return_attn)
#
#             if return_attn:
#                 attn = self._attn
#                 self._attn = None
#                 attn = torch.sparse_coo_tensor(
#                     complete_edge_index,
#                     attn,
#                 ).to_dense().transpose(0, 1)
#
#             out = rearrange(out, 'n h d -> n (h d)')
#         else:
#             out, attn = self.self_attn(qk, v, ptr, return_attn=return_attn)
#
#         output = self.out_proj(out)
#
#         if return_attn:
#             return output, attn, k_values, importance_scores
#         return output, k_values, importance_scores
#
#     def create_k_mask(self, k_values, src, dst):
#         """
#         创建基于k值的注意力掩码
#
#         只允许距离小于等于 min(k_src, k_dst) 的节点对参与注意力计算
#
#         Args:
#         ----------
#         k_values (Tensor): 每个节点的k值 [num_nodes]
#         src (LongTensor): 源节点索引 [num_edges]
#         dst (LongTensor): 目标节点索引 [num_edges]
#
#         Returns:
#         ----------
#         mask (BoolTensor): 注意力掩码 [num_edges]
#         """
#         # 简化实现 - 实际应用中应使用图扩散或近似算法
#         # 这里假设所有节点对的距离都是1（直接连接的边）
#         # 在实际应用中，应替换为图上的最短路径距离计算
#         dist = torch.ones_like(src, dtype=torch.float)
#
#         # 获取每对节点的最小k值
#         k_src = k_values[src]
#         k_dst = k_values[dst]
#         min_k = torch.min(k_src, k_dst)
#
#         # 创建掩码: 距离 > min_k 的位置为True
#         mask = dist > min_k
#
#         return mask
#
#     def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn):
#         """自注意力操作，计算点积注意力"""
#         # 解包查询和键
#         qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
#         qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
#         v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)
#
#         # 计算注意力分数
#         attn = (qk_i * qk_j).sum(-1) * self.scale
#
#         # 应用基于k值的掩码
#         if edge_attr is not None:
#             # 将掩码位置的注意力分数设为负无穷
#             attn = attn.masked_fill(edge_attr.unsqueeze(-1), float('-inf'))
#
#         # 应用softmax
#         attn = utils.softmax(attn, index, ptr, size_i)
#
#         if return_attn:
#             self._attn = attn
#         attn = self.attn_dropout(attn)
#
#         return v_j * attn.unsqueeze(-1)
#
#     def self_attn(self, qk, v, ptr, return_attn=False):
#         """可以返回注意力的自注意力"""
#         # 此方法需要实现完整的自注意力计算
#         # 由于时间限制，这里保持原始实现
#         # 在实际应用中，应添加基于k值的掩码逻辑
#         qk, mask = pad_batch(qk, ptr, return_mask=True)
#         k, q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qk)
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#
#         dots = dots.masked_fill(
#             mask.unsqueeze(1).unsqueeze(2),
#             float('-inf'),
#         )
#
#         dots = self.attend(dots)
#         dots = self.attn_dropout(dots)
#
#         v = pad_batch(v, ptr)
#         v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
#         out = torch.matmul(dots, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = unpad_batch(out, ptr)
#
#         if return_attn:
#             return out, dots
#         return out, None
#
#
# class TransformerEncoderLayer(nn.TransformerEncoderLayer):
#     r"""结构感知Transformer层，由结构感知自注意力和前馈网络组成。
#
#     Args:
#     ----------
#         d_model (int):      输入中期望的特征数量（必需）。
#         nhead (int):        多头注意力模型中的头数（默认=8）。
#         dim_feedforward (int): 前馈网络模型的维度（默认=512）。
#         dropout:            丢弃率（默认=0.1）。
#         activation:         中间层的激活函数，可以是字符串
#                             ("relu" 或 "gelu") 或一元可调用对象（默认：relu）。
#         batch_norm:         使用批量归一化代替层归一化（默认：True）。
#         pre_norm:           预归一化或后归一化（默认=False）。
#         gnn_type:           用于提取子图表征的基础GNN模型。
#                             可以在gnn_layers.py中实现自定义GNN（默认：gcn）。
#         se:                 使用的结构提取器，可以是gnn或khopgnn（默认：gnn）。
#         k_hop:              基础GNN层数或khopgnn结构提取器的K跳大小（默认=2）。
#     """
#
#     def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
#                  activation="relu", batch_norm=True, pre_norm=False,
#                  gnn_type="gcn", se="gnn", k_hop=2, **kwargs):
#         super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
#
#         # 使用自适应注意力层
#         self.self_attn = AdaptiveAttention(d_model, nhead, dropout=dropout,
#                                            bias=False, gnn_type=gnn_type, se=se, k_hop=k_hop, **kwargs)
#         self.batch_norm = batch_norm
#         self.pre_norm = pre_norm
#         if batch_norm:
#             self.norm1 = nn.BatchNorm1d(d_model)
#             self.norm2 = nn.BatchNorm1d(d_model)
#
#     def forward(self, x, edge_index, complete_edge_index,
#                 subgraph_node_index=None, subgraph_edge_index=None,
#                 subgraph_edge_attr=None,
#                 subgraph_indicator_index=None,
#                 edge_attr=None, degree=None, ptr=None,
#                 return_attn=True,
#                 ):
#
#         if self.pre_norm:
#             x = self.norm1(x)
#
#         # 调用自适应注意力层，获取额外输出
#         if return_attn:
#             x2, attn, k_values, importance_scores = self.self_attn(
#                 x,
#                 edge_index,
#                 complete_edge_index,
#                 edge_attr=edge_attr,
#                 subgraph_node_index=subgraph_node_index,
#                 subgraph_edge_index=subgraph_edge_index,
#                 subgraph_indicator_index=subgraph_indicator_index,
#                 subgraph_edge_attr=subgraph_edge_attr,
#                 ptr=ptr,
#                 return_attn=return_attn
#             )
#         else:
#             x2, k_values, importance_scores = self.self_attn(
#                 x,
#                 edge_index,
#                 complete_edge_index,
#                 edge_attr=edge_attr,
#                 subgraph_node_index=subgraph_node_index,
#                 subgraph_edge_index=subgraph_edge_index,
#                 subgraph_indicator_index=subgraph_indicator_index,
#                 subgraph_edge_attr=subgraph_edge_attr,
#                 ptr=ptr,
#                 return_attn=return_attn
#             )
#             attn = None
#
#         if degree is not None:
#             x2 = degree.unsqueeze(-1) * x2
#         x = x + self.dropout1(x2)
#         if self.pre_norm:
#             x = self.norm2(x)
#         else:
#             x = self.norm1(x)
#         x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         x = x + self.dropout2(x2)
#
#         if not self.pre_norm:
#             x = self.norm2(x)
#
#         if return_attn:
#             return x, attn, k_values, importance_scores
#         return x, k_values, importance_scores