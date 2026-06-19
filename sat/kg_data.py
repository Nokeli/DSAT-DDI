import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Dataset
import networkx as nx
#import kg_model
#from kg_model import KGGraphTransformer
from torch_geometric.loader import DataLoader
from sat.gnn_layers import GNN_TYPES
from sat.models import GraphTransformer
from sat.data import GraphDataset
from torch_geometric.utils import k_hop_subgraph
from sat.position_encoding import POSENCODINGS
# import argparse
# parser = argparse.ArgumentParser(
#     description='Structure-Aware Transformer on SBM datasets',
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--seed', type=int, default=0,
#                     help='random seed')
# parser.add_argument('--dataset', type=str, default="PATTERN",
#                     help='name of dataset')
# parser.add_argument('--num_class', type=int, default=2,
#                     help='num of class')
# parser.add_argument('--num-heads', type=int, default=4, help="number of heads")
# parser.add_argument('--num-layers', type=int, default=2, help="number of layers")
# parser.add_argument('--dim-hidden', type=int, default=128 , help="hidden dimension of Transformer")
# parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
# parser.add_argument('--epochs', type=int, default=200,
#                     help='number of epochs')
# parser.add_argument('--lr', type=float, default=0.001,
#                     help='initial learning rate')
# parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
# parser.add_argument('--batch-size', type=int, default=512,
#                     help='batch size')
# parser.add_argument('--abs_pe', type=str, default="rw", choices=POSENCODINGS.keys(),
#                     help='which absolute PE to use?')
# parser.add_argument('--abs_pe_dim', type=int, default=3, help='dimension for absolute PE')
# parser.add_argument('--outdir', type=str, default='',
#                     help='output path')
# parser.add_argument('--warmup', type=int, default=5000, help="number of iterations for warmup")
# parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
# parser.add_argument('--gnn-type', type=str, default='graph',
#                     choices=GNN_TYPES,
#                     help="GNN structure extractor type")
# parser.add_argument('--k-hop', type=int, default=2, help="number of layers for GNNs")
# parser.add_argument('--weight-class', action='store_true', help='weight classes or not')
#
# parser.add_argument('--se', type=str, default="gnn",
#                     help='Extractor type: khopgnn, or gnn')
# args = parser.parse_args()
# args.batch_norm = not args.layer_norm
# class KnowledgeGraph:
#     def __init__(self, file_path, delimiter=' '):
#         """
#         初始化知识图谱处理器
#         :param file_path: 三元组文件路径
#         :param delimiter: 分隔符，默认为空格
#         """
#         self.triples = self._load_triples(file_path, delimiter)
#         self.entities, self.relations = self._get_unique_elements()
#         self.num_nodes = len(self.entities)
#         self.num_relations = len(self.relations)
#
#         # 创建ID映射
#         #self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}
#         #self.relation_to_idx = {r: i for i, r in enumerate(self.relations)}
#
#         # 初始化嵌入层
#         self.embedding_dim = 128
#         self.node_embedding = nn.Embedding(self.num_nodes, self.embedding_dim)
#         self.rel_embedding = nn.Embedding(self.num_relations, self.embedding_dim)
#
#     def _load_triples(self, file_path, delimiter):
#         """加载三元组文件"""
#         triples = []
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 h, r, t = line.strip().split(delimiter)
#                 triples.append((int(h), int(r), int(t)))
#         return triples
#
#     def _get_unique_elements(self):
#         """获取唯一的实体和关系"""
#         entities = sorted(set([h for h, _, _ in self.triples] + [t for _, t, _ in self.triples]))
#         relations = sorted(set([r for _, _, r in self.triples]))
#         return entities, relations
#
#     def get_graph_data(self):
#         """转换为PyG的Data对象"""
#         # 转换边索引
#         edge_index = torch.tensor([
#             [h for h, _, _ in self.triples],
#             [t for _, _, t in self.triples]
#         ], dtype=torch.long)
#
#         # 转换边属性
#         edge_type = torch.tensor([r for _, _, r in self.triples], dtype=torch.long)
#         edge_attr = self.rel_embedding(edge_type)
#
#         # 节点特征
#         x = self.node_embedding(torch.arange(self.num_nodes))
#
#         return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,edge_type=edge_type)

class KGDataset(Dataset):
    def __init__(self, kg_data, transform=None, pre_transform=None):
        """
        知识图谱PyG Dataset类
        :param kg_data: KnowledgeGraph实例或Data对象
        :param transform: 图变换函数
        :param pre_transform: 预处理变换函数
        """
        super().__init__(None, transform, pre_transform)

        if isinstance(kg_data, KnowledgeGraph):
            self.data = kg_data.get_graph_data()
        elif isinstance(kg_data, Data):
            self.data = kg_data
        else:
            raise TypeError("kg_data must be either KnowledgeGraph or Data object")

    def len(self):
        return 1  # 单图数据集

    def get(self, idx):
        return self.data

    @property
    def num_node_features(self):
        return self.data.num_node_features

    @property
    def num_edge_features(self):
        return self.data.num_edge_features
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn


class KnowledgeGraph:
    def __init__(self, file_path, delimiter=' ',directed=False):
        """
        初始化知识图谱处理器
        :param file_path: 三元组文件路径
        :param delimiter: 分隔符，默认为空格
        """
        self.directed = directed
        self.triples = self._load_triples(file_path, delimiter)
        self.entities, self.relations = self._get_unique_elements()
        self.num_nodes = len(self.entities)
        self.num_relations = len(self.relations)

        # 初始化嵌入层
        self.embedding_dim = 128
        self.node_embedding = nn.Embedding(self.num_nodes, self.embedding_dim)
        self.rel_embedding = nn.Embedding(self.num_relations, self.embedding_dim)

    def _load_triples(self, file_path, delimiter):
        """加载三元组文件"""
        triples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                h, t, r = line.strip().split(delimiter)
                triples.append((int(h), int(t), int(r)))
        return triples

    def _get_unique_elements(self):
        """获取唯一的实体和关系"""
        entities = sorted(set([h for h, _, _ in self.triples] + [t for _, t, _ in self.triples]))
        relations = sorted(set([r for _, _, r in self.triples]))
        return entities, relations
    def gen_pyg_data(self):
        edge_index = torch.tensor([
            [h for h, _, _ in self.triples],
            [t for _, t, _ in self.triples]
        ], dtype=torch.long)
        edge_type = torch.tensor([r for _, _, r in self.triples], dtype=torch.long)
        if not self.directed:
            reverse_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
            edge_index = torch.cat([edge_index, reverse_edges], dim=1)
            edge_type = torch.cat([edge_type,edge_type],dim=0)
        edge_attr = self.rel_embedding(edge_type)
        x = self.node_embedding(torch.arange(self.num_nodes))
        return Data(x =x, edge_index=edge_index,edge_attr=edge_attr,edge_type=edge_type)


class SubgraphExtractor:
    def __init__(self, pyg_data, num_hops=2, num_neighbors=32):
        """
        子图抽取器初始化(带邻居采样功能)

        参数:
            pyg_data: PyG的Data对象，包含完整的图数据
            num_hops: 子图的跳数 (默认2跳)
            num_neighbors: 每跳采样的邻居数量 (默认32个)
        """
        self.data = pyg_data
        self.num_hops = num_hops
        self.num_neighbors = num_neighbors

    def _sample_neighbors(self, node_idx, edge_index, num_nodes):
        """
        辅助方法: 采样固定数量的邻居节点
        """
        # 获取所有邻居节点
        neighbors = edge_index[1][edge_index[0] == node_idx].unique()

        # 如果邻居数多于需求，随机采样
        if len(neighbors) > self.num_neighbors:
            neighbors = neighbors[torch.randperm(len(neighbors))[:self.num_neighbors]]

        return neighbors

    def extract_subgraph(self, entity_id, return_edge_attr=True):
        """
        抽取以指定实体为中心的k跳子图，每跳采样固定数量邻居

        参数:
            entity_id: 中心节点的ID
            return_edge_attr: 是否返回边属性

        返回:
            tuple: (subgraph_data, node_mapping)
        """
        if entity_id < 0 or entity_id >= self.data.num_nodes:
            raise ValueError(f"Invalid entity ID: {entity_id}. Must be between 0 and {self.data.num_nodes - 1}")

        # 初始化采样节点集合(包含中心节点)
        sampled_nodes = {entity_id}
        current_nodes = {entity_id}

        # 逐跳采样
        for _ in range(self.num_hops):
            new_nodes = set()
            for node in current_nodes:
                # 采样固定数量邻居
                neighbors = self._sample_neighbors(node, self.data.edge_index, self.data.num_nodes)
                new_nodes.update(neighbors.tolist())

            # 添加到总节点集
            sampled_nodes.update(new_nodes)
            current_nodes = new_nodes

        # 转换为tensor
        subset = torch.tensor(list(sampled_nodes), dtype=torch.long)

        # 提取子图边
        row, col = self.data.edge_index
        edge_mask = torch.isin(row, subset) & torch.isin(col, subset)
        sub_edge_index = self.data.edge_index[:, edge_mask]

        # 重新编号节点(0到n-1)
        node_mapping = {int(node): i for i, node in enumerate(subset)}
        edge_index = sub_edge_index.apply_(lambda x: node_mapping[x])

        # 构建子图数据
        subgraph_data = Data(
            x=self.data.x[subset],
            edge_index=edge_index,
            center_node_idx=node_mapping[entity_id],
            num_nodes=len(subset)
        )

        # 添加边属性(如果需要)
        if return_edge_attr and hasattr(self.data, 'edge_attr'):
            subgraph_data.edge_attr = self.data.edge_attr[edge_mask]

        # 构建映射字典
        mapping_dict = {
            'subgraph_to_original': {v: k for k, v in node_mapping.items()},
            'original_to_subgraph': node_mapping
        }

        return subgraph_data, mapping_dict

# subgraph = KGDataset(subgraph)
# relation = kg_data.relations
# dataset = KGDataset(kg_data)
# model = KGGraphTransformer(d_model=128, num_relations=71, num_heads=8,
#                  dim_feedforward=512, dropout=0.0, num_layers=4,
#                  batch_norm=False,gnn_type="rgcn",)
# output = model(dataset)
# data = GraphDataset(dataset,return_complete_index=False,use_subgraph_edge_attr=True)
# kgdata = DataLoader(data)
# model = GraphTransformer(in_size=128,
#                         num_class=1,
#                         d_model=128,
#                         dim_feedforward=128,
#                         dropout=0.1,
#                         num_heads=1,
#                         num_layers=1,
#                         batch_norm=False,
#                         abs_pe=False,
#                         abs_pe_dim=0,
#                         gnn_type='graph',
#                         use_edge_attr=True,
#                         num_edge_features=71,
#                         in_embed=False,
#                         edge_dim=128,
#                         k_hop=2,se='gnn',task_type='kg')
# drug_list = [i for i in range(1052)]
# for i in drug_list:
#     subgraph,mapping_dict = subect.extract_subgraph(i)
#     subgraph = KGDataset(subgraph)
#     data = DataLoader(data,batch_size=1)
#     for i in data:
#         output = model(i)

def k_hop_subgraph(node_idx, num_hops, edge_index, rel_index, fixed_num, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    '''
    :param node_idx: 采样的节点id
    :param num_hops: 采样的深度
    :param edge_index: 原始图的边的索引
    :param rel_index: 原始边的索引
    :param fixed_num: 每层采样的节点数
    :param relabel_nodes: 是否重新对节点排序
    :param num_nodes: 原始图中有多少个节点
    :param flow: 方向
    :return:
    subset: 采样后的节点数
    edge_index: 重新编号的边的索引
    rel_index: 边的类型
    mapping_mask: 采样节点的位置
    '''
    np.random.seed(42)
    # num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        #print(col[edge_mask].shape)
        if fixed_num == None:
            subsets.append(col[edge_mask])
        elif col[edge_mask].size(0) > fixed_num:
            neighbors = np.random.choice(a=col[edge_mask].numpy(), size=fixed_num, replace=False)
            subsets.append(torch.LongTensor(neighbors))
        else:
            subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]
    #print(subset)

    rel_index = rel_index[edge_mask] if rel_index is not None else None


    mapping_mask = [False for _ in range(len(subset))]
    mapping_mask[inv] = True


    return subset, edge_index, rel_index, mapping_mask
# kg = KnowledgeGraph('../data/new_kg.txt').gen_pyg_data()
kg = KnowledgeGraph('../data/new_kg.txt')
triples = kg.triples
num_node = kg.num_nodes
num_rel = kg.num_relations
# subect = SubgraphExtractor(kg,num_hops=2)
# subgraph,mapping_dict = subect.extract_subgraph(555)
edge_index = torch.tensor([[h for h, _, _ in triples],[t for _, t, _ in triples]], dtype=torch.long)
raw,col = edge_index
reves_index = torch.stack((col,raw),0)
undirected_edge_index = torch.cat((edge_index, reves_index),1)
edge_type = torch.tensor([r for _, _, r in triples], dtype=torch.long)
undirected_rel_index = torch.cat((edge_type, edge_type), 0)
node_index = torch.tensor([i for i in range(1052)])
num_rel_update = []
subset, sub_edge_index, sub_rel_index, mapping_list = k_hop_subgraph(100,2,undirected_edge_index,undirected_rel_index,relabel_nodes=True,fixed_num=32,num_nodes=num_node)
#subset采样的节点数 sub_edge_index 为重置以后得边 sub_rel_index为子图的关系 mapping_list 显示了哪个是采样的节点
print(subset)
print(sub_edge_index)
print(sub_rel_index)
print(mapping_list)
def calculate_shortest_path(edge_index):

    s_edge_index_value = []

    g = nx.DiGraph()
    g.add_edges_from(edge_index.tolist())

    paths = nx.all_pairs_shortest_path_length(g)
    for node_i, node_ij in paths:
        for node_j, length_ij in node_ij.items():
            s_edge_index_value.append([node_i, node_j, length_ij])

    s_edge_index_value.sort()

    return np.array(s_edge_index_value)
row, col = sub_edge_index
new_s_edge_index = sub_edge_index.transpose(1,0).numpy().tolist()
new_s_value = [1 for _ in range(len(new_s_edge_index))]
new_s_rel = sub_rel_index.numpy().tolist()
node_idx = subset.numpy().tolist()
s_edge_index = new_s_edge_index.copy()
s_value = new_s_value.copy()
s_rel = new_s_rel.copy()
edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
sp_edge_index = edge_index_value[:, :2]
sp_value = edge_index_value[:, 2]

for i in range(len(sp_edge_index)):
    if sp_value[i] == 1:  ##也是保证多关系的边全部在数据里
        continue
    else:
        s_edge_index.append(sp_edge_index[i].tolist())
        s_value.append(sp_value[i])
        s_rel.append(sp_value[i] + num_rel)

assert len(s_edge_index) == len(s_value)
assert len(s_edge_index) == len(s_rel)

num_rel_update.append(np.max(s_rel))

subgraphs[d] = node_idx, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel, torch.max(
    degree(col)).item()