# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch_geometric.utils as utils
from torch_geometric.data import Data
import json
from position_encoding import POSENCODINGS
import numpy as np
import os
from rdkit import Chem

def my_inc(self, key, value, *args, **kwargs):
    if key == 'subgraph_edge_index':
        return self.num_subgraph_nodes
    if key == 'subgraph_node_idx':
        return self.num_nodes
    if key == 'subgraph_indicator':
        return self.num_nodes
    elif 'index' in key:
        return self.num_nodes
    else:
        return 0


class GraphDataset(object):
    def __init__(self, dataset, degree=False, k_hop=2, se="gnn", use_subgraph_edge_attr=False,
                 cache_path=None, return_complete_index=False):
        self.dataset = dataset
        # self.n_features = dataset[0].x.shape[-1]
        self.degree = degree
        self.compute_degree()
        self.abs_pe_list = None
        self.return_complete_index = return_complete_index
        self.k_hop = k_hop
        self.se = se
        self.use_subgraph_edge_attr = use_subgraph_edge_attr
        self.cache_path = cache_path
        if self.se == 'khopgnn':
            Data.__inc__ = my_inc
            self.extract_subgraphs()

    def compute_degree(self):
        if not self.degree:
            self.degree_list = None
            return
        # self.degree_list = []
        #for g in self.dataset:
        deg = 1. / torch.sqrt(1. + utils.degree(self.dataset.edge_index[0], self.dataset.num_nodes))
        self.degree_list= deg

    def extract_subgraphs(self):
        print("Extracting {}-hop subgraphs...".format(self.k_hop))
        # indicate which node in a graph it is; for each graph, the
        # indices will range from (0, num_nodes). PyTorch will then
        # increment this according to the batch size
        self.subgraph_node_index = []

        # Each graph will become a block diagonal adjacency matrix of
        # all the k-hop subgraphs centered around each node. The edge
        # indices get augumented within a given graph to make this
        # happen (and later are augmented for proper batching)
        self.subgraph_edge_index = []

        # This identifies which indices correspond to which subgraph
        # (i.e. which node in a graph)
        self.subgraph_indicator_index = []

        # This gets the edge attributes for the new indices
        if self.use_subgraph_edge_attr:
            self.subgraph_edge_attr = []

        for i in range(len(self.dataset)):
            if self.cache_path is not None:
                filepath = "{}_{}.pt".format(self.cache_path, i)
                if os.path.exists(filepath):
                    continue
            graph = self.dataset[i]
            node_indices = []
            edge_indices = []
            edge_attributes = []
            indicators = []
            edge_index_start = 0

            for node_idx in range(graph.num_nodes):
                sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
                    node_idx,
                    self.k_hop,
                    graph.edge_index,
                    relabel_nodes=True,
                    num_nodes=graph.num_nodes
                )
                node_indices.append(sub_nodes)
                edge_indices.append(sub_edge_index + edge_index_start)
                indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    edge_attributes.append(graph.edge_attr[edge_mask])  # CHECK THIS DIDN"T BREAK ANYTHING
                edge_index_start += len(sub_nodes)

            if self.cache_path is not None:
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    subgraph_edge_attr = torch.cat(edge_attributes)
                else:
                    subgraph_edge_attr = None
                torch.save({
                    'subgraph_node_index': torch.cat(node_indices),
                    'subgraph_edge_index': torch.cat(edge_indices, dim=1),
                    'subgraph_indicator_index': torch.cat(indicators).type(torch.LongTensor),
                    'subgraph_edge_attr': subgraph_edge_attr
                }, filepath)
            else:
                self.subgraph_node_index.append(torch.cat(node_indices))
                self.subgraph_edge_index.append(torch.cat(edge_indices, dim=1))
                self.subgraph_indicator_index.append(torch.cat(indicators))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    self.subgraph_edge_attr.append(torch.cat(edge_attributes))
        print("Done!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset

        # if self.n_features == 1:
        #     data.x = data.x.squeeze(-1)
        # if not isinstance(data.y, list):
        #     data.y = data.y.view(data.y.shape[0], -1)
        n = data.num_nodes
        s = torch.arange(n)
        if self.return_complete_index:
            data.complete_edge_index = torch.vstack((s.repeat_interleave(n), s.repeat(n)))
        data.degree = None
        if self.degree:
            data.degree = self.degree_list
        data.abs_pe = None
        if self.abs_pe_list is not None and len(self.abs_pe_list) == len(self.dataset):
            data.abs_pe = self.abs_pe_list

        # add subgraphs and relevant meta data
        if self.se == "khopgnn":
            if self.cache_path is not None:
                cache_file = torch.load("{}_{}.pt".format(self.cache_path, index))
                data.subgraph_edge_index = cache_file['subgraph_edge_index']
                data.num_subgraph_nodes = len(cache_file['subgraph_node_index'])
                data.subgraph_node_idx = cache_file['subgraph_node_index']
                data.subgraph_edge_attr = cache_file['subgraph_edge_attr']
                data.subgraph_indicator = cache_file['subgraph_indicator_index']
                return data
            data.subgraph_edge_index = self.subgraph_edge_index[index]
            data.num_subgraph_nodes = len(self.subgraph_node_index[index])
            data.subgraph_node_idx = self.subgraph_node_index[index]
            if self.use_subgraph_edge_attr and data.edge_attr is not None:
                data.subgraph_edge_attr = self.subgraph_edge_attr[index]
            data.subgraph_indicator = self.subgraph_indicator_index[index].type(torch.LongTensor)
        else:
            data.num_subgraph_nodes = None
            data.subgraph_node_idx = None
            data.subgraph_edge_index = None
            data.subgraph_indicator = None

        return data
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
        ## self.node_embedding = nn.Embedding(self.num_nodes, self.embedding_dim)
        ## self.rel_embedding = nn.Embedding(self.num_relations, self.embedding_dim)

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
        return Data(x=torch.tensor(self.entities,dtype=torch.long),edge_index=torch.tensor(edge_index,dtype=torch.long),edge_type=torch.tensor(edge_type,dtype=torch.long))
        #return edge_index, edge_type
kg_file = KnowledgeGraph('../data/new_kg.txt')
kg_data = kg_file.gen_pyg_data()
print(kg_data)
result = GraphDataset(kg_data,degree=True, k_hop=2, se="gnn", use_subgraph_edge_attr=False)
print(result)


def subgraph_normalize_adj(sub_edge_index, sub_nodes, original_degree):
    """
    sub_nodes: 子图节点在原图中的ID
    original_degree: 原图中所有节点的度数（用于校正归一化）
    """
    row, col = sub_edge_index
    edge_weight = torch.ones_like(row)

    # 使用原图度数进行归一化（保持拓扑连续性）
    deg_inv_sqrt = 1 / torch.sqrt(original_degree[sub_nodes])
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return utils.to_scipy_sparse_matrix(sub_edge_index, edge_weight)
class PositionEncoding(object):
    def apply_to(self, dataset):
        dataset.abs_pe_list = []
        for i, g in enumerate(dataset):
            pe = self.compute_pe(g)
            dataset.abs_pe_list.append(pe)

        return dataset

class SubgraphRWEncoding(PositionEncoding):
    def __init__(self, dim, use_original_degree=True):
        self.pos_enc_dim = dim
        self.use_original_degree = use_original_degree  # 是否使用原图度数校正

    def compute_pe(self, subgraph, original_degree=None):
        # 改进的归一化
        W = subgraph_normalize_adj(
            subgraph.edge_index,
            subgraph.x,
            original_degree if self.use_original_degree else None
        )

        # 多尺度特征捕获
        pe = torch.zeros((len(subgraph.x), self.pos_enc_dim))
        pe[:, 0] = torch.from_numpy(W.diagonal())

        W_power = W.copy()
        for k in range(1, self.pos_enc_dim):
            W_power = W_power.dot(W)

            # 捕获节点对间的互信息（非仅对角线）
            pe[:, k] = torch.from_numpy(W_power.sum(axis=1).A1)  # 行求和

            # 可选：添加中心性特征
            if k == 2:  # 在特定维度注入中心性
                pe[:, k] += torch.log(original_degree[subgraph.x] + 1)

        return pe.float()
def convert(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.float):
        return float(o)
    print(type(o))
    raise TypeError
def subtreeExtractor(drug_id, edge_index, rel_index,shortest_paths, num_rel, fixed_num, khop,num_nodes):

    all_degree = []
    num_rel_update = []
    subgraphs = {}

    json_path = shortest_paths + "subtree_fixed_" + str(fixed_num) + "_hop_" + str(khop) + "sp.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            subgraphs = json.load(f)
            max_rel = 0
            max_degree = 0
            # for s in subgraphs.keys():
            #     max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
            #     max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree

        return subgraphs
#     raw, col = edge_index
#     reves_index = torch.stack((col, raw), 0)
#     undirected_edge_index = torch.cat((edge_index, reves_index), 1)
#     undirected_rel_index = torch.cat((rel_index, rel_index), 0)
#
#     for d in drug_id:
#         subset, sub_edge_index, sub_rel_index, mapping_list = k_hop_subgraph(int(d), khop, undirected_edge_index, undirected_rel_index, fixed_num, relabel_nodes=True,num_nodes=num_nodes)  ##subset是所有集合的节点，mapping指示的是center node是哪个
#         row, col = sub_edge_index
#         # all_degree.append(torch.max(degree(col)).item())
#
#         ##因为这里面会涉及到multi-relation，所以在添加子图的时候，要把多条边都添加进去
#         new_s_edge_index = sub_edge_index.transpose(1,0).numpy().tolist()
#         new_s_value = [1 for _ in range(len(new_s_edge_index))]
#         new_s_rel = sub_rel_index.numpy().tolist()
#         node_idx = subset.numpy().tolist()
#
#         s_edge_index = new_s_edge_index.copy()
#         s_value = new_s_value.copy()
#         s_rel = new_s_rel.copy()
#
# #        edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
# #        sp_edge_index = edge_index_value[:, :2]
# #        sp_value = edge_index_value[:, 2]
# #
# #         for i in range(len(sp_edge_index)):
# #             if sp_value[i] == 1:  ##也是保证多关系的边全部在数据里
# #                 continue
# #             else:
# #                 s_edge_index.append(sp_edge_index[i].tolist())
# #                 s_value.append(sp_value[i])
# #                 s_rel.append(sp_value[i] + num_rel)
# #
# #         assert len(s_edge_index) == len(s_value)
# #         assert len(s_edge_index) == len(s_rel)
# #
# #         num_rel_update.append(np.max(s_rel))
#
#         subgraphs[d] = node_idx, new_s_edge_index, new_s_rel, mapping_list
#
#     with open(json_path, 'w') as f:
#         json.dump(subgraphs, f, default=convert)
#     return subgraphs
e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}
# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()]), atom.GetDegree()
# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(datapath, ligands):

    smile_graph = {}

    paths = datapath + "./mol_sp.json"

    if os.path.exists(paths):
        with open(paths, 'r') as f:
            smile_graph = json.load(f)
        max_rel = 0
        max_degree = 0
        for s in smile_graph.keys():
            max_rel = max(smile_graph[s][3]) if max(smile_graph[s][3]) > max_rel else max_rel
        #     max_degree = smile_graph[s][7] if smile_graph[s][7] > max_degree else max_degree

        return smile_graph, max_rel

    smiles_max_node_degree = []
    num_rel_mol_update = 0
    rel_max = 0
    for d in range(len(ligands)):
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]))  ##还是smiles序列
        c_size, features, edge_index, rel_index = single_smile_to_graph(lg)
        if c_size == 0: ##证明这个药物只由一个atom组成，这种的不考虑
            continue
        # if max(s_value) > num_rel_mol_update:
        #     num_rel_mol_update = max(s_value)

        smile_graph[d] = c_size, features, edge_index, rel_index
        rel_max_part = max(rel_index)
        if rel_max <= rel_max_part:
            rel_max = rel_max_part
        # smiles_max_node_degree.append(deg)

    with open(paths, 'w') as f:
        json.dump(smile_graph, f)

    return smile_graph, rel_max


# mol smile to mol graph edge index
def single_smile_to_graph(smile):

    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    degrees = []
    for atom in mol.GetAtoms():
        feature, degree = atom_features(atom)
        features.append((feature / sum(feature)).tolist())
        degrees.append(degree)

    mol_index = []  ##begin, end, rel
    for bond in mol.GetBonds():
        mol_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])
        mol_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])

    if len(mol_index) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    mol_index = np.array(sorted(mol_index))
    mol_edge_index = mol_index[:,:2]
    mol_rel_index = mol_index[:,2]
    return c_size, features, mol_edge_index.tolist(), mol_rel_index.tolist()
drug_file = '../data/new_smiles.txt'
SMILES = []
drug = open(drug_file,'r')
drug_lines = drug.readlines()
for i in drug_lines:
    smiles = i.strip()
    SMILES.append(smiles)
smile_graph, rel_nums = smile_to_graph('./', SMILES)
pe = []
for i in range(1052):
    c_size, features, edge_index, rel_index = smile_graph[str(i)]
    data_mol = Data(x=torch.Tensor(np.array(features)), edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                          rel_index=torch.tensor(np.array(rel_index, dtype=int), dtype=torch.long))
    data_mol.__setitem__('c_size', torch.LongTensor([c_size]))
    abs_pe_method = POSENCODINGS['rw']
    abs_pe_encoder = abs_pe_method(16, normalization='sym')
    pe.append(abs_pe_encoder.apply_to(data_mol))
drug_list = [i for i in range(1052)]
edge_index = kg_data.edge_index
edge_type = kg_data.edge_type
triples = kg_file.triples
num_nodes = kg_file.num_nodes
num_rel = kg_file.num_relations
subgraph = subtreeExtractor(drug_list,edge_index,edge_type,'./',num_rel,64,4,num_nodes)
subset, subgraph_edge_index, subgraph_rel, mapping_id = subgraph[str(0)]
data_graph = Data(x=torch.tensor(np.array(subset,dtype=int),dtype=torch.long),
                              edge_index= torch.LongTensor(subgraph_edge_index).transpose(1,0),
                              id=torch.LongTensor(np.array(mapping_id,dtype=bool)),
                              rel_index = torch.tensor(np.array(subgraph_rel,dtype=int),dtype=torch.long))
full_degree = torch.zeros(kg_data.num_nodes)
from torch_scatter import scatter_add
full_degree = scatter_add(torch.ones(kg_data.edge_index.size(1)),
                         kg_data.edge_index[0])
encoder = SubgraphRWEncoding(dim=16, use_original_degree=True)
pe = encoder.compute_pe(data_graph,full_degree)
print(pe)