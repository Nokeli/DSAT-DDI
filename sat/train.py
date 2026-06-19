import torch
from sklearn.model_selection import train_test_split,StratifiedKFold
import argparse
import json
from torch_geometric.data import Data
import torch.nn as nn
import time
import torch_geometric.utils as utils
import numpy as np
from torch_geometric.data import Dataset
# from torch_geometric.utils import k_hop_subgraph
#from data import GraphDataset
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_scatter import scatter_add
#from torch_geometric.loader import DataLoader
from models import GraphTransformer
from gnn_layers import GNN_TYPES
from position_encoding import POSENCODINGS
from rdkit import Chem
from tqdm import *
from model import DrugInteractionModel
import numpy as np
from collections import defaultdict, deque
import os
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score, auc
random.seed(42)

# 设置NumPy随机种子
np.random.seed(42)

# 设置PyTorch的CPU随机种子
torch.manual_seed(42)

# 设置所有GPU的随机种子（多卡时）
torch.cuda.manual_seed_all(42)

# 禁用CUDA的随机性优化（确保确定性计算）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser(
    description='Structure-Aware Transformer on DDI datasets',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--dataset', type=str, default="PATTERN",
                    help='name of dataset')
parser.add_argument('--num_class', type=int, default=2,
                    help='num of class')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--num-heads', type=int, default=4, help="number of heads")
parser.add_argument('--num-layers', type=int, default=2, help="number of layers")
parser.add_argument('--dim-hidden', type=int, default=128, help="hidden dimension of Transformer")
parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
parser.add_argument('--abs_pe', type=str, default="rw", choices=POSENCODINGS.keys(),
                    help='which absolute PE to use?')
parser.add_argument('--abs_pe_dim', type=int, default=3, help='dimension for absolute PE')
parser.add_argument('--outdir', type=str, default='',
                    help='output path')
parser.add_argument('--warmup', type=int, default=5000, help="number of iterations for warmup")
parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
parser.add_argument('--gnn-type', type=str, default='gcn',
                    choices=GNN_TYPES,
                    help="GNN structure extractor type")
parser.add_argument('--k-hop', type=int, default=2, help="number of layers for GNNs")
parser.add_argument('--weight-class', action='store_true', help='weight classes or not')

parser.add_argument('--se', type=str, default="khopgnn",
                    help='Extractor type: khopgnn, or gnn')
random.seed(42)

# 设置NumPy随机种子
np.random.seed(42)

# 设置PyTorch的CPU随机种子
torch.manual_seed(42)

# 设置所有GPU的随机种子（多卡时）
torch.cuda.manual_seed_all(42)

# 禁用CUDA的随机性优化（确保确定性计算）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args = parser.parse_args()
args.batch_norm = not args.layer_norm
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
        return Data(x=torch.tensor(self.entities,dtype=torch.long),edge_index=torch.tensor(edge_index,dtype=torch.long),edge_type=torch.tensor(edge_type,dtype=torch.long)),edge_index, edge_type
#生成新的药物子图：
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
kg_file = KnowledgeGraph('../data/new_kg.txt')
triples = kg_file.triples
num_nodes = kg_file.num_nodes
num_rel = kg_file.num_relations
#生成子图
def convert(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.float):
        return float(o)
    print(type(o))
    raise TypeError
def k_hop_subgraph(node_idx, num_hops, edge_index, rel_index, fixed_num, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):

    np.random.seed(42)
    #num_nodes = maybe_num_nodes(edge_index, num_nodes)

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
    raw, col = edge_index
    reves_index = torch.stack((col, raw), 0)
    undirected_edge_index = torch.cat((edge_index, reves_index), 1)
    undirected_rel_index = torch.cat((rel_index, rel_index), 0)

    for d in drug_id:
        subset, sub_edge_index, sub_rel_index, mapping_list = k_hop_subgraph(int(d), khop, undirected_edge_index, undirected_rel_index, fixed_num, relabel_nodes=True,num_nodes=num_nodes)  ##subset是所有集合的节点，mapping指示的是center node是哪个
        row, col = sub_edge_index
        # all_degree.append(torch.max(degree(col)).item())

        ##因为这里面会涉及到multi-relation，所以在添加子图的时候，要把多条边都添加进去
        new_s_edge_index = sub_edge_index.transpose(1,0).numpy().tolist()
        new_s_value = [1 for _ in range(len(new_s_edge_index))]
        new_s_rel = sub_rel_index.numpy().tolist()
        node_idx = subset.numpy().tolist()

        s_edge_index = new_s_edge_index.copy()
        s_value = new_s_value.copy()
        s_rel = new_s_rel.copy()

#        edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
#        sp_edge_index = edge_index_value[:, :2]
#        sp_value = edge_index_value[:, 2]
#
#         for i in range(len(sp_edge_index)):
#             if sp_value[i] == 1:  ##也是保证多关系的边全部在数据里
#                 continue
#             else:
#                 s_edge_index.append(sp_edge_index[i].tolist())
#                 s_value.append(sp_value[i])
#                 s_rel.append(sp_value[i] + num_rel)
#
#         assert len(s_edge_index) == len(s_value)
#         assert len(s_edge_index) == len(s_rel)
#
#         num_rel_update.append(np.max(s_rel))

        subgraphs[d] = node_idx, new_s_edge_index, new_s_rel, mapping_list

    with open(json_path, 'w') as f:
        json.dump(subgraphs, f, default=convert)
    return subgraphs
kg_data,edge_index,edge_type = kg_file.gen_pyg_data()
drug_list = [i for i in range(1052)]
subgraph = subtreeExtractor(drug_list,edge_index,edge_type,'./',num_rel,64, 4,num_nodes)
full_degree = torch.zeros(num_nodes)
full_degree = scatter_add(torch.ones(kg_data.edge_index.size(1)),
                         kg_data.edge_index[0])
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
encoder = SubgraphRWEncoding(dim=8, use_original_degree=True)
class CustomData(Data):
    def __inc__(self, key, value,*args):
        if key == 'rel_index':  # rel_index 是边类型，不进行偏移
            return 0
        return super().__inc__(key, value)
def kg_gen_pe_emb(drug_list,graph):
    for i in drug_list:
        subset, subgraph_edge_index, subgraph_rel, mapping_id = graph[str(i)]
        data_graph = CustomData(x=torch.tensor(np.array(subset, dtype=int), dtype=torch.long),
                                edge_index=torch.LongTensor(subgraph_edge_index).transpose(1, 0),
                                id=torch.LongTensor(np.array(mapping_id, dtype=bool)),
                                rel_index=torch.tensor(np.array(subgraph_rel, dtype=int), dtype=torch.long))
        pe = encoder.compute_pe(data_graph,full_degree)
        graph[str(i)].append(pe)
    return graph
subgraph = kg_gen_pe_emb(drug_list,subgraph)
def mol_gen_pe_emb(drug_list,graph):
    for i in drug_list:
        c_size, features, edge_index, rel_index = smile_graph[str(i)]
        data_mol = Data(x=torch.Tensor(np.array(features)), edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                        rel_index=torch.tensor(np.array(rel_index, dtype=int), dtype=torch.long))
        data_mol.__setitem__('c_size', torch.LongTensor([c_size]))
        abs_pe_method = POSENCODINGS['rw']
        abs_pe_encoder = abs_pe_method(8, normalization='sym')
        pe = abs_pe_encoder.apply_to(data_mol)
        graph[str(i)].append(pe)
    return graph
smile_graph = mol_gen_pe_emb(drug_list,smile_graph)
class DTADataset(InMemoryDataset):
    def __init__(self, x=None, y=None, sub_graph=None, smile_graph=None):
        super(DTADataset, self).__init__()

        self.labels = y
        self.drug_ID = x
        self.sub_graph = sub_graph
        self.smile_graph = smile_graph
    def read_drug_info(self, drugid):
        c_size, features, edge_index, rel_index,mole_pe = self.smile_graph[str(drugid)]

        subset, subgraph_edge_index, subgraph_rel, mapping_id,pe = self.sub_graph[str(drugid)]
        data_mol = CustomData(x=torch.Tensor(np.array(features)),edge_index=torch.LongTensor(edge_index).transpose(1,0),
                            rel_index=torch.tensor(np.array(rel_index,dtype=int),dtype=torch.long),abs_pe= torch.tensor(mole_pe))
        data_mol.__setitem__('c_size',torch.LongTensor([c_size]))
        n = data_mol.num_nodes
        s = torch.arange(n)
        data_mol.complete_edge_index = torch.vstack((s.repeat_interleave(n), s.repeat(n)))
        data_graph = CustomData(x=torch.tensor(np.array(subset,dtype=int),dtype=torch.long),
                              edge_index= torch.LongTensor(subgraph_edge_index).transpose(1,0),
                              id=torch.LongTensor(np.array(mapping_id,dtype=bool)),
                              rel_index = torch.tensor(np.array(subgraph_rel,dtype=int),dtype=torch.long),
                              abs_pe = pe)
        #data_graph.abs_pe = self.kg_encoder.compute_pe(data_graph,full_degree)
        return data_mol,data_graph

    def __len__(self):
        #self.data_mol1, self.data_drug1, self.data_mol2, self.data_drug2
        return len(self.drug_ID)

    def __getitem__(self, idx):
        drug1_id, drug2_id = self.drug_ID[idx]
        label = int(self.labels[idx])

            # 获取药物1数据
        drug1_mol,drug1_subgraph = self.read_drug_info(str(drug1_id))
        drug2_mol,drug2_subgraph = self.read_drug_info(str(drug2_id))

        #return drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph

        return drug1_mol,drug1_subgraph, drug2_mol,drug2_subgraph,label
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    batchD = Batch.from_data_list([data[3] for data in data_list])
    labels = Batch.from_data_list([data[4] for data in data_list])
    return batchA, batchB, batchC, batchD,labels
def train_file(filename):
    interactions = []
    labels =[]
    with open(filename,'r') as f:
        lines = f.readlines()
        for i in lines:
            drug1,drug2,label = i.strip().split(' ')
            interactions.append([int(drug1),int(drug2),int(label)])
            labels.append(int(label))
    return interactions
interactions = train_file('../data/new_ddi.txt')
interactions = np.array(interactions)
# train_set,test_vaild_set = train_test_split(interactions,test_size=0.2,shuffle=True,random_state=42)
# vaild_set,test_set = train_test_split(test_vaild_set,test_size=0.5,shuffle=True,random_state=42)
# train_inter,train_label = train_set[:,0:2],train_set[:,2]
# vaild_inter,vaild_label = vaild_set[:,0:2],vaild_set[:,2]
# test_inter,test_label = test_set[:,0:2],test_set[:,2]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# for fold, (train_idx, test_vail_idx) in enumerate(kf.split(interactions)):
#     train_data = interactions[train_idx]
#     test_data = interactions[test_vail_idx]
#     vaild_set, test_set = train_test_split(test_data, test_size=0.5, shuffle=True, random_state=42)
#     fold_dir = f'../data/fold_{fold}'
#     np.save(f"{fold_dir}/train.npy", train_data)
#     np.save(f"{fold_dir}/valid.npy", vaild_set)
#     np.save(f"{fold_dir}/test.npy", test_set)
train = np.load("../data/cold_one/fold_{}/train.npy".format(args.fold))
valid = np.load("../data/cold_one/fold_{}/valid.npy".format(args.fold))
test = np.load("../data/cold_one/fold_{}/test.npy".format(args.fold))
#test_file = open('../data/test1.txt','r').readlines()
# test = []
# for i in test_file:
#     i= i.strip().split(' ')
#     test.append([int(i[0]),int(i[1]),int(i[2])])
# test = np.array(test)
# print(test)
train_inter,train_label = train[:,0:2],train[:,2]
vaild_inter,vaild_label = valid[:,0:2],valid[:,2]
test_inter,test_label = test[:,0:2],test[:,2]
train_data = DTADataset(x=train_inter,y=train_label,sub_graph=subgraph,smile_graph=smile_graph)
vaild_data = DTADataset(x=vaild_inter,y=vaild_label,sub_graph=subgraph,smile_graph=smile_graph)
test_data = DTADataset(x=test_inter,y=test_label,sub_graph=subgraph,smile_graph=smile_graph)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,collate_fn=collate)
vaild_loader = DataLoader(vaild_data,batch_size=args.batch_size,collate_fn=collate)
test_loader = DataLoader(test_data,batch_size=args.batch_size,collate_fn=collate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = DrugInteractionModel(number_nodes=num_nodes,in_size = 67,rel_num=13,kg_rel_num=72,num_class=1,d_model=80,dim_feedforward=80,
#                 dropout=0.1,num_heads=2,num_layers=2,batch_norm=False,abs_pe=False,
#                 abs_pe_dim=8,gnn_type='graph',use_edge_attr=False,num_edge_features=4,kg_num_edge_feature = 71,in_embed=False,
#                 edge_dim=80,k_hop=2,se='gnn',task_type='mole',device = device)
model = DrugInteractionModel(number_nodes=num_nodes,in_size = 67,rel_num=13,kg_rel_num=72,num_class=1,d_model=80,dim_feedforward=80,
                dropout=0.1,num_heads=2,num_layers=2,batch_norm=False,abs_pe=False,
                abs_pe_dim=8,gnn_type='gcn',use_edge_attr=True,num_edge_features=4,kg_num_edge_feature =71,in_embed=False,
                edge_dim=80,k_hop=2,se='gnn',task_type='mole',device = device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
model.to(device)
criterion = nn.BCELoss().to(device)
n_iterations = len(train_loader)
start = time.time()
best_vaild_loss = float('inf')
patience = 5
# best_auc / counter 必须在 epoch 循环外初始化，否则每轮归零 ->
# 每个 epoch 都覆盖保存最后一轮模型，早停永不触发。
best_auc = 0.0
counter = 0
# mole_input = mole_input.to(device)
# sub_kg_input = sub_kg_input.to(device)
for i_episode in range(50):
    loop = tqdm(train_loader, ncols=80)
    loop.set_description(f'Epoch[{i_episode}/{50}]')
    model.train()
    true_labels,pred_labels = [],[]
    running_loss = 0.0
    running_loss2 = 0.0
    running_correct = 0.0
    total_samples = 0
    for data in loop:
        optimizer.zero_grad()
        data_mol1 = data[0].to(device)
        data_drug1 = data[1].to(device)
        data_mol2 = data[2].to(device)
        data_drug2 = data[3].to(device)
        labels = data[4].to(device)
        # edge = data[0].to(device)
        # labels = data[1].to(device)
        y_pred, mi_loss = model(data_mol1,data_drug1,data_mol2,data_drug2)
        #y_pred = model(edge,labels,mole_input,sub_kg_input)
        y_pred = y_pred.reshape(-1)
        labels = labels.to(torch.float32)
        model_loss = criterion(y_pred, labels) + mi_loss
        model_loss.backward()
        optimizer.step()
        pred_labels.append(list(y_pred.cpu().detach().numpy().reshape(-1)))
        y_pred = y_pred.cpu().detach().numpy().round()
        labels = labels.cpu().numpy()
        total_samples += labels.shape[0]
        true_labels.append(list(labels))
        running_loss += model_loss.item()
        running_correct += (y_pred == labels).sum().item()
    print(f"epoch {i_episode}/{50};trainging loss: {running_loss / n_iterations:.4f}")
    print(f"epoch {i_episode}/{50};training set acc: {running_correct / total_samples:.4f}")
    total_labels = []
    total_pred_auc = []
    with torch.no_grad():
        model.eval()
        vaild_loop = tqdm(vaild_loader, ncols=80)
        for data in vaild_loop:
            # edge = data[0].to(device)
            # labels = data[1].to(device)
            data_mol1 = data[0].cuda()
            data_drug1 = data[1].cuda()
            data_mol2 = data[2].cuda()
            data_drug2 = data[3].cuda()
            labels = data[4].cuda()
            y_pred, _ = model(data_mol1,data_drug1,data_mol2,data_drug2)
            #y_pred = model(edge,labels,mole_input,sub_kg_input)
            total_pred_auc.append(y_pred.cpu().numpy().reshape(-1))
            labels = labels.cpu().numpy()
            total_labels.append(labels)
        #     y_pred = y_pred.reshape(-1)
        #     labels = labels.to(torch.float32)
        #     loss = criterion(y_pred, labels)
        #     vaild_loss += loss.item()
        # vaild_loss /= len(vaild_loader)
        total_pred_auc = np.concatenate(total_pred_auc)
        total_labels = np.concatenate(total_labels)
        auroc = roc_auc_score(total_labels, total_pred_auc)
        print(f"epoch {i_episode}/{50};vaild_AUC: {auroc:.4f}")
        if auroc > best_auc:
            best_auc = auroc
            counter = 0
            torch.save(model.state_dict(), 'SAT_DDI_drugbank_cold_one_{}.pth'.format(args.fold))  # best-val checkpoint
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping triggered')
                break
end = time.time()
elapsed = end - start
print(f"Training completed in {elapsed // 60}m: {elapsed % 60:.2f}s.")

n_test_samples = 0
n_correct = 0
total_labels = []
total_pred_auc = []
total_pred = []

# Testing phase
model.load_state_dict(torch.load('SAT_DDI_drugbank_cold_one_{}.pth'.format(args.fold)))
with torch.no_grad():
    model.eval()
    test_loop = tqdm(test_loader, ncols=80)
    for data in test_loop:
        data_mol1 = data[0].cuda()
        data_drug1 = data[1].cuda()
        data_mol2 = data[2].cuda()
        data_drug2 = data[3].cuda()
        labels = data[4].cuda()
        # edge = data[0].to(device)
        # labels = data[1].to(device)
        # edge_index = get_edge_index(train_edges_true).to(device)
        y_pred, _ = model(data_mol1,data_drug1,data_mol2,data_drug2)
        #y_pred = model(edge,labels,mole_input,sub_kg_input)
        total_pred_auc.append(y_pred.cpu().numpy().reshape(-1))
        y_pred = y_pred.cpu().numpy().reshape(-1).round()
        total_pred.append(y_pred)
        labels = labels.cpu().numpy()
        total_labels.append(labels)
        n_test_samples += len(data_mol1  )
        n_correct += (y_pred == labels).sum()
    # top_emb = top_emb.cpu().numpy()
    # attr_emb = attr_emb.cpu().numpy()
    # gcn_out = gcn_out.cpu().numpy()
    # np.save('vision_embedding/gcn_out.npy',gcn_out)
    # np.save('vision_embedding/top_embedding.npy', top_emb)
    # np.save('vision_embedding/attr_embedding.npy', attr_emb)
    acc = 100.0 * n_correct / n_test_samples
    total_pred = np.concatenate(total_pred)
    total_labels = np.concatenate(total_labels)
    total_pred_auc = np.concatenate(total_pred_auc)
    lr_precision, lr_recall, _ = precision_recall_curve(total_labels, total_pred_auc)
    aupr = auc(lr_recall, lr_precision)
    auroc = roc_auc_score(total_labels, total_pred_auc)
    f1 = f1_score(total_labels, total_pred)
    print(f"test set accuracy: {acc}")
    print(f"AUPR: {aupr}")
    print(f"AUROC: {auroc}")
    print(f"F1:{f1}")
# print(model)
# for i_episode in range(50):
#     loop = tqdm(train_loader, ncols=80)
#     loop.set_description(f'Epoch[{i_episode}/{50}]')
#     for data in loop:
#         data_mol1 = data[0]
#         data_drug1 = data[1]
#         data_mol2 = data[2]
#         data_drug2 = data[3]
#         result = model(data_mol1,data_drug1,data_mol2,data_drug2)
