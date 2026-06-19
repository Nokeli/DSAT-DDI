import torch
from sklearn.model_selection import train_test_split,StratifiedKFold
import argparse
import json
from torch_geometric.data import Data
import torch.nn as nn
import time
import numpy as np
from torch_geometric.data import Dataset
# from torch_geometric.utils import k_hop_subgraph
from data import GraphDataset
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
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
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score, auc
parser = argparse.ArgumentParser(
    description='Structure-Aware Transformer on DDI datasets',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--dataset', type=str, default="PATTERN",
                    help='name of dataset')
parser.add_argument('--num_class', type=int, default=2,
                    help='num of class')
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
parser.add_argument('--gnn-type', type=str, default='graph',
                    choices=GNN_TYPES,
                    help="GNN structure extractor type")
parser.add_argument('--k-hop', type=int, default=2, help="number of layers for GNNs")
parser.add_argument('--weight-class', action='store_true', help='weight classes or not')

parser.add_argument('--se', type=str, default="khopgnn",
                    help='Extractor type: khopgnn, or gnn')
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
        #edge_attr = self.rel_embedding(edge_type)
        #x = self.node_embedding(torch.arange(self.num_nodes))
        # return Data(x =x, edge_index=edge_index,edge_attr=edge_attr,edge_type=edge_type)
        return edge_index, edge_type
from torch_geometric.data import Data
import torch
from typing import List, Dict

def data_to_dict(data: Data) -> Dict:
    """将PyG Data对象转换为可序列化的字典"""
    return {
        'x': data.x.tolist() if data.x is not None else None,
        'edge_index': data.edge_index.tolist() if data.edge_index is not None else None,
        'edge_attr': data.edge_attr.tolist() if hasattr(data, 'edge_attr') and data.edge_attr is not None else None,
        'num_nodes': data.num_nodes if hasattr(data, 'num_nodes') else None,
        'center_node_idx': data.center_node_idx if hasattr(data, 'center_node_idx') else None
        # 可以添加其他需要的属性
    }
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

    def extract_subgraph_dict(self, entity_ids: List[int], return_edge_attr: bool = True) -> Dict[int, dict]:
        """
        批量生成子图字典并转换为可序列化格式

        参数:
            entity_ids: 中心节点ID列表
            return_edge_attr: 是否返回边属性

        返回:
            字典: {entity_id: subgraph_dict}
        """
        subgraph_dict = {}

        for entity_id in entity_ids:
            # 提取子图
            subgraph_data, _ = self.extract_subgraph(entity_id, return_edge_attr)
            # 转换为可序列化字典
            subgraph_dict[entity_id] = data_to_dict(subgraph_data)

        return subgraph_dict

    def save_subgraph_dict_to_json(self, subgraph_dict: Dict[int, dict], file_path: str):
        """
        将子图字典保存为JSON文件

        参数:
            subgraph_dict: 子图字典
            file_path: 保存路径
        """
        # 创建目录(如果不存在)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 保存为JSON
        with open(file_path, 'w') as f:
            json.dump(subgraph_dict, f, indent=4)

        print(f"子图字典已保存到: {file_path}")

    def extract_subgraph_dict_parallel(self, entity_ids: List[int], return_edge_attr: bool = True) -> Dict[int, Data]:
        """
        并行批量生成子图字典(实验性)

        参数:
            entity_ids: 中心节点ID列表
            return_edge_attr: 是否返回边属性

        返回:
            字典: {entity_id: subgraph_data}
        """
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        subgraph_dict = {}

        def _process_entity(entity_id):
            subgraph_data, _ = self.extract_subgraph(entity_id, return_edge_attr)
            return entity_id, subgraph_data

        # 使用线程池并行处理
        with ThreadPoolExecutor() as executor:
            futures = []
            for entity_id in entity_ids:
                futures.append(executor.submit(_process_entity, entity_id))

            # 使用tqdm显示进度条
            for future in tqdm(futures, desc="Extracting subgraphs"):
                entity_id, subgraph_data = future.result()
                subgraph_dict[entity_id] = subgraph_data

        return subgraph_dict
def extract_subgraphs(edge_index, edge_type, target_nodes, num_hops=2, max_neighbors=32):
    """
    提取目标节点的子图

    参数:
        edge_index: 边索引, shape [2, num_edges]
        edge_type: 边类型, shape [num_edges]
        target_nodes: 需要提取子图的节点列表
        num_hops: 跳数 (默认为2)
        max_neighbors: 每跳最大邻居数 (默认为32)

    返回:
        子图字典列表
    """
    # 构建邻接表
    edge_index_np = edge_index.numpy()
    edge_type_np = edge_type.numpy()
    # 构建邻接表 (考虑双向边)
    adj = defaultdict(list)
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        adj[src].append((dst, edge_type_np[i]))
        adj[dst].append((src, edge_type_np[i]))  # 假设是无向图

    subgraphs = []

    for node in target_nodes:
        node = int(node)  # 确保是Python原生int类型

        # 初始化
        visited = set([node])
        queue = deque([(node, 0)])
        current_nodes = set([node])
        edges = []
        edge_types = []

        # BFS遍历
        while queue:
            current_node, hop = queue.popleft()

            if hop >= num_hops:
                continue

            # 获取邻居并限制数量
            neighbors = adj.get(current_node, [])
            if len(neighbors) > max_neighbors:
                neighbors = neighbors[:max_neighbors]

            for neighbor, e_type in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1))

                # 如果邻居在当前子图中，则记录边
                if neighbor in visited:
                    edges.append((current_node, neighbor))
                    edge_types.append(int(e_type))  # 转换为Python原生int
                    current_nodes.add(neighbor)

        # 创建节点映射 (确保中心节点映射为0)
        unique_nodes = sorted(current_nodes)
        # 确保中心节点在第一个位置
        if node in unique_nodes:
            unique_nodes.remove(node)
            unique_nodes.insert(0, node)

        # 创建映射关系
        node_mapping = {old: new for new, old in enumerate(unique_nodes)}

        # 转换边索引
        new_edges = []
        for src, dst in edges:
            new_edges.append([node_mapping[src], node_mapping[dst]])

        # 构建子图字典
        subgraph = {
            "original_node": node,
            "num_nodes": len(unique_nodes),
            "edge_index": new_edges,
            "edge_type": edge_types,
            "node_mapping": {str(k): v for k, v in node_mapping.items()}
        }

        subgraphs.append(subgraph)

    return subgraphs
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

    ##在这个位置应该计算的是最短路径
    # s_edge_index_value = calculate_shortest_path(mol_edge_index)
    # s_edge_index = s_edge_index_value[:, :2]
    # s_value = s_edge_index_value[:, 2]
    # s_rel = s_value
    # s_rel[np.where(s_value == 1)] = mol_rel_index  ##将直接相连的关
    # s_rel[np.where(s_value != 1)] += 23
    #
    # assert len(s_edge_index) == len(s_value)
    # assert len(s_edge_index) == len(s_rel)

    ##c_size:原子的个数
    ##features:每个原子的特征 c_size * 67
    ##edge_index:边 n_edges * 2
    return c_size, features, mol_edge_index.tolist(), mol_rel_index.tolist()
drug_file = '../data/new_smiles.txt'
SMILES = []
drug = open(drug_file,'r')
drug_lines = drug.readlines()
for i in drug_lines:
    smiles = i.strip()
    SMILES.append(smiles)
smile_graph, rel_nums = smile_to_graph('./', SMILES)
# class MoleculeDataset(Dataset):
#     """自定义分子数据集"""
#
#     def __init__(self, smiles_list, transform=None, pre_transform=None):
#         super().__init__(None, transform, pre_transform)
#         self.smiles_list = smiles_list
#
#     def len(self):
#         return len(self.smiles_list)
#
#     def get(self, idx):
#         return smiles_to_mol_graph(self.smiles_list[idx])
class SubgraphLoader:
    """子图数据加载和处理类"""

    def __init__(self, json_path: str,num_nodes,num_relations):
        """
        初始化子图加载器

        参数:
            json_path: 包含子图数据的JSON文件路径
        """
        self.json_path = json_path
        self.subgraphs = self._load_subgraphs()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedding_dim =128
        self.node_embedding = nn.Embedding(self.num_nodes, self.embedding_dim)
        self.rel_embedding = nn.Embedding(self.num_relations, self.embedding_dim)
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.xavier_uniform_(self.rel_embedding.weight)
    def _load_subgraphs(self):
        """从JSON文件加载原始子图数据"""
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def to_pyg_data(self) -> List[Data]:
        """
        将子图数据转换为PyG Data对象列表

        返回:
            List[Data]: PyG Data对象列表
        """
        pyg_subgraphs = []
        for subgraph_dict in self.subgraphs:
            # 转换边索引和边类型为张量
            edge_index = torch.tensor(subgraph_dict['edge_index'], dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(subgraph_dict['edge_type'], dtype=torch.long)
            edge_att = self.rel_embedding(edge_type)
            node_index = torch.tensor(
                [int(key) for key in subgraph_dict['node_mapping'].keys()],  # 确保转换为int
                dtype=torch.long
            )
            x = self.node_embedding(node_index)
            # 创建Data对象
            data = Data(
                x = x,
                edge_index=edge_index,
                edge_type=edge_type,
                num_nodes=subgraph_dict['num_nodes'],
                edge_attr = edge_att,
                original_node=0)
            pyg_subgraphs.append(data)
        return pyg_subgraphs
class SubgraphDataset(Dataset):
    """自定义子图数据集，继承自PyG的Dataset类"""

    def __init__(self, 
                 json_path: str,
                 num_nodes: int,
                 num_relations: int,
                 transform = None,
                 pre_transform = None,
                 device: str = 'cuda'):
        """
        初始化子图数据集
        
        参数:
            json_path: 包含子图数据的JSON文件路径
            num_nodes: 图中最大节点数
            num_relations: 图中最大关系类型数
            transform: 动态图转换函数
            pre_transform: 预处理转换函数
            device: 计算设备
        """
        super().__init__(None, transform, pre_transform)
        self.json_path = json_path
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.device = device
        
        # 初始化嵌入层
        self.embedding_dim = 128
        self.node_embedding = nn.Embedding(num_nodes, self.embedding_dim).to(device)
        self.rel_embedding = nn.Embedding(num_relations, self.embedding_dim).to(device)
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.xavier_uniform_(self.rel_embedding.weight)
        
        # 加载数据
        self.subgraphs = self._load_subgraphs()
        
    def _load_subgraphs(self) -> List[dict]:
        """从JSON文件加载原始子图数据"""
        with open(self.json_path, 'r') as f:
            return json.load(f)
    
    def len(self) -> int:
        """返回数据集中的子图数量"""
        return len(self.subgraphs)
    
    def get(self, idx: int) -> Data:
        """
        获取指定索引的子图数据
        
        参数:
            idx: 子图索引
            
        返回:
            Data: PyG的Data对象
        """
        subgraph_dict = self.subgraphs[idx]
        
        # 转换边索引和边类型
        edge_index = torch.tensor(subgraph_dict['edge_index'], 
                                dtype=torch.long).t().contiguous().to(self.device)
        edge_type = torch.tensor(subgraph_dict['edge_type'],
                               dtype=torch.long).to(self.device)
        
        # 生成节点和边特征
        node_index = torch.tensor(
            [int(key) for key in subgraph_dict['node_mapping'].keys()],
            dtype=torch.long
        ).to(self.device)
        
        x = self.node_embedding(node_index)
        edge_attr = self.rel_embedding(edge_type)
        
        # 创建Data对象
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=subgraph_dict['num_nodes'],
            original_node=0
        )
        
        # 应用预处理（如果定义）  
        return data
    
    # def to_pyg_data(self) -> List[Data]:
    #     """兼容旧接口：返回所有子图的Data对象列表"""
    #     return [self.get(i) for i in range(len(self))]
# drug_file = '../data/new_smiles.txt'
# SMILES = []
# drug = open(drug_file,'r')
# drug_lines = drug.readlines()
# for i in drug_lines:
#     smiles = i.strip()
#     SMILES.append(smiles)
# dataset = MoleculeDataset(SMILES)
# print(dataset)
#new_data = GraphDataset(dataset,k_hop=2,se='gnn',degree=True)
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

    ## subset: LongTensor
    ## edge_index: LongTensor
    ## subgraph_rel: Tensor
    return subgraphs
# kg = kg_file.gen_pyg_data()
edge_index,edge_type = kg_file.gen_pyg_data()
drug_list = [i for i in range(1052)]
subgraph = subtreeExtractor(drug_list,edge_index,edge_type,'./',num_rel,32,2,num_nodes)
# sub_dict = extract_subgraphs(edge_index, edge_type, entity_list)
# with open("subgraphs.json", "w") as f:
#     json.dump(sub_dict, f, indent=2)
# class DTADataset(InMemoryDataset):
#     def __init__(self, x=None, y=None, sub_graph=None, smile_graph=None, ):
#         super(DTADataset, self).__init__()
#
#         self.labels = y
#         self.drug_ID = x
#         self.sub_graph = sub_graph
#         self.smile_graph = smile_graph
#         #self._validate_graph_coverage()
#         #self.data_mol1, self.data_drug1, self.data_mol2, self.data_drug2 = self.process(x, y, sub_graph, smile_graph)
#     def __len__(self):
#         #self.data_mol1, self.data_drug1, self.data_mol2, self.data_drug2
#         return len(self.drug_ID)
#
#     def __getitem__(self, idx):
#         drug1_id, drug2_id = self.drug_ID[idx]
#         label = int(self.labels[idx])
#
#             # 获取药物1数据
#         drug1_mol = self.smile_graph[drug1_id]
#         drug1_subgraph = self.sub_graph[drug1_id]
#
#             # 获取药物2数据
#         drug2_mol = self.smile_graph[drug2_id]
#         drug2_subgraph = self.sub_graph[drug2_id]
#
#         #return drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph
#
#         return drug1_mol,drug1_subgraph, drug2_mol,drug2_subgraph,label
class CustomData(Data):
    def __inc__(self, key, value,*args):
        if key == 'rel_index':  # rel_index 是边类型，不进行偏移
            return 0
        return super().__inc__(key, value)
class DTADataset(InMemoryDataset):
    def __init__(self, x=None, y=None, sub_graph=None, smile_graph=None, ):
        super(DTADataset, self).__init__()

        self.labels = y
        self.drug_ID = x
        self.sub_graph = sub_graph
        self.smile_graph = smile_graph
        #self._validate_graph_coverage()
        #self.data_mol1, self.data_drug1, self.data_mol2, self.data_drug2 = self.process(x, y, sub_graph, smile_graph)

    def read_drug_info(self, drugid):
        c_size, features, edge_index, rel_index = self.smile_graph[str(drugid)]
        subset, subgraph_edge_index, subgraph_rel, mapping_id = self.sub_graph[str(drugid)]
        data_mol = CustomData(x=torch.Tensor(np.array(features)),edge_index=torch.LongTensor(edge_index).transpose(1,0),
                            rel_index=torch.tensor(np.array(rel_index,dtype=int),dtype=torch.long))
        data_mol.__setitem__('c_size',torch.LongTensor([c_size]))
        data_graph = CustomData(x=torch.tensor(np.array(subset,dtype=int),dtype=torch.long),
                              edge_index= torch.LongTensor(subgraph_edge_index).transpose(1,0),
                              id=torch.LongTensor(np.array(mapping_id,dtype=bool)),
                              rel_index = torch.tensor(np.array(subgraph_rel,dtype=int),dtype=torch.long))
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
# def collate(data_list):
#     """
#     Custom collate function to batch the data.
#
#     Args:
#         data_list: List of tuples returned by __getitem__
#
#     Returns:
#         Tuple of batched graphs and labels
#     """
#     # Unpack the data list
#     drug1_mols = [data[0] for data in data_list]
#     drug1_subgraphs = [data[1] for data in data_list]
#     drug2_mols = [data[2] for data in data_list]
#     drug2_subgraphs = [data[3] for data in data_list]
#     labels = torch.stack([data[4] for data in data_list])
#
#     # Create batches
#     batchA = Batch.from_data_list(drug1_mols)
#     batchB = Batch.from_data_list(drug1_subgraphs)
#     batchC = Batch.from_data_list(drug2_mols)
#     batchD = Batch.from_data_list(drug2_subgraphs)
#
#     return batchA, batchB, batchC,batchD, labels
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
train_set,test_vaild_set = train_test_split(interactions,test_size=0.2,shuffle=True,random_state=0)
vaild_set,test_set = train_test_split(test_vaild_set,test_size=0.5,shuffle=True,random_state=0)
train_inter,train_label = train_set[:,0:2],train_set[:,2]
vaild_inter,vaild_label = vaild_set[:,0:2],vaild_set[:,2]
test_inter,test_label = test_set[:,0:2],test_set[:,2]
# loader = SubgraphLoader("subgraphs.json",kg_file.num_nodes,kg_file.num_relations)
# pyg_subgraphs = loader.to_pyg_data()
# pyg_subgraphs = SubgraphDataset(
#     json_path="subgraphs.json",
#     num_nodes=kg_file.num_nodes,
#     num_relations=kg_file.num_relations,
#     device='cuda')
#test_batch = Batch.from_data_list([pyg_subgraphs[0], pyg_subgraphs[1]])
#print(test_batch)  # 应为True
# mole_loader = DataLoader(dataset,batch_size=1052,shuffle=False,collate_fn=lambda batch: Batch.from_data_list(batch))
# for i in mole_loader:
#     mole_input = i
# print(mole_input)
# sub_kg_loader = DataLoader(pyg_subgraphs,batch_size=1052,shuffle=False,collate_fn=lambda batch: Batch.from_data_list(batch))
# for i in sub_kg_loader:
#     sub_kg_input = i
# print(sub_kg_input)
# class DDIDataset(Dataset):
#     '''Customized dataset processing class'''
#
#     def __init__(self, x, y):
#         self.x = torch.from_numpy(x)
#         self.y = torch.from_numpy(y)
#         self.n_samples = self.x.shape[0]
#
#     def __getitem__(self, index):
#         return self.x[index], self.y[index]
#
#     def __len__(self):
#         return self.n_samples

train_data = DTADataset(x=train_inter,y=train_label,sub_graph=subgraph,smile_graph=smile_graph)
vaild_data = DTADataset(x=vaild_inter,y=vaild_label,sub_graph=subgraph,smile_graph=smile_graph)
test_data = DTADataset(x=test_inter,y=test_label,sub_graph=subgraph,smile_graph=smile_graph)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,collate_fn=collate)
vaild_loader = DataLoader(vaild_data,batch_size=args.batch_size,collate_fn=collate)
test_loader = DataLoader(test_data,batch_size=args.batch_size,collate_fn=collate)
# train_data = DDIDataset(train_inter,train_label)
# vaild_data = DDIDataset(vaild_inter,vaild_label)
# test_data = DDIDataset(test_inter,test_label)
# train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
# vaild_loader = DataLoader(vaild_data,batch_size=args.batch_size)
# test_loader = DataLoader(test_data,batch_size=args.batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DrugInteractionModel(number_nodes=num_nodes,in_size = 67,rel_num=13,kg_rel_num=72,num_class=1,d_model=80,dim_feedforward=80,
                dropout=0.1,num_heads=2,num_layers=1,batch_norm=False,abs_pe=False,
                abs_pe_dim=0,gnn_type='graph',use_edge_attr=True,num_edge_features=4,kg_num_edge_feature = 71,in_embed=False,
                edge_dim=128,k_hop=2,se='gnn',task_type='mole',device = device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
model.to(device)
criterion = nn.BCELoss().to(device)
n_iterations = len(train_loader)
start = time.time()
best_vaild_loss = float('inf')
patience = 5
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
        y_pred = model(data_mol1,data_drug1,data_mol2,data_drug2)
        #y_pred = model(edge,labels,mole_input,sub_kg_input)
        y_pred = y_pred.reshape(-1)
        labels = labels.to(torch.float32)
        model_loss = criterion(y_pred, labels)
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
    with torch.no_grad():
        model.eval()
        vaild_loss = 0
        vaild_loop = tqdm(vaild_loader, ncols=80)
        for data in vaild_loop:
            # edge = data[0].to(device)
            # labels = data[1].to(device)
            data_mol1 = data[0].cuda()
            data_drug1 = data[1].cuda()
            data_mol2 = data[2].cuda()
            data_drug2 = data[3].cuda()
            labels = data[4].cuda()
            y_pred= model(data_mol1,data_drug1,data_mol2,data_drug2)
            #y_pred = model(edge,labels,mole_input,sub_kg_input)
            y_pred = y_pred.reshape(-1)
            labels = labels.to(torch.float32)
            loss = criterion(y_pred, labels)
            vaild_loss += loss.item()
        vaild_loss /= len(vaild_loader)
        print(f"epoch {i_episode}/{50};vaild_lost: {vaild_loss:.4f}")
        if vaild_loss < best_vaild_loss:
            best_vaild_loss = vaild_loss
            counter = 0
            torch.save(model.state_dict(), 'SAT_DDI.pth')
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
model.load_state_dict(torch.load('SAT_DDI.pth'))
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
        y_pred= model(data_mol1,data_drug1,data_mol2,data_drug2)
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
