from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
from torch_geometric.data import Data
import numpy as np
from sat.data import GraphDataset
from torch_geometric.loader import DataLoader
from models import GraphTransformer
from sat.gnn_layers import GNN_TYPES
from sat.position_encoding import POSENCODINGS
import argparse
parser = argparse.ArgumentParser(
    description='Structure-Aware Transformer on SBM datasets',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--dataset', type=str, default="PATTERN",
                    help='name of dataset')
parser.add_argument('--num_class', type=int, default=2,
                    help='num of class')
parser.add_argument('--num-heads', type=int, default=4, help="number of heads")
parser.add_argument('--num-layers', type=int, default=2, help="number of layers")
parser.add_argument('--dim-hidden', type=int, default=128 , help="hidden dimension of Transformer")
parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--batch-size', type=int, default=512,
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

parser.add_argument('--se', type=str, default="gnn",
                    help='Extractor type: khopgnn, or gnn')
args = parser.parse_args()
args.batch_norm = not args.layer_norm

def smiles_to_mol_graph(smiles: str,
                        use_positions: bool = False,
                        explicit_hydrogen: bool = True) -> Data:
    """
    将SMILES字符串转换为PyG的Data对象
    Args:
        smiles: 输入SMILES字符串
        use_positions: 是否使用2D/3D坐标作为节点特征
        explicit_hydrogen: 是否显式处理氢原子
    Returns:
        pyg.Data对象
    """
    # SMILES转RDKit分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # 可选：添加氢原子（默认开启）
    if explicit_hydrogen:
        mol = Chem.AddHs(mol)

    # 获取原子特征矩阵 [num_atoms, num_features]
    x = atom_features(mol, use_positions)

    # 获取键信息 [2, num_edges]
    edge_index, edge_attr = bond_features(mol)

    # 转换为PyG Data对象
    return Data(
        x=torch.FloatTensor(x),
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.FloatTensor(edge_attr) if edge_attr is not None else None,
        num_nodes=mol.GetNumAtoms()
    )


def atom_features(mol, use_positions=False):
    """提取原子级特征"""
    feats = []
    for atom in mol.GetAtoms():
        # 基础原子特征
        features = [
            float(atom.GetAtomicNum()),  # 原子序数
            float(atom.GetDegree()),  # 连接度
            float(atom.GetFormalCharge()),  # 形式电荷
            float(atom.GetHybridization().real),  # 杂化类型
            float(atom.GetIsAromatic()),  # 是否芳香族
            float(atom.GetMass() / 100.0),  # 归一化原子质量
            float(atom.GetNumImplicitHs()),  # 隐式氢数
            float(atom.GetTotalNumHs(includeNeighbors=True)),  # 总氢数
        ]

        # 可选：添加2D坐标
        if use_positions and mol.GetNumConformers() > 0:
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            features.extend([pos.x, pos.y, pos.z if hasattr(pos, 'z') else 0.0])

        feats.append(features)

    return np.array(feats)


def bond_features(mol):
    """提取键信息"""
    edges = []
    edge_feats = []

    for bond in mol.GetBonds():
        # 双向连接（无向图）
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

        # 键特征
        features = [
            float(bond.GetBondTypeAsDouble()),  # 键类型
            float(bond.GetIsConjugated()),  # 是否共轭
            float(bond.IsInRing()),  # 是否在环中
            float(bond.GetStereo())  # 立体化学
        ]
        edge_feats.append(features)
        edge_feats.append(features)  # 双向相同特征

    if len(edges) == 0:
        # 处理单原子分子
        edges.append([0, 0])
        edge_feats.append([0.0, 0.0, 0.0, 0.0])

    edge_index = np.array(edges).T  # [2, num_edges]
    edge_attr = np.array(edge_feats) if edge_feats else None

    return edge_index, edge_attr


from torch_geometric.data import Dataset


class MoleculeDataset(Dataset):
    """自定义分子数据集"""

    def __init__(self, smiles_list, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.smiles_list = smiles_list

    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        return smiles_to_mol_graph(self.smiles_list[idx])
drug_file = 'E:\学习\D盘\代码\work7\su_data\kegg\drug_smiles.txt'
SMILES = []
drug = open(drug_file,'r')
drug_lines = drug.readlines()
for i in drug_lines:
    drug_id,smiles = i.strip().split('\t')
    SMILES.append(smiles)
dataset = MoleculeDataset(SMILES)
#new_data = GraphDataset(dataset,k_hop=2,se='gnn',degree=True)
data = DataLoader(dataset,batch_size=20,shuffle=False)
model = GraphTransformer(in_size=8,
                             num_class=args.num_class,
                             d_model=args.dim_hidden,
                             dim_feedforward=2*args.dim_hidden,
                             dropout=args.dropout,
                             num_heads=args.num_heads,
                             num_layers=args.num_layers,
                             batch_norm=args.batch_norm,
                             abs_pe=args.abs_pe,
                             abs_pe_dim=args.abs_pe_dim,
                             gnn_type=args.gnn_type,
                             k_hop=args.k_hop,
                             use_edge_attr=True,
                             num_edge_features=4,
                             se=args.se,
                             in_embed=False,
                             edge_embed=False)
print(len(data))
for i in data:
    print(i)
    output = model(i)