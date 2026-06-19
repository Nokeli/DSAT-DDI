# from models import GraphTransformer
# import torch
# from torch import nn
# from torch.nn import Linear,BCEWithLogitsLoss,Sequential, ReLU, BatchNorm1d as BN
# import math
# import torch.nn.functional as F
# import torch_geometric.nn as gnn
# class Discriminator(nn.Module):
#     def __init__(self, n_h):
#         super(Discriminator, self).__init__()
#         self.f_k = nn.Bilinear(n_h, n_h, 1)

#         for m in self.modules():
#             self.weights_init(m)

#     def weights_init(self, m):
#         if isinstance(m, nn.Bilinear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)

#     def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
#         # c: 1, 512; h_pl: 1, 2708, 512; h_mi: 1, 2708, 512
#         # c_x = torch.unsqueeze(c, 1)
#         # c_x = c_x.expand_as(h_pl)

#         c_x = c
#         sc_1 = self.f_k(h_pl, c_x)
#         sc_2 = self.f_k(h_mi, c_x)

#         if s_bias1 is not None:
#             sc_1 += s_bias1
#         if s_bias2 is not None:
#             sc_2 += s_bias2

#         logits = torch.cat((sc_1, sc_2), 0)

#         return logits

# def init_params(module, layers=2):
#     if isinstance(module, torch.nn.Linear):
#         module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
#         if module.bias is not None:
#             module.bias.data.zero_()
#     if isinstance(module, torch.nn.Embedding):
#         module.weight.data.normal_(mean=0.0, std=0.02)
# class NodeFeatures(torch.nn.Module):
#     def __init__(self, feature_num=1, embedding_dim=1,rel_num=1,layer=2, type='mole'):
#         super(NodeFeatures, self).__init__()

#         if type == 'mole': ##代表有feature num
#             self.node_encoder = Linear(feature_num, embedding_dim)
#             self.rel_encoder = torch.nn.Embedding(rel_num,embedding_dim)
#         else:
#             self.node_encoder = torch.nn.Embedding(feature_num, embedding_dim)
#             self.rel_encoder = torch.nn.Embedding(rel_num,embedding_dim)
#         self.apply(lambda module: init_params(module, layers=layer))

#     # def reset_parameters(self):
#     #     self.node_encoder.reset_parameters()
#     #     self.rel_encoder.reset_parameters()

#     def forward(self, data):
#         node_feature = self.node_encoder(data.x)
#         rel_feature = self.rel_encoder(data.rel_index)
#         return node_feature, rel_feature
# class DNN(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(DNN, self).__init__()

#         # self.layers = nn.Sequential(nn.Linear(num_inputs, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
#         #                             nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
#         #                             nn.Linear(256, num_outputs)
#         #                             )
#         self.layers = nn.Sequential(nn.Linear(num_inputs, 256),nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
#                                     nn.Linear(256, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.1),
#                                     nn.Linear(64, num_outputs)
#                                     )

#     def forward(self, x):
#         output = torch.sigmoid(self.layers(x))
#         return output
# class CrossAttentionFusion(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(CrossAttentionFusion, self).__init__()
#         self.query_proj = nn.Linear(input_dim, hidden_dim)
#         self.key_proj = nn.Linear(input_dim, hidden_dim)
#         self.value_proj = nn.Linear(input_dim, hidden_dim)

#     def forward(self, h_i, h_j):
#         """
#         h_i, h_j: [batch_size, input_dim]
#         Output:  h_ij: [batch_size, hidden_dim * 4]
#         """
#         Q_i = self.query_proj(h_i)
#         K_j = self.key_proj(h_j)
#         V_j = self.value_proj(h_j)

#         Q_j = self.query_proj(h_j)
#         K_i = self.key_proj(h_i)
#         V_i = self.value_proj(h_i)

#         # attention weights
#         alpha_i = F.softmax((Q_i * K_j).sum(dim=-1, keepdim=True) / Q_i.size(-1) ** 0.5, dim=1)
#         alpha_j = F.softmax((Q_j * K_i).sum(dim=-1, keepdim=True) / Q_j.size(-1) ** 0.5, dim=1)

#         h_i_tilde = alpha_i * V_j
#         h_j_tilde = alpha_j * V_i

#         h_diff = torch.abs(h_i_tilde - h_j_tilde)
#         h_mul = h_i_tilde * h_j_tilde

#         h_ij = torch.cat([h_i_tilde, h_j_tilde, h_diff, h_mul], dim=-1)
#         return h_ij
# class DrugInteractionModel(nn.Module):
#     def __init__(self,number_nodes =128,in_size = 8,rel_num=128,kg_rel_num=128,d_model=128,dim_feedforward=128,
#                 dropout=0.1,num_heads=2,num_layers=1,batch_norm=False,abs_pe=False,
#                 abs_pe_dim=0,gnn_type='graph',use_edge_attr=True,num_edge_features=1,kg_num_edge_feature = 71,in_embed=False,
#                 edge_dim=128,k_hop=2,se='gnn',task_type=None,device='cuda',**kwargs):
#         super().__init__()
#         self.number_nodes = number_nodes
#         self.rel_num = rel_num
#         self.kg_rel_num = kg_rel_num
#         self.in_size = in_size
#         self.d_model = d_model
#         self.dim_feedforward = dim_feedforward
#         self.dropout = dropout
#         self.num_head = num_heads
#         self.num_layers = num_layers
#         self.batch_norm = False
#         self.abs_pe = abs_pe
#         self.abs_pe_dim = abs_pe_dim
#         self.gnn_type = gnn_type
#         self.use_edge_attr = use_edge_attr
#         self.num_edge_feature = num_edge_features
#         self.kg_num_edge_feature = kg_num_edge_feature
#         self.in_embed = in_embed
#         self.edge_dim = edge_dim
#         self.k_hop = k_hop
#         self.se = se
#         self.task_type = task_type
#         self.fusion_hidden_dim = 256
#         self.device = device
#         self.disc = Discriminator(self.d_model)
#         self.b_xent = BCEWithLogitsLoss()
#         self.mol_feature = NodeFeatures(feature_num=self.in_size,embedding_dim=self.d_model,rel_num=self.rel_num,type='mole')
#         self.kg_feature = NodeFeatures(feature_num =self.number_nodes,embedding_dim=self.d_model,rel_num=self.kg_rel_num,type='node')
#         self.mole_encoder = GraphTransformer(in_size=in_size,
#                         num_class=1,
#                         d_model=self.d_model,
#                         dim_feedforward=self.dim_feedforward,
#                         dropout=self.dropout,
#                         num_heads=self.num_head,
#                         num_layers=self.num_layers,
#                         batch_norm=False,
#                         abs_pe=self.abs_pe,
#                         abs_pe_dim=self.abs_pe_dim,
#                         gnn_type=self.gnn_type,
#                         use_edge_attr=self.use_edge_attr,
#                         num_edge_features=self.num_edge_feature,
#                         in_embed= self.in_embed,
#                         edge_dim= self.edge_dim,
#                         k_hop=self.k_hop,se=self.se,task_type=self.task_type)
#         self.kg_encoder = GraphTransformer(in_size=self.number_nodes,
#                         num_class=1,
#                         d_model=self.d_model,
#                         dim_feedforward=self.dim_feedforward,
#                         dropout=self.dropout,
#                         num_heads=self.num_head,
#                         num_layers=self.num_layers,
#                         batch_norm=False,
#                         abs_pe=self.abs_pe,
#                         abs_pe_dim=self.abs_pe_dim,
#                         gnn_type=self.gnn_type,
#                         use_edge_attr=self.use_edge_attr,
#                         num_edge_features=self.kg_num_edge_feature,
#                         in_embed= self.in_embed,
#                         edge_dim= self.edge_dim,
#                         k_hop=self.k_hop,se=self.se,task_type='kg')
#         self.classifier = DNN(d_model*4, 1)
#         self.fc1 = nn.Sequential(
#             nn.Linear(d_model * 2, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128, d_model)
#         )
#         self.cross_att = CrossAttentionFusion(d_model * 2,d_model)
#         # self.fusion_mlp = nn.Sequential(
#         #     nn.Linear(d_model*4, self.fusion_hidden_dim),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim // 2),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(self.fusion_hidden_dim // 2, 1)
#         # )

#     def MI(self, graph_embeddings, sub_embeddings):
#         idx = torch.arange(graph_embeddings.shape[0] - 1, -1, -1)
#         idx[len(idx) // 2] = idx[len(idx) // 2 + 1]
#         shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx.to(self.device))
#         c_0_list, c_1_list = [], []
#         for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
#             c_0_list.append(c_0.expand_as(sub))  ##pos
#             c_1_list.append(c_1.expand_as(sub))  ##neg
#         c_0, c_1, sub = torch.cat(c_0_list), torch.cat(c_1_list), torch.cat(sub_embeddings)
#         return self.disc(sub, c_0, c_1)

#     def loss_MI(self, logits):
#         num_logits = logits.shape[0] // 2
#         temp = torch.rand(num_logits)
#         lbl = torch.cat([torch.ones_like(temp), torch.zeros_like(temp)], dim=0).float().to(self.device)

#         return self.b_xent(logits.view([1, -1]), lbl.view([1, -1]))
#     def forward(self, mole_data1,kg_data1,mole_data2,kg_data2):
#         mole_data1.x, mole_data1.edge_attr = self.mol_feature(mole_data1)
#         mole_data2.x, mole_data2.edge_attr = self.mol_feature(mole_data2)
#         mole_re1,mole_sub1 = self.mole_encoder(mole_data1)
#         mole_re2,mole_sub2 = self.mole_encoder(mole_data2)
#         # torch.save(mole_re1,'drug1_re.pt')
#         # torch.save(mole_re2,'drug2_re.pt')
#         kg_data1.x, kg_data1.edge_attr = self.kg_feature(kg_data1)
#         kg_data2.x, kg_data2.edge_attr = self.kg_feature(kg_data2)
#         kg_re1,kg_sub1 = self.kg_encoder(kg_data1)
#         kg_re2,kg_sub2= self.kg_encoder(kg_data2)
#         mole1 = torch.concat([mole_re1,kg_re1],dim=-1)
#         mole2 = torch.concat([mole_re2,kg_re2],dim=-1)
#         combine = torch.concat([mole1,mole2],dim=-1)
#         h_ij = self.cross_att(mole1,mole2)
#         logits = self.classifier(h_ij)
#         #logits = self.classifier(combine)
#         # loss_s_m = self.loss_MI(self.MI(mole1, mole_sub1)) + self.loss_MI(
#         #     self.MI(mole2, mole_sub2))
#         # loss_s_d = self.loss_MI(self.MI(kg_re1, kg_sub1)) + self.loss_MI(
#         #     self.MI(kg_re2, kg_sub2))
#         #logits = torch.sigmoid(logits)
#         return logits
#     # def forward(self, data,label,mole_data,kg_data):
#     #     mole_re1 = self.mole_encoder(mole_data)
#     #     #mole_re2 = self.mole_encoder(mole_data2)
#     #     #kg_re1 = self.kg_encoder(kg_data)
#     #     #kg_re2 = self.kg_encoder(kg_data2)
#     #     #mole1 = torch.concat([mole_re1,kg_re1],dim=-1)
#     #     #mole2 = torch.concat([mole_re2,kg_re2],dim=-1)
#     #     #drug_embeding = torch.concat([mole_re1,kg_re1],dim=-1)
#     #     #combine = torch.concat([mole1,mole2],dim=-1)
#     #     data = data.long()
#     #     combine = torch.hstack([mole_re1[data[:, 0]], mole_re1[data[:, 1]]])
#     #     # logits = self.fusion_mlp(combine)
#     #     # logits = torch.sigmoid(logits)
#     #     logits = self.classifier(combine)
#     #     return logits


from models import GraphTransformer
import torch
from torch import nn
from torch.nn import Linear,BCEWithLogitsLoss,Sequential, ReLU, BatchNorm1d as BN
import math
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import degree as tg_degree
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c: 1, 512; h_pl: 1, 2708, 512; h_mi: 1, 2708, 512
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_pl)

        c_x = c
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)

        return logits

def init_params(module, layers=2):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
class NodeFeatures(torch.nn.Module):
    def __init__(self, feature_num=1, embedding_dim=1,rel_num=1,layer=2, type='mole', max_degree=512):
        super(NodeFeatures, self).__init__()

        if type == 'mole': ##代表有feature num
            self.node_encoder = Linear(feature_num, embedding_dim)
            self.rel_encoder = torch.nn.Embedding(rel_num,embedding_dim)
        else:
            self.node_encoder = torch.nn.Embedding(feature_num, embedding_dim)
            self.rel_encoder = torch.nn.Embedding(rel_num,embedding_dim)
        # 结构信号：把节点在(子)图里的度映射成 embedding，加到节点特征上。
        # 度是推理时从 edge_index 现算的，对冷启动新药同样有效（即使其 ID embedding 未训练好）。
        self.max_degree = max_degree
        self.degree_encoder = torch.nn.Embedding(max_degree, embedding_dim, padding_idx=0)
        self.apply(lambda module: init_params(module, layers=layer))

    # def reset_parameters(self):
    #     self.node_encoder.reset_parameters()
    #     self.rel_encoder.reset_parameters()

    def forward(self, data):
        node_feature = self.node_encoder(data.x)
        row, col = data.edge_index
        x_degree = tg_degree(col, num_nodes=node_feature.size(0)).clamp(max=self.max_degree - 1).long()
        node_feature = node_feature + self.degree_encoder(x_degree)
        rel_feature = self.rel_encoder(data.rel_index)
        return node_feature, rel_feature
class DNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DNN, self).__init__()

        # self.layers = nn.Sequential(nn.Linear(num_inputs, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
        #                             nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
        #                             nn.Linear(256, num_outputs)
        #                             )
        self.layers = nn.Sequential(nn.Linear(num_inputs, 256),nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
                                    nn.Linear(256, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.1),
                                    nn.Linear(64, num_outputs)
                                    )

    def forward(self, x):
        output = torch.sigmoid(self.layers(x))
        return output
class CrossAttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CrossAttentionFusion, self).__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, h_i, h_j):
        """
        h_i, h_j: [batch_size, input_dim]
        Output:  h_ij: [batch_size, hidden_dim * 4]
        """
        Q_i = self.query_proj(h_i)
        K_j = self.key_proj(h_j)
        V_j = self.value_proj(h_j)

        Q_j = self.query_proj(h_j)
        K_i = self.key_proj(h_i)
        V_i = self.value_proj(h_i)

        # attention weights
        alpha_i = F.softmax((Q_i * K_j).sum(dim=-1, keepdim=True) / Q_i.size(-1) ** 0.5, dim=1)
        alpha_j = F.softmax((Q_j * K_i).sum(dim=-1, keepdim=True) / Q_j.size(-1) ** 0.5, dim=1)

        h_i_tilde = alpha_i * V_j
        h_j_tilde = alpha_j * V_i

        h_diff = torch.abs(h_i_tilde - h_j_tilde)
        h_mul = h_i_tilde * h_j_tilde

        h_ij = torch.cat([h_i_tilde, h_j_tilde, h_diff, h_mul], dim=-1)
        return h_ij
class DrugInteractionModel(nn.Module):
    def __init__(self,number_nodes =128,in_size = 8,rel_num=128,kg_rel_num=128,d_model=128,dim_feedforward=128,
                dropout=0.1,num_heads=2,num_layers=1,batch_norm=False,abs_pe=False,
                abs_pe_dim=0,gnn_type='graph',use_edge_attr=True,num_edge_features=1,kg_num_edge_feature = 71,in_embed=False,
                edge_dim=128,k_hop=2,se='gnn',task_type=None,device='cuda',**kwargs):
        super().__init__()
        self.number_nodes = number_nodes
        self.rel_num = rel_num
        self.kg_rel_num = kg_rel_num
        self.in_size = in_size
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_head = num_heads
        self.num_layers = num_layers
        self.batch_norm = False
        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        self.gnn_type = gnn_type
        self.use_edge_attr = use_edge_attr
        self.num_edge_feature = num_edge_features
        self.kg_num_edge_feature = kg_num_edge_feature
        self.in_embed = in_embed
        self.edge_dim = edge_dim
        self.k_hop = k_hop
        self.se = se
        self.task_type = task_type
        self.fusion_hidden_dim = 256
        self.device = device
        # 互信息(InfoMax)自监督正则的权重：分子子结构 / KG子图结构。沿用 TIGER 默认。
        self.mol_coeff = 0.2
        self.mi_coeff = 0.5
        # 反超路线：分子桥接的归纳式 KG 中心节点。
        # 把分子图表示投影到 KG 节点空间，注入到子图中心(药物)节点，
        # 使 KG 通道对未见新药也能产生有意义的表示(不再只靠转导式身份 lookup)。
        self.mol_to_kg = nn.Linear(self.d_model, self.d_model)
        self.id_dropout = 0.3  # 训练时模拟"未见药物"的比例(冷启动模拟)，可调
        self.disc = Discriminator(self.d_model)
        self.b_xent = BCEWithLogitsLoss()
        self.mol_feature = NodeFeatures(feature_num=self.in_size,embedding_dim=self.d_model,rel_num=self.rel_num,type='mole')
        self.kg_feature = NodeFeatures(feature_num =self.number_nodes,embedding_dim=self.d_model,rel_num=self.kg_rel_num,type='node')
        self.mole_encoder = GraphTransformer(in_size=in_size,
                        num_class=1,
                        d_model=self.d_model,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        num_heads=self.num_head,
                        num_layers=self.num_layers,
                        batch_norm=False,
                        abs_pe=self.abs_pe,
                        abs_pe_dim=self.abs_pe_dim,
                        gnn_type=self.gnn_type,
                        use_edge_attr=self.use_edge_attr,
                        num_edge_features=self.num_edge_feature,
                        in_embed= self.in_embed,
                        edge_dim= self.edge_dim,
                        k_hop=self.k_hop,se=self.se,task_type=self.task_type)
        self.kg_encoder = GraphTransformer(in_size=self.number_nodes,
                        num_class=1,
                        d_model=self.d_model,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        num_heads=self.num_head,
                        num_layers=self.num_layers,
                        batch_norm=False,
                        abs_pe=self.abs_pe,
                        abs_pe_dim=self.abs_pe_dim,
                        gnn_type=self.gnn_type,
                        use_edge_attr=self.use_edge_attr,
                        num_edge_features=self.kg_num_edge_feature,
                        in_embed= self.in_embed,
                        edge_dim= self.edge_dim,
                        k_hop=self.k_hop,se=self.se,task_type='kg')
        self.classifier = DNN(d_model*4, 1)
        self.fc1 = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model)
        )
        self.cross_att = CrossAttentionFusion(d_model * 2,d_model)
        # self.fusion_mlp = nn.Sequential(
        #     nn.Linear(d_model*4, self.fusion_hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(self.fusion_hidden_dim // 2, 1)
        # )

    def MI(self, graph_embeddings, sub_embeddings):
        idx = torch.arange(graph_embeddings.shape[0] - 1, -1, -1)
        idx[len(idx) // 2] = idx[len(idx) // 2 + 1]
        shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx.to(self.device))
        c_0_list, c_1_list = [], []
        for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
            c_0_list.append(c_0.expand_as(sub))  ##pos
            c_1_list.append(c_1.expand_as(sub))  ##neg
        c_0, c_1, sub = torch.cat(c_0_list), torch.cat(c_1_list), torch.cat(sub_embeddings)
        return self.disc(sub, c_0, c_1)

    def loss_MI(self, logits):
        num_logits = logits.shape[0] // 2
        temp = torch.rand(num_logits)
        lbl = torch.cat([torch.ones_like(temp), torch.zeros_like(temp)], dim=0).float().to(self.device)

        return self.b_xent(logits.view([1, -1]), lbl.view([1, -1]))
    def _inductive_center(self, kg_data, mol_re):
        """用分子表示为 KG 子图中心(药物)节点生成归纳式特征。
        冷启动药物的身份 lookup 未训练(≈噪声)，而分子结构始终可得；
        把分子表示注入中心节点 -> KG 通道可泛化到未见新药。
        训练时以 id_dropout 概率把中心节点的身份/结构特征清零(模拟"未见")，
        迫使模型在身份缺失时也能仅靠分子注入做出预测。"""
        center_idx = kg_data.id.nonzero().flatten()      # 每个子图的中心节点位置 [B]
        mol_proj = self.mol_to_kg(mol_re)                # [B, d]
        x = kg_data.x.clone()
        center = x[center_idx]
        if self.training and self.id_dropout > 0:
            drop = (torch.rand(center_idx.size(0), device=x.device) < self.id_dropout).unsqueeze(-1)
            center = torch.where(drop, torch.zeros_like(center), center)
        x[center_idx] = center + mol_proj
        kg_data.x = x
        return kg_data

    def forward(self, mole_data1,kg_data1,mole_data2,kg_data2):
        mole_data1.x, mole_data1.edge_attr = self.mol_feature(mole_data1)
        mole_data2.x, mole_data2.edge_attr = self.mol_feature(mole_data2)
        mole_re1,mole_sub1,_ = self.mole_encoder(mole_data1)
        mole_re2,mole_sub2,_ = self.mole_encoder(mole_data2)
        # torch.save(mole_re1,'drug1_re.pt')
        # torch.save(mole_re2,'drug2_re.pt')
        kg_data1.x, kg_data1.edge_attr = self.kg_feature(kg_data1)
        kg_data2.x, kg_data2.edge_attr = self.kg_feature(kg_data2)
        # 分子桥接的归纳式中心节点(反超路线核心)
        kg_data1 = self._inductive_center(kg_data1, mole_re1)
        kg_data2 = self._inductive_center(kg_data2, mole_re2)
        kg_re1,kg_sub1,_ = self.kg_encoder(kg_data1)
        kg_re2,kg_sub2,_ = self.kg_encoder(kg_data2)
        mole1 = torch.concat([mole_re1,kg_re1],dim=-1)
        mole2 = torch.concat([mole_re2,kg_re2],dim=-1)
        combine = torch.concat([mole1,mole2],dim=-1)
        h_ij = self.cross_att(mole1,mole2)
        logits = self.classifier(h_ij)
        #logits = self.classifier(combine)
        # InfoMax 自监督正则：迫使每个通道的池化表示能预测自己的子结构，
        # 让表示扎根于可迁移的结构而非药物身份记忆 -> 提升冷启动泛化。
        # 锚点用通道各自的 d_model 维表示（与 Discriminator(d_model) 对齐）。
        loss_s_m = self.loss_MI(self.MI(mole_re1, mole_sub1)) + self.loss_MI(
            self.MI(mole_re2, mole_sub2))
        loss_s_d = self.loss_MI(self.MI(kg_re1, kg_sub1)) + self.loss_MI(
            self.MI(kg_re2, kg_sub2))
        mi_loss = self.mol_coeff * loss_s_m + self.mi_coeff * loss_s_d
        #logits = torch.sigmoid(logits)
        return logits, mi_loss








