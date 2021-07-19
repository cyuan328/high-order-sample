import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

# 得到qkv 和 利用qkv计算attention 的这两个步骤解耦

"""
    Graph Transformer
    
"""
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 100 
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
                                    dropout, self.layer_norm, self.batch_norm, self.residual) 
                                    for _ in range(n_layers-1)])
        # self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual))
        self.layers.append(GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual))
        
        # add fc here
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.MLP_layer = MLPReadout(out_dim, n_classes)
        self.smooth_distances = []


        # self.Q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # self.K = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None, compute_dis=False):

        # input embedding
        h = self.embedding_h(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        h = self.in_feat_dropout(h)

        
        # GraphTransformer Layers
        if compute_dis:
            self.smooth_distances = []
            for conv in self.layers:
                h, dis = conv.forward(g, h, compute_dis=True)
                self.smooth_distances.append(dis)
        else:
            for conv in self.layers:
                h = conv.forward(g, h, compute_dis=False)            
        h_in2 = h
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        h = h_in2 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        elif self.batch_norm:
            h = self.batch_norm2(h)   

        # output
        h_out = self.MLP_layer(h)

        return h_out
    
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss



        
