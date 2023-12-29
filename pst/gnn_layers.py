# -*- coding: utf-8 -*-
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import Tensor, nn
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch_geometric import utils
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree
from torch_scatter import scatter

GNN_TYPES = [
    "graph",
    "graphsage",
    "gcn",
    "gin",
    "gine",
    "pna",
    "pna2",
    "pna3",
    "mpnn",
    "pna4",
    "rwgnn",
    "khopgnn",
    "gat",
    "gatedgcn",
    "gatedgcn_resi",
    "gatedgcn_edge",
]

EDGE_GNN_TYPES = [
    "gine",
    "gcn",
    "graphsage",
    "graph",
    "pna",
    "pna2",
    "pna3",
    "mpnn",
    "pna4",
    "gat",
    "gatedgcn",
    "gatedgcn_resi",
    "gatedgcn_edge",
    "pifold",
]


def get_simple_gnn_layer(gnn_type, embed_dim, use_edge_attr=False, **kwargs):
    edge_dim = embed_dim if use_edge_attr else None
    if gnn_type == "graph":
        if edge_dim is None:
            return gnn.GraphConv(embed_dim, embed_dim)
        else:
            return SAGEConv(embed_dim, edge_dim, aggr="add")
    elif gnn_type == "graphsage":
        if edge_dim is None:
            return gnn.SAGEConv(embed_dim, embed_dim)
        else:
            return SAGEConv(embed_dim, edge_dim)
    elif gnn_type == "gcn":
        if edge_dim is None:
            return gnn.GCNConv(embed_dim, embed_dim)
        else:
            return GCNConv(embed_dim, edge_dim)
    elif gnn_type == "gatedgcn":
        return GatedGCN(embed_dim, edge_dim)
    elif gnn_type == "gatedgcn_resi":
        return GatedGCN(embed_dim, edge_dim, residual=True)
    elif gnn_type == "gatedgcn_edge":
        return GatedGCN(embed_dim, edge_dim)
    elif gnn_type == "gin":
        mlp = mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        return gnn.GINConv(mlp, train_eps=True)
    elif gnn_type == "gine":
        mlp = mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        return gnn.GINEConv(mlp, train_eps=True, edge_dim=edge_dim)
    elif gnn_type == "pna":
        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        deg = kwargs.get("deg", None)
        layer = gnn.PNAConv(
            embed_dim,
            embed_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=True,
            edge_dim=edge_dim,
        )
        return layer
    elif gnn_type == "pna2":
        aggregators = ["mean", "sum", "max"]
        scalers = ["identity"]
        deg = kwargs.get("deg", None)
        layer = gnn.PNAConv(
            embed_dim,
            embed_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=True,
            edge_dim=edge_dim,
        )
        return layer
    elif gnn_type == "pna2_ram":
        aggregators = ["mean", "sum", "max"]
        scalers = ["identity"]
        deg = kwargs.get("deg", None)
        layer = PNAConv_towers(
            embed_dim,
            embed_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=4,
            divide_input=True,
            edge_dim=edge_dim,
        )
        return layer
    elif gnn_type == "pna3":
        aggregators = ["mean", "sum", "max"]
        scalers = ["identity"]
        deg = kwargs.get("deg", None)
        layer = gnn.PNAConv(
            embed_dim,
            embed_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=1,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
            edge_dim=edge_dim,
        )
        return layer
    elif gnn_type == "pna4":
        aggregators = ["mean", "sum", "max"]
        scalers = ["identity"]
        deg = kwargs.get("deg", None)
        layer = gnn.PNAConv(
            embed_dim,
            embed_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=8,
            pre_layers=1,
            post_layers=1,
            divide_input=True,
            edge_dim=edge_dim,
        )
        return layer

    elif gnn_type == "mpnn":
        aggregators = ["sum"]
        scalers = ["identity"]
        deg = kwargs.get("deg", None)
        layer = gnn.PNAConv(
            embed_dim,
            embed_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=True,
            edge_dim=edge_dim,
        )
        return layer
    elif gnn_type == "gat":
        return gnn.GATConv(embed_dim, embed_dim, edge_dim=edge_dim)
    elif gnn_type == "pifold":
        return NeighborAttention(embed_dim)
    else:
        raise ValueError("Not implemented!")


class GCNConv(gnn.MessagePassing):
    def __init__(self, embed_dim, edge_dim):
        super(GCNConv, self).__init__(aggr="add")

        self.linear = nn.Linear(embed_dim, embed_dim)
        self.root_emb = nn.Embedding(1, embed_dim)

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = nn.Linear(edge_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = utils.degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr=edge_embedding, norm=norm
        ) + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class SAGEConv(gnn.MessagePassing):
    def __init__(self, embed_dim, edge_dim, aggr="mean"):
        super(SAGEConv, self).__init__(aggr=aggr)

        self.embed_dim = embed_dim
        self.edge_dim = edge_dim

        self.lin_l = nn.Linear(embed_dim, embed_dim)
        self.lin_r = nn.Linear(embed_dim, embed_dim)

        self.edge_encoder = nn.Linear(edge_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr, size=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)

        edge_embedding = self.edge_encoder(edge_attr)

        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        out += self.lin_r(x_r)

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)


import math

from torch_scatter import scatter_softmax, scatter_sum


class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        self.W_V = nn.Sequential(
            nn.Linear(num_hidden * 2, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
        )
        self.Bias = nn.Sequential(
            nn.Linear(num_hidden * 3, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_heads),
        )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, h_V, edge_index, edge_attr):
        h_E = edge_attr
        center_id, dst_idx = edge_index
        h_E = torch.cat([h_E, h_V[dst_idx]], dim=-1)
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)

        w = self.Bias(torch.cat([h_V[center_id], h_E], dim=-1)).view(E, n_heads, 1)
        attend_logits = w / math.sqrt(d)

        V = self.W_V(h_E).view(-1, n_heads, d)
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)
        h_V = scatter_sum(attend * V, center_id, dim=0).view([-1, self.num_hidden])

        return self.W_O(h_V)


class GatedGCN(gnn.MessagePassing):
    """
    GatedGCN layer
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """

    def __init__(self, embed_dim, edge_dim, residual=False):
        super().__init__()
        self.A = nn.Linear(embed_dim, embed_dim)
        self.B = nn.Linear(embed_dim, embed_dim)
        self.C = nn.Linear(edge_dim, embed_dim)
        self.D = nn.Linear(embed_dim, embed_dim)
        self.E = nn.Linear(embed_dim, embed_dim)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        # self.EquivStablePE = equivstable_pe
        # if self.EquivStablePE:
        #     self.mlp_r_ij = nn.Sequential(
        #         nn.Linear(1, out_dim), nn.ReLU(),
        #         nn.Linear(out_dim, 1),
        #         nn.Sigmoid())

        self.bn_node_x = nn.BatchNorm1d(embed_dim)
        self.bn_edge_e = nn.BatchNorm1d(embed_dim)
        self.e = None
        self.residual = residual

    def forward(self, x, edge_index, edge_attr, return_edge=False):
        # x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        e = edge_attr
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        # pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, e=e, Ax=Ax)

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = F.relu(x)
        e = F.relu(e)

        # x = F.dropout(x, self.dropout, training=self.training)
        # e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            e = e_in + e

        # batch.x = x
        # batch.edge_attr = e
        if return_edge:
            return x, e

        return x

    def message(self, Dx_i, Ex_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        # if self.EquivStablePE:
        #     r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
        #     r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
        #     sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size, reduce="sum")

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size, reduce="sum")

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out
