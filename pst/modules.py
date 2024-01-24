import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .gnn_layers import EDGE_GNN_TYPES, get_simple_gnn_layer
from .rotary_embedding import RotaryEmbedding
from .utils import (
    from_nested_tensor,
    to_dense_batch,
    to_nested_tensor,
    unpad_dense_batch,
)


class MultiheadAttention(nn.Module):
    """Multi-head attention using PyG interface
    accept Batch object given by PyG
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        dropout=0.0,
        bias=True,
        use_rotary_embeddings=True,
        gnn_type="gin",
        k_hop=2,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = head_dim

        self.struct_extractor = StructureExtractor(
            embed_dim,
            gnn_type=gnn_type,
            num_layers=k_hop,
            **kwargs,
        )

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.k_struct_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_struct_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_struct_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

        self.rot_emb = None
        if use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)

        if self.bias:
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.q_proj.bias, 0.0)

        nn.init.zeros_(self.k_struct_proj.weight)
        nn.init.zeros_(self.q_struct_proj.weight)
        nn.init.zeros_(self.v_struct_proj.weight)

    def forward(self, x, edge_index, edge_attr=None, ptr=None):
        # Compute query/key/value matrix
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if edge_index is not None:
            x_struct = self.struct_extractor(x, edge_index, edge_attr)
            q = q + self.q_struct_proj(x_struct)
            k = k + self.k_struct_proj(x_struct)
            v = v + self.v_struct_proj(x_struct)

        out, attn = self.self_attn_v2((q, k), v, ptr)
        return self.out_proj(out), attn

    def self_attn_v2(self, qk, v, ptr):
        qk, mask = to_dense_batch(qk, ptr, return_mask=True)
        q, k = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qk)

        v = to_dense_batch(v, ptr)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        if self.rot_emb:
            q, k = self.rot_emb(q, k)

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, ~mask.unsqueeze(1).unsqueeze(2), dropout_p=dropout_p
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        out = unpad_dense_batch(out, mask, ptr)

        return out, None

    def self_attn_nested_v2(self, qk, v, ptr, return_attn=False):
        k, q = qk
        k, q = map(lambda t: to_nested_tensor(t, ptr=ptr), qk)
        batch_size = q.size(0)
        q = (
            q.reshape(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            k.reshape(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        v = to_nested_tensor(v, ptr=ptr)
        v = (
            v.reshape(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        out = from_nested_tensor(out)

        if return_attn:
            return out, dots
        return out, None


class StructureExtractor(nn.Module):
    def __init__(
        self, embed_dim, gnn_type="gin", num_layers=2, use_edge_attr=False, **kwargs
    ):
        super().__init__()

        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.use_edge_attr = use_edge_attr

        self.gnn = nn.ModuleList(
            [
                get_simple_gnn_layer(gnn_type, embed_dim, use_edge_attr, **kwargs)
                for _ in range(num_layers)
            ]
        )

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(embed_dim) for _ in range(num_layers)]
        )

        self.out_projs = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)]
        )

    def forward(self, x, edge_index, edge_attr=None):
        out = 0
        for i, gnn_layer in enumerate(self.gnn):
            if (
                self.use_edge_attr
                and self.gnn_type in EDGE_GNN_TYPES
                and edge_attr is not None
            ):
                if self.gnn_type == "gatedgcn_edge":
                    x, edge_attr = gnn_layer(
                        x, edge_index, edge_attr=edge_attr, return_edge=True
                    )
                else:
                    x = gnn_layer(x, edge_index, edge_attr=edge_attr)
            else:
                x = gnn_layer(x, edge_index)

            x = self.batch_norms[i](x)

            out = out + self.out_projs[i](x)

            if i == self.num_layers - 1:
                break

            if not "gatedgcn" in self.gnn_type and not "pifold" in self.gnn_type:
                x = F.relu(x)

            if self.gnn_type == "gatedgcn_edge":
                edge_attr = F.dropout(
                    edge_attr, self.gnn_dropout, training=self.training
                )

        return out


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=512,
        dropout=0.0,
        attn_dropout=0.0,
        activation="gelu",
        gnn_type="gin",
        k_hop=2,
        **kwargs,
    ):
        super().__init__()

        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            bias=True,
            gnn_type=gnn_type,
            k_hop=k_hop,
            **kwargs,
        )

        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)

        self.final_layer_norm = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None, ptr=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            x,
            edge_index,
            edge_attr=edge_attr,
            ptr=ptr,
        )
        x = residual + self.dropout1(x)

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        x = residual + self.dropout2(x)

        return x, edge_attr


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x
