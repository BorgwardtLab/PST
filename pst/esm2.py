import esm
import torch
from torch import nn

from .modules import RobertaLMHead, TransformerLayer


class PST(nn.Module):
    def __init__(
        self,
        embed_dim=320,
        num_heads=20,
        num_layers=4,
        token_dropout=True,
        edge_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.alphabet_size = len(self.alphabet)
        self.padding_idx = self.alphabet.padding_idx
        self.mask_idx = self.alphabet.mask_idx
        self.token_dropout = token_dropout

        self.embed_edge = None
        use_edge_attr = False
        if edge_dim is not None:
            self.embed_edge = nn.Sequential(
                nn.Linear(edge_dim, embed_dim),
                nn.GELU(),
                nn.BatchNorm1d(embed_dim),
                nn.Linear(embed_dim, embed_dim),
            )
            use_edge_attr = True

        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim,
                    num_heads,
                    4 * embed_dim,
                    use_rotary_embeddings=True,
                    use_edge_attr=use_edge_attr,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.emb_layer_norm_after = nn.LayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(self, data, return_repr=False, aggr=None):
        seq, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        if self.embed_edge is not None:
            edge_attr = self.embed_edge(edge_attr)

        x = self.embed_tokens(seq)
        if self.token_dropout:
            from torch_scatter import scatter_mean

            x.masked_fill_((seq == self.mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8
            mask_ratio_observed = scatter_mean(
                (seq == self.mask_idx).to(x.dtype), data.batch
            )
            mask_ratio = (1 - mask_ratio_train) / (1 - mask_ratio_observed)
            x = x * mask_ratio.index_select(-1, data.batch)[:, None]

        hidden_reprs = []
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr=edge_attr, ptr=data.ptr)
            if return_repr and aggr is not None:
                hidden_reprs.append(x)

        x = self.emb_layer_norm_after(x)

        if return_repr:
            if aggr is not None:
                hidden_reprs[-1] = x
                if aggr == "mean":
                    x = sum(hidden_reprs) / len(hidden_reprs)
                else:
                    x = torch.cat(hidden_reprs, dim=-1)
            return x

        x = self.lm_head(x)

        return x

    def mask_forward(self, batch):
        node_true = batch.masked_node_label
        node_pred = self(batch)[batch.masked_node_indices]
        return node_pred, node_true

    @torch.no_grad()
    def mask_predict(self, batch):
        return self(batch)[batch.masked_node_indices]

    def train_struct_only(self, mode=True):
        if mode:
            for n, p in self.named_parameters():
                if "struct" not in n:
                    p.requires_grad = False

    @classmethod
    def from_model_name(cls, model_name, **model_args):
        esm, _ = torch.hub.load("facebookresearch/esm:main", model_name)
        model = get_model(cls, model_name, **model_args)
        model.load_state_dict(esm.state_dict(), strict=False)
        return model

    def save(self, model_path, cfg):
        torch.save({"cfg": cfg, "state_dict": self.state_dict()}, model_path)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        checkpoint = torch.load(model_path, **kwargs)
        cfg, state_dict = checkpoint["cfg"], checkpoint["state_dict"]
        model = get_model(
            cls,
            cfg.model.name,
            k_hop=cfg.model.k_hop,
            gnn_type=getattr(cfg.model, "gnn_type", "gin"),
            edge_dim=getattr(cfg.model, "edge_dim", None),
        )
        model.load_state_dict(state_dict)
        return model, cfg

    @classmethod
    def from_pretrained_url(
        cls, model_name, model_path, train_struct_only=True, **kwargs
    ):
        import os
        from .utils import download_url_content
        name_converter = {
            "pst_t6": ["esm2_t6_8M_UR50D", "train_all"],
            "pst_t6_so": ["esm2_t6_8M_UR50D", "train_struct_only"],
            "pst_t12": ["esm2_t12_35M_UR50D", "train_all"],
            "pst_t12_so": ["esm2_t12_35M_UR50D", "train_struct_only"],
            "pst_t30": ["esm2_t30_150M_UR50D", "train_all"],
            "pst_t30_so": ["esm2_t30_150M_UR50D", "train_struct_only"],
            "pst_t33": ["esm2_t33_650M_UR50D", "train_all"],
            "pst_t33_so": ["esm2_t33_650M_UR50D", "train_struct_only"],
        }

        all_models = {
            "esm2_t6_8M_UR50D": {
                "train_struct_only": "https://datashare.biochem.mpg.de/s/ARzKycmMQePvLXs/download",
                "train_all": "https://datashare.biochem.mpg.de/s/ac9ufZ0NB2IrkZL/download",
            },
            "esm2_t12_35M_UR50D": {
                "train_struct_only": "https://datashare.biochem.mpg.de/s/qRvDPTfExZkq38f/download",
                "train_all": "https://datashare.biochem.mpg.de/s/fOSIwJAIKLYjFe3/download",
            },
            "esm2_t30_150M_UR50D": {
                "train_struct_only": "https://datashare.biochem.mpg.de/s/p73BABG81gZKElL/download",
                "train_all": "https://datashare.biochem.mpg.de/s/a3yugJJMe0I0oEL/download",
            },
            "esm2_t33_650M_UR50D": {
                "train_struct_only": "https://datashare.biochem.mpg.de/s/xGpS7sIG7k8DZX0/download",
                "train_all": "https://datashare.biochem.mpg.de/s/RpWYV4o4ka3gHvX/download",
            },
        }
        train_struct_only_key = (
            "train_struct_only" if train_struct_only else "train_all"
        )
        original_model_name = model_name
        if model_name.startswith("pst"):
            model_name, train_struct_only_key = name_converter[model_name]
        model_url = all_models[model_name][train_struct_only_key]
        if not os.path.isfile(str(model_path)):
            print(f"Downloading {original_model_name} to {model_path}...")
            download_url_content(model_url, str(model_path))
        return cls.from_pretrained(model_path, **kwargs)


def get_model(cls, model_name, **model_args):
    if model_name == "esm2_t6_8M_UR50D":
        model = cls(embed_dim=320, num_heads=20, num_layers=6, **model_args)
    elif model_name == "esm2_t12_35M_UR50D":
        model = cls(embed_dim=480, num_heads=20, num_layers=12, **model_args)
    elif model_name == "esm2_t30_150M_UR50D":
        model = cls(embed_dim=640, num_heads=20, num_layers=30, **model_args)
    elif model_name == "esm2_t33_650M_UR50D":
        model = cls(embed_dim=1280, num_heads=20, num_layers=33, **model_args)
    elif model_name == "esm2_t36_3B_UR50D":
        model = cls(embed_dim=2560, num_heads=20, num_layers=36, **model_args)
    else:
        raise ValueError("Model not implemented!")
    return model


import torch_geometric.nn as gnn


class ProteinNet(nn.Module):
    def __init__(
        self, base_model, num_class, out_head="linear", aggr=None, dropout=0.5
    ):
        super().__init__()
        self.base_model = base_model
        self.embed_dim = self.base_model.embed_dim

        self.num_class = num_class

        if out_head == "linear":
            self.out_head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(self.embed_dim, self.num_class)
            )
        else:
            self.out_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim // 4, self.num_class),
            )

        self.aggr = aggr
        if aggr == "concat":
            self.norm = nn.LayerNorm(self.embed_dim * len(base_model.layers))
            self.lin_proj = nn.ModuleList(
                [
                    nn.Linear(self.embed_dim, self.embed_dim, bias=False)
                    for i in range(len(base_model.layers))
                ]
            )

    def head_parameters(self):
        if self.aggr == "concat":
            import itertools

            return itertools.chain(
                self.out_head.parameters(),
                self.norm.parameters(),
                self.lin_proj.parameters(),
            )
        else:
            return self.out_head.parameters()

    def forward(self, data, include_seq=False):
        out = self.base_model(data, return_repr=True, aggr=self.aggr)
        out, batch = out[data.idx_mask], data.batch[data.idx_mask]
        out = gnn.global_mean_pool(out, batch)
        if include_seq:
            data.edge_index = None
            out_seq = self.base_model(data, return_repr=True, aggr=self.aggr)
            out_seq = out_seq[data.idx_mask]
            out_seq = gnn.global_mean_pool(out_seq, batch)
            out = (out + out_seq) * 0.5
        if self.aggr == "concat":
            out = self.norm(out)
            new_out = 0
            for i in range(len(self.lin_proj)):
                new_out = new_out + self.lin_proj[i](
                    out[:, i * self.embed_dim : (i + 1) * self.embed_dim]
                )
            out = new_out / len(self.lin_proj)
        out = self.out_head(out)
        return out
