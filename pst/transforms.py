from functools import partial

import esm
import torch
from proteinshake.utils import residue_alphabet
from torch_geometric import utils
from torch_geometric.data import Data


esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")


class PretrainingAttr(object):
    def __call__(self, data):
        if isinstance(data, tuple):
            data, _ = data
        new_data = Data()
        new_data.x = data.x
        # new_data.residue_idx = torch.arange(1, data.num_nodes + 1)
        new_data.edge_index = data.edge_index.long()
        if hasattr(data, "edge_attr"):
            new_data.edge_attr = data.edge_attr
        return new_data


class Proteinshake2ESM(object):
    def __init__(self, mask_cls_idx=False):
        self.token_map = torch.LongTensor(
            [esm_alphabet.get_idx(aa) for aa in residue_alphabet]
        )
        self.cls_idx = torch.LongTensor([esm_alphabet.cls_idx])
        self.eos_idx = torch.LongTensor([esm_alphabet.eos_idx])
        self.mask_cls_idx = mask_cls_idx

    def __call__(self, data):
        data.x = self.token_map[data.x]
        data.x = torch.cat([self.cls_idx, data.x, self.eos_idx])
        data.edge_index = data.edge_index + 1
        if self.mask_cls_idx:
            data.idx_mask = torch.ones((len(data.x),), dtype=torch.bool)
            data.idx_mask[0] = data.idx_mask[-1] = False
        return data


class RandomCrop(object):
    def __init__(self, max_len=1024):
        self.max_len = max_len

    def __call__(self, data):
        num_nodes = data.num_nodes
        if num_nodes > self.max_len:
            start_idx = torch.randint(0, num_nodes - self.max_len, size=(1,))
            node_mask = torch.zeros(num_nodes, dtype=torch.bool)
            node_mask[start_idx : start_idx + self.max_len] = True
            edge_index, edge_attr = utils.subgraph(
                node_mask, data.edge_index, data.edge_attr, relabel_nodes=True
            )
            data.x = data.x[node_mask]
            data.edge_index = edge_index
            data.edge_attr = edge_attr
            if hasattr(data, "residue_idx"):
                data.residue_idx = data.residue_idx[node_mask]
        return data


class MaskNode(object):
    def __init__(
        self, mask_idx=esm_alphabet.mask_idx, mask_rate=0.15, probs=[0.8, 0.1, 0.1]
    ):
        self.mask_idx = mask_idx
        self.mask_rate = mask_rate
        self.probs = probs
        if probs is not None:
            self.probs = torch.tensor(self.probs)

    def __call__(self, data):
        num_nodes = data.num_nodes
        subset_mask = torch.rand(num_nodes) < self.mask_rate

        data.masked_node_indices = subset_mask
        data.masked_node_label = data.x[subset_mask]

        if self.probs is None:
            data.x[subset_mask] = self.mask_idx
        else:
            x_mask = data.x[subset_mask]
            if len(x_mask) > 0:
                probs = torch.multinomial(self.probs, len(x_mask), replacement=True)
                x_mask[probs == 0] = self.mask_idx
                x_mask[probs == 1] = x_mask[probs == 1].random_(self.mask_idx)
                data.x[subset_mask] = x_mask
        # data.x[subset_mask] = self.mask_idx

        return data


class MutationDataset(object):
    def __init__(
        self,
        graph,
        protein_dict,
        mutations,
        mask_idx=esm_alphabet.mask_idx,
        strategy="masked",
        use_transform=True,
    ):
        transform = Proteinshake2ESM()
        if use_transform:
            self.graph = transform(graph)
        else:
            self.graph = graph
        self.protein_dict = protein_dict
        self.mutations = mutations["mutations"].values
        self.y = mutations["y"].values
        self.mask_idx = mask_idx
        self.strategy = strategy

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, index):
        data = self.graph.clone()
        data.y = self.y[index]
        mutations = self.mutations[index].split()
        mask = torch.tensor(list(map(lambda x: int(x[1:-1]), mutations)))
        wt_indices = torch.tensor(
            list(map(lambda x: esm_alphabet.get_idx(x[0]), mutations))
        )
        mt_indices = torch.tensor(
            list(map(lambda x: esm_alphabet.get_idx(x[-1]), mutations))
        )
        if self.strategy != "mt-all":
            data.masked_node_indices = mask
        data.masked_node_labels = data.x[mask].view(-1, 1)
        if self.strategy == "mt" or self.strategy == "mt-all":
            data.x[mask] = mt_indices
        else:
            data.x[mask] = self.mask_idx
        data.wt_indices = wt_indices.view(-1, 1)
        data.mt_indices = mt_indices.view(-1, 1)
        return data
