from functools import partial
from itertools import repeat
from pathlib import Path

import esm
import torch
from pst.utils import (
    distribute_function,
    featurize_ca,
    flatten_lists,
    make_batches,
    read_pickle,
)
from torch_geometric.data import Data, InMemoryDataset

esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")


class CustomGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        dataset,
        n_jobs=-1,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.n_jobs = n_jobs
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        # Read data into huge `Data` list.
        data_list = [p for p in self.dataset.proteins()]
        if self.pre_transform is not None:
            if self.n_jobs > 1:
                data_list = make_batches(data_list, 20)
                data_list = distribute_function(
                    self.pre_transform,
                    data_list,
                    n_jobs=self.n_jobs,
                )
                data_list = flatten_lists(data_list)
            else:
                data_list = [self.pre_transform(d) for d in data_list]
        else:
            raise ValueError("pre transform must be implemented")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s].clone()
        return data


class GearNetTask(InMemoryDataset):
    def __init__(
        self,
        root,
        n_jobs=-1,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.n_jobs = n_jobs
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"{self.split}.pt"]

    def process(self):
        # Read data into huge `Data` list.
        split_data = read_pickle(Path(self.root) / f"{self.split}.pkl")

        if self.pre_transform is not None:
            data_list = distribute_function(
                self.pre_transform,
                split_data,
                n_jobs=self.n_jobs,
            )
            data_list = [d for d in data_list if d is not None]
        else:
            featurize_fn = partial(featurize_ca, esm_alphabet=esm_alphabet)
            data_list = distribute_function(
                featurize_fn, split_data, n_jobs=self.n_jobs
            )
            data_list = [d for d in data_list if d is not None]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
