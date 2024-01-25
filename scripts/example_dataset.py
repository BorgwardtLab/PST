import os
import torch
import esm
import torch_geometric.nn as gnn
from biopandas.pdb import PandasPdb
from torch_geometric.data import Data, Dataset
from pst.utils import AA_THREE_TO_ONE


class ExampleDataset(Dataset):
    def __init__(
        self,
        root,
        eps=8.0,
        esm_alphabet=esm.data.Alphabet.from_architecture("ESM-1b"),
        num_workers=0,
        transform=None,
        pre_transform=None,
    ):
        self.eps = eps
        self.esm_alphabet = esm_alphabet
        self.num_workers = num_workers
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        raw_files = os.listdir(self.root + "/raw/")
        if "processed" in raw_files:
            raw_files.remove("processed")
        return raw_files

    @property
    def processed_file_names(self):
        return ["data_{i}.pt" for i in range(len(self.raw_file_names))]

    def get_graph_from_pdb(self, fname):
        pdb_contents = PandasPdb().read_pdb(fname).df["ATOM"]
        ca = pdb_contents[pdb_contents["atom_name"] == "CA"]
        structure = ca[["x_coord", "y_coord", "z_coord"]]
        structure = structure.to_numpy()
        structure = torch.tensor(structure, dtype=torch.float)
        edge_index = gnn.radius_graph(
            structure, r=self.eps, loop=False, num_workers=self.num_workers
        )
        edge_index += 1 # shift for cls_idx

        x = torch.cat(
            [
                torch.LongTensor([self.esm_alphabet.cls_idx]),
                torch.LongTensor(
                    [
                        self.esm_alphabet.get_idx(res)
                        for res in self.esm_alphabet.tokenize(
                            "".join(
                                ca["residue_name"]
                                .apply(lambda x: AA_THREE_TO_ONE[x])
                                .tolist()
                            )
                        )
                    ]
                ),
                torch.LongTensor([self.esm_alphabet.eos_idx]),
            ]
        )
        idx_mask = torch.zeros_like(x, dtype=torch.bool)
        idx_mask[1:-1] = True

        return Data(x=x, edge_index=edge_index, idx_mask=idx_mask)

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = self.get_graph_from_pdb(raw_path)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        return data
