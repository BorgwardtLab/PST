"""ec_dataset.py

Dataset for the EC classification task.
"""

import os

import einops
import hydra
import torch
from protein_rep_learning.feat import cif2backbone, download_id, read_ec_split
from protein_rep_learning.utils import (distribute_function,
                                        featurize_backbone, get_device,
                                        load_obj, prepare, save_obj)
from pyprojroot import here
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm.rich import tqdm


class ECDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        n_jobs=-1,
        split="train",
        check_downloaded=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.n_jobs = n_jobs
        self.split = split
        self.check_downloaded = check_downloaded
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
        split_data = read_ec_split(self.split)
        if self.check_downloaded:
            distribute_function(
                func=download_id,
                X=split_data["pdb_id"].tolist(),
                n_jobs=self.n_jobs,
                description="Downloading",
                dest_dir_cif=f"{self.root}/cif/",
                dest_dir_fasta=f"{self.root}/fasta/",
            )

        split_data = read_ec_split(self.split)
        split_data["chain_id"] = (
            split_data["pdb_id_chain"].apply(lambda x: x[5:]).str.upper()
        )
        split_data["cif_path"] = split_data["pdb_id"].apply(
            lambda x: f"{self.root}/cif/{x}.cif"
        )
        split_data["fasta_path"] = split_data["pdb_id"].apply(
            lambda x: f"{self.root}/fasta/{x}.fasta"
        )

        if not os.path.exists(f"{self.root}/{self.split}_proteins.pkl"):
            proteins = split_data.to_dict(orient="records")
            proteins = distribute_function(
                func=cif2backbone,
                X=proteins,
                n_jobs=self.n_jobs,
                description="To protein_dict",
            )
            proteins = [p for p in proteins if p is not None]
            save_obj(proteins, f"{self.root}/{self.split}_proteins.pkl")
        else:
            proteins = load_obj(f"{self.root}/{self.split}proteins.pkl")

        X, S, mask = prepare(proteins)
        data_list = [Data(x=x, s=s, mask=mask) for x, s, mask in zip(X, S, mask)]
        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


@hydra.main(
    version_base="1.3", config_path=str(here() / "config"), config_name="config"
)
def main(cfg):
    dataset = ECDataset(
        root="data/", n_jobs=cfg.compute.n_jobs, split="test", check_downloaded=True
    )
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    device = get_device()
    for batch in tqdm(loader):
        # unsqueeze
        X = einops.rearrange(batch.x, "(b w) a c -> b w a c", b=cfg.training.batch_size)
        mask = einops.rearrange(batch.mask, "(b w) -> b w", b=cfg.training.batch_size)
        S = einops.rearrange(batch.s, "(b w) -> b w", b=cfg.training.batch_size)
        X, S, mask = X.to(device), S.to(device), mask.to(device)
        X, S, _V, _E, E_idx, batch_id = featurize_backbone(X, S, mask)
        Data(x=_V, s=S, mask=mask, edge_index=E_idx, batch=batch_id, edge_attr=_E)


if __name__ == "__main__":
    main()
