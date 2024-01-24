import logging
import os
from pathlib import Path

import esm
import hydra
import torch
import torch_geometric.nn as gnn
from biopandas.pdb import PandasPdb
from omegaconf import OmegaConf
from pyprojroot import here
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch
from tqdm import tqdm

from pst.esm2 import PST
from pst.utils import AA_THREE_TO_ONE, download_url_content

logger = logging.getLogger(__name__)


class ExampleDataset(Dataset):
    def __init__(
        self,
        root,
        eps,
        esm_alphabet,
        num_workers=-1,
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
        edge_index += 1
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


@torch.no_grad()
def compute_repr(data_loader, model, cfg):
    embeddings = []
    for batch_idx, data in enumerate(tqdm(data_loader)):
        data = data.to(cfg.device)
        out = model(data, return_repr=True)
        out, batch = out[data.idx_mask], data.batch[data.idx_mask]
        embeddings = embeddings + list(unbatch(out, batch))
    return embeddings


@hydra.main(
    version_base="1.3",
    config_path=str(here() / "config"),
    config_name="pst_inference_config",
)
def main(cfg):
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")

    pretrained_path = Path(f".cache/pretrained_models/{cfg.model}/model.pt")
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.model == "esm2_t6_8M_UR50D":
        download_url_content(
            "https://datashare.biochem.mpg.de/s/ac9ufZ0NB2IrkZL/download",
            str(pretrained_path),
        )
    elif cfg.model == "esm2_t12_35M_UR50D":
        download_url_content(
            "https://datashare.biochem.mpg.de/s/fOSIwJAIKLYjFe3/download",
            str(pretrained_path),
        )
    elif cfg.model == "esm2_t30_150M_UR50D":
        download_url_content(
            "https://datashare.biochem.mpg.de/s/a3yugJJMe0I0oEL/download",
            str(pretrained_path),
        )
    elif cfg.model == "esm1b_t33_650M_UR50D":
        download_url_content(
            "https://datashare.biochem.mpg.de/s/RpWYV4o4ka3gHvX/download",
            str(pretrained_path),
        )

    model, model_cfg = PST.from_pretrained(
        pretrained_path, map_location=torch.device("cpu")
    )
    model.eval()
    model.to(cfg.device)
    esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")

    dataset = ExampleDataset(
        root="./examples", eps=8, num_workers=cfg.num_workers, esm_alphabet=esm_alphabet
    )

    data_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    X = compute_repr(data_loader, model, cfg)
    for x in X:
        print("Shape of representation (length, d_model):")
        print(x.shape)


if __name__ == "__main__":
    main()
