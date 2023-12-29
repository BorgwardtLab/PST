# -*- coding: utf-8 -*-
"""ca_only_feat.py

Since only CAs are available, we need to extract the following features for the proteins:
- RBF distance matrix for CAs
- Angles between CAs
"""

from pathlib import Path

import esm
import hydra
from pyprojroot import here
from torch_geometric.loader import DataLoader
from tqdm import tqdm

esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")


@hydra.main(
    version_base="1.3", config_path=str(here() / "config"), config_name="config"
)
def main(cfg):
    train_dataset = EnzymeCommissionGearNet(
        root=Path(cfg.paths.td_ec),
        n_jobs=cfg.compute.n_jobs,
        split="train",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.compute.num_workers,
    )
    for batch in tqdm(train_dataloader):
        print(batch)

    valid_dataset = EnzymeCommissionGearNet(
        root=Path(cfg.paths.td_ec),
        n_jobs=cfg.compute.n_jobs,
        split="valid",
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.compute.num_workers,
    )
    for batch in tqdm(valid_dataloader):
        batch

    test_dataset = EnzymeCommissionGearNet(
        root=Path(cfg.paths.td_ec),
        n_jobs=cfg.compute.n_jobs,
        split="test",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.compute.num_workers,
    )
    for batch in tqdm(test_dataloader):
        batch


if __name__ == "__main__":
    main()
