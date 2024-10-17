# -*- coding: utf-8 -*-
import json
import logging
import pickle
from pathlib import Path
from timeit import default_timer as timer

import biotite.structure as struc
import esm
import fastpdb
import hydra
import numpy as np
import pandas as pd
import torch
import torch_geometric.nn as gnn
import torchdrug
from easydict import EasyDict as edict
from omegaconf import OmegaConf
from pyprojroot import here
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import radius_neighbors_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from torchdrug import core, datasets, models, tasks  # noqa
from tqdm import tqdm

from pst.downstream import (
    convert_to_numpy,
    mask_cls_idx,
    preprocess,
)
from pst.downstream.mlp import train_and_eval_mlp
from pst.esm2 import PST

log = logging.getLogger(__name__)

esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")

AA_THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",
}


@torch.no_grad()
def compute_repr(data_loader, model, cfg):
    embeddings = []
    for batch_idx, data in enumerate(tqdm(data_loader, desc="Computing embeddings")):
        data = data.to(cfg.device)
        out = model(data, return_repr=True, aggr=cfg.aggr)
        out, batch = out[data.idx_mask], data.batch[data.idx_mask]
        out = gnn.global_mean_pool(out, batch)

        if cfg.include_seq:
            data.edge_index = None
            out_seq = model(data, return_repr=True, aggr=cfg.aggr)
            out_seq = out_seq[data.idx_mask]
            out_seq = gnn.global_mean_pool(out_seq, batch)
            out = (out + out_seq) * 0.5

        out = out.cpu()

        embeddings = embeddings + list(torch.chunk(out, len(data.ptr) - 1))

    return torch.cat(embeddings)

def create_graph_from_pdb(idx, protein):
    # Load the protein structure
    pdb_file = fastpdb.PDBFile.read(here() / "datasets" / "esmfold" / "structures" / f"{idx}.pdb")
    structure = pdb_file.get_structure(model=1) 
    
    coords = structure.coord
    element = structure.element
    resname = structure.res_name
    resid = structure.res_id
    chain_id = structure.chain_id
    atom_name = structure.atom_name

    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "element": element,
            "resname": resname,
            "atom_name": atom_name,
            "resid": resid,
            "chain_id": chain_id,
        }
    )

    # Extract CA atom coordinates
    coordinates = df.loc[df["atom_name"] == "CA", ["x", "y", "z"]].values

    sequence = "".join(
        df.loc[df.atom_name == "CA"].resname.map(AA_THREE_TO_ONE).tolist()
    )
    x = torch.LongTensor(
        [esm_alphabet.get_idx(res) for res in esm_alphabet.tokenize(sequence)]
    )

    # Create edge index and edge attributes
    edge_index, edge_attr = from_scipy_sparse_matrix(
        radius_neighbors_graph(coordinates, 8.0)
    )
    return Data(edge_index=edge_index, x=x, edge_attr=edge_attr)

def get_structures(dataset, task, eps=8):
    data_loader = torchdrug.data.DataLoader(dataset, batch_size=1, shuffle=False)
    structures = []
    labels = []
    idx_range = dataset.indices
    for idx, protein in tqdm(zip(idx_range, data_loader), total=len(list(idx_range)), desc="Get structures"):
        graph = create_graph_from_pdb(idx, protein)
        # x, edge_index, edge_attr = create_graph_from_contact_map()
        labels.append(protein["targets"])
        structures.append(
            graph
        )

    return structures, torch.cat(labels)


@hydra.main(
    version_base="1.3", config_path=str(here() / "config"), config_name="pst_gearnet_esmfold"
)
def main(cfg):
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.include_seq and "so" not in cfg.model:
        cfg.model = f"{cfg.model}_so"

    pretrained_path = Path(cfg.pretrained) / f"{cfg.model}.pt"
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)

    model, model_cfg = PST.from_pretrained_url(
        cfg.model, pretrained_path,
    )

    model.eval()
    model.to(cfg.device)

    task = core.Configurable.load_config_dict(
        edict(OmegaConf.to_container(cfg.task, resolve=True))
    )

    structure_path = (
        Path(cfg.data.esmfold_structures_path) / f"structures_{model_cfg.data.graph_eps}.pt"
    )
    if structure_path.exists():
        tmp = torch.load(structure_path)
        train_str, y_tr = tmp["train_str"], tmp["y_tr"]
        val_str, y_val = tmp["val_str"], tmp["y_val"]
        test_str, y_te = tmp["test_str"], tmp["y_te"]
        del tmp
    else:
        # To make torchdrug work, one has to delete unrecognized attributes...
        cfg.dataset.__delattr__('name')
        dataset = core.Configurable.load_config_dict(
            OmegaConf.to_container(cfg.dataset, resolve=True)
        )
        train_dset, val_dset, test_dset = dataset.split()

        train_str, y_tr = get_structures(train_dset, task, eps=model_cfg.data.graph_eps)
        val_str, y_val = get_structures(val_dset, task, eps=model_cfg.data.graph_eps)
        test_str, y_te = get_structures(test_dset, task, eps=model_cfg.data.graph_eps)
        torch.save(
            {
                "train_str": train_str,
                "val_str": val_str,
                "test_str": test_str,
                "y_tr": y_tr,
                "y_val": y_val,
                "y_te": y_te,
            },
            structure_path,
        )

    # this is awful i know, todo: proper transform and dataset
    train_str = [mask_cls_idx(data) for data in train_str]
    val_str = [mask_cls_idx(data) for data in val_str]
    test_str = [mask_cls_idx(data) for data in test_str]

    train_loader = DataLoader(
        train_str,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_str,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_str,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # compute embeddings
    tic = timer()
    X_tr = compute_repr(train_loader, model, cfg)
    X_val = compute_repr(val_loader, model, cfg)
    X_te = compute_repr(test_loader, model, cfg)
    compute_time = timer() - tic
    preprocess(X_tr)
    preprocess(X_val)
    preprocess(X_te)

    X_tr, X_val, X_te, y_tr, y_val, y_te = convert_to_numpy(
        X_tr, X_val, X_te, y_tr, y_val, y_te
    )
    X_mask = np.isnan(X_tr.sum(1))
    X_tr, y_tr = X_tr[~X_mask], y_tr[~X_mask]
    log.info(f"X_tr shape: {X_tr.shape} y_tr shape: {y_tr.shape}")

    if cfg.use_pca is not None:
        from sklearn.decomposition import PCA

        cfg.use_pca = 1024 if X_tr.shape[1] < 10000 else 2048
        pca = PCA(cfg.use_pca)
        pca = pca.fit(X_tr)
        X_tr = pca.transform(X_tr)
        X_val = pca.transform(X_val)
        X_te = pca.transform(X_te)
        log.info(f"PCA done. X_tr shape: {X_tr.shape}")

    X_tr, y_tr = torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()
    X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    X_te, y_te = torch.from_numpy(X_te).float(), torch.from_numpy(y_te).float()

    train_and_eval_mlp(
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_te,
        y_te,
        cfg,
        task,
        batch_size=32,
        epochs=100,
    )


if __name__ == "__main__":
    main()
