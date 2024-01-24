import logging
from pathlib import Path
from timeit import default_timer as timer

import esm
import hydra
import pandas as pd
import torch
import torch_geometric.nn as gnn
from omegaconf import OmegaConf
from pyprojroot import here
from sklearn.neighbors import radius_neighbors_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from tqdm import tqdm

from pst.esm2 import PST
from pst.downstream.mlp import train_and_eval_linear
from pst.downstream import (
    preprocess,
    convert_to_numpy,
    mask_cls_idx,
)

log = logging.getLogger(__name__)

esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")


@torch.no_grad()
def compute_repr(data_loader, model, cfg):
    embeddings = []
    for batch_idx, data in enumerate(tqdm(data_loader)):
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


def get_structures(dataset, eps=8.0):
    structures = []
    labels = []
    for protein in tqdm(dataset):
        sequence = protein.seq
        if len(sequence) == 0:
            continue
        coords = protein.pos
        labels.append(torch.tensor(protein.y))

        torch_sequence = torch.LongTensor(
            [esm_alphabet.get_idx(res) for res in esm_alphabet.tokenize(sequence)]
        )
        graph_adj = radius_neighbors_graph(coords, radius=eps, mode="connectivity")
        edge_index = from_scipy_sparse_matrix(graph_adj)[0].long()
        torch_sequence = torch.cat(
            [
                torch.LongTensor([esm_alphabet.cls_idx]),
                torch_sequence,
                torch.LongTensor([esm_alphabet.eos_idx]),
            ]
        )
        edge_index = edge_index + 1  # shift for cls_idx

        edge_attr = None

        structures.append(
            Data(edge_index=edge_index, x=torch_sequence, edge_attr=edge_attr)
        )

    return structures, torch.stack(labels)


@hydra.main(
    version_base="1.3", config_path=str(here() / "config"), config_name="pst_gearnet"
)
def main(cfg):
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.include_seq:
        pretrained_path = Path(cfg.pretrained) / "pst_so.pt"
    else:
        pretrained_path = Path(cfg.pretrained) / "pst.pt"

    pretrained_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        model, model_cfg = PST.from_pretrained_url(
            cfg.model, pretrained_path, cfg.include_seq
        )
    except:
        model, model_cfg = PST.from_pretrained_url(
            cfg.model,
            pretrained_path,
            cfg.include_seq,
            map_location=torch.device("cpu"),
        )

    model.eval()
    model.to(cfg.device)

    scop_data = torch.load(Path(cfg.dataset.path) / "data.pt")

    structure_path = (
        Path(cfg.dataset.path) / f"structures_{model_cfg.data.graph_eps}.pt"
    )
    if structure_path.exists():
        tmp = torch.load(structure_path)
        train_str, y_tr = tmp["train_str"], tmp["y_tr"]
        val_str, y_val = tmp["val_str"], tmp["y_val"]
        test_str, y_te = tmp["test_str"], tmp["y_te"]
        stratified_indices = tmp["stratified_indices"]
        del tmp
    else:
        train_str, y_tr = get_structures(
            scop_data["train"], eps=model_cfg.data.graph_eps
        )
        val_str, y_val = get_structures(scop_data["val"], eps=model_cfg.data.graph_eps)
        test_data = (
            scop_data["test_family"]
            + scop_data["test_superfamily"]
            + scop_data["test_fold"]
        )
        n_fm = len(scop_data["test_family"])
        n_sf = len(scop_data["test_superfamily"])
        n_fo = len(scop_data["test_fold"])
        test_str, y_te = get_structures(test_data, eps=model_cfg.data.graph_eps)
        stratified_indices = {}
        stratified_indices["family"] = torch.arange(0, n_fm)
        stratified_indices["superfamily"] = torch.arange(n_fm, n_fm + n_sf)
        stratified_indices["fold"] = torch.arange(n_fm + n_sf, n_fm + n_sf + n_fo)
        torch.save(
            {
                "train_str": train_str,
                "val_str": val_str,
                "test_str": test_str,
                "y_tr": y_tr,
                "y_val": y_val,
                "y_te": y_te,
                "stratified_indices": stratified_indices,
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

    X_tr, y_tr = torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()
    X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
    X_te, y_te = torch.from_numpy(X_te).float(), torch.from_numpy(y_te).long()

    val_score, test_score, test_stratified_score = train_and_eval_linear(
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_te,
        y_te,
        1195,
        stratified_indices,
        use_cuda=torch.cuda.is_available(),
    )

    results = [
        {
            "test_top1": test_score[0],
            "test_family": test_stratified_score["family"][0],
            "test_superfamily": test_stratified_score["superfamily"][0],
            "test_fold": test_stratified_score["fold"][0],
            "val_acc": val_score,
            "compute_time": compute_time,
        }
    ]

    pd.DataFrame(results).to_csv(f"{cfg.logs.path}/results.csv")


if __name__ == "__main__":
    main()
