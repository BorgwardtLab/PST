import copy
import json
import logging
from pathlib import Path
from timeit import default_timer as timer

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from esm import FastaBatchedDataset, pretrained
from omegaconf import OmegaConf
from pyprojroot import here
from sklearn.metrics import make_scorer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

log = logging.getLogger(__name__)


@torch.no_grad()
def compute_repr(data_loader, model, cfg):
    for batch_idx, (labels, strs, toks) in enumerate(tqdm(data_loader)):
        if torch.cuda.is_available() and not cfg.nogpu:
            toks = toks.to(device="cuda", non_blocking=True)

        # out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
        if hasattr(cfg, "aggr") and cfg.aggr == "concat":
            out = model(
                toks,
                repr_layers=[i for i in range(model.num_layers)],
                return_contacts=False,
            )
            out = torch.cat(
                [out["representations"][l] for l in out["representations"]], dim=-1
            ).to(device="cpu")
        else:
            out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
            out = out["representations"][model.num_layers].to(device="cpu")
        if batch_idx == 0:
            embeddings = [None] * len(data_loader.dataset)

        for i, label in enumerate(labels):
            truncate_len = min(cfg.truncation_seq_length, len(strs[i]))
            embedding = out[i, 1 : truncate_len + 1].mean(0, keepdim=True)
            embeddings[label] = embedding

    return torch.cat(embeddings)


def get_sequences(dataset):
    sequences = []
    labels = []
    for protein in tqdm(dataset):
        sequences.append(protein.seq)
        labels.append(torch.tensor(protein.y))
    return sequences, torch.stack(labels)


def get_loader(seqs, alphabet, cfg):
    seq_ids = list(range(len(seqs)))
    dataset = FastaBatchedDataset(seq_ids, seqs)
    batches = dataset.get_batch_indices(cfg.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(cfg.truncation_seq_length),
        batch_sampler=batches,
    )
    return data_loader


def preprocess(X):
    X -= X.mean(dim=-1, keepdim=True)
    X /= X.norm(dim=-1, keepdim=True)
    return X


def convert_to_numpy(*args):
    out = []
    for t in args:
        out.append(t.numpy().astype("float64"))
    return out


@hydra.main(
    version_base="1.3", config_path=str(here() / "config"), config_name="esm_gearnet"
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")

    model, alphabet = pretrained.load_model_and_alphabet(cfg.model)
    model.eval()

    scop_data = torch.load(Path(cfg.dataset.path) / "data.pt")

    train_seq, y_tr = get_sequences(scop_data["train"])
    val_seq, y_val = get_sequences(scop_data["val"])
    test_seq_family, y_te_family = get_sequences(scop_data["test_family"])
    test_seq_superfamily, y_te_superfamily = get_sequences(
        scop_data["test_superfamily"]
    )
    test_seq_fold, y_te_fold = get_sequences(scop_data["test_fold"])
    test_seq = test_seq_family + test_seq_superfamily + test_seq_fold
    y_te = torch.cat([y_te_family, y_te_superfamily, y_te_fold])  # .view(-1, 1)
    stratified_indices = {}
    stratified_indices["family"] = torch.arange(0, len(y_te_family))
    stratified_indices["superfamily"] = torch.arange(
        len(y_te_family), len(y_te_family) + len(y_te_superfamily)
    )
    stratified_indices["fold"] = torch.arange(
        len(y_te_family) + len(y_te_superfamily),
        len(y_te_family) + len(y_te_superfamily) + len(y_te_fold),
    )

    train_loader = get_loader(train_seq, alphabet, cfg)
    val_loader = get_loader(val_seq, alphabet, cfg)
    test_loader = get_loader(test_seq, alphabet, cfg)

    if torch.cuda.is_available() and not cfg.nogpu:
        model = model.cuda()
        log.info("Transferred model to GPU")

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

    if cfg.solver == "linear":
        X_tr, y_tr = torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()
        X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
        X_te, y_te = torch.from_numpy(X_te).float(), torch.from_numpy(y_te).long()
        from data_utils.mlp_utils import train_and_eval_linear

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
    else:
        raise NotImplementedError

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
