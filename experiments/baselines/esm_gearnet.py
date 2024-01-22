import json
import logging
from pathlib import Path
from timeit import default_timer as timer

import hydra
import numpy as np
import pandas as pd
import torch
import torchdrug
from easydict import EasyDict as edict
from esm import FastaBatchedDataset, pretrained
from omegaconf import OmegaConf
from pyprojroot import here

from torchdrug import core, datasets, models, tasks  # noqa
from tqdm import tqdm

from pst.downstream.mlp import train_and_eval_mlp
from pst.downstream import preprocess, convert_to_numpy

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


def get_sequences(dataset, task):
    data_loader = torchdrug.data.DataLoader(dataset, batch_size=1, shuffle=False)
    sequences = []
    labels = []
    for protein in tqdm(data_loader):
        out = task.graph_construction_model(protein["graph"])
        sequences = sequences + out.to_sequence()
        labels.append(protein["targets"])
    return sequences, torch.cat(labels)


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


@hydra.main(
    version_base="1.3", config_path=str(here() / "config"), config_name="esm_gearnet"
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")

    model, alphabet = pretrained.load_model_and_alphabet(cfg.model)
    model.eval()

    task = core.Configurable.load_config_dict(
        edict(OmegaConf.to_container(cfg.task, resolve=True))
    )

    sequence_path = Path(cfg.dataset.path) / "sequences.pt"
    if sequence_path.exists():
        tmp = torch.load(sequence_path)
        train_seq, y_tr = tmp["train_seq"], tmp["y_tr"]
        val_seq, y_val = tmp["val_seq"], tmp["y_val"]
        test_seq, y_te = tmp["test_seq"], tmp["y_te"]
        del tmp
    else:
        dataset = core.Configurable.load_config_dict(
            OmegaConf.to_container(cfg.dataset, resolve=True)
        )

        train_dset, val_dset, test_dset = dataset.split()

        train_seq, y_tr = get_sequences(train_dset, task)

        val_seq, y_val = get_sequences(val_dset, task)

        test_seq, y_te = get_sequences(test_dset, task)

        torch.save(
            {
                "train_seq": train_seq,
                "val_seq": val_seq,
                "test_seq": test_seq,
                "y_tr": y_tr,
                "y_val": y_val,
                "y_te": y_te,
            },
            sequence_path,
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
    X_mask = np.isnan(X_tr.sum(1))
    X_tr, y_tr = X_tr[~X_mask], y_tr[~X_mask]
    log.info(f"X_tr shape: {X_tr.shape} y_tr shape: {y_tr.shape}")
    if cfg.use_pca is not None:
        from sklearn.decomposition import PCA

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
