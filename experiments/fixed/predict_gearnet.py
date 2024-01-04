import json
import logging
import pickle
from pathlib import Path
from timeit import default_timer as timer

import esm
import hydra
import numpy as np
import pandas as pd
import torch
import torch_geometric.nn as gnn
import torchdrug
from easydict import EasyDict as edict
from omegaconf import OmegaConf
from pyprojroot import here
from sklearn.metrics import make_scorer
from sklearn.neighbors import radius_neighbors_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from torchdrug import core
from tqdm import tqdm

from pst.esm2 import PST

log = logging.getLogger(__name__)

esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")


def rbf(D, num_rbf):
    D_min, D_max, D_count = 0.0, 20.0, num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1, 1, 1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def get_rbf(X):
    D = torch.sqrt(torch.sum((X[:, None, :] - X[None, :, :]) ** 2, -1) + 1e-6)
    return rbf(D, 16)


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


def get_structures(dataset, task, use_rbfs=False, eps=8):
    data_loader = torchdrug.data.DataLoader(dataset, batch_size=1, shuffle=False)
    structures = []
    labels = []
    for protein in tqdm(data_loader):
        out = task.graph_construction_model(protein["graph"])
        sequence = out.to_sequence()[0]
        if len(sequence) == 0:
            continue
        coords = out.node_position
        labels.append(protein["targets"])

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

        if use_rbfs:
            rbf_dist = get_rbf(coords).squeeze()
            edge_attr = rbf_dist[edge_index[0, :], edge_index[1, :]]
        else:
            edge_attr = None

        structures.append(
            Data(edge_index=edge_index, x=torch_sequence, edge_attr=edge_attr)
        )

    return structures, torch.cat(labels)


def convert_to_numpy(*args):
    out = []
    for t in args:
        out.append(t.numpy().astype("float64"))
    return out


def mask_cls_idx(data):
    data.idx_mask = torch.ones((len(data.x),), dtype=torch.bool)
    data.idx_mask[0] = data.idx_mask[-1] = False
    return data


@hydra.main(
    version_base="1.3", config_path=str(here() / "config"), config_name="sat_gearnet"
)
def main(cfg):
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.use_edge_attr:
        pretrained_path = (
            Path(cfg.pretrained.prefix) / "with_edge_attr" / cfg.pretrained.name
        )
    else:
        if cfg.include_seq:
            pretrained_path = (
                Path(cfg.pretrained.prefix) / "train_struct_only" / cfg.pretrained.name
            )
        else:
            pretrained_path = Path(cfg.pretrained.prefix) / cfg.pretrained.name

    model, model_cfg = PST.from_pretrained(pretrained_path)
    model.eval()
    model.to(cfg.device)

    task = core.Configurable.load_config_dict(
        edict(OmegaConf.to_container(cfg.task, resolve=True))
    )

    structure_path = (
        Path(cfg.dataset.path) / f"structures_{model_cfg.data.graph_eps}.pt"
    )
    if structure_path.exists():
        tmp = torch.load(structure_path)
        train_str, y_tr = tmp["train_str"], tmp["y_tr"]
        val_str, y_val = tmp["val_str"], tmp["y_val"]
        test_str, y_te = tmp["test_str"], tmp["y_te"]
        del tmp
    else:
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

    def scorer(y_true, y_pred, return_all_metric=False):
        out = task.evaluate(torch.from_numpy(y_pred), torch.from_numpy(y_true))
        out = {key: value.item() for key, value in out.items()}
        if return_all_metric:
            return out
        return out[cfg.metric]

    scoring = make_scorer(scorer, needs_threshold=True)

    if cfg.solver == "sklearn_cv":
        from data_utils.sklearn_utils import SklearnPredictor
        from sklearn.model_selection import GridSearchCV, PredefinedSplit

        estimator = SklearnPredictor("multi_label")

        grid = estimator.get_grid()
        grid = {key: np.logspace(-3, 1, 9) for key, value in grid.items()}

        test_split_index = [-1] * len(y_tr) + [0] * len(y_val)
        X_tr_val, y_tr_val = np.concatenate((X_tr, X_val), axis=0), np.concatenate(
            (y_tr, y_val)
        )

        splits = PredefinedSplit(test_fold=test_split_index)

        clf = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring=scoring,
            cv=splits,
            refit=False,
            n_jobs=-1,
        )

        tic = timer()
        clf.fit(X_tr_val, y_tr_val)
        log.info(pd.DataFrame.from_dict(clf.cv_results_).sort_values("rank_test_score"))
        estimator.set_params(**clf.best_params_)
        clf = estimator
        clf.fit(X_tr, y_tr)
        clf_time = timer() - tic
    elif cfg.solver == "mlp":
        X_tr, y_tr = torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()
        X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
        X_te, y_te = torch.from_numpy(X_te).float(), torch.from_numpy(y_te).float()
        from data_utils.mlp_utils import train_and_eval_mlp

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
        return
    elif cfg.solver == "chainensemble":
        from data_utils.sklearn_utils import LinearSVCChainEnsemble

        clf = LinearSVCChainEnsemble(100, scoring)

        tic = timer()

        # with open('chainensemble-100.pkl','rb') as f:
        #    clf.estimators = pickle.load(f)

        clf.fit(X_tr, y_tr, X_val, y_val)
        with open("chainensemble-100.pkl", "wb") as f:
            pickle.dump(clf.estimators, f)

        clf_time = timer() - tic
    else:
        raise Exception("Not implemented")

    try:
        y_pred = clf.decision_function(X_te)
    except:
        y_pred = clf.predict_proba(X_te)
    if isinstance(y_pred, list):
        if y_pred[0].ndim > 1:
            y_pred = [y[:, 1] for y in y_pred]
        y_pred = np.asarray(y_pred).T
    test_score = scorer(y_te, y_pred, return_all_metric=True)
    log.info("Test score: ")
    log.info(test_score)

    # y_val_pred = clf.predict(X_val)
    try:
        y_val_pred = clf.decision_function(X_val)
    except:
        y_val_pred = clf.predict_proba(X_val)
    if isinstance(y_val_pred, list):
        if y_val_pred[0].ndim > 1:
            y_val_pred = [y[:, 1] for y in y_val_pred]
        y_val_pred = np.asarray(y_val_pred).T
    val_score = scorer(y_val, y_val_pred, return_all_metric=True)

    results = [
        {
            "test_score": test_score[cfg.metric],
            "val_score": val_score[cfg.metric],
            "compute_time": compute_time,
            "clf_time": clf_time,
        }
    ]

    pd.DataFrame(results).to_csv(f"{cfg.logs.path}/results.csv")
    if cfg.solver == "flaml":
        with open(f"{cfg.logs.path}/best_config.txt", "w") as f:
            f.write(json.dumps(clf.best_config))


if __name__ == "__main__":
    main()
