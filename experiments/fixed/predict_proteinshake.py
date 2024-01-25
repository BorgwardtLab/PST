import json
import logging
from pathlib import Path
from timeit import default_timer as timer

import hydra
import numpy as np
import pandas as pd
import torch
import torch_geometric.nn as gnn
from omegaconf import OmegaConf
from proteinshake.transforms import Compose
from pyprojroot import here
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from pst.downstream import compute_metrics, get_task, prepare_data, mask_cls_idx
from pst.esm2 import PST
from pst.transforms import PretrainingAttr, Proteinshake2ESM
from pst.downstream.sklearn_wrapper import SklearnPredictor

log = logging.getLogger(__name__)


@torch.no_grad()
def compute_repr(data_loader, model, task, cfg):
    embeddings = []
    for batch_idx, data in enumerate(tqdm(data_loader)):
        data = data.to(cfg.device)
        out = model(data, return_repr=True)
        out, batch = out[data.idx_mask], data.batch[data.idx_mask]
        if "protein" in task.task_type[0]:
            out = gnn.global_mean_pool(out, batch)
        if cfg.include_seq:
            data.edge_index = None
            out_seq = model(data, return_repr=True)
            out_seq = out_seq[data.idx_mask]
            if "protein" in task.task_type[0]:
                out_seq = gnn.global_mean_pool(out_seq, batch)
            out = (out + out_seq) * 0.5
        out = out.cpu()
        if "protein" in task.task_type[0]:
            embeddings = embeddings + list(torch.chunk(out, len(data.ptr) - 1))
        else:
            embeddings = embeddings + list(
                torch.split(out, tuple(torch.diff(data.ptr) - 2))
            )
    return embeddings


@hydra.main(
    version_base="1.3",
    config_path=str(here() / "config"),
    config_name="pst_proteinshake",
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

    task = get_task(cfg.task.class_name)(root=cfg.task.path, split=cfg.split)
    dataset = task.dataset.to_graph(eps=model_cfg.data.graph_eps).pyg(
        transform=Compose(
            [
                PretrainingAttr(),
                Proteinshake2ESM(mask_cls_idx=True),
            ]
        )
    )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    tic = timer()
    X = compute_repr(data_loader, model, task, cfg)
    compute_time = timer() - tic
    X_tr, y_tr, X_val, y_val, X_te, y_te = prepare_data(X, task)
    log.info(f"X_tr shape: {X_tr.shape} y_tr shape: {y_tr.shape}")

    ## Solving the problem with sklearn
    estimator = SklearnPredictor(task.task_out)

    grid = estimator.get_grid()

    scoring = lambda y_true, y_pred: compute_metrics(y_true, y_pred, task)[
        cfg.task.metric
    ]
    if task.task_out == "multi_label" or task.task_out == "binary":
        scoring = make_scorer(scoring, needs_threshold=True)
    else:
        scoring = make_scorer(scoring)

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
    #####

    if task.task_out == "multi_label" or task.task_out == "binary":
        try:
            y_pred = clf.decision_function(X_te)
        except:
            y_pred = clf.predict(X_te)
    else:
        y_pred = clf.predict(X_te)
    if isinstance(y_pred, list):
        if y_pred[0].ndim > 1:
            y_pred = [y[:, 1] for y in y_pred]
        y_pred = np.asarray(y_pred).T
    test_score = compute_metrics(y_te, y_pred, task)[cfg.task.metric]
    log.info(f"Test score: {test_score:.3f}")

    if task.task_out == "multi_label" or task.task_out == "binary":
        try:
            y_val_pred = clf.decision_function(X_val)
        except:
            y_val_pred = clf.predict(X_val)
    else:
        y_val_pred = clf.predict(X_val)
    if isinstance(y_val_pred, list):
        if y_val_pred[0].ndim > 1:
            y_val_pred = [y[:, 1] for y in y_val_pred]
        y_val_pred = np.asarray(y_val_pred).T
    val_score = compute_metrics(y_val, y_val_pred, task)[cfg.task.metric]

    results = [
        {
            "test_score": test_score,
            "val_score": val_score,
            "compute_time": compute_time,
            "clf_time": clf_time,
        }
    ]

    pd.DataFrame(results).to_csv(f"{cfg.logs.path}/results.csv")


if __name__ == "__main__":
    main()
