import importlib

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import compute_metrics


def get_task(task_name):
    all_task_classes = importlib.import_module("proteinshake.tasks")
    return getattr(all_task_classes, task_name)


def prepare_data(X, task, use_pca=False):
    train_idx, val_idx, test_idx = task.train_index, task.val_index, task.test_index
    if not "pair" in task.task_type[0]:
        y_tr = [task.target(task.proteins[idx]) for idx in train_idx]
        y_val = [task.target(task.proteins[idx]) for idx in val_idx]
        y_te = [task.target(task.proteins[idx]) for idx in test_idx]
        if "residue" in task.task_type[0]:
            y_tr, y_val, y_te = map(np.concatenate, (y_tr, y_val, y_te))
        else:
            y_tr, y_val, y_te = map(np.asarray, (y_tr, y_val, y_te))
        if task.task_type[1] == "multi_label":
            col = y_tr.sum(0) > 0
            y_tr = y_tr[:, col]
            y_val = y_val[:, col]
            y_te = y_te[:, col]
        if "molecule" in task.task_in:
            X = [
                torch.cat(
                    [
                        X[i],
                        torch.tensor(p["protein"]["fp_maccs"]).view(1, -1),
                        torch.tensor(p["protein"]["fp_morgan_r2"]).view(1, -1),
                    ],
                    dim=-1,
                )
                for i, p in enumerate(task.dataset.proteins())
            ]
        X_tr = torch.cat([X[i] for i in train_idx]).numpy()
        X_val = torch.cat([X[i] for i in val_idx]).numpy()
        X_te = torch.cat([X[i] for i in test_idx]).numpy()
        if "molecule" in task.task_in and use_pca:
            from sklearn.decomposition import PCA

            mol_dim = len(task.proteins[0]["protein"]["fp_maccs"]) + len(
                task.proteins[0]["protein"]["fp_morgan_r2"]
            )
            pca = PCA(64)
            X_tr = pca.fit_transform(X_tr)
            X_val = pca.transform(X_val)
            X_te = pca.transform(X_te)
    else:
        y_tr = np.asarray(
            [task.target(task.proteins[i], task.proteins[j]) for i, j in train_idx]
        )
        y_val = np.asarray(
            [task.target(task.proteins[i], task.proteins[j]) for i, j in val_idx]
        )
        y_te = np.asarray(
            [task.target(task.proteins[i], task.proteins[j]) for i, j in test_idx]
        )
        X_tr = torch.cat([(X[i] + X[j]) / 2.0 for i, j in train_idx]).numpy()
        X_val = torch.cat([(X[i] + X[j]) / 2.0 for i, j in val_idx]).numpy()
        X_te = torch.cat([(X[i] + X[j]) / 2.0 for i, j in test_idx]).numpy()
    return (
        X_tr.astype("float64"),
        y_tr,
        X_val.astype("float64"),
        y_val,
        X_te.astype("float64"),
        y_te,
    )
