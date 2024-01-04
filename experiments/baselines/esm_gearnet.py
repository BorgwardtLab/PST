import json
import logging
from pathlib import Path
from timeit import default_timer as timer

import hydra
import numpy as np
import pandas as pd
import torch
import torchdrug

# from data_utils import get_task, compute_metrics, prepare_data
from easydict import EasyDict as edict
from esm import FastaBatchedDataset, pretrained
from omegaconf import OmegaConf
from pyprojroot import here

# from torchdrug.data import DataLoader
from sklearn.metrics import make_scorer
from torchdrug import core, datasets, models, tasks  # noqa
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

    def scorer(y_true, y_pred, return_all_metric=False):
        out = task.evaluate(torch.from_numpy(y_pred), torch.from_numpy(y_true))
        out = {key: value.item() for key, value in out.items()}
        if return_all_metric:
            return out
        return out[cfg.metric]

    scoring = make_scorer(scorer, needs_threshold=True)

    if cfg.solver == "flaml":
        from data_utils.sklearn_utils import MyMultiOutputClassifier
        from flaml import AutoML

        clf = AutoML()
        # def custom_metric(
        #     X_test, y_test, estimator, labels,
        #     X_train, y_train, weight_test=None, weight_train=None,
        #     config=None, groups_test=None, groups_train=None,
        # ):
        #     if cfg.task.type == "classification":
        #         y_pred = estimator.predict_proba(X_test)
        #     else:
        #         y_pred = estimator.predict(X_test)
        #     val_score = task.evaluate(y_pred, y_test)[cfg.metric]
        #     return -val_score, {'val_score': val_score}
        clf = MyMultiOutputClassifier(clf, n_jobs=-1)
        settings = {
            "metric": "f1",
            "task": "classification",
            "estimator_list": ["lgbm"],
            "seed": cfg.seed,
            "time_budget": cfg.budget,
            "auto_augment": False,
            "n_jobs": 1,
            "verbose": 1,
        }
        clf.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, **settings)
        clf_time = clf.best_config_train_time
    elif cfg.solver == "sklearn_cv":
        from data_utils.sklearn_utils import SklearnPredictor
        from sklearn.model_selection import GridSearchCV, PredefinedSplit

        estimator = SklearnPredictor("multi_label")

        grid = estimator.get_grid()
        grid = {key: np.logspace(-3, 4, 15) for key, value in grid.items()}

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
    elif cfg.solver == "rf":
        from flaml import tune
        from flaml.automl.model import RandomForestEstimator
        from flaml.automl.task.factory import task_factory
        from sklearn.ensemble import RandomForestClassifier

        flaml_search_space = RandomForestEstimator.search_space(
            X_tr.shape,
            task_factory("classification"),
        )

        config_search_space = {
            hp: space["domain"] for hp, space in flaml_search_space.items()
        }
        low_cost_partial_config = {
            hp: space["low_cost_init_value"]
            for hp, space in flaml_search_space.items()
            if "low_cost_init_value" in space
        }
        points_to_evaluate = [
            {
                hp: space["init_value"]
                for hp, space in flaml_search_space.items()
                if "init_value" in space
            }
        ]

        # estimator = RandomForestClassifier(n_jobs=-1)

        def train_rf(config):
            params = RandomForestEstimator(
                task_factory("classification"), **config
            ).params
            # estimator = estimator.set_params(**params)
            estimator = RandomForestClassifier(n_jobs=-1, **params)
            estimator.fit(X_tr, y_tr)

            y_pred = estimator.predict_proba(X_val)
            if y_pred[0].ndim > 1:
                y_pred = [y[:, 1] for y in y_pred]
            y_pred = np.asarray(y_pred).T
            val_score = scorer(y_val, y_pred, return_all_metric=True)
            return val_score

        tic = timer()
        analysis = tune.run(
            train_rf,
            metric=cfg.metric,
            mode="max",
            config=config_search_space,
            low_cost_partial_config=low_cost_partial_config,
            points_to_evaluate=points_to_evaluate,
            time_budget_s=cfg.budget,
            num_samples=-1,
        )

        print(analysis.best_result)
        best_params = RandomForestEstimator(
            task_factory("classification"), **analysis.best_config
        ).params
        print(best_params)
        clf = RandomForestClassifier(n_jobs=-1, **best_params)
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
    else:
        raise NotImplementedError
    # y_pred = clf.predict(X_te)
    try:
        y_pred = clf.decision_function(X_te)
    except:
        print("here")
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
