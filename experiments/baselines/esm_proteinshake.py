import json
import logging
from timeit import default_timer as timer

import hydra
import numpy as np
import pandas as pd
import torch
from data_utils import compute_metrics, get_task, prepare_data
from esm import FastaBatchedDataset, pretrained
from omegaconf import OmegaConf
from pyprojroot import here
from tqdm import tqdm

log = logging.getLogger(__name__)


@torch.no_grad()
def compute_repr(data_loader, model, task, cfg):
    for batch_idx, (labels, strs, toks) in enumerate(tqdm(data_loader)):
        if torch.cuda.is_available() and not cfg.nogpu:
            toks = toks.to(device="cuda", non_blocking=True)

        out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
        out = out["representations"][model.num_layers].to(device="cpu")
        if batch_idx == 0:
            embeddings = [None] * len(data_loader.dataset)

        for i, label in enumerate(labels):
            truncate_len = min(cfg.truncation_seq_length, len(strs[i]))
            if "protein" in task.task_type[0]:
                embedding = out[i, 1 : truncate_len + 1].mean(0, keepdim=True)
            else:
                embedding = out[i, 1 : truncate_len + 1]
            embeddings[label] = embedding

    return embeddings


@hydra.main(
    version_base="1.3",
    config_path=str(here() / "config"),
    config_name="esm_proteinshake",
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")

    model, alphabet = pretrained.load_model_and_alphabet(cfg.model)
    model.eval()

    task = get_task(cfg.task.class_name)(root=cfg.task.path, split=cfg.split)
    seqs = [p["protein"]["sequence"] for p in task.dataset.proteins()]
    seq_ids = list(range(len(seqs)))

    dataset = FastaBatchedDataset(seq_ids, seqs)
    batches = dataset.get_batch_indices(cfg.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(cfg.truncation_seq_length),
        batch_sampler=batches,
    )

    if torch.cuda.is_available() and not cfg.nogpu:
        model = model.cuda()
        log.info("Transferred model to GPU")

    tic = timer()
    X = compute_repr(data_loader, model, task, cfg)
    compute_time = timer() - tic
    X_tr, y_tr, X_val, y_val, X_te, y_te = prepare_data(X, task)
    print(f"X_tr shape: {X_tr.shape} y_tr shape: {y_tr.shape}")

    if cfg.solver == "flaml":
        from flaml import AutoML

        clf = AutoML()

        def custom_metric(
            X_test,
            y_test,
            estimator,
            labels,
            X_train,
            y_train,
            weight_test=None,
            weight_train=None,
            config=None,
            groups_test=None,
            groups_train=None,
        ):
            if cfg.task.type == "classification":
                y_pred = estimator.predict_proba(X_test)
            else:
                y_pred = estimator.predict(X_test)
            val_score = compute_metrics(y_test, y_pred, task)[cfg.task.metric]
            return -val_score, {"val_score": val_score}

        settings = {
            "metric": custom_metric,
            "task": cfg.task.type,
            "estimator_list": ["lgbm"],
            "seed": cfg.seed,
            "time_budget": cfg.budget,
        }
        clf.fit(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val, **settings)
        clf_time = clf.best_config_train_time
    elif cfg.solver == "sklearn":
        import copy

        from data_utils.sklearn_utils import SklearnPredictor

        clf = SklearnPredictor(task.task_out)

        grid = clf.get_grid()

        best_val_score = 0.0
        best_model = None
        tic = timer()
        for param, values in grid.items():
            for value in values:
                clf.set_params(**{param: value})
                clf.fit(X_tr, y_tr)
                val_score = compute_metrics(y_val, clf.predict(X_val), task)[
                    cfg.task.metric
                ]
                test_score = compute_metrics(y_te, clf.predict(X_te), task)[
                    cfg.task.metric
                ]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model = copy.deepcopy(clf)
                log.info(
                    f"trained model with {param}={value}, val score: {val_score} test score: {test_score}"
                )
        clf_time = timer() - tic
        clf = best_model
    elif cfg.solver == "sklearn_cv":
        import copy

        from data_utils.sklearn_utils import SklearnPredictor
        from sklearn.metrics import make_scorer
        from sklearn.model_selection import GridSearchCV, PredefinedSplit

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
        print(pd.DataFrame.from_dict(clf.cv_results_).sort_values("rank_test_score"))
        estimator.set_params(**clf.best_params_)
        clf = estimator
        clf.fit(X_tr, y_tr)
        clf_time = timer() - tic
    elif cfg.solver == "cyanure":
        import copy

        from cyanure import preprocess
        from data_utils.cyanure_utils import CyanurePredictor

        if cfg.task.name != "structure_similarity":
            preprocess(X_tr, normalize=True, columns=False)
            preprocess(X_val, normalize=True, columns=False)
            preprocess(X_te, normalize=True, columns=False)

        clf = CyanurePredictor(task.task_out)

        grid = {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        }

        best_val_score = 0.0
        best_model = None
        tic = timer()
        for param, values in grid.items():
            for value in values:
                clf.set_params(**{param: value})
                clf.fit(X_tr, y_tr)
                val_score = compute_metrics(y_val, clf.predict(X_val), task)[
                    cfg.task.metric
                ]
                test_score = compute_metrics(y_te, clf.predict(X_te), task)[
                    cfg.task.metric
                ]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model = copy.deepcopy(clf)
                log.info(
                    f"trained model with {param}={value}, val score: {val_score} test score: {test_score}"
                )
        clf_time = timer() - tic
        clf = best_model

    # y_pred = clf.predict(X_te)
    try:
        y_pred = clf.decision_function(X_te)
    except:
        y_pred = clf.predict(X_te)
    if isinstance(y_pred, list):
        if y_pred[0].ndim > 1:
            y_pred = [y[:, 1] for y in y_pred]
        y_pred = np.asarray(y_pred).T
    test_score = compute_metrics(y_te, y_pred, task)[cfg.task.metric]
    log.info(f"Test score: {test_score:.3f}")

    # y_val_pred = clf.predict(X_val)
    try:
        y_val_pred = clf.decision_function(X_val)
    except:
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
    if cfg.solver == "flaml":
        with open(f"{cfg.logs.path}/best_config.txt", "w") as f:
            f.write(json.dumps(clf.best_config))


if __name__ == "__main__":
    main()
