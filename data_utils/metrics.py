import numpy as np
from scipy.stats import spearmanr
from sklearn import metrics


def compute_metrics(y_true, y_score, task):
    _, task_type = task.task_type
    y_pred = y_score
    if task_type == "multi_class" or task_type == "multi-class":
        if y_score.ndim > 1 and y_score.shape[-1] > 1:
            y_pred = y_score.argmax(-1)
        scores = task.evaluate(y_true, y_pred)
    elif task_type == "multi_label":
        # y_pred = (y_score > 0.5).astype('int')
        scores = task.evaluate(y_true, y_score)
    elif task_type == "binary":
        if isinstance(y_pred, list):
            scores = task.evaluate(y_true, y_pred)
        else:
            if y_score.ndim > 1 and y_score.shape[-1] > 1:
                y_score = y_score[:, 1]
            y_pred = (y_score > 0.0).astype("int")
            scores = task.evaluate(y_true, y_pred)
            scores["accuracy"] = metrics.accuracy_score(y_true, y_pred)
            scores["auc"] = metrics.roc_auc_score(y_true, y_score)
            scores["aupr"] = metrics.average_precision_score(y_true, y_score)
    elif task_type == "regression":
        scores = task.evaluate(y_true, y_pred)
        scores["neg_mse"] = -scores["mse"]
        scores["mae"] = metrics.mean_absolute_error(y_true, y_pred)
        scores["spearmanr"] = spearmanr(y_true, y_pred).correlation
        scores["r2"] = metrics.r2_score(y_true, y_pred)
    else:
        scores = task.evaluate(y_true, y_pred)
    return scores
