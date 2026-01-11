from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.metrics import normalized_mutual_info_score


def calc_nmi(true_labels, pred_labels) -> float:
    return normalized_mutual_info_score(true_labels, pred_labels)


def calc_ari(true_labels, pred_labels) -> float:
    return adjusted_rand_score(true_labels, pred_labels)


def calc_acc(true_labels, pred_labels) -> float:
    return clustering_accuracy(true_labels, pred_labels)


def clustering_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_labels = {v: i for i, v in enumerate(np.unique(y_true))}
    pred_labels = {v: i for i, v in enumerate(np.unique(y_pred))}
    yt = np.array([true_labels[v] for v in y_true])
    yp = np.array([pred_labels[v] for v in y_pred])

    w = confusion_matrix(yt, yp)
    row_ind, col_ind = linear_sum_assignment(-w)
    return w[row_ind, col_ind].sum() / w.sum()