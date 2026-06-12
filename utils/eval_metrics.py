# eval_metrics.py
# -*- coding: utf-8 -*-
"""
Clustering evaluation metrics in pure Python:
- AMI, ARI via scikit-learn
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def clustering_evaluation_metrics(ground_truth, predicted):
    """Return AMI and ARI for given ground-truth and predicted labels."""
    gt = np.asarray(ground_truth).ravel()
    pr = np.asarray(predicted).ravel()
    if gt.shape[0] != pr.shape[0]:
        raise ValueError("ground_truth and predicted must have the same length.")

    ami = adjusted_mutual_info_score(gt, pr)
    ari = adjusted_rand_score(gt, pr)
    return ami, ari
