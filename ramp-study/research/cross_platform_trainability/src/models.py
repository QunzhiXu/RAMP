"""The three model families swept in every job (amendment 6).

kNN matters most: it works only if corresponding cells across platforms are
genuinely near each other, whereas a parametric model can compensate for
misalignment by learning a boundary and so conceal it.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

K_NEIGHBORS = 30
CAPS = ['knn', 'linear', 'mlp4', 'mlp8', 'mlp16', 'mlp32', 'mlp64', 'mlp128', 'mlp256']
PARAMETRIC = [c for c in CAPS if c != 'knn']


def family(cap):
    return 'knn' if cap == 'knn' else ('linear' if cap == 'linear' else 'mlp')


def clf(cap, seed):
    if cap == 'knn':
        return KNeighborsClassifier(n_neighbors=K_NEIGHBORS, weights='distance')
    if cap == 'linear':
        return LogisticRegression(max_iter=3000, C=1.0)
    return MLPClassifier(hidden_layer_sizes=(int(cap[3:]),), max_iter=400,
                         early_stopping=True, n_iter_no_change=12, random_state=seed)


def reg(cap, seed):
    if cap == 'knn':
        return KNeighborsRegressor(n_neighbors=K_NEIGHBORS, weights='distance')
    if cap == 'linear':
        return Ridge(alpha=1.0)
    return MLPRegressor(hidden_layer_sizes=(int(cap[3:]),), max_iter=300,
                        early_stopping=True, n_iter_no_change=10, random_state=seed)


def n_params(cap, d, k):
    """NaN for kNN: no comparable parameter count, so M2 skips it."""
    if cap == 'knn':
        return np.nan
    if cap == 'linear':
        return d * k + k
    h = int(cap[3:])
    return d * h + h + h * k + k
