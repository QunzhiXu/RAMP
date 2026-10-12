"""Batch mixing against biological conservation, the two-axis view.

Job 3 (platform discrimination, lower is better) is reported on its own in the
main grid. On its own it has a loophole: an arm that destroys the data entirely
drives platform accuracy to chance and wins. The standard answer in the
integration literature is to report mixing against conservation together, so
that flattening is visible as a loss on the second axis.

These are the widely used quantities, implemented directly rather than pulled
from scib (which is not installed here). LISI is the k-nearest-neighbour
inverse Simpson index, NOT the perplexity-calibrated Gaussian-kernel form of the
original: the simplification is stated wherever the numbers are reported.
"""
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from scipy.stats import chi2

K_LISI = 90
K_KBET = 50
SUBSAMPLE = 20_000


def _codes(labels):
    u, c = np.unique(labels, return_inverse=True)
    return c, len(u)


def lisi(nn_idx, labels):
    """Inverse Simpson index of labels among each cell's neighbours.

    Ranges from 1 (neighbourhood is one label) to the number of labels
    (neighbourhood matches a uniform mix)."""
    c, n = _codes(labels)
    out = np.empty(len(nn_idx))
    for i, row in enumerate(nn_idx):
        p = np.bincount(c[row], minlength=n) / len(row)
        out[i] = 1.0 / np.sum(p ** 2)
    return float(out.mean())


def kbet(nn_idx, batch):
    """Fraction of neighbourhoods whose batch composition departs from global.

    Lower means better mixed."""
    c, n = _codes(batch)
    glob = np.bincount(c, minlength=n) / len(c)
    k = nn_idx.shape[1]
    exp = glob * k
    ok = exp > 0
    stat = np.empty(len(nn_idx))
    for i, row in enumerate(nn_idx):
        obs = np.bincount(c[row], minlength=n)
        stat[i] = np.sum((obs[ok] - exp[ok]) ** 2 / exp[ok])
    return float((chi2.sf(stat, df=ok.sum() - 1) < 0.05).mean())


def evaluate(X, celltype, dataset, seed=0, subsample=SUBSAMPLE):
    """Return one row of batch-mixing and bio-conservation scores."""
    rng = np.random.default_rng(seed)
    if len(X) > subsample:
        sel = rng.choice(len(X), subsample, replace=False)
        X, celltype, dataset = X[sel], celltype[sel], dataset[sel]
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)

    k = max(K_LISI, K_KBET) + 1
    nn = NearestNeighbors(n_neighbors=k).fit(Xs)
    idx = nn.kneighbors(Xs, return_distance=False)[:, 1:]

    cc, n_ct = _codes(celltype)
    km = KMeans(n_clusters=n_ct, n_init=10, random_state=seed).fit_predict(Xs)

    i_lisi, c_lisi = lisi(idx[:, :K_LISI], dataset), lisi(idx[:, :K_LISI], celltype)
    _, n_ds = _codes(dataset)
    return dict(
        # --- batch mixing (higher iLISI / lower kBET and ASW_batch is better)
        iLISI=i_lisi,
        iLISI_norm=(i_lisi - 1) / max(n_ds - 1, 1),
        kBET_reject=kbet(idx[:, :K_KBET], dataset),
        ASW_batch=float(silhouette_score(Xs, dataset, sample_size=10_000,
                                         random_state=seed)),
        # --- biological conservation (higher is better, except cLISI)
        cLISI=c_lisi,
        cLISI_norm=1 - (c_lisi - 1) / max(n_ct - 1, 1),
        ARI=float(adjusted_rand_score(cc, km)),
        NMI=float(normalized_mutual_info_score(cc, km)),
        ASW_celltype=float(silhouette_score(Xs, celltype, sample_size=10_000,
                                            random_state=seed)),
        n_cells=len(Xs), n_celltypes=n_ct, n_datasets=n_ds)
