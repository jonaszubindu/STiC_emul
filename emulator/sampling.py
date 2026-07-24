"""Plain-numpy mini-batch k-means and stratified draw (vendored from
STIC_vers_2025/nn_emulator/sampling.py; no sklearn dependency)."""

import numpy as np


def _kmeanspp_init(X, k, rng):
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    centers[0] = X[rng.integers(n)]
    d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        p = d2 / d2.sum()
        centers[i] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, np.sum((X - centers[i]) ** 2, axis=1))
    return centers


def _assign(X, centers, chunk=100000):
    """Nearest-center labels, chunked to bound memory."""
    labels = np.empty(X.shape[0], dtype=np.int32)
    c2 = 0.5 * np.sum(centers ** 2, axis=1)
    for i0 in range(0, X.shape[0], chunk):
        sl = slice(i0, i0 + chunk)
        labels[sl] = np.argmax(X[sl] @ centers.T - c2, axis=1)
    return labels


def minibatch_kmeans(X, k, rng, batch=8192, iters=200):
    """Standard mini-batch k-means (Sculley 2010) in plain numpy."""
    centers = _kmeanspp_init(X[rng.choice(X.shape[0],
                                          min(50000, X.shape[0]),
                                          replace=False)], k, rng)
    counts = np.zeros(k)
    for _ in range(iters):
        idx = rng.choice(X.shape[0], min(batch, X.shape[0]), replace=False)
        B = X[idx]
        lab = _assign(B, centers)
        for j in np.unique(lab):
            pts = B[lab == j]
            counts[j] += pts.shape[0]
            eta = pts.shape[0] / counts[j]
            centers[j] = (1.0 - eta) * centers[j] + eta * pts.mean(axis=0)
    return centers


def stratified_draw(labels, n_total, alloc_exp, min_per_cluster, rng):
    """Allocate n_total draws over clusters ~ size**alloc_exp, then draw."""
    k = labels.max() + 1
    sizes = np.bincount(labels, minlength=k).astype(float)
    w = np.where(sizes > 0, sizes ** alloc_exp, 0.0)
    alloc = np.floor(n_total * w / w.sum()).astype(int)
    alloc = np.minimum(np.maximum(alloc, min_per_cluster), sizes.astype(int))
    while alloc.sum() > n_total:
        j = np.argmax(alloc)
        alloc[j] -= 1
    room = sizes.astype(int) - alloc
    while alloc.sum() < n_total and room.sum() > 0:
        j = np.argmax(room)
        alloc[j] += 1
        room[j] -= 1
    sel = []
    for j in range(k):
        if alloc[j] <= 0:
            continue
        members = np.where(labels == j)[0]
        sel.append(rng.choice(members, alloc[j], replace=False))
    return np.concatenate(sel)
