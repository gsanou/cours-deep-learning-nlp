"""Exported from notebook pf03_baselines.ipynb — edit the notebook source (src/pf03_baselines.md), not this file."""

"""Window one-hot features for per-residue baselines."""
import numpy as np
from scipy.sparse import csr_matrix, vstack

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def window_features(seq, w=6):
    n, k = len(seq), len(AA20) + 1
    rows, cols = [], []
    for t in range(n):
        for o in range(-w, w + 1):
            j = t + o
            c = AA20.find(seq[j]) if 0 <= j < n else -1
            rows.append(t)
            cols.append((o + w) * k + (c if c >= 0 else k - 1))
    return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, (2 * w + 1) * k))


def stack_features(seqs, w=6):
    return vstack([window_features(s, w) for s in seqs]).tocsr()
