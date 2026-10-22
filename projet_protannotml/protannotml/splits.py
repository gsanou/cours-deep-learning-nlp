"""Exported from notebook pf02_splits_homologie.ipynb — edit the notebook source (src/pf02_splits_homologie.md), not this file."""

"""Homology-aware splitting for ProtAnnotML (identity on IMGT-aligned germlines, group splits, leakage)."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*X"
AA_INDEX = {a: i for i, a in enumerate(ALPHABET)}


def encode_aligned(position_dicts, last=104):
    """List of {IMGT position: aa} (V-REGION) -> int array (n, last), -1 for gaps."""
    X = np.full((len(position_dicts), last), -1, dtype=np.int16)
    for i, pos in enumerate(position_dicts):
        for p, aa in pos.items():
            base = int(str(p).split(".")[0])
            if base <= last:
                X[i, base - 1] = AA_INDEX.get(aa, AA_INDEX["X"])
    return X


def identity_matrix(X, Y=None):
    """Pairwise identity on positions occupied in both sequences (matches / co-occupied positions)."""
    Y = X if Y is None else Y
    k = len(ALPHABET)

    def onehot(A):
        O = np.zeros((A.shape[0], A.shape[1], k), dtype=np.float32)
        r, c = np.nonzero(A >= 0)
        O[r, c, A[r, c]] = 1
        return O.reshape(A.shape[0], -1)

    matches = onehot(X) @ onehot(Y).T
    both = (X >= 0).astype(np.float32) @ (Y >= 0).astype(np.float32).T
    return matches / np.maximum(both, 1)


def cluster_at(identity, threshold):
    """Single-linkage clusters: connected components of the graph identity >= threshold."""
    n_clusters, labels = connected_components(csr_matrix(identity >= threshold), directed=False)
    return n_clusters, labels


def group_split(examples, group_of_domain, fractions=(0.8, 0.1, 0.1), seed=0):
    """Assign whole groups to train/val/test. group_of_domain(domain_dict) -> hashable group id.

    Returns {"train": [...ids], "val": [...], "test": [...], "dropped": [...]}.
    """
    rng = np.random.default_rng(seed)
    ex_groups = [{group_of_domain(d) for d in ex["domains"] if d["type"] == "V"} for ex in examples]
    count = {}
    for gs in ex_groups:
        for g in gs:
            count[g] = count.get(g, 0) + 1
    order = list(count)
    rng.shuffle(order)
    total, acc, side = sum(count.values()), 0, {}
    bounds = np.cumsum(fractions) * total
    for g in order:
        side[g] = "train" if acc < bounds[0] else "val" if acc < bounds[1] else "test"
        acc += count[g]
    out = {"train": [], "val": [], "test": [], "dropped": []}
    for ex, gs in zip(examples, ex_groups):
        sides = {side[g] for g in gs}
        out[sides.pop() if len(sides) == 1 else "dropped"].append(ex["id"])
    return out


def leakage_report(examples, split, allele_index, identity):
    """Max germline identity of each test V domain to the training V alleles, and exact-allele overlap.
    allele_index maps 'species|allele' keys to rows of the identity matrix."""
    key = lambda d: f"{d['species']}|{d['v_allele']}"
    by_id = {ex["id"]: ex for ex in examples}
    train_alleles = {key(d) for i in split["train"] for d in by_id[i]["domains"] if d["type"] == "V"}
    train_idx = np.array(sorted(allele_index[a] for a in train_alleles))
    max_id, seen = [], []
    for i in split["test"]:
        for d in by_id[i]["domains"]:
            if d["type"] == "V":
                max_id.append(identity[allele_index[key(d)], train_idx].max())
                seen.append(key(d) in train_alleles)
    return np.array(max_id), np.array(seen)
