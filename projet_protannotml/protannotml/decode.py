"""Exported from notebook pf03_baselines.ipynb — edit the notebook source (src/pf03_baselines.md), not this file."""

"""Constrained decoders: monotone IMGT numbering, ordered regions, valid BIO tags."""
import numpy as np

NEG = -1e9


def viterbi(log_em, allowed, start_allowed=None):
    """Generic Viterbi. log_em (L, S); allowed (S, S) boolean transition mask; start_allowed (S,) boolean."""
    L, S = log_em.shape
    trans = np.where(allowed, 0.0, NEG)
    delta = log_em[0] + (0.0 if start_allowed is None else np.where(start_allowed, 0.0, NEG))
    back = np.zeros((L, S), dtype=np.int32)
    for t in range(1, L):
        scores = delta[:, None] + trans
        back[t] = scores.argmax(0)
        delta = scores.max(0) + log_em[t]
    path = [int(delta.argmax())]
    for t in range(L - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def monotone_viterbi(log_em):
    """Best strictly increasing state path (states already sorted in chain order), O(L*S) with prefix maxima."""
    L, S = log_em.shape
    idx = np.arange(S)
    delta = log_em[0].copy()
    back = np.zeros((L, S), dtype=np.int32)
    for t in range(1, L):
        cm = np.maximum.accumulate(delta)
        arg = np.maximum.accumulate(np.where(delta == cm, idx, 0))
        prev_best = np.concatenate([[NEG], cm[:-1]])        # max over s' < s
        back[t] = np.concatenate([[0], arg[:-1]])
        delta = prev_best + log_em[t]
    path = [int(delta.argmax())]
    for t in range(L - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def ordered_regions_mask(n_states):
    """Regions FR1..FR4 in order: stay, or move to the next region."""
    allowed = np.eye(n_states, dtype=bool)
    allowed[np.arange(n_states - 1), np.arange(1, n_states)] = True
    return allowed


BIO_TAGS = ["O", "B-V", "I-V", "B-C", "I-C"]


def bio_mask(tags=BIO_TAGS):
    """I-X only after B-X or I-X; never start with I-X."""
    S = len(tags)
    allowed = np.ones((S, S), dtype=bool)
    for j, t in enumerate(tags):
        if t.startswith("I-"):
            allowed[:, j] = [tags[i] in ("B-" + t[2:], t) for i in range(S)]
    start = np.array([not t.startswith("I-") for t in tags])
    return allowed, start


def spans_from_bio(tags):
    """BIO tags -> list of (type, start, end) spans (end exclusive)."""
    spans, cur = [], None
    for i, t in enumerate(list(tags) + ["O"]):
        if cur and (t == "O" or t.startswith("B-") or t[2:] != cur[0]):
            spans.append((cur[0], cur[1], i)); cur = None
        if t.startswith("B-") or (t.startswith("I-") and cur is None):
            cur = (t[2:], i)
    return spans


def span_match(pred, gold, tol=0):
    """Fraction of gold spans with a predicted span of the same type and both bounds within tol."""
    hits = sum(any(p[0] == g[0] and abs(p[1] - g[1]) <= tol and abs(p[2] - g[2]) <= tol for p in pred) for g in gold)
    return hits, len(gold)
