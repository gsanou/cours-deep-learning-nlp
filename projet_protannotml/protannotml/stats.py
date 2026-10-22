"""Exported from notebook pf05_evaluation.ipynb — edit the notebook source (src/pf05_evaluation.md), not this file."""

"""Evaluation statistics for ProtAnnotML: cluster bootstrap, paired comparisons, selective prediction."""
import numpy as np
from scipy import stats


def bootstrap_ci(values, groups=None, n_boot=2000, alpha=0.05, seed=0, statistic=np.mean):
    """Percentile CI of statistic(values). If groups is given, resample whole groups (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if groups is None:
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        boots = np.array([statistic(values[i]) for i in idx])
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in uniq}
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            boots.append(statistic(values[np.concatenate([members[g] for g in pick])]))
        boots = np.array(boots)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(statistic(values)), float(lo), float(hi)


def paired_bootstrap_diff(a, b, groups=None, n_boot=2000, seed=0):
    """CI and two-sided p-value for mean(a - b) on paired per-item scores, optionally by cluster."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    est, lo, hi = bootstrap_ci(d, groups, n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed + 1)
    if groups is None:
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    else:
        groups = np.asarray(groups); uniq = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in uniq}
        boots = np.array([d[np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])].mean() for _ in range(n_boot)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return est, lo, hi, max(p, 1 / n_boot)


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar test on paired binary outcomes (binomial test on discordant pairs)."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, float(stats.binomtest(n10, n01 + n10, 0.5).pvalue)


def risk_coverage(confidence, correct):
    """Sort by decreasing confidence; return coverage and error rate among the accepted predictions."""
    order = np.argsort(-np.asarray(confidence))
    c = np.asarray(correct, dtype=float)[order]
    k = np.arange(1, len(c) + 1)
    return k / len(c), 1 - np.cumsum(c) / k


def aurc(confidence, correct):
    """Area under the risk-coverage curve (lower is better)."""
    cov, risk = risk_coverage(confidence, correct)
    return float(np.mean(risk))
