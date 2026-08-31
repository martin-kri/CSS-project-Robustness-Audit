"""
stats_helpers.py
Shared statistics functions used by the three analysis scripts:
Wilson confidence intervals, Newcombe effect sizes, permutation and
two-proportion tests, Holm correction, and a mean bootstrap.
"""

import glob
import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

RNG = np.random.default_rng(42)   # fixed seed so results are reproducible


def fmt_p(p):
    """Format a p-value, avoiding 'p = 0' from floating-point underflow."""
    if p < 1e-16:
        return "p < 1e-16"
    if p < 0.001:
        return f"p = {p:.1e}"
    return f"p = {p:.3f}"


def wilson(k, n, alpha=0.05):
    """95% CI for a proportion. At 100% cooperation (1050/1050) this gives
    [99.64%, 100%] instead of the textbook formula's uninformative [100%, 100%]."""
    if n == 0:
        return (np.nan, np.nan)
    return proportion_confint(k, n, alpha=alpha, method="wilson")


def rate_line(label, k, n):
    """One formatted line: rate, interval, sample size."""
    p = k / n
    lo, hi = wilson(k, n)
    note = ""
    if k == n:
        note = f"   [0 counter-examples in {n}; true rate > {1-3/n:.2%}]"
    elif k == 0:
        note = f"   [0 occurrences in {n}; true rate < {3/n:.2%}]"
    return f"{label:42s} {p:7.2%}  95% CI [{lo:6.2%}, {hi:6.2%}]  n = {n}{note}"


def newcombe_diff(k1, n1, k2, n2, alpha=0.05):
    """
    Difference between two proportions, with a confidence interval that stays
    valid when one of them is 0% or 100%. Returns (difference, low, high).
    """
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, alpha)
    l2, u2 = wilson(k2, n2, alpha)
    d = p1 - p2
    return (d,
            d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def two_prop_test(k1, n1, k2, n2):
    """Standard two-proportion z-test. Returns the p-value."""
    p1, p2 = k1 / n1, k2 / n2
    pbar = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return 2 * (1 - stats.norm.cdf(abs((p1 - p2) / se)))


def holm(tests, title):
    """
    tests: list of (label, p_value).
    Adjusts for how many tests you ran, then prints the verdict.
    """
    if not tests:
        return
    labels, ps = zip(*tests)
    reject, p_adj, _, _ = multipletests(ps, alpha=0.05, method="holm")
    print(f"\n  Holm correction, {title} ({len(ps)} tests):")
    for lab, p, pa, r in zip(labels, ps, p_adj, reject):
        print(f"    {lab:40s} {fmt_p(p):14s} -> adjusted {fmt_p(pa):14s} "
              f"{'significant' if r else 'not significant'}")


def bootstrap_mean_ci(x, n_boot=10000, alpha=0.05, seed=42):
    """Confidence interval for a mean, by resampling the data."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if np.allclose(x, x[0]):        # every value identical: no uncertainty to show
        return x.mean(), x[0], x[0]
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (x.mean(),
            np.percentile(draws, 100 * alpha / 2),
            np.percentile(draws, 100 * (1 - alpha / 2)))


def perm_test_means(x, y, n_perm=20000, seed=42):
    """Permutation test for a difference in means. Returns (difference, p-value)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = x.mean() - y.mean()
    pool = np.concatenate([x, y])
    nx, count = len(x), 0
    rng = np.random.default_rng(seed)
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:nx].mean() - pool[nx:].mean()) >= abs(obs) - 1e-12:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def load_all(pattern, model_from_filename):
    """Read every CSV matching the pattern, tagging each with its model."""
    frames = []
    for path in sorted(glob.glob(pattern)):
        d = pd.read_csv(path)
        d["model"] = model_from_filename(path)
        d["source_file"] = os.path.basename(path)
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return pd.concat(frames, ignore_index=True)


def model_from_filename(path):
    return "Llama" if "llama" in path.lower() else "Gemma"