"""
WAIC and PSIS-LOO from a pointwise log-likelihood matrix ll of shape
(draws, cells).  Vehtari, Gelman & Gabry (2017); Session 20.

PSIS: the importance ratios for leave-one-out are r_s = 1/p(y_c | theta_s).
Their right tail is fitted with a generalised Pareto distribution
(Zhang & Stephens 2009) and replaced by the fitted quantiles; khat is the
fitted shape and flags cells where the estimate is unreliable (khat > 0.7).
"""
import numpy as np


def _lse(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)),
                      axis=axis)


def waic(ll):
    S = ll.shape[0]
    lppd = _lse(ll, axis=0) - np.log(S)
    p_w = np.var(ll, axis=0, ddof=1)
    elpd = lppd - p_w
    C = len(elpd)
    return dict(elpd=float(elpd.sum()), p_eff=float(p_w.sum()),
                waic=float(-2 * elpd.sum()),
                se=float(2 * np.sqrt(C) * np.std(elpd, ddof=1)),
                pointwise=elpd)


def _gpdfit(x):
    """x: sorted positive excesses. Returns (k, sigma)."""
    n = len(x)
    m = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    bs = bs / (3.0 * x[int(n / 4 + 0.5) - 1]) + 1.0 / x[-1]
    ks = np.mean(np.log1p(-np.outer(bs, x)), axis=1)
    L = n * (np.log(-bs / ks) - ks - 1.0)
    w = 1.0 / np.sum(np.exp(L[None, :] - L[:, None]), axis=1)
    b = np.sum(bs * w) / np.sum(w)
    k = float(np.mean(np.log1p(-b * x)))
    sigma = -k / b
    k = k * n / (n + 10.0) + 0.5 * 10.0 / (n + 10.0)   # weak prior on k
    return k, sigma


def _psis_smooth(lw):
    """Pareto-smooth a vector of log importance weights. Returns (lw, khat)."""
    S = len(lw)
    lw = lw - np.max(lw)
    M = int(min(0.2 * S, 3.0 * np.sqrt(S)))
    if M < 5:
        return lw, 0.0
    idx = np.argsort(lw)
    tail_idx = idx[-M:]
    cut = lw[idx[-M - 1]]
    x = np.exp(lw[tail_idx]) - np.exp(cut)
    if np.any(x <= 0) or np.all(x < 1e-300):
        return lw, 0.0
    k, sigma = _gpdfit(np.sort(x))
    if not np.isfinite(k) or k >= 1.0:
        return lw, float(k)
    q = (np.arange(1, M + 1) - 0.5) / M
    qgpd = sigma * np.expm1(-k * np.log1p(-q)) / k
    smoothed = np.log(np.exp(cut) + qgpd)
    order = tail_idx[np.argsort(lw[tail_idx])]
    lw = lw.copy()
    lw[order] = np.minimum(smoothed, np.max(lw))
    return lw, float(k)


def psis_loo(ll):
    S, C = ll.shape
    elpd = np.empty(C)
    ks = np.empty(C)
    for c in range(C):
        lwc, k = _psis_smooth(-ll[:, c])
        ks[c] = k
        elpd[c] = _lse(lwc + ll[:, c]) - _lse(lwc)
    return dict(elpd=float(elpd.sum()), looic=float(-2 * elpd.sum()),
                se=float(2 * np.sqrt(C) * np.std(elpd, ddof=1)),
                khat_max=float(np.max(ks)), khat_bad=int(np.sum(ks > 0.7)),
                khat=ks, pointwise=elpd)
