"""Split R-hat, effective sample size, autocorrelation — Gelman et al. (BDA3 §11.4)."""
import numpy as np

def split_rhat(x):
    """x: (chains, draws). Split each chain in half, then the usual R-hat."""
    x = np.asarray(x, dtype=float)
    m, n = x.shape
    n2 = n // 2
    s = np.concatenate([x[:, :n2], x[:, n2:2 * n2]], axis=0)   # (2m, n2)
    M, N = s.shape
    means = s.mean(axis=1)
    varis = s.var(axis=1, ddof=1)
    W = varis.mean()
    B = N * means.var(ddof=1)
    if W <= 0:
        return np.nan
    var_hat = (N - 1) / N * W + B / N
    return float(np.sqrt(var_hat / W))

def _acf(v, maxlag):
    v = v - v.mean()
    d = np.correlate(v, v, mode="full")[len(v) - 1:]
    d = d / d[0] if d[0] != 0 else d
    return d[:maxlag + 1]

def acf(x, maxlag=40):
    """Mean autocorrelation across chains."""
    x = np.asarray(x, dtype=float)
    return np.mean([_acf(c, maxlag) for c in x], axis=0)

def ess(x):
    """Effective sample size, Geyer initial-positive-sequence estimator."""
    x = np.asarray(x, dtype=float)
    m, n = x.shape
    maxlag = min(n - 2, 500)
    rho = np.mean([_acf(c, maxlag) for c in x], axis=0)
    # pair the autocorrelations and stop at the first non-positive pair sum
    s = 0.0
    for t in range(1, maxlag - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        s += pair
    return float(m * n / (1.0 + 2.0 * s))

def summarise(draws, name):
    """R-hat / ESS / mean / sd / 95% CI for a scalar parameter."""
    x = np.asarray(draws, dtype=float)
    flat = x.ravel()
    return dict(param=name, mean=flat.mean(), sd=flat.std(ddof=1),
                lo=np.percentile(flat, 2.5), hi=np.percentile(flat, 97.5),
                rhat=split_rhat(x), ess=ess(x))
