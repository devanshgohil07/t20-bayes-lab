"""Posterior predictive checks for M3 (and M2 for contrast).

For every cell we draw y_rep ~ N(theta + delta, sigma^2/n) from the posterior
predictive and compare the replicated data with the observed data, both as a
funnel and through four test statistics.
"""
import numpy as np, json, os
import dataio
from gibbs import IPL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

def replicate(model, z, D, n_rep=400, rng=None):
    rng = rng or np.random.default_rng(11)
    d = D["full"] if model == "M3" else D["ipl"]
    th = z[f"{model}_theta"]; th = th.reshape(-1, th.shape[-1])
    dl = z[f"{model}_delta"]; dl = dl.reshape(-1, dl.shape[-1])
    s2 = z[f"{model}_sigma2"].reshape(-1)
    S = th.shape[0]
    idx = rng.choice(S, n_rep, replace=False)
    mean = th[np.ix_(idx, d.p)] + dl[idx][:, d.l]
    yrep = mean + np.sqrt(s2[idx][:, None] / d.n[None, :]) * rng.normal(size=(n_rep, d.C))
    return d, yrep

LEAGUE_NAMES = ["ipl", "cpl", "bbl", "t20i"]

def stats(y, n, l, L):
    """Test statistics a cricket person would recognise, plus the usual spread checks."""
    out = {
        "sd of cell strike rates":      float(np.std(y)),
        "IQR of cell strike rates":     float(np.percentile(y, 75) - np.percentile(y, 25)),
        "highest cell strike rate":     float(np.max(y)),
        "lowest cell strike rate":      float(np.min(y)),
        "sd, cells under 100 balls":    float(np.std(y[n < 100])) if (n < 100).any() else np.nan,
        "sd, cells over 300 balls":     float(np.std(y[n >= 300])) if (n >= 300).any() else np.nan,
        "share of cells above SR 150":  float(np.mean(y > 150)),
        "share of cells below SR 100":  float(np.mean(y < 100)),
        "share of impossible (SR < 0)": float(np.mean(y < 0)),
    }
    for j in range(L):
        m = l == j
        # ball-weighted league mean: what a scorecard would actually report
        out[f"{LEAGUE_NAMES[j]} run rate (ball-weighted)"] = (
            float(np.sum(n[m] * y[m]) / np.sum(n[m])) if m.any() else np.nan)
    return out

if __name__ == "__main__":
    D = dataio.load()
    z = np.load(os.path.join(OUT, "draws.npz"))
    res = {}
    for model in ("M2", "M3"):
        d, yrep = replicate(model, z, D)
        obs = stats(d.y, d.n, d.l, d.L)
        rep = [stats(r, d.n, d.l, d.L) for r in yrep]
        pval = {k: float(np.mean([r[k] >= obs[k] for r in rep])) for k in obs}
        res[model] = dict(obs=obs, pval=pval,
                          rep_mean={k: float(np.mean([r[k] for r in rep])) for k in obs})
        print(f"\n{model} posterior predictive p-values (P(T_rep >= T_obs))")
        for k in obs:
            if np.isnan(obs[k]):
                continue
            print(f"   {k:12s} obs={obs[k]:9.3f}  rep mean={res[model]['rep_mean'][k]:9.3f}"
                  f"   p={pval[k]:.3f}")
        np.savez_compressed(os.path.join(OUT, f"ppc_{model}.npz"),
                            yrep=yrep, y=d.y, n=d.n, l=d.l)
    json.dump(res, open(os.path.join(OUT, "ppc.json"), "w"), indent=1)
