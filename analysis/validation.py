"""
Out-of-sample validation: train on 2021-2024, predict every player's 2025 IPL
strike rate, stratified by how many IPL balls the player had *before* 2025.

The stratification is the point of the study. For players with no prior IPL
record the conventional no-pooling estimate does not exist at all.
"""
import numpy as np, json, os
import dataio
from gibbs import Priors, run_chains, IPL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "out")

BUCKETS = [("0 balls, overseas record", None), ("0 balls, no record", None),
           ("1-100", (1, 100)), ("101-300", (101, 300)), ("300+", (301, np.inf))]


def flat(a):
    return a.reshape(-1, *a.shape[2:])


def theta_draws_for(model, fits, D, gidx, rng):
    """Posterior draws of theta on the IPL scale for global player ids `gidx`."""
    f = fits[model]
    S = f["sigma2"].size
    if model == "M0":
        return np.repeat(flat(f["theta"])[:, :1], len(gidx), axis=1)
    if model == "M3":
        return flat(f["theta"])[:, gidx]
    loc = D["ipl_remap"][gidx]                     # -1 where the player has no IPL data
    th = np.empty((S, len(gidx)))
    have = loc >= 0
    if have.any():
        th[:, have] = flat(f["theta"])[:, loc[have]]
    if (~have).any():
        if model == "M1":
            th[:, ~have] = np.nan                  # undefined: no pooling has nothing
        else:                                      # M2: draw from the fitted population
            mu, tau2 = flat(f["mu"]), flat(f["tau2"])
            th[:, ~have] = (mu[:, None]
                            + np.sqrt(tau2)[:, None] * rng.normal(size=(S, (~have).sum())))
    return th


def run(fits, D):
    rng = np.random.default_rng(7)
    t = D["test"]
    ipl_balls = np.zeros(len(D["players"]))
    ipl_balls[D["ipl_players"]] = np.bincount(D["ipl"].p, weights=D["ipl"].n,
                                              minlength=D["ipl"].P)
    other = np.bincount(D["full"].p[D["full"].l != IPL],
                        weights=D["full"].n[D["full"].l != IPL],
                        minlength=len(D["players"]))
    b = ipl_balls[t["p"]]
    o = other[t["p"]]
    masks = {
        "0 balls, overseas record": (b == 0) & (o > 0),
        "0 balls, no record":       (b == 0) & (o == 0),
        "1-100":                    (b >= 1) & (b <= 100),
        "101-300":                  (b > 100) & (b <= 300),
        "300+":                     b > 300,
        "ALL":                      np.ones(len(b), bool),
        "ALL with IPL record":      b > 0,
    }

    res = {}
    for model in ("M0", "M1", "M2", "M3"):
        th = theta_draws_for(model, fits, D, t["p"], rng)
        s2 = flat(fits[model]["sigma2"])[:, None]
        yrep = th + np.sqrt(s2 / t["n"][None, :]) * rng.normal(size=th.shape)
        yhat = th.mean(axis=0)
        lo, hi = np.percentile(yrep, [2.5, 97.5], axis=0)
        # pointwise log predictive density, Monte Carlo over draws
        sd = np.sqrt(s2 / t["n"][None, :] + 0.0)
        lpd = (-0.5 * np.log(2 * np.pi * sd ** 2)
               - (t["y"][None, :] - th) ** 2 / (2 * sd ** 2))
        m = np.max(lpd, axis=0)
        lppd = m + np.log(np.mean(np.exp(lpd - m), axis=0))
        res[model] = dict(yhat=yhat, lo=lo, hi=hi, lppd=lppd)

    rows = []
    for name, msk in masks.items():
        if msk.sum() == 0:
            continue
        row = {"bucket": name, "n": int(msk.sum())}
        for model in ("M0", "M1", "M2", "M3"):
            r = res[model]
            ok = msk & np.isfinite(r["yhat"])
            if ok.sum() == 0:
                row[model] = None
                continue
            err = r["yhat"][ok] - t["y"][ok]
            cov = np.mean((t["y"][ok] >= r["lo"][ok]) & (t["y"][ok] <= r["hi"][ok]))
            # bootstrap se of RMSE
            bs = [np.sqrt(np.mean(rng.choice(err ** 2, ok.sum())))
                  for _ in range(2000)]
            row[model] = dict(rmse=float(np.sqrt(np.mean(err ** 2))),
                              rmse_se=float(np.std(bs)),
                              mae=float(np.mean(np.abs(err))),
                              cov95=float(cov), elpd=float(np.sum(r["lppd"][ok])),
                              n_defined=int(ok.sum()))
        rows.append(row)
    return rows, res, masks, b


if __name__ == "__main__":
    import dataio
    D = dataio.load()
    z = np.load(os.path.join(OUT, "draws.npz"))
    fits = {m: {k.split("_", 1)[1]: z[k] for k in z.files if k.startswith(m + "_")}
            for m in ("M0", "M1", "M2", "M3")}
    rows, res, masks, b = run(fits, D)

    print(f"\n{'bucket':26s} {'n':>4s} | " +
          " | ".join(f"{m:^22s}" for m in ("M0", "M1", "M2", "M3")))
    print(f"{'':26s} {'':>4s} | " + " | ".join(f"{'RMSE (se)   cov95':^22s}" for _ in range(4)))
    for r in rows:
        line = f"{r['bucket']:26s} {r['n']:4d} | "
        cells = []
        for m in ("M0", "M1", "M2", "M3"):
            v = r[m]
            cells.append("      not defined     " if v is None else
                         f"{v['rmse']:6.2f} ({v['rmse_se']:4.2f}) {v['cov95']*100:5.1f}%")
        print(line + " | ".join(cells))

    print("\nholdout elpd (higher is better), all defined cells:")
    for m in ("M0", "M1", "M2", "M3"):
        tot = [r[m]["elpd"] for r in rows if r["bucket"] == "ALL" and r[m]]
        tot2 = [r[m]["elpd"] for r in rows if r["bucket"] == "ALL with IPL record" and r[m]]
        print(f"  {m}: all cells {tot[0] if tot else float('nan'):9.2f}   "
              f"with IPL record {tot2[0] if tot2 else float('nan'):9.2f}")

    json.dump([{k: v for k, v in r.items()} for r in rows],
              open(os.path.join(OUT, "validation.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "validation_pred.npz"),
                        **{f"{m}_{k}": v for m, r in res.items() for k, v in r.items()},
                        prior_balls=b, y=D["test"]["y"], n=D["test"]["n"],
                        p=D["test"]["p"])
    print("\nsaved out/validation.json")
