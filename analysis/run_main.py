"""Fit M0-M3, save draws, and print the MCMC diagnostics table (T3)."""
import numpy as np, json, os, time
import dataio, diagnostics as dg
from gibbs import Priors, run_chains, IPL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "out"); os.makedirs(OUT, exist_ok=True)

D   = dataio.load()
pri = Priors()
SET = dict(chains=4, iters=12000, burn=4000, thin=4)

fits = {}
for model in ("M0", "M1", "M2", "M3"):
    data = D["full"] if model == "M3" else D["ipl"]
    t0 = time.time()
    fits[model] = run_chains(data, pri, model, **SET)
    print(f"{model}: {fits[model]['sigma2'].size} draws in {time.time()-t0:.1f}s "
          f"(C={data.C}, P={data.P})")

np.savez_compressed(os.path.join(OUT, "draws.npz"),
    **{f"{m}_{k}": v for m, f in fits.items() for k, v in f.items() if v is not None})

# ---------------------------------------------------------- diagnostics ---
rows = []
for m in ("M0", "M1", "M2", "M3"):
    f = fits[m]
    params = [("sigma2", f["sigma2"])]
    if m in ("M2", "M3"):
        params += [("mu", f["mu"]), ("tau2", f["tau2"])]
    if m == "M3":
        params += [("omega2", f["omega2"])]
        for j, name in enumerate(D["leagues"]):
            if j != IPL:
                params.append((f"delta[{name}]", f["delta"][:, :, j]))
    # a few representative thetas: most, median and least data
    d = D["full"] if m == "M3" else D["ipl"]
    tot = np.bincount(d.p, weights=d.n, minlength=d.P)
    order = np.argsort(-tot)
    for tag, i in (("most", order[0]), ("median", order[len(order)//2])):
        params.append((f"theta[{tag} data]", f["theta"][:, :, i]))
    for name, x in params:
        s = dg.summarise(x, name); s["model"] = m; rows.append(s)

print(f"\n{'model':5s} {'parameter':22s} {'mean':>10s} {'sd':>9s} "
      f"{'2.5%':>10s} {'97.5%':>10s} {'Rhat':>7s} {'ESS':>8s}")
for r in rows:
    print(f"{r['model']:5s} {r['param']:22s} {r['mean']:10.3f} {r['sd']:9.3f} "
          f"{r['lo']:10.3f} {r['hi']:10.3f} {r['rhat']:7.4f} {r['ess']:8.0f}")

# worst R-hat / smallest ESS over ALL thetas, the number that actually matters
for m in ("M2", "M3"):
    th = fits[m]["theta"]
    rh = np.array([dg.split_rhat(th[:, :, i]) for i in range(th.shape[2])])
    es = np.array([dg.ess(th[:, :, i]) for i in range(0, th.shape[2], 5)])
    print(f"\n{m}: worst theta Rhat = {np.nanmax(rh):.4f}   "
          f"min theta ESS (every 5th) = {es.min():.0f}   median ESS = {np.median(es):.0f}")

json.dump([{k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in r.items()} for r in rows],
          open(os.path.join(OUT, "diagnostics.json"), "w"), indent=1)

# Worst-case diagnostics over every theta in M3, which is the number the report quotes.
th = fits["M3"]["theta"]
rh = np.array([dg.split_rhat(th[:, :, i]) for i in range(th.shape[2])])
es = np.array([dg.ess(th[:, :, i]) for i in range(0, th.shape[2], 10)])
json.dump({"rhatWorst": float(np.nanmax(rh)), "essMin": float(es.min()),
           "essMedian": float(np.median(es))},
          open(os.path.join(OUT, "diag_summary.json"), "w"), indent=1)
print("\nsaved out/draws.npz and out/diagnostics.json")
