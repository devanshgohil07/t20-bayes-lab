"""Write everything the web app needs into app/data/."""
import numpy as np, json, os, shutil
import dataio, diagnostics as dg
from gibbs import Priors, IPL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, APP = os.path.join(ROOT, "out"), os.path.join(ROOT, "app", "data")
os.makedirs(APP, exist_ok=True)

D = dataio.load()
z = np.load(os.path.join(OUT, "draws.npz"))
full, ipl = D["full"], D["ipl"]
th = z["M3_theta"].reshape(-1, full.P)
dl = z["M3_delta"].reshape(-1, 4)
mu, tau2, s2, om2 = (z[f"M3_{k}"].reshape(-1) for k in ("mu", "tau2", "sigma2", "omega2"))

KEEP = 200
idx = np.linspace(0, th.shape[0] - 1, KEEP).astype(int)

# ---- cells (the app re-runs the sampler on these) -------------------------
shutil.copy(os.path.join(ROOT, "data", "cells.json"), os.path.join(APP, "cells.json"))

json.dump({"leagues": D["leagues"], "iplIndex": IPL,
           "delta": [[round(float(v), 2) for v in dl[idx, j]] for j in range(4)],
           "mu": [round(float(v), 2) for v in mu[idx]],
           "tau2": [round(float(v), 1) for v in tau2[idx]],
           "sigma2": [round(float(v), 0) for v in s2[idx]],
           "omega2": [round(float(v), 1) for v in om2[idx]]},
          open(os.path.join(APP, "offsets.json"), "w"))

# ---- P(theta > c) on a grid, from ALL 8,000 draws -------------------------
GRID = np.arange(100, 201, 1)
prob = np.stack([(th > c).mean(axis=0) for c in GRID], axis=1)   # (P, len(GRID))
json.dump({"thresholds": GRID.tolist(),
           "p": [[round(float(v), 3) for v in row] for row in prob]},
          open(os.path.join(APP, "prob.json"), "w"))

# ---- per-player summary ---------------------------------------------------
balls = np.zeros((full.P, 4))
for c in range(full.C):
    balls[full.p[c], full.l[c]] = full.n[c]
sr = np.full((full.P, 4), np.nan)
for c in range(full.C):
    sr[full.p[c], full.l[c]] = full.y[c]
post = th.mean(axis=0); psd = th.std(axis=0, ddof=1)
lo, hi = np.percentile(th, [2.5, 97.5], axis=0)
json.dump({"players": D["players"],
           "balls": balls.astype(int).tolist(),
           "sr": [[None if np.isnan(v) else round(float(v), 1) for v in r] for r in sr],
           "mean": [round(float(v), 1) for v in post],
           "sd": [round(float(v), 1) for v in psd],
           "lo": [round(float(v), 1) for v in lo],
           "hi": [round(float(v), 1) for v in hi]},
          open(os.path.join(APP, "summary.json"), "w"))

# ---- alternative priors: posterior mean and sd per player ----------------
# The three stakeholder priors and the stress test. P(theta>c) for these is taken
# from a Normal approximation to the posterior, which agrees with the draws to
# within 0.013 in probability (checked against the baseline).
sens = np.load(os.path.join(OUT, "sensitivity.npz"))
alt = {}
for name in ("Scout", "Analyst", "CFO", "Stress"):
    key = f"{name}_theta_mean"
    if key not in sens.files:
        continue
    alt[name] = {"mean": [round(float(v), 1) for v in sens[key]],
                 "sd": [round(float(v), 2) for v in sens[f"{name}_theta_sd"]]}
json.dump(alt, open(os.path.join(APP, "altpriors.json"), "w"))

# ---- validation, diagnostics, ppc, headline numbers ----------------------
br = json.load(open(os.path.join(ROOT, "data", "ballruns.json")))
v = np.array(br["values"], float); c = np.array(br["counts"], float)
m1 = (v * c).sum() / c.sum(); var = (v**2 * c).sum() / c.sum() - m1**2

overlap = [[len(set(full.p[full.l == a]) & set(full.p[full.l == b])) for b in range(4)]
           for a in range(4)]
json.dump({
    "priors": Priors().__dict__,
    "ballRuns": {"values": br["values"], "counts": br["counts"],
                 "mean": round(float(m1), 4), "var": round(float(var), 4),
                 "sigma2": round(float(var) * 1e4, 0),
                 "sdAt100": round(float(np.sqrt(var) * 10), 1)},
    "fitted": {"mu": round(float(mu.mean()), 2), "tau2": round(float(tau2.mean()), 1),
               "sigma2": round(float(s2.mean()), 0), "omega2": round(float(om2.mean()), 1),
               "sigmaEff": round(float(np.sqrt(s2.mean())), 1),
               "sdAt100Fitted": round(float(np.sqrt(s2.mean()) / 10), 1),
               "designEffect": round(float(s2.mean() / (var * 1e4)), 2),
               "delta": [round(float(dl[:, j].mean()), 2) for j in range(4)],
               "deltaLo": [round(float(np.percentile(dl[:, j], 2.5)), 2) for j in range(4)],
               "deltaHi": [round(float(np.percentile(dl[:, j], 97.5)), 2) for j in range(4)]},
    "overlap": overlap,
    "leagueProfile": json.load(open(os.path.join(ROOT, "data", "league_profile.json"))),
    "counts": {"players": full.P, "cells": full.C, "iplPlayers": ipl.P,
               "balls": int(full.n.sum()),
               "cellsByLeague": [int((full.l == j).sum()) for j in range(4)]},
    "diagnostics": json.load(open(os.path.join(OUT, "diag_summary.json"))),
    "ppc": json.load(open(os.path.join(OUT, "ppc.json")))["M3"],
    "compare": json.load(open(os.path.join(OUT, "compare.json"))),
    "sensitivity": json.load(open(os.path.join(OUT, "sensitivity.json"))),
}, open(os.path.join(APP, "meta.json"), "w"), indent=1)

for f in sorted(os.listdir(APP)):
    print(f"  app/data/{f}  {os.path.getsize(os.path.join(APP,f))/1024:.0f} KB")
