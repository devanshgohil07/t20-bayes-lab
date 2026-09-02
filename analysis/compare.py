"""WAIC / PSIS-LOO for M0-M3, evaluated on the common set of IPL cells."""
import numpy as np, json, os
import dataio
from waic_loo import waic, psis_loo
from gibbs import IPL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

D = dataio.load()
z = np.load(os.path.join(OUT, "draws.npz"))
mask_ipl = D["full"].l == IPL

rows = []
for m in ("M0", "M1", "M2", "M3"):
    ll = z[f"{m}_loglik"]
    ll = ll.reshape(-1, ll.shape[-1])
    if m == "M3":
        ll = ll[:, mask_ipl]              # restrict to the common cells
    w, l = waic(ll), psis_loo(ll)
    rows.append(dict(model=m, waic=w["waic"], waic_se=w["se"], p_waic=w["p_eff"],
                     elpd_waic=w["elpd"], looic=l["looic"], loo_se=l["se"],
                     elpd_loo=l["elpd"], khat_max=l["khat_max"], khat_bad=l["khat_bad"]))

best = max(r["elpd_waic"] for r in rows)
print(f"{'model':6s} {'elpd_waic':>11s} {'p_waic':>8s} {'WAIC':>10s} {'se':>7s} "
      f"{'dWAIC':>8s} | {'elpd_loo':>10s} {'LOOIC':>10s} {'khat>0.7':>9s}")
for r in rows:
    print(f"{r['model']:6s} {r['elpd_waic']:11.1f} {r['p_waic']:8.1f} {r['waic']:10.1f} "
          f"{r['waic_se']:7.1f} {2*(best-r['elpd_waic']):8.1f} | "
          f"{r['elpd_loo']:10.1f} {r['looic']:10.1f} {r['khat_bad']:9d}")
json.dump(rows, open(os.path.join(OUT, "compare.json"), "w"), indent=1)
