"""
Prior sensitivity, expressed as three people who would plausibly disagree about the
same decision rather than as an arbitrary grid of hyperparameters.

  Scout    trusts what he has seen with his own eyes: a wide spread of real ability,
           so a big innings should count.
  Analyst  the report's baseline.
  CFO      has watched franchises overpay for one good season: assumes most of what
           looks like ability is noise, so shrink hard.

Same data, same sampler, same seeds. Only the prior changes.
"""
import numpy as np, json, os
import dataio
from gibbs import Priors, run_chains

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

STAKEHOLDERS = {
    "Scout":   Priors(m0=140, s0=25, a_tau=3, b_tau=800, a_om=3, b_om=128),
    "Analyst": Priors(),
    "CFO":     Priors(m0=130, s0=10, a_tau=6, b_tau=250, a_om=3, b_om=128),
    # A deliberate stress test, not a plausible belief: someone who insists that
    # almost all apparent difference between T20 batters is noise (tau ~ 2.5).
    "Stress":  Priors(m0=130, s0=5, a_tau=40, b_tau=250, a_om=3, b_om=128),
}

def prior_summary(p):
    return dict(mu=f"N({p.m0:.0f}, {p.s0:.0f}^2)",
                tau=f"IG({p.a_tau:.0f}, {p.b_tau:.0f})  ->  E[tau] ~ {np.sqrt(p.b_tau/(p.a_tau-1)):.0f}",
                omega=f"IG({p.a_om:.0f}, {p.b_om:.0f})")

if __name__ == "__main__":
    D = dataio.load()
    res = {}
    for name, pr in STAKEHOLDERS.items():
        f = run_chains(D["full"], pr, "M3", chains=4, study=f"sens{name}")
        th = f["theta"].reshape(-1, D["full"].P)
        res[name] = dict(theta_mean=th.mean(axis=0),
                         theta_sd=th.std(axis=0, ddof=1),
                         prob140=(th > 140).mean(axis=0),
                         mu=float(f["mu"].mean()), tau2=float(f["tau2"].mean()),
                         sigma2=float(f["sigma2"].mean()),
                         delta=f["delta"].reshape(-1, 4).mean(axis=0))
        print(f"{name:8s} mu={res[name]['mu']:7.2f}  tau={np.sqrt(res[name]['tau2']):6.2f}  "
              f"sigma={np.sqrt(res[name]['sigma2']):7.1f}  "
              f"delta={np.round(res[name]['delta'],2)}")

    tot = np.bincount(D["full"].p, weights=D["full"].n, minlength=D["full"].P)
    elig = tot >= 150
    print("\nShortlist agreement: P(theta > 140) > 0.80, players with 150+ balls")
    lists = {}
    for name in STAKEHOLDERS:
        sel = set(np.where(elig & (res[name]["prob140"] > 0.80))[0])
        lists[name] = sel
        print(f"  {name:8s} {len(sel):3d} players")
    base = lists["Analyst"]
    for name in ("Scout", "CFO", "Stress"):
        inter = len(base & lists[name])
        print(f"  Analyst vs {name:8s}: {inter} shared, "
              f"{len(lists[name] - base)} only {name}, {len(base - lists[name])} only Analyst, "
              f"Jaccard {inter/len(base | lists[name]):.2f}")

    # top-10 stability
    top = {n: [D["players"][i] for i in np.argsort(-res[n]["prob140"] * elig)[:10]]
           for n in STAKEHOLDERS}
    print("\nTop 10 by P(theta>140):")
    for n in STAKEHOLDERS:
        print(f"  {n:8s} " + ", ".join(top[n]))
    print(f"\nNames common to all three top-10 lists: "
          f"{len(set(top['Scout']) & set(top['Analyst']) & set(top['CFO']))}")

    np.savez_compressed(os.path.join(OUT, "sensitivity.npz"),
                        **{f"{n}_{k}": v for n, r in res.items() for k, v in r.items()
                           if isinstance(v, np.ndarray)})
    json.dump({n: dict(mu=r["mu"], tau2=r["tau2"], sigma2=r["sigma2"],
                       delta=list(map(float, r["delta"])),
                       n_shortlist=len(lists[n]), top10=top[n],
                       shortlist=sorted(D["players"][int(i)] for i in lists[n]))
               for n, r in res.items()},
              open(os.path.join(OUT, "sensitivity.json"), "w"), indent=1)
