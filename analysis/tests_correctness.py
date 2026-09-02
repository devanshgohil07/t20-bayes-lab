"""
Mandatory correctness tests for the Gibbs sampler (spec Part 3).
These replace a brms/Stan cross-check, which the sandbox cannot run.

  1. Single player, single league  -> matches the closed-form Normal-Normal
                                      posterior of Session 5.
  2. Simulated recovery           -> known theta, delta, sigma2, tau2, omega2
                                      all inside their 95% credible intervals.
  3. Limiting cases               -> tau -> 0 gives complete pooling,
                                      tau -> infinity gives no pooling  (PS5 Q1(e)).
"""
import numpy as np, sys
from gibbs import Data, Priors, gibbs, run_chains, IPL
from rng import rng as seeded

FAIL = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {detail}")
    if not cond:
        FAIL.append(name)

# ---------------------------------------------------------------- test 1 ---
print("\nTEST 1 - single player, single league vs closed form")
n, ybar, sigma2, tau2, mu0 = 120.0, 148.0, 25465.0, 225.0, 130.0
d = Data(p=np.array([0]), l=np.array([0]), n=np.array([n]),
         y=np.array([ybar]), P=1, L=4)
# pin mu = mu0 (s0 -> 0) and sigma2 = sigma2 (IG concentrated), fix tau2
pr = Priors(m0=mu0, s0=1e-6, a_sig=1e9, b_sig=1e9 * sigma2)
out = gibbs(d, pr, model="M2", iters=200000, burn=20000, thin=2,
            rng=seeded("t1"), tau2_fixed=tau2)
th = out["theta"][:, 0]

prec  = n / sigma2 + 1.0 / tau2
m_cf  = (n * ybar / sigma2 + mu0 / tau2) / prec
sd_cf = 1.0 / np.sqrt(prec)
w     = (n * tau2) / (sigma2 + n * tau2)          # PS5 Q1(b) shrinkage weight
m_w   = w * ybar + (1 - w) * mu0

mcse = th.std(ddof=1) / np.sqrt(len(th))
print(f"    closed form  mean={m_cf:.4f}  sd={sd_cf:.4f}   w={w:.4f}  w-form mean={m_w:.4f}")
print(f"    sampler      mean={th.mean():.4f}  sd={th.std(ddof=1):.4f}   MCSE={mcse:.4f}")
check("closed-form mean equals shrinkage-weight form (exact)", abs(m_cf - m_w) < 1e-10,
      f"|diff|={abs(m_cf-m_w):.2e}")
check("sampler mean matches closed form within 4 MCSE",
      abs(th.mean() - m_cf) < 4 * mcse, f"|diff|={abs(th.mean()-m_cf):.4f}")
check("sampler sd matches closed form to 3 dp",
      abs(th.std(ddof=1) - sd_cf) < 5e-3 * sd_cf, f"|diff|={abs(th.std(ddof=1)-sd_cf):.4f}")

# ---------------------------------------------------------------- test 2 ---
print("\nTEST 2 - simulated recovery of theta, delta, sigma2, tau2, omega2")
g = seeded("t2")
P, L = 300, 4
TRUE = dict(mu=132.0, tau2=200.0, omega2=70.0, sigma2=25000.0)
theta_t = g.normal(TRUE["mu"], np.sqrt(TRUE["tau2"]), P)
delta_t = g.normal(0, np.sqrt(TRUE["omega2"]), L); delta_t[IPL] = 0.0
p_, l_, n_ = [], [], []
for i in range(P):                      # every player in IPL + 1-3 other leagues
    ls = [IPL] + list(g.choice([1, 2, 3], size=g.integers(1, 4), replace=False))
    for lg in ls:
        p_.append(i); l_.append(lg); n_.append(g.integers(25, 600))
p_, l_, n_ = np.array(p_), np.array(l_), np.array(n_, dtype=float)
y_ = theta_t[p_] + delta_t[l_] + g.normal(0, np.sqrt(TRUE["sigma2"] / n_))
sim = Data(p=p_, l=l_, n=n_, y=y_, P=P, L=L)

dr = run_chains(sim, Priors(), model="M3", chains=4, study="sim")
def ci(a): 
    f = np.asarray(a).reshape(-1) if np.asarray(a).ndim <= 2 else a
    return np.percentile(f, [2.5, 97.5])
for k in ("mu", "tau2", "omega2", "sigma2"):
    lo, hi = ci(dr[k]); t = TRUE[k]
    check(f"{k} recovered", lo <= t <= hi, f"true={t:.1f}  CI=({lo:.1f}, {hi:.1f})")
dl = dr["delta"].reshape(-1, L)
for j in range(1, L):
    lo, hi = np.percentile(dl[:, j], [2.5, 97.5])
    check(f"delta[{j}] recovered", lo <= delta_t[j] <= hi,
          f"true={delta_t[j]:.2f}  CI=({lo:.2f}, {hi:.2f})")
tht = dr["theta"].reshape(-1, P)
lo, hi = np.percentile(tht, [2.5, 97.5], axis=0)
cov = float(np.mean((lo <= theta_t) & (theta_t <= hi)))
check("theta 95% coverage in [0.90, 0.99]", 0.90 <= cov <= 0.99, f"coverage={cov:.3f}")

# ---------------------------------------------------------------- test 3 ---
print("\nTEST 3 - limiting cases (PS5 Q1(e))")
import dataio
D = dataio.load()
ipl = D["ipl"]
ybar, tot = dataio.weighted_mean_by_player(ipl)

# 3a. Complete pooling: M0's sampler vs the closed-form pooled posterior.
m0 = run_chains(ipl, Priors(), "M0", chains=4, study="lim0")
s2 = m0["sigma2"].reshape(-1).mean()
prec_cf = ipl.n.sum() / s2 + 1.0 / Priors().s0 ** 2
mean_cf = (np.sum(ipl.n * ipl.y) / s2 + Priors().m0 / Priors().s0 ** 2) / prec_cf
th0 = m0["theta"].reshape(-1, ipl.P)[:, 0]
mcse0 = th0.std(ddof=1) / np.sqrt(len(th0))
check("M0 matches the closed-form complete-pooling posterior",
      abs(th0.mean() - mean_cf) < 4 * mcse0,
      f"sampler={th0.mean():.3f}  closed form={mean_cf:.3f}  MCSE={mcse0:.3f}")

# 3b. tau -> 0: shrinkage is monotone in tau^2 and theta collapses on the pooled value.
taus, spread, dist = [400.0, 100.0, 25.0, 4.0, 1.0], [], []
for t2 in taus:
    dd = run_chains(ipl, Priors(), "M2", chains=2, iters=8000, burn=3000,
                    study=f"lim{t2}", tau2_fixed=t2)
    tm = dd["theta"].reshape(-1, ipl.P).mean(axis=0)
    spread.append(tm.std()); dist.append(abs(tm.mean() - mean_cf))
print("    tau^2 :", taus)
print("    sd(E[theta]) :", [round(v, 3) for v in spread])
check("posterior spread of theta decreases monotonically as tau^2 -> 0",
      all(spread[i] > spread[i + 1] for i in range(len(spread) - 1)),
      f"{[round(v,2) for v in spread]}")
check("theta collapses onto the complete-pooling value",
      spread[-1] < 0.05 * spread[0] and dist[-1] < 1.0,
      f"sd={spread[-1]:.3f} (from {spread[0]:.2f}); |mean - pooled|={dist[-1]:.3f}")

# 3c. tau -> infinity: no pooling. Compare on the Monte Carlo error scale.
dinf = run_chains(ipl, Priors(), "M2", chains=4, study="limI", tau2_fixed=1e8)
tin = dinf["theta"].reshape(-1, ipl.P)
z = np.abs(tin.mean(axis=0) - ybar) / (tin.std(axis=0, ddof=1) / np.sqrt(tin.shape[0]))
check("tau -> infinity reproduces the raw (no-pooling) means",
      np.max(z) < 4.5, f"max standardised deviation = {np.max(z):.2f} MCSE")

print("\n" + ("ALL CORRECTNESS TESTS PASSED" if not FAIL else f"FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
