"""
Gibbs samplers for the model ladder M0-M3.

Cells c = 1..C with player p[c], league l[c], balls n[c], strike rate y[c].

    M3   y[c] ~ N( theta[p[c]] + delta[l[c]], sigma2 / n[c] )
         theta[i] ~ N(mu, tau2)      delta[l] ~ N(0, omega2),  delta[IPL] = 0
         mu ~ N(m0, s0^2)   tau2 ~ IG(a_tau, b_tau)
         omega2 ~ IG(a_om, b_om)     sigma2 ~ IG(a_sig, b_sig)

All six full conditionals are closed form; there is no Metropolis step.
M2 is M3 restricted to IPL cells with delta identically zero.
M1 is no pooling (flat prior on each theta), M0 complete pooling (one theta).

Everything is vectorised over cells with np.bincount, so a 12,000-iteration
chain over 839 cells runs in a couple of seconds.
"""
import numpy as np
from dataclasses import dataclass, field

IPL = 0

# ----------------------------------------------------------------- priors ---
@dataclass
class Priors:
    m0: float = 130.0      # mu ~ N(m0, s0^2)
    s0: float = 20.0
    a_tau: float = 3.0     # tau2 ~ IG(a_tau, b_tau)   E[tau2] = b/(a-1) = 225 -> sd 15
    b_tau: float = 450.0
    a_om: float = 3.0      # omega2 ~ IG(a_om, b_om)   E[omega2] = 64 -> sd 8
    b_om: float = 128.0
    a_sig: float = 3.0     # sigma2 ~ IG(a_sig, b_sig) E[sigma2] = 26000
    b_sig: float = 52000.0


@dataclass
class Data:
    p: np.ndarray          # player index per cell
    l: np.ndarray          # league index per cell
    n: np.ndarray          # balls faced
    y: np.ndarray          # strike rate
    P: int                 # number of players
    L: int                 # number of leagues

    @property
    def C(self) -> int:
        return len(self.y)


def inv_gamma(rng, a, b, size=None):
    """X ~ IG(a, b)  <=>  1/X ~ Gamma(shape=a, rate=b)."""
    return 1.0 / rng.gamma(shape=a, scale=1.0 / b, size=size)


# --------------------------------------------------------------- sampler ---
def gibbs(data: Data, priors: Priors, model: str = "M3",
          iters: int = 12000, burn: int = 4000, thin: int = 4,
          rng: np.random.Generator = None,
          tau2_fixed: float = None, sigma2_fixed: float = None,
          keep_theta: bool = True):
    """Run one chain. Returns a dict of draws (post burn-in, thinned)."""
    rng = rng or np.random.default_rng(0)
    P, L, C = data.P, data.L, data.C
    p, l, n, y = data.p, data.l, data.n, data.y

    n_by_player = np.bincount(p, weights=n, minlength=P)
    has_data = n_by_player > 0

    # --- initialise -------------------------------------------------------
    ybar = np.divide(np.bincount(p, weights=n * y, minlength=P), n_by_player,
                     out=np.full(P, float(priors.m0)), where=has_data)
    theta = np.where(has_data, ybar, priors.m0).astype(float)
    delta = np.zeros(L)
    mu = float(np.mean(theta[has_data])) if has_data.any() else priors.m0
    tau2 = 225.0 if tau2_fixed is None else tau2_fixed
    omega2 = 64.0
    sigma2 = 25000.0 if sigma2_fixed is None else sigma2_fixed

    if model == "M0":                      # complete pooling: one shared theta
        theta = np.full(P, mu)

    keep = np.arange(burn, iters, thin)
    K = len(keep)
    out = {k: np.empty(K) for k in ("mu", "tau2", "omega2", "sigma2")}
    out["theta"] = np.empty((K, P)) if keep_theta else None
    out["delta"] = np.empty((K, L))
    out["loglik"] = np.empty((K, C))
    k = 0

    for it in range(iters):
        # ---- 1. theta ----------------------------------------------------
        if model == "M0":
            # single theta: precision-weighted over all cells + prior on mu
            prec = n.sum() / sigma2 + 1.0 / priors.s0**2
            mean = (np.sum(n * (y - delta[l])) / sigma2 + priors.m0 / priors.s0**2) / prec
            th0 = mean + rng.normal() / np.sqrt(prec)
            theta[:] = th0
            mu = th0
        else:
            r = y - delta[l]
            Sn = np.bincount(p, weights=n, minlength=P) / sigma2
            Sr = np.bincount(p, weights=n * r, minlength=P) / sigma2
            if model == "M1":              # no pooling: flat prior on theta
                prec = np.where(Sn > 0, Sn, 1.0)
                mean = np.where(Sn > 0, Sr / prec, priors.m0)
                theta = mean + rng.normal(size=P) / np.sqrt(prec)
            else:                          # M2 / M3: hierarchical
                prec = Sn + 1.0 / tau2
                mean = (Sr + mu / tau2) / prec
                theta = mean + rng.normal(size=P) / np.sqrt(prec)

        # ---- 2. delta (M3 only; delta[IPL] pinned to 0) -------------------
        if model == "M3":
            s = y - theta[p]
            Sn_l = np.bincount(l, weights=n, minlength=L) / sigma2
            Ss_l = np.bincount(l, weights=n * s, minlength=L) / sigma2
            for j in range(L):
                if j == IPL:
                    delta[j] = 0.0
                    continue
                prec = Sn_l[j] + 1.0 / omega2
                delta[j] = Ss_l[j] / prec + rng.normal() / np.sqrt(prec)

        # ---- 3. mu -------------------------------------------------------
        if model in ("M2", "M3"):
            prec = P / tau2 + 1.0 / priors.s0**2
            mean = (theta.sum() / tau2 + priors.m0 / priors.s0**2) / prec
            mu = mean + rng.normal() / np.sqrt(prec)

        # ---- 4. tau2 -----------------------------------------------------
        if model in ("M2", "M3") and tau2_fixed is None:
            tau2 = inv_gamma(rng, priors.a_tau + P / 2.0,
                             priors.b_tau + 0.5 * np.sum((theta - mu) ** 2))

        # ---- 5. omega2 ---------------------------------------------------
        if model == "M3":
            free = np.array([j for j in range(L) if j != IPL])
            omega2 = inv_gamma(rng, priors.a_om + len(free) / 2.0,
                               priors.b_om + 0.5 * np.sum(delta[free] ** 2))

        # ---- 6. sigma2 ---------------------------------------------------
        if sigma2_fixed is None:
            resid = y - theta[p] - delta[l]
            sigma2 = inv_gamma(rng, priors.a_sig + C / 2.0,
                               priors.b_sig + 0.5 * np.sum(n * resid ** 2))

        # ---- store -------------------------------------------------------
        if it >= burn and (it - burn) % thin == 0 and k < K:
            out["mu"][k] = mu
            out["tau2"][k] = tau2
            out["omega2"][k] = omega2
            out["sigma2"][k] = sigma2
            out["delta"][k] = delta
            if keep_theta:
                out["theta"][k] = theta
            resid = y - theta[p] - delta[l]
            out["loglik"][k] = (-0.5 * np.log(2 * np.pi * sigma2 / n)
                                - n * resid ** 2 / (2 * sigma2))
            k += 1

    return out


def run_chains(data, priors, model="M3", chains=4, iters=12000, burn=4000,
               thin=4, study="main", tau2_fixed=None, sigma2_fixed=None):
    """Run `chains` independent chains and stack the draws."""
    from rng import chain_rng
    res = [gibbs(data, priors, model, iters, burn, thin,
                 chain_rng(model, c, study), tau2_fixed, sigma2_fixed)
           for c in range(chains)]
    stacked = {}
    for key in ("mu", "tau2", "omega2", "sigma2"):
        stacked[key] = np.stack([r[key] for r in res])            # (chains, K)
    for key in ("theta", "delta", "loglik"):
        if res[0][key] is not None:
            stacked[key] = np.stack([r[key] for r in res])        # (chains,K,·)
    return stacked
