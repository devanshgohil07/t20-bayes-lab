"""Every figure in the report. Run from analysis/:  python3 figures.py"""
import numpy as np, json, os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
import dataio, diagnostics as dg
from style import *
from gibbs import Priors, IPL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, FIGS = os.path.join(ROOT, "out"), os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)

D = dataio.load()
z = np.load(os.path.join(OUT, "draws.npz"))
full, ipl = D["full"], D["ipl"]
P3 = lambda k: z[f"M3_{k}"].reshape(-1, *z[f"M3_{k}"].shape[2:])
th3 = P3("theta"); dl3 = P3("delta"); s23 = P3("sigma2"); mu3 = P3("mu"); tau3 = P3("tau2")
raw_ipl, ipl_balls = dataio.weighted_mean_by_player(ipl)

# Per-ball variance on the strike-rate scale, taken from the data rather than hard coded.
_br = json.load(open(os.path.join(ROOT, "data", "ballruns.json")))
_v = np.array(_br["values"], float); _c = np.array(_br["counts"], float)
_m1 = (_v * _c).sum() / _c.sum()
SIGMA2_BALL = ((_v ** 2 * _c).sum() / _c.sum() - _m1 ** 2) * 1e4

# ============================================================ F1  funnel ===
def f1():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0),
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    iplmean = np.sum(ipl.n * ipl.y) / np.sum(ipl.n)
    sig = np.sqrt(SIGMA2_BALL)
    nn = np.logspace(np.log10(20), np.log10(ipl.n.max() * 1.2), 200)
    other = full.l != IPL
    ax.scatter(full.n[other], full.y[other], s=6, color="#c9ced6", alpha=.45, lw=0,
               label="CPL / BBL / T20I")
    ax.scatter(ipl.n, ipl.y, s=16, color=LEAGUE["ipl"], alpha=.85, lw=0, label="IPL")
    for k, ls in ((1.96, "-"), (1.0, (0, (4, 3)))):
        ax.plot(nn, iplmean + k * sig / np.sqrt(nn), color=ACCENT, lw=1.2, ls=ls)
        ax.plot(nn, iplmean - k * sig / np.sqrt(nn), color=ACCENT, lw=1.2, ls=ls)
    ax.axhline(iplmean, color=INK, lw=1.0, ls="--")
    ax.set_xscale("log"); ax.set_ylim(20, 265)
    ax.set_xlabel("balls faced, 2021–2024 (log scale)")
    ax.set_ylabel("strike rate")
    ax.legend(fontsize=8.5, loc="lower right", markerscale=1.8, handletextpad=.4)
    ax.annotate("±2 sampling SD\nat 40 balls",
                xy=(40, iplmean + 1.96 * sig / np.sqrt(40)), xytext=(150, 243),
                fontsize=8.5, color=ACCENT, ha="left",
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=.8))
    ax = axes[1]
    o = np.argsort(-ipl.y)[:15]
    yy = np.arange(len(o))[::-1]
    ax.barh(yy, ipl.n[o], color=LEAGUE["ipl"], alpha=.85, height=.72)
    for i, k in enumerate(o):
        ax.text(ipl.n[k] + 8, yy[i], f"SR {ipl.y[k]:.0f}", va="center", fontsize=8,
                color=MUTED)
    ax.set_yticks(yy); ax.set_yticklabels([f"#{i+1}" for i in range(len(o))], fontsize=8)
    ax.set_xlabel("IPL balls faced behind that strike rate")
    ax.set_xlim(0, max(ipl.n[o]) * 1.5)
    med = int(np.median(ipl.n[o]))
    noise = np.sqrt(SIGMA2_BALL / med)
    ax.set_title("The top of the raw leaderboard", loc="left", fontsize=9.5, color=INK, pad=6)
    fig.tight_layout()
    figtitle(fig, "F1 · Strike rate against balls faced",
             "Red curves: ±1 and ±2 sampling SD around the IPL average.")
    ax.set_xlabel(f"IPL balls faced (median {med}, worth ±{noise:.0f} SR of noise)")
    save(fig, "F1_funnel", FIGS)

# =============================================================== F2  DAG ===
def f2():
    fig, ax = plt.subplots(figsize=(7.4, 4.0)); ax.axis("off"); ax.grid(False)
    def node(x, y, txt, kind="param", w=.115, h=.105):
        fc = "#eef2f7" if kind == "param" else ("#dde6ef" if kind == "data" else PAPER)
        ec = LEAGUE["ipl"] if kind != "hyper" else MUTED
        ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, mutation_scale=.02,
                                    boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=1.1))
        ax.text(x, y, txt, ha="center", va="center", fontsize=10)
        return (x, y, w, h)
    def arrow(a, b):
        ax.add_patch(FancyArrowPatch((a[0], a[1] - a[3]/2), (b[0], b[1] + b[3]/2),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=MUTED, lw=1.0, shrinkA=1, shrinkB=1))
    m0 = node(.13, .88, "$m_0, s_0^2$", "hyper"); at = node(.35, .88, "$a_\\tau, b_\\tau$", "hyper")
    ao = node(.66, .88, "$a_\\omega, b_\\omega$", "hyper"); asg = node(.88, .88, "$a_\\sigma, b_\\sigma$", "hyper")
    mu = node(.13, .62, "$\\mu$"); tau = node(.35, .62, "$\\tau^2$")
    om = node(.66, .62, "$\\omega^2$"); sg = node(.88, .62, "$\\sigma^2$")
    th = node(.24, .36, "$\\theta_i$"); de = node(.66, .36, "$\\delta_\\ell$")
    y  = node(.45, .10, "$y_{i\\ell}$", "data")
    for a, b in ((m0, mu), (at, tau), (ao, om), (asg, sg), (mu, th), (tau, th),
                 (om, de), (th, y), (de, y), (sg, y)):
        arrow(a, b)
    ax.add_patch(Rectangle((.10, .265), .30, .19, fill=False, ec=MUTED, lw=.9, ls=(0,(4,3))))
    ax.text(.385, .285, "players $i=1..P$", fontsize=8, color=MUTED, ha="right")
    ax.add_patch(Rectangle((.55, .265), .24, .19, fill=False, ec=MUTED, lw=.9, ls=(0,(4,3))))
    ax.text(.775, .285, "leagues $\\ell \\neq$ IPL", fontsize=8, color=MUTED, ha="right")
    ax.add_patch(Rectangle((.28, .03), .34, .155, fill=False, ec=MUTED, lw=.9, ls=(0,(4,3))))
    ax.text(.605, .048, "cells $i \\times \\ell$", fontsize=8, color=MUTED, ha="right")
    ax.text(.02, .10, "what we\nobserve", fontsize=8.5, color=MUTED, ha="left", va="center")
    ax.text(.02, .36, "what we\nwant", fontsize=8.5, color=ACCENT, ha="left", va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    title(ax, "F2 · Directed acyclic graph for M3",
          "Arrows read 'depends on'. Fixing $\\delta_{\\rm IPL}=0$ puts every $\\theta_i$ on the IPL scale.")
    save(fig, "F2_dag", FIGS)

# ================================================== F3  prior predictive ===
def f3():
    pr, g = Priors(), np.random.default_rng(3)
    S = 3000
    mu = g.normal(pr.m0, pr.s0, S)
    tau2 = 1 / g.gamma(pr.a_tau, 1 / pr.b_tau, S)
    om2 = 1 / g.gamma(pr.a_om, 1 / pr.b_om, S)
    s2 = 1 / g.gamma(pr.a_sig, 1 / pr.b_sig, S)
    idx = g.integers(0, full.C, S)
    theta = mu + np.sqrt(tau2) * g.normal(size=S)
    delta = np.where(full.l[idx] == IPL, 0.0, np.sqrt(om2) * g.normal(size=S))
    ysim = theta + delta + np.sqrt(s2 / full.n[idx]) * g.normal(size=S)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    ax = axes[0]
    ax.hist(ysim, bins=60, range=(0, 300), color=LEAGUE["ipl"], alpha=.5, density=True,
            label="simulated from the prior")
    ax.hist(full.y, bins=60, range=(0, 300), histtype="step", color=ACCENT, lw=1.6,
            density=True, label="actually observed")
    ax.set_xlabel("strike rate"); ax.set_yticks([])
    ax.legend(fontsize=8.5)
    ax = axes[1]
    ax.hist(np.sqrt(tau2), bins=50, color=LEAGUE["bbl"], alpha=.6, density=True)
    ax.axvline(np.sqrt(np.median(tau2)), color=INK, lw=1.1, ls="--")
    ax.set_xlabel("prior spread of player ability, $\\tau$ (strike-rate points)")
    ax.set_yticks([]); ax.set_xlim(0, 60)
    fig.tight_layout()
    figtitle(fig, "F3 · Prior predictive check",
             "Cell strike rates simulated from the priors alone, before seeing any data. "
             "Prior median $\\tau$ = %.1f." % np.sqrt(np.median(tau2)), top=0.80)
    save(fig, "F3_prior_predictive", FIGS)

# ============================================================ F4 traces ====
PARAMS = [("mu", "$\\mu$  (population mean, IPL scale)"),
          ("tau2", "$\\tau^2$  (between-player variance)"),
          ("omega2", "$\\omega^2$  (between-league variance)"),
          ("sigma2", "$\\sigma^2$  (effective per-ball variance)")]

def f4():
    fig, axes = plt.subplots(4, 2, figsize=(7.6, 7.2),
                             gridspec_kw=dict(width_ratios=[3, 1]))
    for r, (k, lab) in enumerate(PARAMS):
        x = z[f"M3_{k}"]
        for c in range(x.shape[0]):
            axes[r, 0].plot(x[c], lw=.5, alpha=.75, color=LCOL[c])
        axes[r, 0].set_ylabel(lab, fontsize=9)
        axes[r, 1].hist(x.reshape(-1), bins=45, orientation="horizontal",
                        color=MUTED, alpha=.6)
        axes[r, 1].set_ylim(axes[r, 0].get_ylim()); axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
        axes[r, 0].text(.985, .93, f"$\\hat R$ = {dg.split_rhat(x):.4f}   ESS = {dg.ess(x):.0f}",
                        transform=axes[r, 0].transAxes, ha="right", fontsize=8, color=MUTED)
        if r < 3:
            axes[r, 0].set_xticklabels([])
    axes[3, 0].set_xlabel("draw (after 4,000 burn-in, thinned by 4)")
    fig.tight_layout()
    figtitle(fig, "F4 · Trace plots for the four global parameters (M3)",
             "Each colour is one chain; the panel on the right is the pooled posterior.",
             top=0.93)
    save(fig, "F4_traces", FIGS)

# ============================================================== F5  ACF ====
def f5():
    fig, axes = plt.subplots(1, 4, figsize=(8.2, 2.6), sharey=True)
    for a, (k, lab) in zip(axes, PARAMS):
        r = dg.acf(z[f"M3_{k}"], 40)
        a.bar(np.arange(len(r)), r, color=LEAGUE["ipl"], width=.8)
        a.axhline(0, color=MUTED, lw=.8)
        a.set_title(lab.split("  ")[0], fontsize=10)
        a.set_xlabel("lag")
        a.set_ylim(-.15, 1.02)
    axes[0].set_ylabel("autocorrelation", fontsize=9)
    fig.tight_layout()
    figtitle(fig, "F5 · Autocorrelation of the thinned draws (M3)",
             "Values near zero by lag 5 indicate that the sampler is mixing well.", top=0.74)
    save(fig, "F5_acf", FIGS)

# ============================================================== F6  PPC ====
def f6():
    d = np.load(os.path.join(OUT, "ppc_M3.npz"))
    yrep, y, n = d["yrep"], d["y"], d["n"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    ax = axes[0]
    for r in yrep[:60]:
        ax.scatter(n, r, s=4, color=MUTED, alpha=.06, lw=0)
    ax.scatter(n, y, s=7, color=ACCENT, alpha=.8, lw=0)
    ax.set_xscale("log"); ax.set_xlabel("balls faced (log scale)"); ax.set_ylabel("strike rate")
    ax.axhline(0, color=INK, lw=.8, ls=":")
    ax.set_ylim(-60, 280)
    ax.text(.98, .04, "grey = 60 replicated datasets\nred = the real one", transform=ax.transAxes,
            ha="right", fontsize=8.5, color=MUTED)
    ax = axes[1]
    reps = np.array([np.std(r[n < 100]) for r in yrep])
    ax.hist(reps, bins=35, color=MUTED, alpha=.6)
    ax.axvline(np.std(y[n < 100]), color=ACCENT, lw=2)
    ax.set_xlabel("SD of strike rate among cells under 100 balls")
    ax.set_yticks([])
    ax.text(.03, .93, "the model expects short spells to be\nnoisier than they are",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")
    fig.tight_layout()
    figtitle(fig, "F6 · Posterior predictive check",
             "Left: data the fitted model would generate, against the data observed. "
             "Right: one test statistic, with the observed value in red.", top=0.83)
    save(fig, "F6_ppc", FIGS)

# ========================================================= F7  shrinkage ===
def f7(topk=22):
    gid = D["ipl_players"]
    post = th3[:, gid].mean(axis=0)
    raw = raw_ipl
    rank_raw = (-raw).argsort().argsort() + 1
    rank_post = (-post).argsort().argsort() + 1
    sel = np.argsort(-raw)[:topk]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 5.6),
                             gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axes[0]
    yy = np.arange(len(sel))[::-1]
    mu_bar = mu3.mean()
    for k, i in enumerate(sel):
        y0 = yy[k]
        col = ACCENT if post[i] < raw[i] else LEAGUE["ipl"]
        ax.annotate("", xy=(post[i], y0), xytext=(raw[i], y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3, mutation_scale=9))
        ax.scatter(raw[i], y0, s=16, color=MUTED, zorder=3)
        ax.scatter(post[i], y0, s=26, color=col, zorder=4)
        ax.text(252, y0, f"#{rank_raw[i]} → #{rank_post[i]}", fontsize=7.6, va="center",
                color=col)
    ax.axvline(mu_bar, color=INK, lw=1, ls="--")
    ax.text(mu_bar - 2, len(sel) - .3, "$\\mu$", fontsize=10, ha="right")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{D['players'][gid[i]]}  ·  {ipl_balls[i]:.0f} balls" for i in sel],
                       fontsize=7.6)
    ax.set_xlim(95, 300); ax.set_xticks([100, 150, 200, 250])
    ax.set_xlabel("strike rate")
    ax.grid(axis="y", visible=False)
    fell = int(np.sum(rank_post[sel] > rank_raw[sel]))
    ax.set_title("Raw strike rate and posterior mean", loc="left", fontsize=9.5, color=INK,
                 pad=6)

    ax = axes[1]
    tot = np.bincount(full.p, weights=full.n, minlength=full.P)[gid]
    dhat = dl3.mean(axis=0)
    adj_num = np.bincount(full.p, weights=full.n * (full.y - dhat[full.l]), minlength=full.P)[gid]
    adj = adj_num / tot
    mu_bar = mu3.mean()
    realised = np.clip((post - mu_bar) / np.where(np.abs(adj - mu_bar) < 1e-9, np.nan, adj - mu_bar),
                       0, 1.2)
    nn = np.logspace(np.log10(tot.min()), np.log10(tot.max()), 200)
    w = nn * tau3.mean() / (s23.mean() + nn * tau3.mean())
    ax.plot(nn, w, color=LEAGUE["ipl"], lw=2, zorder=3,
            label="$w = n\\tau^2/(\\sigma^2 + n\\tau^2)$")
    ax.scatter(tot, realised, s=13, color=MUTED, alpha=.55, lw=0,
               label="each IPL batter, realised")
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("total balls faced across all four leagues (log scale)")
    ax.set_ylabel("weight on the batter's own record")
    ax.legend(fontsize=8.5, loc="upper left")
    for nb, dy in ((50, .14), (200, -.11), (800, -.13)):
        wv = nb * tau3.mean() / (s23.mean() + nb * tau3.mean())
        ax.annotate(f"{nb} balls → {wv:.0%} own record", xy=(nb, wv),
                    xytext=(nb * 1.7, wv + dy), fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", lw=.7, color=MUTED))
    ax.set_title("Shrinkage weight, theory and practice", loc="left", fontsize=9.5,
                 color=INK, pad=6)
    fig.tight_layout()
    figtitle(fig, "F7 · The leaderboard after partial pooling",
             f"Top {topk} by raw IPL strike rate. Grey = raw, coloured = posterior mean. "
             f"{fell} of {topk} move down.", top=0.86)
    save(fig, "F7_shrinkage", FIGS)

# =========================================================== F8  offsets ===
def f8():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    from scipy.stats import gaussian_kde
    xs = np.linspace(-20, 8, 400)
    for j in (1, 2, 3):
        k = gaussian_kde(dl3[:, j])
        ax.fill_between(xs, k(xs), color=LCOL[j], alpha=.32, lw=0)
        ax.plot(xs, k(xs), color=LCOL[j], lw=1.7)
    ax.axvline(0, color=INK, lw=1.4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
    top = ax.get_ylim()[1]
    # One label per league, stacked so that they can never collide with each other.
    for r, j in enumerate((1, 2, 3)):
        m = float(dl3[:, j].mean()); lo, hi = np.percentile(dl3[:, j], [2.5, 97.5])
        ax.text(-19.5, top * (0.95 - 0.085 * r),
                f"{LNAME[j]}   {m:+.1f}   [{lo:+.1f}, {hi:+.1f}]",
                fontsize=9, color=LCOL[j], va="top", fontweight="600")
    ax.text(0.4, top * 0.95, "IPL = 0 by construction", fontsize=8.5, color=INK, va="top")
    ax.set_xlabel("league scoring-environment offset $\\delta_\\ell$  (strike-rate points vs IPL)")
    ax.set_yticks([])
    title(ax, "F8 · What a strike rate is worth in each league",
          "Posterior for $\\delta_\\ell$: how much a league's numbers must be adjusted to read as IPL numbers.")
    save(fig, "F8_offsets", FIGS)

# ============================================================ F9  RMSE =====
def f9():
    rows = json.load(open(os.path.join(OUT, "validation.json")))
    order = ["0 balls, overseas record", "0 balls, no record", "1-100", "101-300",
             "300+", "ALL with IPL record"]
    labels = ["no IPL record,\nbut overseas data", "no record\nanywhere", "1–100\nIPL balls",
              "101–300\nIPL balls", "300+\nIPL balls", "all with\nIPL record"]
    models = ["M0", "M1", "M2", "M3"]
    cols = [MUTED, "#9aa3ad", LEAGUE["bbl"], ACCENT]
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    W = .2
    for k, m in enumerate(models):
        xs, hs, es = [], [], []
        for i, b in enumerate(order):
            r = next(x for x in rows if x["bucket"] == b)
            if r[m] is None:
                ax.text(i + (k - 1.5) * W, 2, "no\nestimate", ha="center", fontsize=7,
                        color=ACCENT, rotation=90, va="bottom")
                continue
            xs.append(i + (k - 1.5) * W); hs.append(r[m]["rmse"]); es.append(r[m]["rmse_se"])
        ax.bar(xs, hs, W * .92, yerr=es, color=cols[k], label=m,
               error_kw=dict(lw=.8, ecolor=MUTED, capsize=2))
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=8.3)
    ax.set_ylabel("RMSE on 2025 IPL strike rate")
    ax.legend(ncol=4, fontsize=8.5, loc="upper right")
    title(ax, "F9 · Predictive accuracy by prior IPL exposure",
          "Predicting 2025 from 2021–2024. Error bars are ±1 bootstrap standard error.")
    save(fig, "F9_validation", FIGS)

# ========================================================= F10 shortlist ===
def f10(threshold=140.0, k=18):
    gid = np.arange(len(D["players"]))
    tot = np.bincount(full.p, weights=full.n, minlength=full.P)
    elig = tot >= 150
    prob = (th3 > threshold).mean(axis=0)
    cand = np.where(elig)[0]
    top = cand[np.argsort(-prob[cand])][:k][::-1]
    m = th3[:, top].mean(axis=0)
    lo, hi = np.percentile(th3[:, top], [2.5, 97.5], axis=0)
    names = [D["players"][i] for i in top]
    rawv = np.full(len(top), np.nan)
    loc = D["ipl_remap"][top]
    rawv[loc >= 0] = raw_ipl[loc[loc >= 0]]

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ypos = np.arange(len(top))
    ax.hlines(ypos, lo, hi, color=MUTED, lw=1.6, alpha=.8)
    ax.scatter(m, ypos, s=26, color=ACCENT, zorder=3, label="posterior mean $\\theta_i$")
    ok = ~np.isnan(rawv)
    ax.scatter(rawv[ok], ypos[ok], s=22, marker="|", color=LEAGUE["ipl"], zorder=3,
               label="raw IPL strike rate")
    ax.axvline(threshold, color=INK, lw=1.2, ls="--")
    ax.text(threshold + 1, len(top) - .4, f"threshold {threshold:.0f}", fontsize=8.5)
    for i, gi in enumerate(top):
        ax.text(ax.get_xlim()[1], i, f"  {prob[gi]:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("strike rate on the IPL scale")
    ax.legend(fontsize=8.5, loc="lower left", framealpha=.9, facecolor=PAPER,
              edgecolor="none")
    title(ax, "F10 · The shortlist, with credible intervals",
          f"Top {k} by $P(\\theta_i > {threshold:.0f} \\mid$ data$)$, shown at right. Bars are 95% credible intervals.")
    save(fig, "F10_shortlist", FIGS)

if __name__ == "__main__":
    for f in (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10):
        f()
    print("all figures written")
