"""Every table in the report, written as standalone LaTeX fragments into tables/.

T1 to T7, where T1 and T2 each split into two fragments, so nine files in all.
Numbers quoted in the prose are handled separately by report_numbers.py."""
import numpy as np, json, os
import dataio, diagnostics as dg
from gibbs import Priors, IPL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, TAB = os.path.join(ROOT, "out"), os.path.join(ROOT, "tables")
os.makedirs(TAB, exist_ok=True)
D = dataio.load()
z = np.load(os.path.join(OUT, "draws.npz"))
full, ipl = D["full"], D["ipl"]
LN = ["IPL", "CPL", "BBL", "T20I"]

def write(name, body):
    open(os.path.join(TAB, name + ".tex"), "w").write(body)
    print("  wrote tables/" + name + ".tex")

# ------------------------------------------------------------------- T1 ---
def t1():
    lines = []
    for j in range(4):
        m = full.l == j
        lines.append((LN[j], int(m.sum()), int(full.n[m].sum()),
                      float(np.sum(full.n[m] * full.y[m]) / full.n[m].sum()),
                      int(np.median(full.n[m])), float(np.min(full.y[m])), float(np.max(full.y[m]))))
    piv = np.zeros((4, 4), int)
    for a in range(4):
        pa = set(full.p[full.l == a])
        for b in range(4):
            piv[a, b] = len(pa & set(full.p[full.l == b]))
    s = ["\\begin{tabular}{lrrrrr}", "\\toprule",
         "League & Cells & Balls & Run rate & Median balls & Range of cell SR \\\\", "\\midrule"]
    for r in lines:
        s.append(f"{r[0]} & {r[1]:,} & {r[2]:,} & {r[3]:.1f} & {r[4]:,} & {r[5]:.0f}--{r[6]:.0f} \\\\")
    s += [f"\\midrule All & {full.C:,} & {int(full.n.sum()):,} & "
          f"{np.sum(full.n*full.y)/full.n.sum():.1f} & {int(np.median(full.n)):,} & "
          f"{full.y.min():.0f}--{full.y.max():.0f} \\\\", "\\bottomrule", "\\end{tabular}"]
    o = ["\\begin{tabular}{l" + "r" * 4 + "}", "\\toprule",
         " & " + " & ".join(LN) + " \\\\", "\\midrule"]
    for a in range(4):
        o.append(LN[a] + " & " + " & ".join(
            (f"\\textbf{{{piv[a,b]}}}" if a == b else str(piv[a, b])) for b in range(4)) + " \\\\")
    o += ["\\bottomrule", "\\end{tabular}"]
    write("T1a_data", "\n".join(s)); write("T1b_overlap", "\n".join(o))

# ------------------------------------------------------------------- T2 ---
def t2():
    p = Priors()
    br = json.load(open(os.path.join(ROOT, "data", "ballruns.json")))
    v = np.array(br["values"], float); c = np.array(br["counts"], float)
    m1 = (v * c).sum() / c.sum(); m2 = (v**2 * c).sum() / c.sum(); var = m2 - m1**2
    rows = [
        ("$\\mu$", f"$N({p.m0:.0f}, {p.s0:.0f}^2)$",
         "T20 batting averages a strike rate near 130; $\\pm40$ covers anything sane."),
        ("$\\tau^2$", f"$\\mathrm{{IG}}({p.a_tau:.0f}, {p.b_tau:.0f})$",
         f"Prior mean {p.b_tau/(p.a_tau-1):.0f}, i.e.\\ $\\tau\\approx{np.sqrt(p.b_tau/(p.a_tau-1)):.0f}$: "
         "most batters within $\\pm30$ of the mean."),
        ("$\\omega^2$", f"$\\mathrm{{IG}}({p.a_om:.0f}, {p.b_om:.0f})$",
         f"Prior mean {p.b_om/(p.a_om-1):.0f}, i.e.\\ $\\omega\\approx{np.sqrt(p.b_om/(p.a_om-1)):.0f}$: "
         "leagues differ by single-digit strike-rate points."),
        ("$\\sigma^2$", f"$\\mathrm{{IG}}({p.a_sig:.0f}, {p.b_sig:,.0f})$",
         f"Prior mean {p.b_sig/(p.a_sig-1):,.0f}, set near the value implied by the ball-by-ball "
         f"run distribution: $\\mathrm{{Var}}(X)={var:.2f}$ runs per ball, so "
         f"$\\sigma^2\\approx{var*1e4:,.0f}$ if balls were independent."),
    ]
    s = ["\\begin{tabular}{llp{8.2cm}}", "\\toprule",
         "Parameter & Prior & Where the number comes from \\\\", "\\midrule"]
    s += [f"{a} & {b} & {c} \\\\[2pt]" for a, b, c in rows]
    s += ["\\bottomrule", "\\end{tabular}"]
    write("T2_priors", "\n".join(s))
    counts = {int(k): int(x) for k, x in zip(br["values"], br["counts"])}
    tot = sum(counts.values())
    d = ["\\begin{tabular}{lrrrrrrr}", "\\toprule", "Runs off the bat & " +
         " & ".join(str(k) for k in sorted(counts)) + " \\\\", "\\midrule",
         "Share of balls & " + " & ".join(f"{counts[k]/tot:.3f}" for k in sorted(counts)) + " \\\\",
         "\\bottomrule", "\\end{tabular}"]
    write("T2b_ballruns", "\n".join(d))
    return m1, var

# ------------------------------------------------------------------- T3 ---
def t3():
    rows = []
    names = [("mu", "$\\mu$"), ("tau2", "$\\tau^2$"), ("omega2", "$\\omega^2$"),
             ("sigma2", "$\\sigma^2$")]
    for k, lab in names:
        x = z[f"M3_{k}"]
        rows.append((lab, x.reshape(-1).mean(), x.reshape(-1).std(ddof=1),
                     *np.percentile(x, [2.5, 97.5]), dg.split_rhat(x), dg.ess(x)))
    for j in (1, 2, 3):
        x = z["M3_delta"][:, :, j]
        rows.append((f"$\\delta_{{\\mathrm{{{LN[j]}}}}}$", x.reshape(-1).mean(),
                     x.reshape(-1).std(ddof=1), *np.percentile(x, [2.5, 97.5]),
                     dg.split_rhat(x), dg.ess(x)))
    th = z["M3_theta"]
    # taken from run_main so that the table and the prose quote the same numbers
    dsum = json.load(open(os.path.join(OUT, "diag_summary.json")))
    s = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         "Parameter & Mean & SD & 2.5\\% & 97.5\\% & $\\hat{R}$ & ESS \\\\", "\\midrule"]
    for r in rows:
        s.append(f"{r[0]} & {r[1]:,.2f} & {r[2]:,.2f} & {r[3]:,.2f} & {r[4]:,.2f} "
                 f"& {r[5]:.4f} & {r[6]:,.0f} \\\\")
    s += ["\\midrule",
          f"all {th.shape[2]} $\\theta_i$ (worst) & \\multicolumn{{4}}{{l}}{{}} & "
          f"{dsum['rhatWorst']:.4f} & {dsum['essMin']:,.0f} \\\\",
          "\\bottomrule", "\\end{tabular}"]
    write("T3_diagnostics", "\n".join(s))
    return dsum["rhatWorst"], dsum["essMin"]

# ------------------------------------------------------------------- T4 ---
def t4():
    rows = json.load(open(os.path.join(OUT, "compare.json")))
    best = max(r["elpd_waic"] for r in rows)
    desc = {"M0": "complete pooling", "M1": "no pooling", "M2": "partial pooling, IPL only",
            "M3": "partial pooling + league offsets"}
    s = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
         "Model & & $\\widehat{\\mathrm{elpd}}_{\\mathrm{WAIC}}$ & $p_{\\mathrm{WAIC}}$ & "
         "$\\Delta$WAIC & $\\widehat{\\mathrm{elpd}}_{\\mathrm{LOO}}$ & LOOIC & "
         "cells with $\\hat k>0.7$ \\\\", "\\midrule"]
    for r in rows:
        s.append(f"{r['model']} & {desc[r['model']]} & {r['elpd_waic']:.1f} & {r['p_waic']:.1f} & "
                 f"{2*(best-r['elpd_waic']):.1f} & {r['elpd_loo']:.1f} & {r['looic']:.1f} & "
                 f"{r['khat_bad']} / 194 \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    write("T4_waic_loo", "\n".join(s))

# ------------------------------------------------------------------- T5 ---
def t5():
    rows = json.load(open(os.path.join(OUT, "validation.json")))
    order = [("0 balls, overseas record", "No IPL record, has overseas data"),
             ("0 balls, no record", "No record in any of the four leagues"),
             ("1-100", "1--100 IPL balls"), ("101-300", "101--300 IPL balls"),
             ("300+", "300+ IPL balls"), ("ALL with IPL record", "All with an IPL record"),
             ("ALL", "All 103 holdout batters")]
    s = ["\\begin{tabular}{lrrrrr}", "\\toprule",
         "Prior IPL exposure & $n$ & M0 & M1 & M2 & M3 \\\\",
         "& & \\multicolumn{4}{c}{RMSE (bootstrap SE)} \\\\", "\\midrule"]
    for key, lab in order:
        r = next(x for x in rows if x["bucket"] == key)
        cells = []
        best = min(x[1]["rmse"] for x in [(m, r[m]) for m in ("M0","M1","M2","M3")] if x[1])
        for m in ("M0", "M1", "M2", "M3"):
            v = r[m]
            if v is None:
                cells.append("\\emph{undefined}")
            else:
                t = f"{v['rmse']:.1f} ({v['rmse_se']:.1f})"
                cells.append("\\textbf{" + t + "}" if abs(v["rmse"] - best) < 1e-9 else t)
        s.append(f"{lab} & {r['n']} & " + " & ".join(cells) + " \\\\")
    s.append("\\midrule\n\\multicolumn{6}{l}{\\emph{Coverage of the 95\\% posterior "
             "predictive interval, all 103 batters}} \\\\")
    r = next(x for x in rows if x["bucket"] == "ALL")
    s.append("Nominal 95\\% & 103 & " + " & ".join(
        ("---" if r[m] is None else f"{r[m]['cov95']*100:.0f}\\%") for m in ("M0","M1","M2","M3")) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    write("T5_validation", "\n".join(s))

# ------------------------------------------------------------------- T6 ---
def t6():
    j = json.load(open(os.path.join(OUT, "sensitivity.json")))
    import sensitivity as S
    s = ["\\begin{tabular}{lllrrrr}", "\\toprule",
         "Stakeholder & Prior on $\\mu$ & Prior on $\\tau^2$ & $E[\\mu\\mid y]$ & "
         "$E[\\tau\\mid y]$ & Shortlist & Top-10 shared \\\\", "\\midrule"]
    base = set(j["Analyst"]["top10"])
    for name in ("Scout", "Analyst", "CFO", "Stress"):
        p = S.STAKEHOLDERS[name]; r = j[name]
        s.append(f"{name} & $N({p.m0:.0f},{p.s0:.0f}^2)$ & "
                 f"$\\mathrm{{IG}}({p.a_tau:.0f},{p.b_tau:.0f})$ & {r['mu']:.1f} & "
                 f"{np.sqrt(r['tau2']):.1f} & {r['n_shortlist']} & "
                 f"{len(base & set(r['top10']))} / 10 \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    write("T6_sensitivity", "\n".join(s))

# ------------------------------------------------------------------- T7 ---
def t7():
    prof = json.load(open(os.path.join(ROOT, "data", "league_profile.json")))
    order = {"ipl": "IPL", "cpl": "CPL", "bbl": "BBL", "t20i": "T20I"}
    s = ["\\begin{tabular}{lrrrrr}", "\\toprule",
         "League & Balls & Strike rate & Dot balls & Fours & Balls per six \\\\", "\\midrule"]
    for x in prof:
        s.append(f"{order[x['league']]} & {x['balls']:,} & {x['sr']:.1f} & "
                 f"{x['dot']:.1f}\\% & {x['four']:.1f}\\% & {x['ballsPerSix']:.1f} \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    write("T7_league_profile", "\n".join(s))


if __name__ == "__main__":
    t1(); m1, var = t2(); rh, es = t3(); t4(); t5(); t6(); t7()
    print(f"\nkey numbers: E[X]={m1:.4f}  Var={var:.4f}  sigma^2={var*1e4:,.0f}  "
          f"sigma={np.sqrt(var)*100:.1f}  SD@100balls={np.sqrt(var)*10:.1f}")
    print(f"worst theta Rhat {rh:.4f}, min theta ESS {es:.0f}")
