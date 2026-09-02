"""Every number quoted in the report, written out as LaTeX macros so that the prose
can never disagree with the analysis."""
import numpy as np, json, os
import dataio
from gibbs import Priors, IPL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, TAB = os.path.join(ROOT, "out"), os.path.join(ROOT, "tables")
D = dataio.load(); z = np.load(os.path.join(OUT, "draws.npz"))
full, ipl = D["full"], D["ipl"]
th = z["M3_theta"].reshape(-1, full.P); dl = z["M3_delta"].reshape(-1, 4)
mu, tau2, s2, om2 = (z[f"M3_{k}"].reshape(-1) for k in ("mu", "tau2", "sigma2", "omega2"))
br = json.load(open(os.path.join(ROOT, "data", "ballruns.json")))
v = np.array(br["values"], float); cnt = np.array(br["counts"], float)
m1 = (v * cnt).sum() / cnt.sum(); var = (v ** 2 * cnt).sum() / cnt.sum() - m1 ** 2
val = json.load(open(os.path.join(OUT, "validation.json")))
cmp_ = json.load(open(os.path.join(OUT, "compare.json")))
ppc = json.load(open(os.path.join(OUT, "ppc.json")))["M3"]
sens = json.load(open(os.path.join(OUT, "sensitivity.json")))
t4 = json.load(open(os.path.join(OUT, "test4.json")))
tst = json.load(open(os.path.join(OUT, "tests.json")))
dg = json.load(open(os.path.join(OUT, "diag_summary.json")))
raw_ipl, ipl_balls = dataio.weighted_mean_by_player(ipl)

M = {}
DIG = str.maketrans({"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                     "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"})
def s(k, v_):
    """LaTeX command names may contain only letters, so digits are spelled out."""
    M[k.translate(DIG)] = v_

s("Ncells", f"{full.C:,}"); s("Nplayers", f"{full.P:,}")
s("Nballs", f"{int(full.n.sum()):,}"); s("Nipl", f"{ipl.P:,}")
s("Ntest", f"{len(D['test']['p'])}")
for j, nm in enumerate(["Ipl", "Cpl", "Bbl", "Ttwoi"]):
    s(f"cells{nm}", f"{int((full.l == j).sum()):,}")
ov = [[len(set(full.p[full.l == a]) & set(full.p[full.l == b])) for b in range(4)] for a in range(4)]
s("bridgeIplTtwoi", str(ov[0][3])); s("bridgeIplCpl", str(ov[0][1])); s("bridgeIplBbl", str(ov[0][2]))

s("Ndeliveries", f"{int(cnt.sum()):,}")   # every delivery in the window, before the
                                          # 25-ball floor removes the smallest cells
s("ballMean", f"{m1:.3f}"); s("ballVar", f"{var:.3f}")
s("sigmaSqBall", f"{var*1e4:,.0f}"); s("sigmaBall", f"{np.sqrt(var)*100:.1f}")
s("sdHundred", f"{np.sqrt(var)*10:.1f}")
s("sigmaSqFit", f"{s2.mean():,.0f}"); s("sigmaFit", f"{np.sqrt(s2.mean()):.0f}")
s("designEffect", f"{s2.mean()/(var*1e4):.2f}")
s("sdHundredFit", f"{np.sqrt(s2.mean())/10:.1f}")
s("effBalls", f"{100/(s2.mean()/(var*1e4)):.0f}")

s("muFit", f"{mu.mean():.1f}"); s("tauFit", f"{np.sqrt(tau2.mean()):.1f}")
s("omegaFit", f"{np.sqrt(om2.mean()):.1f}")
for j, nm in enumerate(["Ipl", "Cpl", "Bbl", "Ttwoi"]):
    s(f"delta{nm}", f"{dl[:,j].mean():.1f}")
    s(f"delta{nm}Lo", f"{np.percentile(dl[:,j],2.5):.1f}")
    s(f"delta{nm}Hi", f"{np.percentile(dl[:,j],97.5):.1f}")

s("rhatWorst", f"{dg['rhatWorst']:.4f}"); s("essMin", f"{dg['essMin']:,.0f}")
s("essMedian", f"{dg['essMedian']:,.0f}")

def vb(bucket, model, field):
    r = next(x for x in val if x["bucket"] == bucket)
    return r[model][field] if r[model] else None
s("rmseZeroOverseasMthree", f"{vb('0 balls, overseas record','M3','rmse'):.1f}")
s("rmseZeroOverseasMtwo", f"{vb('0 balls, overseas record','M2','rmse'):.1f}")
s("rmseZeroOverseasMzero", f"{vb('0 balls, overseas record','M0','rmse'):.1f}")
s("countZeroOverseas", str(next(x for x in val if x['bucket']=='0 balls, overseas record')['n']))
s("countZeroNone", str(next(x for x in val if x['bucket']=='0 balls, no record')['n']))
s("rmseZeroNoneMthree", f"{vb('0 balls, no record','M3','rmse'):.1f}")
s("rmseZeroNoneMzero", f"{vb('0 balls, no record','M0','rmse'):.1f}")
for m in ("M0", "M1", "M2", "M3"):
    s(f"rmseRecord{m}", f"{vb('ALL with IPL record',m,'rmse'):.1f}")
    s(f"cov{m}", f"{vb('ALL',m,'cov95')*100:.0f}")
    s(f"rmseBigM{m[-1]}", f"{vb('300+',m,'rmse'):.1f}")
s("rmseMidMthree", f"{vb('101-300','M3','rmse'):.1f}")
s("rmseMidMtwo", f"{vb('101-300','M2','rmse'):.1f}")
s("rmseMidMone", f"{vb('101-300','M1','rmse'):.1f}")

for r in cmp_:
    s(f"elpdLoo{r['model']}", f"{r['elpd_loo']:.1f}")
    s(f"elpdWaic{r['model']}", f"{r['elpd_waic']:.1f}")
    s(f"pwaic{r['model']}", f"{r['p_waic']:.1f}")
    s(f"khatBad{r['model']}", str(r["khat_bad"]))

s("ppcSdSmall", f"{ppc['obs']['sd, cells under 100 balls']:.1f}")
s("ppcSdSmallRep", f"{ppc['rep_mean']['sd, cells under 100 balls']:.1f}")
s("ppcSdSmallP", f"{ppc['pval']['sd, cells under 100 balls']:.2f}")
s("ppcAbove", f"{ppc['obs']['share of cells above SR 150']*100:.1f}")
s("ppcAboveRep", f"{ppc['rep_mean']['share of cells above SR 150']*100:.1f}")
s("ppcIplRate", f"{ppc['obs']['ipl run rate (ball-weighted)']:.1f}")
s("ppcIplRateRep", f"{ppc['rep_mean']['ipl run rate (ball-weighted)']:.1f}")
s("ppcIplP", f"{ppc['pval']['ipl run rate (ball-weighted)']:.2f}")
s("ppcNegative", f"{ppc['rep_mean']['share of impossible (SR < 0)']*100:.2f}")

s("shortAnalyst", str(sens["Analyst"]["n_shortlist"]))
s("shortScout", str(sens["Scout"]["n_shortlist"]))
s("shortCfo", str(sens["CFO"]["n_shortlist"]))
s("shortStress", str(sens["Stress"]["n_shortlist"]))
s("topTen", ", ".join(sens["Analyst"]["top10"][:6]))
# how far the three plausible priors agree, computed rather than remembered
import itertools as _it
def _jac(a_, b_):
    A, B = set(a_), set(b_)
    return len(A & B) / len(A | B)
_pairs = [_jac(sens[a_]["shortlist"], sens[b_]["shortlist"])
          for a_, b_ in _it.combinations(("Scout", "Analyst", "CFO"), 2)]
s("jaccardLo", f"{min(_pairs):.2f}"); s("jaccardHi", f"{max(_pairs):.2f}")
s("topTenCommon", str(len(set(sens["Scout"]["top10"]) & set(sens["Analyst"]["top10"])
                          & set(sens["CFO"]["top10"]))))
s("stressKeeps", str(len(set(sens["Stress"]["top10"]) & set(sens["Analyst"]["top10"]))))

s("jsMu", f"{[r for r in t4['rows'] if r['param']=='mu'][0]['js']:.2f}")
s("pyMu", f"{[r for r in t4['rows'] if r['param']=='mu'][0]['python']:.2f}")
s("jsThetaCorr", f"{t4['thetaCorr']:.4f}"); s("jsThetaRmse", f"{t4['thetaRMSE']:.2f}")
s("jsMaxZ", f"{max(r['z'] for r in t4['rows']):.2f}")

# correctness-test results
s("testOneMcse", f"{tst['t1_mcse_ratio']:.2f}")
s("testTwoCoverage", f"{tst['t2_theta_coverage']*100:.1f}")
s("testThreeSpreadFirst", f"{tst['t3_spread_first']:.1f}")
s("testThreeSpreadLast", f"{tst['t3_spread_last']:.2f}")
s("testThreeMaxZ", f"{tst['t3_maxz']:.1f}")

# per-league ball profile
prof = {x["league"]: x for x in json.load(open(os.path.join(ROOT, "data", "league_profile.json")))}
for k, nm in (("ipl", "Ipl"), ("cpl", "Cpl"), ("bbl", "Bbl"), ("t20i", "Ttwoi")):
    s(f"dot{nm}", f"{prof[k]['dot']:.1f}")
    s(f"six{nm}", f"{prof[k]['six']:.1f}")
    s(f"bps{nm}", f"{prof[k]['ballsPerSix']:.1f}")
    s(f"srRaw{nm}", f"{prof[k]['sr']:.1f}")

# headline movers
gid = D["ipl_players"]; post = th[:, gid].mean(axis=0)
rr = (-raw_ipl).argsort().argsort() + 1
rp = (-post).argsort().argsort() + 1
top = np.argsort(-raw_ipl)[:25]
worst = top[np.argmax(rp[top] - rr[top])]
s("moverName", D["players"][gid[worst]].replace("&", "\\&"))
s("moverBalls", f"{ipl_balls[worst]:.0f}")
s("moverRaw", f"{raw_ipl[worst]:.0f}")
s("moverRankRaw", str(rr[worst])); s("moverRankPost", str(rp[worst]))
s("moverRawExact", f"{raw_ipl[worst]:.1f}")
s("moverProb", f"{(th[:, gid[worst]] > 140).mean():.2f}")
riser = top[np.argmin(rp[top] - rr[top])]
s("riserName", D["players"][gid[riser]])
s("riserBalls", f"{ipl_balls[riser]:.0f}")
s("riserRankRaw", str(rr[riser])); s("riserRankPost", str(rp[riser]))
s("countFall", str(int(np.sum(rp[np.argsort(-raw_ipl)[:22]] > rr[np.argsort(-raw_ipl)[:22]]))))
# shrinkage weight for the long-career bucket, on total balls across all four leagues
_tot = np.zeros(full.P); np.add.at(_tot, full.p, full.n)
_big = np.array([_tot[gid[j]] for j in range(len(gid)) if ipl_balls[j] > 300])
_w = lambda n: n * tau2.mean() / (s2.mean() + n * tau2.mean())
s("bigMedianBalls", f"{np.median(_big):,.0f}")
s("bigMedianW", f"{_w(np.median(_big)):.2f}")

o = np.argsort(-ipl.y)[:15]
s("topFifteenMedianBalls", f"{np.median(ipl.n[o]):.0f}")
s("topFifteenNoise", f"{np.sqrt(var*1e4/np.median(ipl.n[o])):.0f}")

with open(os.path.join(TAB, "numbers.tex"), "w") as f:
    for k, v_ in M.items():
        f.write("\\newcommand{\\n%s}{%s}\n" % (k, v_))
print(f"wrote tables/numbers.tex with {len(M)} macros")
for k in ("sdHundred", "sdHundredFit", "designEffect", "moverName", "moverRankRaw",
          "moverRankPost", "riserName", "riserRankRaw", "riserRankPost", "topTen"):
    print(f"  {k} = {M[k]}")
