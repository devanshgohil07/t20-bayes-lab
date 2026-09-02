# Beyond Strike Rate

**Estimating T20 batting ability across four competitions with a hierarchical Bayesian model.**

| | |
|---|---|
| **Read the report** | [`report/report.pdf`](report/report.pdf), 18 pages |
| **Use the interactive site** | https://devanshgohil07.github.io/t20-bayes-lab/ |
| **Everything else** | is in this repository. The map is at the bottom. |

---

## The problem

An IPL auction ranks batters by strike rate and prices them off that ranking. A strike rate is
an average, and an average over a small number of balls moves around on its own.

Before any model, count. A ball off the bat produces 0, 1, 2, 4 or 6 runs, and almost nothing
else. Across the 225,001 deliveries in our window the mean is 1.320 runs per ball with variance
2.798. A strike rate multiplies by 100, so the per-ball variance on the strike-rate scale is
about 27,978, and the standard deviation of a strike rate over *n* balls is σ/√n. Put *n* = 100:

> **A strike rate over 100 balls carries ±16.7 points of sampling noise.**
> This number needs no model. It follows from counting runs.

So the raw leaderboard is not a neutral summary of the data. It is an estimate, and a noisy one
for anyone with a short record. It also has nothing at all to say about a batter who has never
played in the IPL, which is exactly the batter a franchise is most often bidding on.

We replace it with a model that says how much of a batter's number to believe, puts a CPL or
BBL strike rate on the IPL scale, and returns each batter as a probability rather than a rank.

## The model

A cell is one batter in one competition: the balls he faced there and the strike rate he made.
We use 839 cells covering 600 batters, from the IPL, CPL, BBL and men's T20 internationals,
2021 to 2024. The 2025 IPL is held back for testing.

```
y[c]      ~ N( theta[p(c)] + delta[l(c)],  sigma^2 / n[c] )
theta[i]  ~ N(mu, tau^2)
delta[l]  ~ N(0, omega^2),  delta[IPL] = 0
mu ~ N(130, 20^2)   tau^2 ~ IG(3, 450)   omega^2 ~ IG(3, 128)   sigma^2 ~ IG(3, 52000)
```

- `theta[i]` is what we want: the batter's ability, as the strike rate we expect from him in
  the IPL.
- `delta[l]` is the competition's scoring environment. Fixing `delta[IPL] = 0` is what makes
  every `theta` readable as an IPL number.
- The likelihood variance is `sigma^2 / n[c]`, because `y[c]` is a mean over `n[c]` balls. A
  batter with 600 balls pulls harder than one with 30, which is the whole point.
- The offsets are identified by batters who appear in more than one competition: 106 link the
  IPL to T20 internationals, 38 to the CPL, 36 to the BBL.
- All six full conditionals are closed form, so the sampler is plain Gibbs. No Metropolis step,
  no probabilistic programming language.

The decision is kept separate from the inference. The model returns a posterior; the franchise
says what it is buying and how sure it needs to be:

```
shortlist batter i   if   P(theta[i] > c | data) > p
```

Our baseline is `c = 140`, `p = 0.80`, which shortlists 28 batters. Both are business choices,
not statistical ones, and the site lets you move them.

## How we built it

Six steps, in order.

**1. Build the data.** Cricsheet publishes every delivery of the IPL, CPL, BBL and men's T20
internationals. `prep/build_data.py` reads all four archives, drops wides (a wide is not a ball
the batter faced), keeps no-balls, and aggregates deliveries into player × league cells. Cells
under 25 balls are dropped, which is what lets us treat a cell strike rate as an approximately
Normal average.

**2. Check the model is identifiable before fitting it.** The league offsets are only knowable
from batters who played in more than one competition. Step 1 prints the league-by-league overlap
matrix, and that is the go or no-go check: with no bridge players, nothing identifies the
offsets and the model is not worth fitting. This is also where we found that the unfiltered T20I
file spans 105 national teams and had to be restricted (see the notes below).

**3. Test the sampler before trusting it.** `analysis/tests_correctness.py` runs three checks:
one batter in one league must reproduce the closed-form conjugate posterior; data simulated from
known parameters must be recovered inside their intervals; and shrinkage must behave correctly
in the limits. `analysis/test4_js_vs_python.py` adds a fourth, running the browser sampler
headlessly and comparing it with the Python one. Only then did we fit anything real.

**4. Fit the ladder.** `analysis/run_main.py` fits four models: complete pooling (M0), no
pooling (M1, which is the raw leaderboard written as a model), partial pooling on IPL data alone
(M2), and partial pooling with league offsets (M3). Comparing against the simpler rungs is what
shows the hierarchy earns its extra parameters.

**5. Check the fit.** Convergence with split R-hat and effective sample size
(`analysis/diagnostics.py`), posterior predictive checks (`analysis/ppc.py`), and WAIC with
PSIS-LOO (`analysis/compare.py`). The posterior predictive check is the one that found a real
misfit, which the report reports rather than hides.

**6. Validate out of sample and stress the priors.** `analysis/validation.py` predicts every
batter's 2025 IPL strike rate from 2021 to 2024 data, stratified by how much IPL history each
batter had beforehand. The table structure was fixed before it was run.
`analysis/sensitivity.py` then refits under the priors of a scout, an analyst and a finance
head, plus one deliberately implausible prior to find where the conclusion breaks.

Every figure, table and quoted number is then generated from the fitted draws, so nothing in the
report is typed by hand.

To re-run it yourself, after downloading the Cricsheet CSV archives into `raw/`:

```bash
python3 prep/build_data.py                     # cells, holdout, ball runs, league profile
cd analysis
python3 tests_correctness.py                   # checks 1 to 3
for s in run_main validation compare ppc sensitivity; do python3 $s.py; done
for s in figures tables report_numbers export_app; do python3 $s.py; done
cd ../report && pdflatex report.tex && pdflatex report.tex
```

Python 3.11 with numpy, pandas, scipy and matplotlib. No R, no Stan, no `brms`. Seeding is a
pure function of `MASTER_SEED` in `analysis/rng.py`, so two runs give byte-identical output,
figures included. The fit takes about twenty seconds.

## What we found

| | |
|---|---|
| Sampling noise in a 100-ball strike rate, from counting runs | **±16.7** |
| Fitted σ² against the value implied by independent balls | **1.93×** |
| The fitted model's own read of a 100-ball strike rate | **±23.2** |
| League offsets against the IPL | CPL −7.9, BBL −4.9, T20I −5.2 (no interval covers zero) |
| 2025 holdout, batters with an IPL record (RMSE) | M3 **25.0** < M2 25.5 < M0 26.8 < raw leaderboard 29.9 |
| 2025 holdout, no IPL record but an overseas one | M3 **33.7**; the raw leaderboard gives no estimate at all |
| Coverage of the 95% interval | 94–95% with pooling, **79%** without |

The last row is the one a franchise should care about. Intervals that cover what they claim to
cover are what make a probability threshold mean anything.

## Where everything is

**The report** is `report/report.pdf`, built from `report/report.tex` and `preamble.tex`.

**The site** is at https://devanshgohil07.github.io/t20-bayes-lab/ and its source is in `app/`:
`index.html`, `css/style.css`, five files in `js/`, and the exported posteriors in `data/`
(516 KB). The sampler in `app/js/gibbs.js` is a second, independent implementation of the same
six conditionals, written in JavaScript; check 4 is what confirms the two agree.

**The code** that produces every result:

```
prep/build_data.py       Cricsheet archives -> cells, holdout, ball runs, league profile
analysis/
  gibbs.py               the six full conditionals and the M0 to M3 ladder
  rng.py                 seeding
  dataio.py              loaders; the full view and the re-indexed IPL view
  diagnostics.py         split R-hat, Geyer ESS, autocorrelation
  waic_loo.py            WAIC and PSIS-LOO with generalised Pareto tail fitting
  tests_correctness.py   checks 1 to 3
  test4_js_vs_python.py  check 4, drives the browser sampler headlessly
  run_main.py            fits the ladder, saves draws and diagnostics
  validation.py          the 2025 IPL holdout study
  compare.py             WAIC and LOO on the common 194 IPL cells
  ppc.py                 posterior predictive checks
  sensitivity.py         scout, analyst, finance and stress priors
  figures.py, style.py   every figure
  tables.py              every table
  report_numbers.py      every number quoted in the report, as a LaTeX macro
  export_app.py          the JSON the site reads
```

`analysis/gibbs.py` is the best place to start: it is the six conditionals of Appendix A written
out in under two hundred lines.

**The figures** are in `figs/`, each as both PDF and PNG:

| | |
|---|---|
| F1 | Strike rate against balls faced |
| F2 | The model as a directed acyclic graph |
| F3 | Prior predictive check |
| F4 | Trace plots for the four global parameters |
| F5 | Autocorrelation of the thinned draws |
| F6 | Posterior predictive check |
| F7 | The leaderboard after partial pooling |
| F8 | What a strike rate is worth in each league |
| F9 | Predictive accuracy by prior IPL exposure |
| F10 | The shortlist, with credible intervals |

`figs/F10_shortlist_ALLT20I_before.pdf` is kept on purpose: the shortlist from before the T20I
restriction, discussed below.

**The tables** are in `tables/`, as LaTeX fragments:

| | |
|---|---|
| T1a, T1b | The four competitions after filtering; the bridge-player overlap matrix |
| T2, T2b | The priors and where each number comes from; the distribution of runs off the bat |
| T3 | Posterior summaries and convergence diagnostics |
| T4 | WAIC and PSIS-LOO for the four models |
| T5 | Out-of-sample accuracy on the 2025 IPL |
| T6 | Prior sensitivity |
| T7 | How each competition scores |

`tables/numbers.tex` holds every other number the report quotes, as a macro.

**The data** is in `data/`: `cells.json` (the 839 training cells), `holdout.json` (the 2025
IPL), `ballruns.json` (the per-ball run distribution) and `league_profile.json` (each
competition's dot, four and six rates). The raw Cricsheet archives and the posterior draws are
not in the repository; both are rebuilt by the commands above.

## Three notes about the data and the fit

**T20 internationals are restricted to Full Member against Full Member matches.** The
unfiltered archive covers 105 national teams, from India and Australia down to Estonia and
Mongolia. That is not one scoring environment, so a single offset cannot describe it. Our first
shortlist was headed by batters from associate nations with very high strike rates against much
weaker bowling. That shortlist is kept at `figs/F10_shortlist_ALLT20I_before.pdf`, because the
failure is instructive. The restriction cost one bridge player out of 107.

**The fitted σ² is 1.93 times the value implied by independent balls.** This is not an error.
Form, matchups, pitch and match situation all correlate within a spell, so 100 balls carry
roughly the information of 52. Section 11 of the report gives the conjugate correction and says
why we did not adopt it.

**The league offsets are not a ranking of league quality.** δ measures the scoring environment,
not the standard of the competition. The CPL shows the difference. CricViz, using the same
bridge-player logic on a different quantity, rate the CPL among the strongest competitions in
world T20; our offset puts it lowest of the four. Both can be true. The CPL clears the rope more
often than the IPL, once every 14.6 balls against 15.7, but plays 42.2% dot balls against the
IPL's 36.6%, so the same batter scores more slowly there. `data/league_profile.json` holds the
per-league profile behind that claim.

## Data

Ball-by-ball archives from [Cricsheet](https://cricsheet.org), used under the terms given there.
2021 to 2024 for fitting, the 2025 IPL held back for testing.
