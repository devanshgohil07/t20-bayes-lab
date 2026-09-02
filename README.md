# Beyond Strike Rate

**Estimating T20 batting ability across four competitions with a hierarchical Bayesian model.**

**Live site:** https://devanshgohil07.github.io/t20-bayes-lab/  
**Report:** [`report/report.pdf`](report/report.pdf)

Group project · Bayesian Statistics

---

## The problem

An IPL auction ranks batters by strike rate and prices them off that ranking. A strike rate is
an average, and an average over a small number of balls moves around on its own.

Before any model, count. A ball off the bat produces 0, 1, 2, 4 or 6 runs, and almost nothing
else. Across the 225,001 deliveries in our window the mean is 1.320 runs per ball with variance
2.798. A strike rate multiplies by 100, so the per-ball variance on the strike-rate scale is
about 27,978, and the standard deviation of a strike rate over *n* balls is σ/√n.

Put *n* = 100:

> **A strike rate over 100 balls carries ±16.7 points of sampling noise.**
> This number needs no model. It follows from counting runs.

So the raw leaderboard is not a neutral summary of the data. It is an estimate, and a noisy one
for anyone with a short record. This project replaces it with a model that says how much of a
batter's number to believe, and that can put a CPL or BBL strike rate on the IPL scale.

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

Each piece, in one line:

- `theta[i]` is what we want: the batter's ability, expressed as the strike rate we expect from
  him in the IPL.
- `delta[l]` is the competition's scoring environment. Fixing `delta[IPL] = 0` is what makes
  every `theta` readable as an IPL number.
- The likelihood variance is `sigma^2 / n[c]`, because `y[c]` is a mean over `n[c]` balls. A
  batter with 600 balls therefore pulls harder than one with 30, which is the whole point.
- The offsets are identified by batters who appear in more than one competition. 106 link the
  IPL to T20 internationals, 38 to the CPL, 36 to the BBL.
- All six full conditionals are closed form, so the sampler is plain Gibbs. No Metropolis step,
  no probabilistic programming language.

The prior for `sigma^2` is centred on the 27,978 counted above. The others are weakly
informative and the report shows that the ranking does not depend on them.

## The decision

Inference and the decision are kept apart. The model returns a posterior for each batter. The
franchise says what it is buying and how sure it needs to be:

```
shortlist batter i   if   P(theta[i] > c | data) > p
```

Our baseline is `c = 140`, `p = 0.80`, which shortlists 28 batters. Both numbers are business
choices, not statistical ones, and the site lets you move them.

## What the model gives

| | |
|---|---|
| Sampling noise in a 100-ball strike rate, from counting runs | **±16.7** |
| Fitted σ² against the value implied by independent balls | **1.93×** |
| The fitted model's own read of a 100-ball strike rate | **±23.2** |
| League offsets against the IPL | CPL −7.9, BBL −4.9, T20I −5.2 (no interval covers zero) |
| 2025 holdout, batters with an IPL record (RMSE) | M3 **25.0** < M2 25.5 < M0 26.8 < raw leaderboard 29.9 |
| 2025 holdout, no IPL record but an overseas one | M3 **33.7**; the raw leaderboard gives no estimate at all |
| Coverage of the 95% interval | 94–95% with pooling, **79%** without |

The second row is the one worth pausing on. The last row is the one a franchise should care
about: intervals that cover what they claim to cover are what make a probability threshold
mean anything.

## Reproducing the analysis

Python 3.11 with numpy, pandas, scipy and matplotlib. No R, no Stan, no `brms`.

```bash
# 1. Data. Download the CSV (csv2) archives from https://cricsheet.org/matches/
#    and unzip into raw/ipl, raw/cpl, raw/bbl, raw/t20i
python3 prep/build_data.py     # writes data/cells.json, holdout.json, ballruns.json,
                               # league_profile.json, and prints the bridge-player matrix

# 2. Check the sampler before trusting it
cd analysis
python3 tests_correctness.py   # tests 1 to 3

# 3. Fit and analyse
python3 run_main.py            # M0 to M3, draws and diagnostics
python3 validation.py          # the 2025 IPL holdout study
python3 compare.py             # WAIC and PSIS-LOO
python3 ppc.py                 # posterior predictive checks
python3 sensitivity.py         # scout, analyst, finance and stress priors

# 4. Outputs
python3 figures.py             # F1 to F10, as PDF and PNG
python3 tables.py              # T1 to T7, as LaTeX fragments
python3 report_numbers.py      # every number quoted in the report, as a LaTeX macro
python3 export_app.py          # the JSON the site reads

# 5. The report
cd ../report && pdflatex report.tex && pdflatex report.tex
```

The bridge-player matrix printed by step 1 is the go or no-go check. If batters did not appear
in more than one competition, nothing would identify the offsets and the model would not be
worth fitting.

Every run is seeded from `MASTER_SEED = 20260901` in `analysis/rng.py`. Chain seeds come from
`zlib.crc32` and not Python's built-in `hash()`, which is salted per process, so two runs of the
same script give byte-identical output, figures included.

`raw/` and the posterior draws in `out/*.npz` are not in the repository. Steps 1 and 3 rebuild
them, and the fit takes about twenty seconds.

## Running the site locally

No build step, no dependencies, no bundler.

```bash
cd app && python3 -m http.server 8899
# then open http://127.0.0.1:8899
```

Opening `index.html` from the filesystem will not work, because the page fetches its JSON and
browsers block that over `file://`.

Test 4 checks the JavaScript sampler against the Python one and needs the server running:

```bash
python3 analysis/test4_js_vs_python.py
```

## Deploying the site

The site is static. Upload the contents of `app/` to a repository, then
**Settings, Pages, Deploy from a branch, `main` / `(root)`**.

## The code

Every number, table and figure in the report is produced by the code here; nothing is typed by
hand. The report does not reprint any of it, so this is the place to read it.

Two files are worth opening first. `analysis/gibbs.py` is the six full conditionals of
Appendix A written out in under two hundred lines. `app/js/gibbs.js` is the same sampler written
a second time, in another language, from the same derivations. Test 4 checks that the two agree,
which is the closest thing we have to an independent implementation.

```
prep/build_data.py     Cricsheet -> cells.json / holdout.json / ballruns.json / league_profile.json
analysis/
  gibbs.py             the six full conditionals and the M0 to M3 ladder
  rng.py               seeding
  dataio.py            loaders; the `full` view and the re-indexed `ipl` view
  diagnostics.py       split R-hat, Geyer ESS, autocorrelation
  waic_loo.py          WAIC and PSIS-LOO with generalised Pareto tail fitting
  tests_correctness.py       tests 1 to 3
  test4_js_vs_python.py      test 4, drives the browser sampler headlessly
  run_main.py  validation.py  compare.py  ppc.py  sensitivity.py
  figures.py  style.py  tables.py  report_numbers.py  export_app.py
app/
  index.html  css/style.css
  js/rng.js  js/gibbs.js  js/worker.js  js/plots.js  js/app.js
  data/*.json                 exported posteriors, 516 KB
report/report.tex      the report; preamble.tex the style, report.pdf the built version
figs/ (22)             F1 to F10, as PDF and PNG
tables/ (10)           T1 to T7, plus numbers.tex
data/ (4)              cells, holdout, the per-ball run distribution, the league profile
```

## Three points about the data and the fit

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
per-league ball profile behind that claim.

## Data

Ball-by-ball archives from [Cricsheet](https://cricsheet.org), used under the terms given there.
2021 to 2024 for fitting, the 2025 IPL held back for testing.
