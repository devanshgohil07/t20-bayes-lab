# Beyond Strike Rate

**Estimating T20 batting ability across four leagues with a hierarchical Bayesian model.**

A hierarchical Bayesian model of T20 batting ability across four competitions, with an
interactive companion that runs the sampler in your browser.

**Live site:** https://devanshgohil07.github.io/t20-bayes-lab/  
**Report:** [`report/report.pdf`](report/report.pdf)

Group project · Bayesian Statistics

---

## The idea in three sentences

A strike rate is an average over a small number of balls, and such averages are noisy:
counting runs off the bat across all 225,001 deliveries in our window shows that a 100-ball
strike rate carries **±16.7 points of sampling noise** before any question about ability is
raised. We fit a partial pooling model to 839 player × league cells (221,164 legal balls) from
the IPL, CPL, BBL and men's T20 internationals, with one offset per league, so that a CPL strike
rate can be read on the IPL scale. Each batter
is returned as a posterior distribution on that scale, which turns an auction shortlist from a
ranking into a probability statement.

The model, for cell *c* with batter *p(c)*, league *ℓ(c)*, *n* balls and strike rate *y*:

```
y[c]      ~ N( theta[p(c)] + delta[l(c)],  sigma^2 / n[c] )
theta[i]  ~ N(mu, tau^2)          the batter's ability, on the IPL scale
delta[l]  ~ N(0, omega^2)         the league's scoring environment, delta[IPL] = 0
mu ~ N(130, 20^2)   tau^2 ~ IG(3, 450)   omega^2 ~ IG(3, 128)   sigma^2 ~ IG(3, 52000)
```

All six full conditionals are closed form, so the sampler is pure Gibbs: no Metropolis step
and no probabilistic programming language.

## What we found

| | |
|---|---|
| Sampling noise in a 100-ball strike rate, from counting runs | **±16.7** |
| Fitted σ² vs the independence value | **1.93×**, so balls are not conditionally independent |
| The model's own read of a 100-ball strike rate | **±23.2** |
| League offsets vs the IPL | CPL −7.9, BBL −4.9, T20I −5.2 (all intervals exclude zero) |
| 2025 holdout, batters with an IPL record | M3 **25.0** < M2 25.5 < M0 26.8 < raw leaderboard 29.9 |
| 2025 holdout, no IPL record but overseas data | M3 **33.7**; the raw leaderboard has no estimate at all |
| 95% interval coverage | 94–95% for the pooled models, **79%** for no pooling |

## Reproducing it

Python 3.11 with numpy, pandas, scipy and matplotlib. No R, no Stan, no `brms`.

```bash
# 1. Data. Download the CSV (csv2) archives from https://cricsheet.org/matches/
#    and unzip into raw/ipl, raw/cpl, raw/bbl, raw/t20i
python3 prep/build_data.py          # -> data/cells.json, holdout.json, ballruns.json,
                                    #    league_profile.json
                                    #    prints the bridge-player matrix (the go/no-go check)

# 2. Check the sampler before trusting it
cd analysis
python3 tests_correctness.py        # tests 1-3

# 3. Fit and analyse
python3 run_main.py                 # M0-M3, draws + diagnostics
python3 validation.py               # the 2025 IPL holdout study
python3 compare.py                  # WAIC and PSIS-LOO
python3 ppc.py                      # posterior predictive checks
python3 sensitivity.py              # Scout / Analyst / CFO / stress priors

# 4. Outputs
python3 figures.py                  # F1-F10 as PDF and PNG
python3 tables.py                   # T1-T7 as LaTeX fragments
python3 report_numbers.py           # every quoted number, as a LaTeX macro
python3 export_app.py               # the JSON the site reads

# 5. The paper
cd ../report && pdflatex report.tex && pdflatex report.tex
```

Every run is seeded from `MASTER_SEED = 20260901` in `analysis/rng.py`. Chain seeds are derived
with `zlib.crc32` rather than Python's built-in `hash()`, which is salted per process, so two
runs of the same script give byte-identical output, figures included.

`raw/` and the posterior draws in `out/*.npz` are not in the repository; steps 1 and 3 above
regenerate them, and the fit takes about fifteen seconds.

## Running the site locally

No build step, no dependencies, no bundler.

```bash
cd app && python3 -m http.server 8899
# then open http://127.0.0.1:8899
```

Opening `index.html` directly from the filesystem will not work, because the page fetches its
JSON and browsers block that over `file://`.

Test 4, which checks the JavaScript sampler against the Python one, needs the server running:

```bash
python3 analysis/test4_js_vs_python.py
```

## Deploying

The site is static. Upload the contents of `app/` to a repository, then
**Settings → Pages → Deploy from a branch → `main` / `(root)`**.

## Where the code lives

Every number, table and figure in the report is produced by the code in this repository;
nothing is typed by hand. The report does not reprint any of it, so this is the place to read
it. The two files worth opening first are `analysis/gibbs.py`, which is the six full
conditionals derived in Appendix A of the report written out in about a hundred lines, and
`app/js/gibbs.js`, which is the same sampler written a second time from the same derivations.
Test 4 checks that the two agree.

```
prep/build_data.py     Cricsheet -> cells.json / holdout.json / ballruns.json / league_profile.json
analysis/
  gibbs.py             the six full conditionals and the M0-M3 ladder
  rng.py               seeding
  dataio.py            loaders; `full` and re-indexed `ipl` views
  diagnostics.py       split R-hat, Geyer ESS, autocorrelation
  waic_loo.py          WAIC and PSIS-LOO with generalised Pareto tail fitting
  tests_correctness.py       tests 1-3
  test4_js_vs_python.py      test 4, drives the browser sampler headlessly
  run_main.py  validation.py  compare.py  ppc.py  sensitivity.py
  figures.py  style.py  tables.py  report_numbers.py  export_app.py
app/
  index.html  css/style.css
  js/rng.js  js/gibbs.js  js/worker.js  js/plots.js  js/app.js
  data/*.json                 exported posteriors, 516 KB
report/report.tex      the paper; preamble.tex the style, report.pdf the built version
figs/ (22)             F1-F10 as PDF and PNG
tables/ (10)           T1-T7 as LaTeX fragments, plus numbers.tex
data/ (4)              cells, holdout, the per-ball run distribution, the league profile
```

## Three notes worth reading before you argue with the results

**T20 internationals are restricted to Full Member v Full Member matches.** The unfiltered
archive covers 105 national teams, from India and Australia to Estonia and Mongolia, which is
not one scoring environment and cannot be described by a single offset. Our first shortlist was
headed by batters from associate nations who had recorded very high strike rates against much
weaker bowling; `figs/F10_shortlist_ALLT20I_before.pdf` is that shortlist, kept deliberately.
The restriction cost one bridge player out of 107.

**The fitted σ² is 1.93× the value implied by assuming balls are independent.** That is not an
error but a measurement: form, matchups, pitch and match situation all correlate within a spell,
so 100 balls carry roughly the information of 52. Section 11 of the report gives the
conjugate correction and explains why we did not adopt it.

**The league offsets are not a ranking of league quality.** δ measures the scoring environment,
not the standard of the competition, and the CPL is the case that shows the difference. CricViz,
using the same bridge-player logic on a different quantity, rate the CPL among the strongest
competitions in world T20; our offset puts it lowest of the four. Both are true. The CPL clears
the rope more often than the IPL does, once every 14.6 balls against 15.7, but plays 42.2% dot
balls against the IPL's 36.6%, so the same batter scores more slowly there.
`data/league_profile.json` holds the per-league ball profile behind that claim.

## Data

Ball-by-ball archives from [Cricsheet](https://cricsheet.org), used under the terms given there.
2021–2024 for fitting, the 2025 IPL held out for testing.
