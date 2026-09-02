"""
Beyond Strike Rate: data preparation.

Reads the four Cricsheet CSV (csv2) archives, restricts to 2021-2025, aggregates
ball-by-ball deliveries to player x league cells, and writes:

    data/cells.json    training cells   (2021-2024, all four leagues)
    data/holdout.json  test cells       (2025, IPL only)
    data/ballruns.json empirical distribution of runs off the bat (for sigma^2)

Also prints the league x league bridge-player overlap matrix, which is the
identification check for the league-offset parameters delta[l].

Cricsheet ships an `all_matches.csv` inside each csv2 zip; we use it directly.
Unzip each archive into raw/<league>/ and run:  python3 prep/build_data.py
"""
import pandas as pd, numpy as np, json, os, glob

LEAGUES   = {"ipl": 0, "cpl": 1, "bbl": 2, "t20i": 3}
YEARS     = range(2021, 2026)
MIN_BALLS = 25

USECOLS = ["match_id", "start_date", "striker", "runs_off_bat", "wides",
           "batting_team", "bowling_team"]

# T20 internationals are played by 105 different teams, from India and Australia
# down to Estonia and Mongolia. That is not one scoring environment, so a single
# league offset cannot describe it. We keep only matches in which BOTH sides are
# ICC Full Members, which is the T20I most people mean.
FULL_MEMBERS = {"India", "Australia", "England", "South Africa", "New Zealand",
                "Pakistan", "Sri Lanka", "Bangladesh", "West Indies",
                "Afghanistan", "Ireland", "Zimbabwe"}

frames = []
def ensure_all_matches(name):
    """The t20s_male_csv2 archive ships per-match files only; build the
    aggregate file once by concatenating them (headers are identical)."""
    path = f"raw/{name}/all_matches.csv"
    if os.path.exists(path):
        return path
    files = sorted(f for f in glob.glob(f"raw/{name}/*.csv")
                   if not f.endswith("_info.csv"))
    with open(path, "w", newline="") as out:
        for k, f in enumerate(files):
            with open(f) as fh:
                header = fh.readline()
                if k == 0:
                    out.write(header)
                out.writelines(fh)
    return path

for name, idx in LEAGUES.items():
    d = pd.read_csv(ensure_all_matches(name), usecols=USECOLS, low_memory=False)
    d["league"] = idx
    frames.append(d)
    print(f"{name:5s} raw deliveries: {len(d):>9,}")

df = pd.concat(frames, ignore_index=True)
df["year"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year
df = df[df["year"].isin(YEARS)]

# A wide is NOT a ball faced by the batter.
df = df[df["wides"].isna()]
print(f"\nDeliveries 2021-2025, wides excluded: {len(df):,}")

before = (df["league"] == LEAGUES["t20i"]).sum()
keep = (df["league"] != LEAGUES["t20i"]) | (df["batting_team"].isin(FULL_MEMBERS)
                                            & df["bowling_team"].isin(FULL_MEMBERS))
df = df[keep]
after = (df["league"] == LEAGUES["t20i"]).sum()
print(f"T20I deliveries kept (Full Member v Full Member only): {after:,} of {before:,}")

def aggregate(frame):
    g = (frame.groupby(["striker", "league"])
              .agg(n=("runs_off_bat", "size"), runs=("runs_off_bat", "sum"))
              .reset_index())
    g = g[g["n"] >= MIN_BALLS]
    g["y"] = 100.0 * g["runs"] / g["n"]
    return g

train = aggregate(df[df["year"] <= 2024])
test  = aggregate(df[(df["year"] == 2025) & (df["league"] == LEAGUES["ipl"])])

players = sorted(set(train["striker"]) | set(test["striker"]))
pidx = {p: i for i, p in enumerate(players)}

os.makedirs("data", exist_ok=True)

json.dump({
  "meta": {"window": "2021-2024", "minBalls": MIN_BALLS,
           "leagues": list(LEAGUES), "iplIndex": 0},
  "players": players,
  "cells": [{"p": pidx[r.striker], "l": int(r.league),
             "n": int(r.n), "y": round(float(r.y), 3)}
            for r in train.itertuples()],
}, open("data/cells.json", "w"))

json.dump({
  "meta": {"season": 2025, "league": "ipl"},
  "cells": [{"p": pidx[r.striker], "n": int(r.n), "y": round(float(r.y), 3)}
            for r in test.itertuples()],
}, open("data/holdout.json", "w"))

# Per-league ball outcome profile. A league's strike rate is the net of how often the
# ball goes for nothing and how often it clears the rope, and those two move separately.
prof = []
for name, idx in LEAGUES.items():
    r = df[(df["league"] == idx) & (df["year"] <= 2024)]["runs_off_bat"].to_numpy(float)
    prof.append({"league": name, "balls": int(len(r)),
                 "sr": round(100 * r.mean(), 1),
                 "dot": round(100 * (r == 0).mean(), 1),
                 "four": round(100 * (r == 4).mean(), 1),
                 "six": round(100 * (r == 6).mean(), 1),
                 "ballsPerSix": round(len(r) / max((r == 6).sum(), 1), 1)})
json.dump(prof, open("data/league_profile.json", "w"), indent=1)
print("\nPer-league ball outcome profile (2021-2024):")
print(f"  {'league':6s} {'balls':>8s} {'SR':>7s} {'dot%':>6s} {'four%':>6s} {'six%':>6s} {'balls/six':>10s}")
for x in prof:
    print(f"  {x['league']:6s} {x['balls']:8,d} {x['sr']:7.1f} {x['dot']:6.1f} "
          f"{x['four']:6.1f} {x['six']:6.1f} {x['ballsPerSix']:10.1f}")

# Empirical per-ball run distribution (training window) -> sigma^2 for the prior.
tr = df[df["year"] <= 2024]
vc = tr["runs_off_bat"].value_counts().sort_index()
json.dump({"values": [int(v) for v in vc.index], "counts": [int(c) for c in vc.values]},
          open("data/ballruns.json", "w"))

x  = tr["runs_off_bat"].to_numpy(dtype=float)
m1, m2 = x.mean(), (x**2).mean()
var = m2 - m1**2
print(f"\nPer-ball runs off the bat: E[X]={m1:.4f}  E[X^2]={m2:.4f}  Var={var:.4f}")
print(f"sigma^2 on the x100 strike-rate scale = {var*1e4:,.0f}   sigma = {np.sqrt(var)*100:.1f}")
print(f"SD of observed strike rate off 100 balls = {np.sqrt(var)*100/10:.1f} SR points")

# --- go / no-go: bridge-player overlap matrix ---
piv = train.pivot_table(index="striker", columns="league",
                        values="n", aggfunc="sum").notna().astype(int)
print("\nLeague x League overlap (players appearing in both):")
print(piv.T.values @ piv.values)
print("\nLeague order:", list(LEAGUES))
print(f"\nPlayers: {len(players)}   Train cells: {len(train)}   Test cells: {len(test)}")
print("Cells per league:", train.groupby('league').size().to_dict())
