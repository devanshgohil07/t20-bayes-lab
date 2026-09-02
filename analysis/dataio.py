"""Load cells.json / holdout.json into Data containers.

Two views are returned:
  full : every player, every league          -> used by M3
  ipl  : IPL cells only, players *reindexed* -> used by M0, M1, M2
`ipl_players` maps a local IPL index back to the global player index.
"""
import json, numpy as np, os
from gibbs import Data, IPL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load():
    cells = json.load(open(os.path.join(ROOT, "data", "cells.json")))
    hold  = json.load(open(os.path.join(ROOT, "data", "holdout.json")))
    players = cells["players"]
    P, L = len(players), len(cells["meta"]["leagues"])
    c = cells["cells"]
    full = Data(p=np.array([x["p"] for x in c]),
                l=np.array([x["l"] for x in c]),
                n=np.array([x["n"] for x in c], dtype=float),
                y=np.array([x["y"] for x in c]), P=P, L=L)

    m = full.l == IPL
    ipl_players = np.unique(full.p[m])                 # global ids with IPL data
    remap = -np.ones(P, dtype=int); remap[ipl_players] = np.arange(len(ipl_players))
    ipl = Data(p=remap[full.p[m]], l=full.l[m], n=full.n[m], y=full.y[m],
               P=len(ipl_players), L=L)

    test = dict(p=np.array([x["p"] for x in hold["cells"]]),
                n=np.array([x["n"] for x in hold["cells"]], dtype=float),
                y=np.array([x["y"] for x in hold["cells"]]))
    return dict(players=players, leagues=cells["meta"]["leagues"], meta=cells["meta"],
                full=full, ipl=ipl, ipl_players=ipl_players, ipl_remap=remap, test=test)

def weighted_mean_by_player(d: Data):
    """Precision-weighted (i.e. ball-weighted) raw mean per player; nan if none."""
    tot = np.bincount(d.p, weights=d.n, minlength=d.P)
    num = np.bincount(d.p, weights=d.n * d.y, minlength=d.P)
    out = np.full(d.P, np.nan)
    out[tot > 0] = num[tot > 0] / tot[tot > 0]
    return out, tot
