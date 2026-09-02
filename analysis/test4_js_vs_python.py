"""
Correctness test 4: the JavaScript sampler in the app must reproduce the Python one.

Two independent implementations of the same six full conditionals, on the same data,
with different random number generators (numpy PCG64 vs mulberry32 + Box-Muller).
If they agree, the conditionals are almost certainly right in both.

Runs the app's Web Worker in headless Chromium. Start a server first:
    (cd app && python3 -m http.server 8899)
    python3 analysis/test4_js_vs_python.py
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = os.environ.get("APP_URL", "http://127.0.0.1:8899/index.html")

JS = """
async () => {
  const cells = await (await fetch('data/cells.json')).json();
  const d = {
    p: Int32Array.from(cells.cells.map(c => c.p)),
    l: Int32Array.from(cells.cells.map(c => c.l)),
    n: Float64Array.from(cells.cells.map(c => c.n)),
    y: Float64Array.from(cells.cells.map(c => c.y)),
    P: cells.players.length, L: 4
  };
  const out = [];
  for (let c = 0; c < 4; c++) {
    out.push(Gibbs.run({ data: d, model: 'M3', iters: 12000, burn: 4000, thin: 4,
                         seed: 20260901 + 7919 * c }));
  }
  const flat = k => [].concat.apply([], out.map(o => o[k]));
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const sd = a => { const m = mean(a);
    return Math.sqrt(a.reduce((x, y) => x + (y - m) * (y - m), 0) / (a.length - 1)); };
  const res = {};
  ['mu', 'tau2', 'omega2', 'sigma2'].forEach(k => {
    res[k] = { mean: mean(flat(k)), sd: sd(flat(k)),
               rhat: Gibbs.splitRhat(out.map(o => o[k])),
               ess: Gibbs.ess(out.map(o => o[k])) };
  });
  [1, 2, 3].forEach(j => {
    const a = [].concat.apply([], out.map(o => o.delta[j]));
    res['delta' + j] = { mean: mean(a), sd: sd(a),
                         rhat: Gibbs.splitRhat(out.map(o => o.delta[j])),
                         ess: Gibbs.ess(out.map(o => o.delta[j])) };
  });
  res.thetaMean = Array.from(out[0].thetaMean);
  res.nKept = out[0].nKept;
  return res;
}
"""

def main():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page()
        pg.goto(URL)
        pg.wait_for_timeout(1500)
        js = pg.evaluate(JS)
        b.close()

    z = np.load(os.path.join(ROOT, "out", "draws.npz"))
    py = {}
    for k in ("mu", "tau2", "omega2", "sigma2"):
        x = z[f"M3_{k}"].reshape(-1)
        py[k] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), n=x.size)
    dl = z["M3_delta"].reshape(-1, 4)
    for j in (1, 2, 3):
        py[f"delta{j}"] = dict(mean=float(dl[:, j].mean()), sd=float(dl[:, j].std(ddof=1)),
                               n=dl.shape[0])

    print(f"\n{'parameter':10s} {'python':>12s} {'javascript':>12s} {'diff':>9s} "
          f"{'MCSE':>8s} {'diff/MCSE':>10s}")
    ok = True
    rows = []
    for k in ("mu", "tau2", "omega2", "sigma2", "delta1", "delta2", "delta3"):
        p, j = py[k], js[k]
        # both runs are Monte Carlo, so the comparison is on the combined MCSE
        essP = z[f"M3_{k}"].size if k in ("mu", "tau2", "omega2", "sigma2") else dl.shape[0]
        mcse = np.sqrt(p["sd"] ** 2 / max(js[k]["ess"], 1) + p["sd"] ** 2 / max(js[k]["ess"], 1))
        d = j["mean"] - p["mean"]
        z_ = abs(d) / mcse
        ok &= z_ < 4
        rows.append(dict(param=k, python=p["mean"], js=j["mean"], diff=d, mcse=mcse, z=z_,
                         rhat_js=j["rhat"], ess_js=j["ess"]))
        print(f"{k:10s} {p['mean']:12.3f} {j['mean']:12.3f} {d:9.3f} {mcse:8.3f} {z_:10.2f}")

    th_py = z["M3_theta"].reshape(-1, len(js["thetaMean"])).mean(axis=0)
    th_js = np.array(js["thetaMean"])
    corr = float(np.corrcoef(th_py, th_js)[0, 1])
    mad = float(np.max(np.abs(th_py - th_js)))
    rmse = float(np.sqrt(np.mean((th_py - th_js) ** 2)))
    print(f"\nplayer abilities: correlation {corr:.5f}, RMSE {rmse:.3f}, "
          f"largest gap {mad:.2f} strike-rate points (one JS chain vs four Python chains)")
    ok &= corr > 0.999 and rmse < 1.5

    json.dump(dict(rows=rows, thetaCorr=corr, thetaRMSE=rmse, thetaMaxDiff=mad, passed=bool(ok)),
              open(os.path.join(ROOT, "out", "test4.json"), "w"), indent=1)
    print("\n" + ("TEST 4 PASSED: the two implementations agree within Monte Carlo error"
                  if ok else "TEST 4 FAILED"))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
