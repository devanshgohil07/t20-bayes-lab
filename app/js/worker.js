/* Web Worker wrapper: keeps the sampler off the UI thread so the page stays alive. */
importScripts('rng.js', 'gibbs.js');

self.onmessage = function (e) {
  const msg = e.data;
  if (msg.cmd !== 'run') return;
  const chains = msg.chains || 4;
  const out = [];
  for (let c = 0; c < chains; c++) {
    out.push(self.Gibbs.run(Object.assign({}, msg.opts, {
      seed: 20260901 + 7919 * c,
      onProgress: function (f) {
        self.postMessage({ type: 'progress', value: (c + f) / chains });
      }
    })));
  }
  const keys = ['mu', 'tau2', 'omega2', 'sigma2'];
  const diag = {};
  keys.forEach(k => {
    const cs = out.map(o => o[k]);
    diag[k] = { rhat: self.Gibbs.splitRhat(cs), ess: self.Gibbs.ess(cs),
                acf: self.Gibbs.acf(cs[0], 30) };
  });
  diag.delta = [0, 1, 2, 3].map(j => {
    const cs = out.map(o => o.delta[j]);
    return { rhat: self.Gibbs.splitRhat(cs), ess: self.Gibbs.ess(cs) };
  });
  self.postMessage({
    type: 'done',
    chains: out.map(o => ({ mu: o.mu, tau2: o.tau2, omega2: o.omega2,
                            sigma2: o.sigma2, delta: o.delta,
                            watchDraws: o.watchDraws, nKept: o.nKept })),
    thetaMean: Array.from(out[0].thetaMean),
    thetaSd: Array.from(out[0].thetaSd),
    diag: diag
  });
};
