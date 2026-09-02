/* The same six full conditionals as analysis/gibbs.py, in the browser.

   y[c] ~ N( theta[p[c]] + delta[l[c]], sigma2 / n[c] )
   theta[i] ~ N(mu, tau2)      delta[l] ~ N(0, omega2),  delta[IPL] = 0
   mu ~ N(m0, s0^2)  tau2 ~ IG(a_tau,b_tau)  omega2 ~ IG(a_om,b_om)  sigma2 ~ IG(a_sig,b_sig)

   Models: M0 complete pooling, M1 no pooling, M2 partial pooling (IPL only),
   M3 partial pooling with league offsets. */
(function (root) {
  'use strict';
  const IPL = 0;

  const DEFAULT_PRIORS = {
    m0: 130, s0: 20, a_tau: 3, b_tau: 450,
    a_om: 3, b_om: 128, a_sig: 3, b_sig: 52000
  };

  function runGibbs(opts) {
    const data = opts.data;                 // {p,l,n,y,P,L}
    const pr = Object.assign({}, DEFAULT_PRIORS, opts.priors || {});
    const model = opts.model || 'M3';
    const iters = opts.iters || 6000,
          burn = opts.burn || 2000,
          thin = opts.thin || 2;
    const rng = new root.RNG(opts.seed === undefined ? 1 : opts.seed);
    const onProgress = opts.onProgress;
    const tau2Fixed = opts.tau2Fixed;

    const P = data.P, L = data.L, C = data.n.length;
    const p = data.p, l = data.l, n = data.n, y = data.y;

    const nByPlayer = new Float64Array(P), sumByPlayer = new Float64Array(P);
    for (let c = 0; c < C; c++) { nByPlayer[p[c]] += n[c]; sumByPlayer[p[c]] += n[c] * y[c]; }

    const theta = new Float64Array(P);
    for (let i = 0; i < P; i++) theta[i] = nByPlayer[i] > 0 ? sumByPlayer[i] / nByPlayer[i] : pr.m0;
    const delta = new Float64Array(L);
    let mu = pr.m0, tau2 = tau2Fixed !== undefined && tau2Fixed !== null ? tau2Fixed : 225,
        omega2 = 64, sigma2 = 25000;
    if (model === 'M0') { mu = 0; for (let i = 0; i < P; i++) mu += theta[i] / P; }

    const nKeep = Math.max(1, Math.floor((iters - burn) / thin));
    const keep = { mu: [], tau2: [], omega2: [], sigma2: [],
                   delta: [[], [], [], []], theta: null, iters: [] };
    const thetaSum = new Float64Array(P), thetaSq = new Float64Array(P);
    const watch = opts.watch === undefined ? -1 : opts.watch;
    const watchDraws = [];
    let kept = 0;

    const Sn = new Float64Array(P), Sr = new Float64Array(P);
    const SnL = new Float64Array(L), SsL = new Float64Array(L);

    for (let it = 0; it < iters; it++) {
      /* 1. theta */
      if (model === 'M0') {
        let tot = 0, num = 0;
        for (let c = 0; c < C; c++) { tot += n[c]; num += n[c] * (y[c] - delta[l[c]]); }
        const prec = tot / sigma2 + 1 / (pr.s0 * pr.s0);
        const mean = (num / sigma2 + pr.m0 / (pr.s0 * pr.s0)) / prec;
        const th0 = mean + rng.randn() / Math.sqrt(prec);
        theta.fill(th0); mu = th0;
      } else {
        Sn.fill(0); Sr.fill(0);
        for (let c = 0; c < C; c++) {
          Sn[p[c]] += n[c] / sigma2;
          Sr[p[c]] += n[c] * (y[c] - delta[l[c]]) / sigma2;
        }
        if (model === 'M1') {
          for (let i = 0; i < P; i++) {
            if (Sn[i] > 0) theta[i] = Sr[i] / Sn[i] + rng.randn() / Math.sqrt(Sn[i]);
            else theta[i] = pr.m0 + pr.s0 * rng.randn();
          }
        } else {
          for (let i = 0; i < P; i++) {
            const prec = Sn[i] + 1 / tau2;
            theta[i] = (Sr[i] + mu / tau2) / prec + rng.randn() / Math.sqrt(prec);
          }
        }
      }

      /* 2. delta (M3 only, delta[IPL] pinned to zero) */
      if (model === 'M3') {
        SnL.fill(0); SsL.fill(0);
        for (let c = 0; c < C; c++) {
          SnL[l[c]] += n[c] / sigma2;
          SsL[l[c]] += n[c] * (y[c] - theta[p[c]]) / sigma2;
        }
        for (let j = 0; j < L; j++) {
          if (j === IPL) { delta[j] = 0; continue; }
          const prec = SnL[j] + 1 / omega2;
          delta[j] = SsL[j] / prec + rng.randn() / Math.sqrt(prec);
        }
      }

      /* 3. mu */
      if (model === 'M2' || model === 'M3') {
        const prec = P / tau2 + 1 / (pr.s0 * pr.s0);
        let s = 0; for (let i = 0; i < P; i++) s += theta[i];
        const mean = (s / tau2 + pr.m0 / (pr.s0 * pr.s0)) / prec;
        mu = mean + rng.randn() / Math.sqrt(prec);
      }

      /* 4. tau2 */
      if ((model === 'M2' || model === 'M3') && (tau2Fixed === undefined || tau2Fixed === null)) {
        let ss = 0; for (let i = 0; i < P; i++) { const d = theta[i] - mu; ss += d * d; }
        tau2 = rng.randInvGamma(pr.a_tau + P / 2, pr.b_tau + 0.5 * ss);
      }

      /* 5. omega2 */
      if (model === 'M3') {
        let ss = 0, k = 0;
        for (let j = 0; j < L; j++) if (j !== IPL) { ss += delta[j] * delta[j]; k++; }
        omega2 = rng.randInvGamma(pr.a_om + k / 2, pr.b_om + 0.5 * ss);
      }

      /* 6. sigma2 */
      let ss = 0;
      for (let c = 0; c < C; c++) {
        const r = y[c] - theta[p[c]] - delta[l[c]];
        ss += n[c] * r * r;
      }
      sigma2 = rng.randInvGamma(pr.a_sig + C / 2, pr.b_sig + 0.5 * ss);

      /* store */
      if (it >= burn && (it - burn) % thin === 0) {
        keep.mu.push(mu); keep.tau2.push(tau2);
        keep.omega2.push(omega2); keep.sigma2.push(sigma2);
        for (let j = 0; j < L; j++) keep.delta[j].push(delta[j]);
        keep.iters.push(it);
        for (let i = 0; i < P; i++) { thetaSum[i] += theta[i]; thetaSq[i] += theta[i] * theta[i]; }
        if (watch >= 0) watchDraws.push(theta[watch]);
        kept++;
      }
      if (onProgress && (it % 250 === 0)) onProgress(it / iters);
    }

    const mean = new Float64Array(P), sd = new Float64Array(P);
    for (let i = 0; i < P; i++) {
      mean[i] = thetaSum[i] / kept;
      sd[i] = Math.sqrt(Math.max(0, thetaSq[i] / kept - mean[i] * mean[i]));
    }
    keep.thetaMean = mean; keep.thetaSd = sd; keep.watchDraws = watchDraws;
    keep.nKept = kept;
    return keep;
  }

  /* ---- diagnostics, same definitions as analysis/diagnostics.py ---- */
  function splitRhat(chains) {
    const halves = [];
    chains.forEach(c => {
      const h = Math.floor(c.length / 2);
      halves.push(c.slice(0, h)); halves.push(c.slice(h, 2 * h));
    });
    const M = halves.length, N = halves[0].length;
    if (N < 4) return NaN;
    const means = halves.map(h => h.reduce((a, b) => a + b, 0) / N);
    const vars = halves.map((h, k) =>
      h.reduce((a, b) => a + (b - means[k]) * (b - means[k]), 0) / (N - 1));
    const W = vars.reduce((a, b) => a + b, 0) / M;
    const gm = means.reduce((a, b) => a + b, 0) / M;
    const B = N * means.reduce((a, b) => a + (b - gm) * (b - gm), 0) / (M - 1);
    if (W <= 0) return NaN;
    return Math.sqrt(((N - 1) / N * W + B / N) / W);
  }

  function acf(x, maxlag) {
    const n = x.length, m = x.reduce((a, b) => a + b, 0) / n;
    const d = x.map(v => v - m);
    let c0 = 0; for (let i = 0; i < n; i++) c0 += d[i] * d[i];
    const out = [];
    for (let k = 0; k <= maxlag; k++) {
      let s = 0; for (let i = 0; i < n - k; i++) s += d[i] * d[i + k];
      out.push(c0 === 0 ? 0 : s / c0);
    }
    return out;
  }

  function ess(chains) {
    const m = chains.length, n = chains[0].length;
    const maxlag = Math.min(n - 2, 200);
    const rho = new Array(maxlag + 1).fill(0);
    chains.forEach(c => { const a = acf(c, maxlag); for (let k = 0; k <= maxlag; k++) rho[k] += a[k] / m; });
    let s = 0;
    for (let t = 1; t < maxlag - 1; t += 2) {
      const pair = rho[t] + rho[t + 1];
      if (pair <= 0) break;
      s += pair;
    }
    return m * n / (1 + 2 * s);
  }

  root.Gibbs = { run: runGibbs, splitRhat, acf, ess, DEFAULT_PRIORS, IPL };
})(typeof self !== 'undefined' ? self : this);
