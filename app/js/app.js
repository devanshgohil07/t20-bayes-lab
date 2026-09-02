/* UI wiring. Loads the precomputed posteriors, draws every tab, and runs the
   browser Gibbs sampler in a Web Worker for the Gibbs lab. */
(function () {
'use strict';
const $ = s => document.querySelector(s);
const $$ = s => Array.prototype.slice.call(document.querySelectorAll(s));
const P = window.Plots, C = P.C;
const LN = ['IPL', 'CPL', 'BBL', 'T20I'];
const fmt = (v, d) => (v === null || v === undefined || isNaN(v)) ? '–' : v.toFixed(d === undefined ? 1 : d);
const commas = v => v.toLocaleString('en-US');

let DATA = {};

Promise.all(['cells', 'summary', 'draws', 'prob', 'offsets', 'meta', 'validation', 'altpriors']
  .map(f => fetch('data/' + f + '.json').then(r => r.json())))
  .then(([cells, summary, draws, prob, offsets, meta, validation, alt]) => {
    DATA = { cells, summary, draws, prob, offsets, meta, validation, alt };
    prep();
    boot();
  })
  .catch(e => {
    document.querySelector('main').insertAdjacentHTML('afterbegin',
      '<p class="note" style="color:#b3122b">Could not load the data files (' + e +
      '). If you are opening this from your own machine, serve the folder over http ' +
      '(<code>python3 -m http.server</code>) rather than double-clicking the file.</p>');
  });

/* ------------------------------------------------------------- prepare --- */
function prep() {
  const s = DATA.summary, m = DATA.meta;
  DATA.P = s.players.length;
  DATA.totBalls = s.balls.map(r => r.reduce((a, b) => a + b, 0));
  DATA.iplBalls = s.balls.map(r => r[0]);
  DATA.iplSR = s.sr.map(r => r[0]);
  DATA.rankRaw = rankOf(DATA.iplSR);
  DATA.rankPost = rankOf(DATA.summary.mean.map((v, i) => DATA.iplBalls[i] > 0 ? v : null));
  DATA.sampler = {
    p: Int32Array.from(DATA.cells.cells.map(c => c.p)),
    l: Int32Array.from(DATA.cells.cells.map(c => c.l)),
    n: Float64Array.from(DATA.cells.cells.map(c => c.n)),
    y: Float64Array.from(DATA.cells.cells.map(c => c.y)),
    P: DATA.P, L: 4
  };
}
function rankOf(vals) {
  const idx = vals.map((v, i) => [v, i]).filter(a => a[0] !== null && !isNaN(a[0]));
  idx.sort((a, b) => b[0] - a[0]);
  const out = new Array(vals.length).fill(null);
  idx.forEach((a, k) => out[a[1]] = k + 1);
  return out;
}
function probAbove(i, c) {
  const g = DATA.prob.thresholds, p = DATA.prob.p[i];
  if (c <= g[0]) return p[0];
  if (c >= g[g.length - 1]) return p[p.length - 1];
  const k = Math.floor(c - g[0]), f = c - g[0] - k;
  return p[k] * (1 - f) + p[Math.min(k + 1, p.length - 1)] * f;
}
function normalTail(mean, sd, c) {           /* P(X > c) for X ~ N(mean, sd^2) */
  const z = (c - mean) / sd;
  return 0.5 * erfc(z / Math.SQRT2);
}
function erfc(x) {                            /* Abramowitz & Stegun 7.1.26 */
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const y = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
    t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
    t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? y : 2 - y;
}

/* ---------------------------------------------------------------- boot --- */
function boot() {
  $$('#nav button').forEach(b => b.onclick = () => show(b.dataset.tab));
  fillKeyNumbers();
  tabProblem(); tabConjugate(); tabGibbs(); tabShrink(); tabOffsets(); tabDecision(); tabMethod();
  let t;
  window.addEventListener('resize', () => {
    clearTimeout(t);
    t = setTimeout(() => { redraw[current] && redraw[current](); }, 180);
  });
}
let current = 'problem';
const redraw = {};
function show(name) {
  current = name;
  $$('#nav button').forEach(b => b.setAttribute('aria-selected', b.dataset.tab === name));
  $$('.tab').forEach(s => s.classList.toggle('on', s.id === 'tab-' + name));
  if (redraw[name]) redraw[name]();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function fillKeyNumbers() {
  const m = DATA.meta, b = m.ballRuns, c = m.counts;
  $('#k-sd').textContent = '±' + b.sdAt100;
  $('#k-cells').textContent = commas(c.cells);
  $('#k-balls').textContent = commas(c.balls);
  $('#k-mean').textContent = b.mean.toFixed(3);
  $('#k-var').textContent = b.var.toFixed(3);
  $('#k-sigma2').textContent = commas(b.sigma2);
  $('#k-sd2').textContent = '±' + b.sdAt100;
  const tot = b.counts.reduce((a, x) => a + x, 0);
  $('#k-ballcount').textContent = commas(tot);
  $('#tbl-ballruns').innerHTML =
    '<table><thead><tr><th>Runs off the bat</th>' +
    b.values.map(v => '<th>' + v + '</th>').join('') + '</tr></thead><tbody><tr>' +
    '<td>share of balls</td>' +
    b.counts.map(x => '<td>' + (x / tot).toFixed(3) + '</td>').join('') +
    '</tr></tbody></table>';
  $('#foot-nums').textContent =
    'Fitted on ' + commas(c.cells) + ' cells covering ' + commas(c.players) +
    ' batters. Worst R̂ across every parameter: ' + m.diagnostics.rhatWorst.toFixed(4) +
    '. Smallest effective sample size: ' + Math.round(m.diagnostics.essMin).toLocaleString() + '.';
}

/* ====================================================== 1. the problem === */
function tabProblem() {
  const cv = $('#c-funnel');
  const s = DATA.summary, cells = DATA.cells.cells;
  const ipl = cells.filter(c => c.l === 0), other = cells.filter(c => c.l !== 0);
  const iplMean = ipl.reduce((a, c) => a + c.n * c.y, 0) / ipl.reduce((a, c) => a + c.n, 0);
  const sig = Math.sqrt(DATA.meta.ballRuns.sigma2);

  function draw() {
    const ch = new P.Chart(cv, { xlog: true, xlim: [22, 2400], ylim: [30, 250],
                                 height: 470, margin: { l: 48, r: 14, t: 16, b: 44 } });
    ch.clear();
    ch.axes({ xlab: 'balls faced, 2021–2024 (log scale)', ylab: 'strike rate',
              xticks: [25, 50, 100, 200, 400, 800, 1600] });
    const xs = [], lo1 = [], hi1 = [], lo2 = [], hi2 = [];
    for (let k = 0; k <= 60; k++) {
      const n = 22 * Math.pow(2400 / 22, k / 60);
      xs.push(n);
      hi2.push(iplMean + 1.96 * sig / Math.sqrt(n)); lo2.push(iplMean - 1.96 * sig / Math.sqrt(n));
      hi1.push(iplMean + sig / Math.sqrt(n)); lo1.push(iplMean - sig / Math.sqrt(n));
    }
    ch.band(xs, lo2, hi2, { color: 'rgba(179,18,43,.10)' });
    ch.band(xs, lo1, hi1, { color: 'rgba(179,18,43,.14)' });
    ch.scatter(other.map(c => c.n), other.map(c => c.y),
               { color: i => C.league[other[i].l], r: 2.1, alpha: .35 });
    ch.scatter(ipl.map(c => c.n), ipl.map(c => c.y), { color: C.league[0], r: 3, alpha: .9 });
    ch.hline(iplMean, { color: C.ink, dash: [5, 4], w: 1 });
    ch.line(xs, hi2, { color: C.accent, w: 1.2 });
    ch.line(xs, lo2, { color: C.accent, w: 1.2 });
    const mb = +$('#s-minballs').value;
    if (mb > 25) ch.vline(mb, { color: C.ink, w: 1.4, dash: [3, 3] });
    ch.text(ch.px.r - 6, ch.px.t + 10, 'IPL average ' + fmt(iplMean),
            { abs: true, align: 'right', color: C.muted });
  }
  function table() {
    const mb = +$('#s-minballs').value;
    $('#v-minballs').textContent = mb;
    const rows = DATA.summary.players.map((nm, i) => ({ nm, i }))
      .filter(r => DATA.iplBalls[r.i] >= mb)
      .sort((a, b) => DATA.iplSR[b.i] - DATA.iplSR[a.i]).slice(0, 10);
    $('#t-leader').innerHTML = rows.map(r =>
      '<tr><td>' + r.nm + '</td><td class="num">' + commas(DATA.iplBalls[r.i]) +
      '</td><td class="num">' + fmt(DATA.iplSR[r.i]) + '</td></tr>').join('');
    const n = DATA.iplBalls.filter(v => v >= mb).length;
    $('#n-leader').textContent = n + ' batters clear ' + mb + ' balls. Noise in a ' + mb +
      '-ball strike rate: ±' + fmt(Math.sqrt(DATA.meta.ballRuns.sigma2 / mb)) + ' points.';
    draw();
  }
  $('#s-minballs').oninput = table;
  redraw.problem = table;
  table();
}

/* ==================================================== 2. conjugate lab === */
function tabConjugate() {
  const sel = $('#sel-player');
  const order = DATA.summary.players.map((nm, i) => ({ nm, i }))
    .filter(r => DATA.iplBalls[r.i] >= 25)
    .sort((a, b) => DATA.iplBalls[b.i] - DATA.iplBalls[a.i]);
  sel.innerHTML = order.map(r => '<option value="' + r.i + '">' + r.nm + ' — ' +
    commas(DATA.iplBalls[r.i]) + ' balls, SR ' + fmt(DATA.iplSR[r.i]) + '</option>').join('');
  sel.value = order[0].i;

  function pick() {
    const i = +sel.value;
    $('#s-nballs').value = Math.min(1200, Math.max(10, DATA.iplBalls[i]));
    render();
  }
  function render() {
    const i = +sel.value;
    const m0 = +$('#s-m0').value, tau = +$('#s-tau').value, n = +$('#s-nballs').value;
    const ybar = DATA.iplSR[i];
    const sigma2 = DATA.meta.fitted.sigma2;
    $('#v-m0').textContent = m0; $('#v-tau').textContent = tau.toFixed(1);
    $('#v-nballs').textContent = commas(n);
    const w = n * tau * tau / (sigma2 + n * tau * tau);
    const post = w * ybar + (1 - w) * m0;
    const postSd = Math.sqrt(1 / (n / sigma2 + 1 / (tau * tau)));
    const likeSd = Math.sqrt(sigma2 / n);
    $('#v-w').textContent = (w * 100).toFixed(0) + '%';
    $('#bar-w').style.width = (w * 100) + '%';
    $('#n-conj').innerHTML = 'Posterior mean <b>' + fmt(post) + '</b> ' +
      '(95% interval ' + fmt(post - 1.96 * postSd) + ' to ' + fmt(post + 1.96 * postSd) + '). ' +
      'The player’s own record is worth ' + (w * 100).toFixed(0) +
      '% of that answer, the population the other ' + ((1 - w) * 100).toFixed(0) + '%.';

    const ch = new P.Chart($('#c-conj'), { xlim: [70, 220], ylim: [0, 1], height: 340 });
    ch.clear();
    const pdf = (x, m, s) => Math.exp(-0.5 * (x - m) * (x - m) / (s * s)) / (s * 2.5066282746);
    const xs = [];
    for (let k = 0; k <= 300; k++) xs.push(70 + 150 * k / 300);
    const curves = [
      { ys: xs.map(x => pdf(x, m0, tau)), c: C.faint },
      { ys: xs.map(x => pdf(x, ybar, likeSd)), c: C.league[0] },
      { ys: xs.map(x => pdf(x, post, postSd)), c: C.accent }
    ];
    const mx = Math.max.apply(null, curves.map(cu => Math.max.apply(null, cu.ys)));
    ch.ylim = [0, mx * 1.15];
    ch.axes({ xlab: 'strike rate', yticks: [] });
    curves.forEach(cu => ch.line(xs, cu.ys, { color: cu.c, w: 2 }));
    ch.vline(ybar, { color: C.league[0], dash: [3, 3], w: 1 });
    ch.vline(post, { color: C.accent, dash: [3, 3], w: 1 });
    ch.text(post, mx * 1.06, 'θ = ' + fmt(post), { color: C.accent, align: 'center', size: 12,
                                                   weight: '600' });
  }
  sel.onchange = pick;
  ['#s-m0', '#s-tau', '#s-nballs'].forEach(id => $(id).oninput = render);
  redraw.conjugate = render;
  pick();
}

/* ======================================================== 3. gibbs lab === */
function tabGibbs() {
  let last = null, running = false;
  const worker = new Worker('js/worker.js');
  worker.onmessage = e => {
    if (e.data.type === 'progress') { $('#bar-run').style.width = (e.data.value * 100) + '%'; return; }
    last = e.data; running = false;
    $('#btn-run').disabled = false;
    $('#btn-run').textContent = 'Run 4 chains';
    $('#bar-run').style.width = '100%';
    render();
  };
  ['#s-iters', '#s-burn', '#s-gm0', '#s-gtau'].forEach(id => $(id).oninput = () => {
    $('#v-iters').textContent = $('#s-iters').value;
    $('#v-burn').textContent = $('#s-burn').value;
    $('#v-gm0').textContent = $('#s-gm0').value;
    $('#v-gtau').textContent = $('#s-gtau').value;
  });
  $('#btn-run').onclick = () => {
    const model = $('#sel-model').value;
    const iters = +$('#s-iters').value, burn = Math.min(+$('#s-burn').value, iters - 500);
    const tau = +$('#s-gtau').value;
    const data = model === 'M3' ? DATA.sampler : iplOnly();
    running = true;
    $('#btn-run').disabled = true; $('#btn-run').textContent = 'Sampling…';
    $('#bar-run').style.width = '0%';
    worker.postMessage({ cmd: 'run', chains: 4, opts: {
      data, model, iters, burn, thin: 2,
      priors: { m0: +$('#s-gm0').value, b_tau: 2 * tau * tau, a_tau: 3 }
    } });
  };
  function iplOnly() {
    const c = DATA.cells.cells.filter(x => x.l === 0);
    const ids = Array.from(new Set(c.map(x => x.p))).sort((a, b) => a - b);
    const map = new Map(ids.map((v, k) => [v, k]));
    return { p: Int32Array.from(c.map(x => map.get(x.p))), l: Int32Array.from(c.map(() => 0)),
             n: Float64Array.from(c.map(x => x.n)), y: Float64Array.from(c.map(x => x.y)),
             P: ids.length, L: 4 };
  }
  function render() {
    if (!last) return;
    const model = $('#sel-model').value;
    const showTau = model === 'M2' || model === 'M3';
    trace($('#c-trace1'), last.chains.map(c => c.sigma2), 'σ²  effective per-ball variance');
    trace($('#c-trace2'), showTau ? last.chains.map(c => c.mu) : last.chains.map(c => c.sigma2),
          showTau ? 'μ  population mean strike rate' : 'σ²  (this model has no μ)');
    acfPlot($('#c-acf'), last.diag.sigma2.acf);
    postPlot($('#c-post'), [].concat.apply([], showTau ? last.chains.map(c => c.mu)
                                                       : last.chains.map(c => c.sigma2)),
             showTau ? 'μ' : 'σ²');
    const rows = [['σ²', 'sigma2']];
    if (showTau) rows.push(['μ', 'mu'], ['τ²', 'tau2']);
    if (model === 'M3') rows.push(['ω²', 'omega2']);
    let html = rows.map(([lab, k]) => {
      const all = [].concat.apply([], last.chains.map(c => c[k]));
      const mean = all.reduce((a, b) => a + b, 0) / all.length;
      const d = last.diag[k];
      return '<tr><td>' + lab + '</td><td class="num">' + fmt(mean, mean > 1000 ? 0 : 2) +
        '</td><td class="num">' + d.rhat.toFixed(4) + '</td><td class="num">' +
        Math.round(d.ess).toLocaleString() + '</td></tr>';
    }).join('');
    if (model === 'M3') {
      [1, 2, 3].forEach(j => {
        const all = [].concat.apply([], last.chains.map(c => c.delta[j]));
        const mean = all.reduce((a, b) => a + b, 0) / all.length;
        html += '<tr><td>δ ' + LN[j] + '</td><td class="num">' + fmt(mean, 2) +
          '</td><td class="num">' + last.diag.delta[j].rhat.toFixed(4) + '</td><td class="num">' +
          Math.round(last.diag.delta[j].ess).toLocaleString() + '</td></tr>';
      });
    }
    $('#t-diag').innerHTML = html;
    const ref = DATA.meta.fitted;
    const allmu = [].concat.apply([], last.chains.map(c => c.mu));
    const mumean = allmu.reduce((a, b) => a + b, 0) / allmu.length;
    $('#n-gibbs').innerHTML = model === 'M3'
      ? 'The Python run of the same model, same priors, gives μ = ' + fmt(ref.mu, 2) +
        ' and σ² = ' + commas(Math.round(ref.sigma2)) + '. This browser run gives μ = ' +
        fmt(mumean, 2) + '. Two independent implementations of the same six conditionals.'
      : 'Switch to M3 to compare against the Python fit reported in the paper.';
  }
  function trace(cv, chains, label) {
    const flat = [].concat.apply([], chains);
    const lo = Math.min.apply(null, flat), hi = Math.max.apply(null, flat);
    const ch = new P.Chart(cv, { xlim: [0, chains[0].length - 1], ylim: [lo, hi],
                                 height: 130, margin: { l: 62, r: 12, t: 16, b: 26 } });
    ch.clear();
    ch.axes({ ny: 3, yfmt: v => Math.abs(v) > 1000 ? (v / 1000).toFixed(0) + 'k' : v.toFixed(1) });
    chains.forEach((c, k) => ch.line(c.map((_, i) => i), c, { color: C.chain[k], w: .7, alpha: .8 }));
    ch.text(ch.px.l, 9, label, { abs: true, color: C.ink, size: 12, weight: '600' });
  }
  function acfPlot(cv, r) {
    const ch = new P.Chart(cv, { xlim: [-.5, r.length - .5], ylim: [-.2, 1.05],
                                 height: 170, margin: { l: 42, r: 10, t: 18, b: 34 } });
    ch.clear(); ch.axes({ xlab: 'lag', ny: 4 });
    const g = ch.ctx; g.fillStyle = C.league[0];
    const w = Math.max(1.5, (ch.px.r - ch.px.l) / r.length - 1.5);
    r.forEach((v, k) => {
      const y0 = ch.Y(0), y1 = ch.Y(v);
      g.fillRect(ch.X(k) - w / 2, Math.min(y0, y1), w, Math.abs(y1 - y0));
    });
    ch.hline(0, { color: C.muted, w: .8 });
    ch.text(ch.px.l, 10, 'autocorrelation of σ²', { abs: true, color: C.ink, size: 12, weight: '600' });
  }
  function postPlot(cv, vals, label) {
    const lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    const ch = new P.Chart(cv, { xlim: [lo, hi], ylim: [0, 1], height: 170,
                                 margin: { l: 42, r: 10, t: 18, b: 34 } });
    ch.clear();
    ch.hist(vals, { bins: 36, color: C.accent, alpha: .45 });
    ch.axes({ yticks: [], nx: 4, xlab: 'posterior of ' + label });
    ch.hist(vals, { bins: 36, color: C.accent, alpha: .45, autoY: false });
    ch.text(ch.px.l, 10, 'posterior ' + label, { abs: true, color: C.ink, size: 12, weight: '600' });
  }
  redraw.gibbs = () => {
    if (!last && !running) { $('#btn-run').click(); return; }
    render();
  };
}

/* ======================================================== 4. shrinkage === */
function tabShrink() {
  $('#v-taufit').textContent = Math.sqrt(DATA.meta.fitted.tau2).toFixed(1);
  $('#s-stau').value = Math.sqrt(DATA.meta.fitted.tau2).toFixed(1);

  function shrunk(tau) {
    /* Re-apply the shrinkage formula at a user-chosen tau, holding sigma2 and mu
       at their fitted values. This is the conditional posterior mean, not a re-fit. */
    const s2 = DATA.meta.fitted.sigma2, mu = DATA.meta.fitted.mu, d = DATA.meta.fitted.delta;
    const out = [];
    for (let i = 0; i < DATA.P; i++) {
      let sn = 0, sr = 0;
      for (let j = 0; j < 4; j++) {
        const n = DATA.summary.balls[i][j];
        if (n > 0) { sn += n; sr += n * (DATA.summary.sr[i][j] - d[j]); }
      }
      if (sn === 0) { out.push(mu); continue; }
      const w = sn * tau * tau / (s2 + sn * tau * tau);
      out.push(w * (sr / sn) + (1 - w) * mu);
    }
    return out;
  }
  function render() {
    const tau = +$('#s-stau').value;
    $('#v-stau').textContent = tau.toFixed(1);
    const post = shrunk(tau);
    const cand = DATA.summary.players.map((nm, i) => i)
      .filter(i => DATA.iplBalls[i] >= 25)
      .sort((a, b) => DATA.iplSR[b] - DATA.iplSR[a]).slice(0, 20);

    const ch = new P.Chart($('#c-shrink'), { xlim: [95, 250], ylim: [-0.6, cand.length - .4],
                                             height: 460, margin: { l: 150, r: 60, t: 18, b: 40 } });
    ch.clear();
    ch.axes({ xlab: 'strike rate', yticks: [], nx: 5 });
    cand.forEach((i, k) => {
      const y = cand.length - 1 - k;
      const falls = post[i] < DATA.iplSR[i];
      const col = falls ? C.accent : C.league[0];
      ch.arrows([DATA.iplSR[i]], [y], [post[i]], [y], { color: col, w: 1.4, alpha: .9 });
      ch.scatter([DATA.iplSR[i]], [y], { color: C.faint, r: 3 });
      ch.scatter([post[i]], [y], { color: col, r: 4.2 });
      ch.text(ch.px.l - 8, ch.Y(y), DATA.summary.players[i], { abs: true, align: 'right',
        color: C.ink, size: 11.5 });
      ch.text(ch.px.r + 6, ch.Y(y), commas(DATA.iplBalls[i]) + 'b', { abs: true, align: 'left',
        color: C.muted, size: 10.5 });
    });
    ch.vline(DATA.meta.fitted.mu, { color: C.ink, dash: [4, 4], w: 1 });
    ch.text(DATA.meta.fitted.mu, cand.length - .55, 'μ', { color: C.ink, size: 12, align: 'center' });

    const rankNew = rankOf(post.map((v, i) => DATA.iplBalls[i] > 0 ? v : null));
    const mv = cand.map(i => ({ i, d: (rankNew[i] || 0) - (DATA.rankRaw[i] || 0) }))
      .sort((a, b) => Math.abs(b.d) - Math.abs(a.d)).slice(0, 8);
    $('#t-movers').innerHTML = mv.map(r =>
      '<tr><td>' + DATA.summary.players[r.i] + '</td><td class="num">' +
      commas(DATA.iplBalls[r.i]) + '</td><td class="num">' + fmt(DATA.iplSR[r.i]) +
      '</td><td class="num">' + fmt(post[r.i]) + '</td><td class="num" style="color:' +
      (r.d > 0 ? C.accent : C.league[0]) + '">#' + DATA.rankRaw[r.i] + ' → #' + rankNew[r.i] +
      '</td></tr>').join('');
  }
  $('#s-stau').oninput = render;
  redraw.shrink = render;
  render();
}

/* ========================================================== 5. offsets === */
function tabOffsets() {
  function render() {
    const off = DATA.offsets;
    const ch = new P.Chart($('#c-offsets'), { xlim: [-16, 6], ylim: [0, 1], height: 340 });
    ch.clear();
    const dens = [1, 2, 3].map(j => P.density(off.delta[j], -16, 6, 200));
    const mx = Math.max.apply(null, dens.map(d => Math.max.apply(null, d.ys)));
    ch.ylim = [0, mx * 1.25];
    ch.axes({ xlab: 'δ — strike-rate points relative to the IPL', yticks: [] });
    dens.forEach((d, k) => {
      const j = k + 1;
      ch.band(d.xs, d.xs.map(() => 0), d.ys, { color: hexa(C.league[j], .22) });
      ch.line(d.xs, d.ys, { color: C.league[j], w: 2 });
      const mean = DATA.meta.fitted.delta[j];
      ch.text(mean, Math.max.apply(null, d.ys) * 1.06, LN[j] + '  ' + fmt(mean, 1),
              { color: C.league[j], align: 'center', size: 12, weight: '600' });
    });
    ch.vline(0, { color: C.ink, w: 1.6 });
    ch.text(0.4, mx * 1.18, 'IPL = 0 by construction', { color: C.ink, size: 11 });

    const ov = DATA.meta.overlap;
    $('#tbl-overlap').innerHTML = '<table><thead><tr><th></th>' +
      LN.map(n => '<th>' + n + '</th>').join('') + '</tr></thead><tbody>' +
      ov.map((row, a) => '<tr><td>' + LN[a] + '</td>' + row.map((v, b) =>
        '<td class="num"' + (a === b ? ' style="color:var(--muted)"' : '') + '>' + v + '</td>')
        .join('') + '</tr>').join('') + '</tbody></table>';
    const f = DATA.meta.fitted;
    $('#t-offsets').innerHTML = [1, 2, 3].map(j =>
      '<tr><td>' + LN[j] + '</td><td class="num">' + fmt(f.delta[j], 1) +
      '</td><td class="num">' + fmt(f.deltaLo[j], 1) + ' to ' + fmt(f.deltaHi[j], 1) +
      '</td></tr>').join('');
  }
  redraw.offsets = render;
  render();
}
function hexa(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return 'rgba(' + (n >> 16) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
}

/* ========================================================= 6. decision === */
function tabDecision() {
  function priorOf(name) {
    if (name === 'Analyst') return null;
    return DATA.alt[name];
  }
  function pAbove(i, c, name) {
    const a = priorOf(name);
    if (!a) return probAbove(i, c);
    return normalTail(a.mean[i], a.sd[i], c);
  }
  function render() {
    const c = +$('#s-thresh').value, cut = +$('#s-cut').value, mb = +$('#s-minb2').value;
    const who = $('#sel-prior').value;
    $('#v-thresh').textContent = c; $('#v-cut').textContent = cut.toFixed(2);
    $('#v-minb2').textContent = mb;
    const alt = priorOf(who);
    const mean = alt ? alt.mean : DATA.summary.mean;

    const elig = [];
    for (let i = 0; i < DATA.P; i++) if (DATA.totBalls[i] >= mb) elig.push(i);
    const prob = {}; elig.forEach(i => prob[i] = pAbove(i, c, who));
    const short = elig.filter(i => prob[i] > cut).sort((a, b) => prob[b] - prob[a]);
    const rawShort = elig.filter(i => DATA.iplSR[i] !== null && DATA.iplSR[i] > c)
      .sort((a, b) => DATA.iplSR[b] - DATA.iplSR[a]);

    const top = short.slice(0, 16);
    const ch = new P.Chart($('#c-decision'), { xlim: [c - 22, 205],
      ylim: [-0.6, Math.max(1, top.length) - .4], height: 440,
      margin: { l: 148, r: 52, t: 18, b: 40 } });
    ch.clear(); ch.axes({ xlab: 'strike rate on the IPL scale', yticks: [], nx: 5 });
    top.forEach((i, k) => {
      const y = top.length - 1 - k;
      const lo = alt ? alt.mean[i] - 1.96 * alt.sd[i] : DATA.summary.lo[i];
      const hi = alt ? alt.mean[i] + 1.96 * alt.sd[i] : DATA.summary.hi[i];
      const g = ch.ctx;
      g.strokeStyle = C.faint; g.lineWidth = 2.4;
      g.beginPath(); g.moveTo(ch.X(lo), ch.Y(y)); g.lineTo(ch.X(hi), ch.Y(y)); g.stroke();
      ch.scatter([mean[i]], [y], { color: C.accent, r: 4.2 });
      if (DATA.iplSR[i] !== null) {
        g.strokeStyle = C.league[0]; g.lineWidth = 1.6;
        g.beginPath(); g.moveTo(ch.X(DATA.iplSR[i]), ch.Y(y) - 6);
        g.lineTo(ch.X(DATA.iplSR[i]), ch.Y(y) + 6); g.stroke();
      }
      ch.text(ch.px.l - 8, ch.Y(y), DATA.summary.players[i],
              { abs: true, align: 'right', color: C.ink, size: 11.5 });
      ch.text(ch.px.r + 6, ch.Y(y), prob[i].toFixed(2),
              { abs: true, align: 'left', color: C.muted, size: 10.5 });
    });
    ch.vline(c, { color: C.ink, dash: [4, 4], w: 1.2 });

    const row = (i, val, balls) => '<tr><td>' + DATA.summary.players[i] + '</td><td class="num">' +
      commas(balls) + '</td><td class="num">' + val + '</td><td class="num">' +
      prob[i].toFixed(2) + '</td></tr>';
    $('#t-short').innerHTML = short.slice(0, 14)
      .map(i => row(i, fmt(mean[i]), DATA.totBalls[i])).join('')
      || '<tr><td colspan="4" class="note">Nobody clears that bar.</td></tr>';
    $('#t-raw').innerHTML = rawShort.slice(0, 14)
      .map(i => row(i, fmt(DATA.iplSR[i]), DATA.iplBalls[i])).join('')
      || '<tr><td colspan="4" class="note">Nobody clears that bar.</td></tr>';

    const setA = new Set(short), setB = new Set(rawShort);
    let both = 0; setB.forEach(i => { if (setA.has(i)) both++; });
    $('#n-decision').innerHTML = short.length + ' batters clear ' +
      'P(θ &gt; ' + c + ') &gt; ' + cut.toFixed(2) + ' among the ' + elig.length +
      ' with ' + mb + '+ balls.' + (alt ?
      ' <span class="note">For the alternative priors the probability uses a Normal ' +
      'approximation to the posterior; it agrees with the draws to within 0.013.</span>' : '');
    $('#n-overlap').innerHTML = 'The raw list has ' + rawShort.length + ' names, the Bayesian ' +
      'list has ' + short.length + ', and ' + both + ' appear on both. The disagreements are ' +
      'the point: the raw list keeps anyone who happened to strike at ' + c +
      ' for a handful of balls, and drops anyone whose IPL record is short or absent even ' +
      'though we have watched them elsewhere.';
  }
  ['#s-thresh', '#s-cut', '#s-minb2'].forEach(id => $(id).oninput = render);
  $('#sel-prior').onchange = render;
  redraw.decision = render;
  render();
}

/* =========================================================== method ===== */
function tabMethod() {
  const m = DATA.meta, f = m.fitted, b = m.ballRuns, c = m.counts;
  $('#method-body').innerHTML = `
  <h3>The data</h3>
  <p>Cricsheet ball-by-ball records for four competitions, 2021 to 2025. We drop wides,
  because a wide is not a ball the batter faced, and aggregate the rest to
  <b>player × league cells</b>: <span class="eq">n</span> balls faced and
  <span class="eq">y</span>, the strike rate over those balls. Cells with fewer than 25 balls
  are dropped. Training uses 2021–2024 (${commas(c.cells)} cells, ${commas(c.players)} batters);
  testing uses the 2025 IPL.</p>
  <p>T20 internationals are restricted to matches in which both sides are ICC Full Members.
  The full archive covers 105 national teams, which is not one scoring environment; see the
  note on the League offsets tab.</p>

  <h3>The model</h3>
  <p class="eq" style="font-size:16px;line-height:2.1">
    y<sub>iℓ</sub> | θ, δ, σ² &nbsp;~&nbsp; N( θ<sub>i</sub> + δ<sub>ℓ</sub>, σ²/n<sub>iℓ</sub> )<br>
    θ<sub>i</sub> | μ, τ² &nbsp;~&nbsp; N(μ, τ²) &nbsp;&nbsp;&nbsp; the batter's ability, on the IPL scale<br>
    δ<sub>ℓ</sub> | ω² &nbsp;~&nbsp; N(0, ω²) &nbsp;&nbsp;&nbsp; the league's scoring environment,
    with δ<sub>IPL</sub> = 0<br>
    μ ~ N(130, 20²) &nbsp;&nbsp; τ² ~ IG(3, 450) &nbsp;&nbsp; ω² ~ IG(3, 128)
    &nbsp;&nbsp; σ² ~ IG(3, 52000)
  </p>
  <p>The Normal likelihood is written for a <em>mean</em> of at least 25 balls, so the central
  limit theorem applies, and the <span class="eq">σ²/n</span> term is how a long career comes to
  count for more than a short one. Fixing <span class="eq">δ<sub>IPL</sub> = 0</span> makes
  every <span class="eq">θ<sub>i</sub></span> readable as this batter's expected IPL strike
  rate, including for batters who have never played in the IPL.</p>

  <h3>The six full conditionals</h3>
  <p>All six are closed form, so the sampler is pure Gibbs with no Metropolis step. With
  <span class="eq">r<sub>c</sub> = y<sub>c</sub> − δ<sub>ℓ(c)</sub></span> and
  <span class="eq">s<sub>c</sub> = y<sub>c</sub> − θ<sub>p(c)</sub></span>:</p>
  <ol style="max-width:74ch">
    <li><span class="eq">θ<sub>i</sub></span>: Normal with precision
      <span class="eq">Σ n<sub>c</sub>/σ² + 1/τ²</span> and mean
      <span class="eq">(Σ n<sub>c</sub>r<sub>c</sub>/σ² + μ/τ²)</span> over that precision.</li>
    <li><span class="eq">δ<sub>ℓ</sub></span> (ℓ ≠ IPL): Normal, precision
      <span class="eq">Σ n<sub>c</sub>/σ² + 1/ω²</span>, mean
      <span class="eq">(Σ n<sub>c</sub>s<sub>c</sub>/σ²)</span> over that precision.</li>
    <li><span class="eq">μ</span>: Normal, precision <span class="eq">P/τ² + 1/s₀²</span>.</li>
    <li><span class="eq">τ²</span>: IG(a<sub>τ</sub> + P/2, b<sub>τ</sub> + ½Σ(θ<sub>i</sub>−μ)²).</li>
    <li><span class="eq">ω²</span>: IG(a<sub>ω</sub> + (L−1)/2, b<sub>ω</sub> + ½Σδ<sub>ℓ</sub>²).</li>
    <li><span class="eq">σ²</span>: IG(a<sub>σ</sub> + C/2,
      b<sub>σ</sub> + ½Σ n<sub>c</sub>(y<sub>c</sub> − θ − δ)²).</li>
  </ol>
  <p>Four chains, 12,000 iterations, 4,000 burn-in, thinned by 4. Worst R̂ across every
  parameter including all ${commas(c.players)} abilities: <b>${m.diagnostics.rhatWorst.toFixed(4)}</b>.
  Smallest effective sample size: <b>${Math.round(m.diagnostics.essMin).toLocaleString()}</b>.</p>

  <h3>What the fit says</h3>
  <p>μ = ${fmt(f.mu, 1)}, τ = ${fmt(Math.sqrt(f.tau2), 1)}, σ = ${fmt(f.sigmaEff, 0)},
  and the league offsets are ${[1,2,3].map(j => LN[j] + ' ' + fmt(f.delta[j], 1)).join(', ')}.</p>
  <p>There are two estimates of the noise in a strike rate, and the gap between them is
  itself a result. Counting runs off the bat gives a per-ball variance of ${commas(b.sigma2)},
  so a 100-ball strike rate carries <b>±${b.sdAt100}</b> points of sampling noise. The fitted
  σ² is ${commas(Math.round(f.sigma2))}, which is <b>${f.designEffect}×</b> larger. This says
  balls are not independent given ability: form, matchups, conditions and match situation all
  correlate within a spell, so 100 balls carry roughly the information of
  ${Math.round(100 / f.designEffect)}. On the model's own terms, a 100-ball strike rate is
  <b>±${f.sdAt100Fitted}</b>.</p>

  <h3>Limitations</h3>
  <ol style="max-width:74ch">
    <li><b>The Normal likelihood.</b> Strike rates are bounded below by zero and skewed right;
      the Normal is neither. We model a <em>mean</em> of at least 25 balls, so the central
      limit theorem applies, and the posterior predictive check finds no impossible values. It
      does show the model expecting short spells to be noisier than they are.</li>
    <li><b>Selection.</b> We observe only batters who were selected, so absolute ability
      estimates are biased upward. The <em>comparison</em> between methods is unaffected, since
      every method is scored on the same batters.</li>
    <li><b>Ability is treated as fixed over five years.</b> It is not. We restrict to 2021–2025
      to limit the damage and report a split-half check.</li>
    <li><b>δ combines pitch, bowling standard and format.</b> We call it a
      scoring-environment offset rather than a quality adjustment, because the decision needs
      the translation between scales, not its cause.</li>
    <li><b>Balls are not conditionally independent.</b> The fitted σ² is ${f.designEffect}
      times the pure sampling value, which is the model measuring exactly this.</li>
    <li><b>Strike rate is not the whole of batting.</b> A batter who strikes at 160 and is
      dismissed every other over is not worth more than one who strikes at 145 and bats
      through. Dismissals are deliberately out of scope.</li>
    <li><b>Offsets are only as good as the bridge.</b> ${DATA.meta.overlap[0][3]} batters link
      the IPL and T20I; ${DATA.meta.overlap[0][1]} link the IPL and CPL and
      ${DATA.meta.overlap[0][2]} the IPL and BBL. The CPL and BBL offsets lean partly on the
      T20I bridge, which is reflected in the width of their credible intervals.</li>
  </ol>`;
  redraw.method = () => {};
}
})();
