/* Small canvas plotting layer. No chart library: the plots here (funnel with a
   sampling band, shrinkage arrows, live traces) are specific enough that a generic
   library would cost more than it saves. */
(function (root) {
  'use strict';

  const C = {
    ink: '#14181f', muted: '#6b7280', faint: '#9aa3ad', line: '#e6e8ec',
    paper: '#ffffff', accent: '#b3122b', accentSoft: 'rgba(179,18,43,.14)',
    league: ['#2f5d8c', '#c1663e', '#4f8a5b', '#8a6bab'],
    chain: ['#2f5d8c', '#c1663e', '#4f8a5b', '#8a6bab']
  };

  function Chart(canvas, o) {
    o = o || {};
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.m = Object.assign({ l: 52, r: 14, t: 14, b: 38 }, o.margin || {});
    this.xlog = !!o.xlog;
    this.xlim = o.xlim || [0, 1];
    this.ylim = o.ylim || [0, 1];
    this.resize(o.height || 300);
  }

  Chart.prototype.resize = function (cssHeight) {
    const dpr = window.devicePixelRatio || 1;
    const w = this.cv.clientWidth || this.cv.parentNode.clientWidth || 600;
    this.cv.width = Math.round(w * dpr);
    this.cv.height = Math.round(cssHeight * dpr);
    this.cv.style.height = cssHeight + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = w; this.H = cssHeight;
    this.px = { l: this.m.l, r: w - this.m.r, t: this.m.t, b: cssHeight - this.m.b };
  };

  Chart.prototype.clear = function () {
    this.ctx.clearRect(0, 0, this.W, this.H);
  };
  Chart.prototype.X = function (v) {
    const a = this.xlog ? Math.log(Math.max(v, 1e-9)) : v;
    const lo = this.xlog ? Math.log(this.xlim[0]) : this.xlim[0];
    const hi = this.xlog ? Math.log(this.xlim[1]) : this.xlim[1];
    return this.px.l + (a - lo) / (hi - lo) * (this.px.r - this.px.l);
  };
  Chart.prototype.Y = function (v) {
    return this.px.b - (v - this.ylim[0]) / (this.ylim[1] - this.ylim[0]) * (this.px.b - this.px.t);
  };

  function niceTicks(lo, hi, n) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  Chart.prototype.axes = function (o) {
    o = o || {};
    const g = this.ctx;
    const xt = this.xlog ? (o.xticks || [25, 50, 100, 200, 400, 800, 1600])
                            .filter(v => v >= this.xlim[0] && v <= this.xlim[1])
                        : (o.xticks || niceTicks(this.xlim[0], this.xlim[1], o.nx || 6));
    const yt = o.yticks || niceTicks(this.ylim[0], this.ylim[1], o.ny || 5);
    g.save();
    g.strokeStyle = C.line; g.lineWidth = 1; g.fillStyle = C.muted;
    g.font = '11px -apple-system,Segoe UI,Roboto,sans-serif';
    yt.forEach(v => {
      const y = Math.round(this.Y(v)) + .5;
      if (y < this.px.t - 1 || y > this.px.b + 1) return;
      g.beginPath(); g.moveTo(this.px.l, y); g.lineTo(this.px.r, y); g.stroke();
      g.textAlign = 'right'; g.textBaseline = 'middle';
      g.fillText(o.yfmt ? o.yfmt(v) : String(v), this.px.l - 8, y);
    });
    g.textAlign = 'center'; g.textBaseline = 'top';
    xt.forEach(v => {
      const x = Math.round(this.X(v)) + .5;
      if (x < this.px.l - 1 || x > this.px.r + 1) return;
      g.strokeStyle = C.line;
      g.beginPath(); g.moveTo(x, this.px.b); g.lineTo(x, this.px.b + 4); g.stroke();
      g.fillText(o.xfmt ? o.xfmt(v) : String(v), x, this.px.b + 8);
    });
    g.strokeStyle = C.faint;
    g.beginPath(); g.moveTo(this.px.l, this.px.b + .5); g.lineTo(this.px.r, this.px.b + .5); g.stroke();
    if (o.xlab) {
      g.fillStyle = C.muted; g.textAlign = 'center'; g.textBaseline = 'bottom';
      g.font = '12px -apple-system,Segoe UI,Roboto,sans-serif';
      g.fillText(o.xlab, (this.px.l + this.px.r) / 2, this.H - 2);
    }
    if (o.ylab) {
      g.save(); g.translate(11, (this.px.t + this.px.b) / 2); g.rotate(-Math.PI / 2);
      g.fillStyle = C.muted; g.textAlign = 'center'; g.textBaseline = 'top';
      g.font = '12px -apple-system,Segoe UI,Roboto,sans-serif';
      g.fillText(o.ylab, 0, 0); g.restore();
    }
    g.restore();
  };

  Chart.prototype.clip = function (fn) {
    const g = this.ctx; g.save();
    g.beginPath();
    g.rect(this.px.l, this.px.t - 4, this.px.r - this.px.l, this.px.b - this.px.t + 8);
    g.clip(); fn(); g.restore();
  };

  Chart.prototype.scatter = function (xs, ys, o) {
    o = o || {}; const g = this.ctx;
    this.clip(() => {
      g.globalAlpha = o.alpha === undefined ? .8 : o.alpha;
      const r = o.r || 2.4;
      if (typeof o.color === 'function') {
        for (let i = 0; i < xs.length; i++) {
          g.fillStyle = o.color(i);
          g.beginPath(); g.arc(this.X(xs[i]), this.Y(ys[i]), r, 0, 6.2832); g.fill();
        }
      } else {
        g.fillStyle = o.color || C.league[0];
        for (let i = 0; i < xs.length; i++) {
          g.beginPath(); g.arc(this.X(xs[i]), this.Y(ys[i]), r, 0, 6.2832); g.fill();
        }
      }
      g.globalAlpha = 1;
    });
  };

  Chart.prototype.line = function (xs, ys, o) {
    o = o || {}; const g = this.ctx;
    this.clip(() => {
      g.strokeStyle = o.color || C.ink; g.lineWidth = o.w || 1.6;
      if (o.dash) g.setLineDash(o.dash);
      g.globalAlpha = o.alpha === undefined ? 1 : o.alpha;
      g.beginPath();
      for (let i = 0; i < xs.length; i++) {
        const x = this.X(xs[i]), y = this.Y(ys[i]);
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    });
  };

  Chart.prototype.band = function (xs, lo, hi, o) {
    o = o || {}; const g = this.ctx;
    this.clip(() => {
      g.fillStyle = o.color || C.accentSoft;
      g.beginPath();
      for (let i = 0; i < xs.length; i++) {
        const x = this.X(xs[i]), y = this.Y(hi[i]);
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      for (let i = xs.length - 1; i >= 0; i--) g.lineTo(this.X(xs[i]), this.Y(lo[i]));
      g.closePath(); g.fill();
    });
  };

  Chart.prototype.vline = function (x, o) {
    o = o || {}; const g = this.ctx;
    g.save(); g.strokeStyle = o.color || C.ink; g.lineWidth = o.w || 1.2;
    if (o.dash) g.setLineDash(o.dash);
    g.beginPath(); g.moveTo(this.X(x), this.px.t); g.lineTo(this.X(x), this.px.b); g.stroke();
    g.restore();
  };
  Chart.prototype.hline = function (y, o) {
    o = o || {}; const g = this.ctx;
    g.save(); g.strokeStyle = o.color || C.ink; g.lineWidth = o.w || 1.2;
    if (o.dash) g.setLineDash(o.dash);
    g.beginPath(); g.moveTo(this.px.l, this.Y(y)); g.lineTo(this.px.r, this.Y(y)); g.stroke();
    g.restore();
  };

  Chart.prototype.arrows = function (x0, y0, x1, y1, o) {
    o = o || {}; const g = this.ctx;
    this.clip(() => {
      g.lineWidth = o.w || 1;
      g.globalAlpha = o.alpha === undefined ? .55 : o.alpha;
      for (let i = 0; i < x0.length; i++) {
        g.strokeStyle = typeof o.color === 'function' ? o.color(i) : (o.color || C.muted);
        const ax = this.X(x0[i]), ay = this.Y(y0[i]), bx = this.X(x1[i]), by = this.Y(y1[i]);
        g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke();
        const a = Math.atan2(by - ay, bx - ax), h = 4;
        g.beginPath(); g.moveTo(bx, by);
        g.lineTo(bx - h * Math.cos(a - .5), by - h * Math.sin(a - .5));
        g.lineTo(bx - h * Math.cos(a + .5), by - h * Math.sin(a + .5));
        g.closePath(); g.fillStyle = g.strokeStyle; g.fill();
      }
      g.globalAlpha = 1;
    });
  };

  Chart.prototype.hist = function (vals, o) {
    o = o || {}; const g = this.ctx;
    const bins = o.bins || 40, lo = this.xlim[0], hi = this.xlim[1];
    const cnt = new Array(bins).fill(0);
    vals.forEach(v => {
      const k = Math.floor((v - lo) / (hi - lo) * bins);
      if (k >= 0 && k < bins) cnt[k]++;
    });
    const mx = Math.max.apply(null, cnt) || 1;
    if (o.autoY !== false) this.ylim = [0, mx * 1.12];
    this.clip(() => {
      g.fillStyle = o.color || C.league[0];
      g.globalAlpha = o.alpha === undefined ? .55 : o.alpha;
      for (let k = 0; k < bins; k++) {
        const x0 = this.X(lo + (hi - lo) * k / bins), x1 = this.X(lo + (hi - lo) * (k + 1) / bins);
        g.fillRect(x0, this.Y(cnt[k]), Math.max(1, x1 - x0 - .6), this.px.b - this.Y(cnt[k]));
      }
      g.globalAlpha = 1;
    });
    return cnt;
  };

  /* Gaussian kernel density on a fixed grid. */
  function density(vals, lo, hi, m) {
    m = m || 220;
    const n = vals.length;
    let mean = 0; vals.forEach(v => mean += v / n);
    let sd = 0; vals.forEach(v => sd += (v - mean) * (v - mean));
    sd = Math.sqrt(sd / (n - 1));
    const h = 1.06 * sd * Math.pow(n, -0.2) || 1;
    const xs = [], ys = [];
    for (let k = 0; k < m; k++) {
      const x = lo + (hi - lo) * k / (m - 1);
      let s = 0;
      for (let i = 0; i < n; i++) {
        const z = (x - vals[i]) / h;
        s += Math.exp(-.5 * z * z);
      }
      xs.push(x); ys.push(s / (n * h * 2.5066282746));
    }
    return { xs, ys };
  }

  Chart.prototype.bars = function (labels, values, o) {
    o = o || {}; const g = this.ctx;
    const n = values.length, pad = .18;
    const w = (this.px.r - this.px.l) / n;
    g.save();
    g.font = '11px -apple-system,Segoe UI,Roboto,sans-serif';
    for (let i = 0; i < n; i++) {
      const x = this.px.l + w * (i + pad), bw = w * (1 - 2 * pad);
      g.fillStyle = typeof o.color === 'function' ? o.color(i) : (o.color || C.league[0]);
      const y = this.Y(values[i]);
      g.fillRect(x, y, bw, this.px.b - y);
      g.fillStyle = C.muted; g.textAlign = 'center'; g.textBaseline = 'top';
      g.fillText(labels[i], x + bw / 2, this.px.b + 7);
      if (o.showValue) {
        g.fillStyle = C.ink; g.textBaseline = 'bottom';
        g.fillText(o.fmt ? o.fmt(values[i]) : values[i].toFixed(1), x + bw / 2, y - 3);
      }
    }
    g.restore();
  };

  Chart.prototype.text = function (x, y, s, o) {
    o = o || {}; const g = this.ctx;
    g.save();
    g.fillStyle = o.color || C.muted;
    g.font = (o.weight ? o.weight + ' ' : '') + (o.size || 11) +
             'px -apple-system,Segoe UI,Roboto,sans-serif';
    g.textAlign = o.align || 'left'; g.textBaseline = o.baseline || 'middle';
    const px = o.abs ? x : this.X(x), py = o.abs ? y : this.Y(y);
    g.fillText(s, px, py);
    g.restore();
  };

  root.Plots = { Chart, C, density, niceTicks };
})(window);
