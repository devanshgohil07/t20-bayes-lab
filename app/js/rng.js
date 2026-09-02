/* Random number generation for the browser.
   randn : Box-Muller (Sessions 8-9 notes)
   randGamma : Marsaglia-Tsang (2000)
   randInvGamma : X ~ Gamma(a, rate b)  =>  1/X ~ InvGamma(a, b)
   A seeded generator (mulberry32) is used so a run in the browser is reproducible. */
(function (root) {
  'use strict';

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function RNG(seed) {
    this.u = mulberry32(seed === undefined ? 20260901 : seed);
    this.spare = null;
  }

  RNG.prototype.unif = function () {
    let u = this.u();
    return u === 0 ? 1e-12 : u;
  };

  RNG.prototype.randn = function () {
    if (this.spare !== null) { const s = this.spare; this.spare = null; return s; }
    const u = this.unif(), v = this.unif();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };

  /* Gamma with shape a and RATE b, so the mean is a/b. */
  RNG.prototype.randGamma = function (a, b) {
    if (a < 1) return this.randGamma(a + 1, b) * Math.pow(this.unif(), 1 / a);
    const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = this.randn(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = this.unif();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v / b;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v / b;
    }
  };

  RNG.prototype.randInvGamma = function (a, b) { return 1 / this.randGamma(a, b); };

  root.RNG = RNG;
})(typeof self !== 'undefined' ? self : this);
