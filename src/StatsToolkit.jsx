import React, { useState, useMemo, useCallback } from "react";
import {
  BarChart, Bar, LineChart, Line, ComposedChart, Scatter, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell
} from "recharts";
import {
  Dice5, Sigma, ScatterChart as ScatterIcon, Repeat, Plus, Trash2,
  RefreshCw, Info
} from "lucide-react";

/* ============================================================
   MATH HELPERS
   ============================================================ */
function lgamma(x) {
  var g = 7;
  var c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  x -= 1;
  var a = c[0];
  var t = x + g + 0.5;
  for (var i = 1; i < g + 2; i++) { a += c[i] / (x + i); }
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function logChoose(n, k) { return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1); }

function binomialPMF(k, n, p) {
  if (k < 0 || k > n) return 0;
  if (p <= 0) return k === 0 ? 1 : 0;
  if (p >= 1) return k === n ? 1 : 0;
  return Math.exp(logChoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p));
}
function poissonPMF(k, lambda) {
  if (k < 0) return 0;
  if (lambda === 0) return k === 0 ? 1 : 0;
  return Math.exp(-lambda + k * Math.log(lambda) - lgamma(k + 1));
}
function geometricPMF(k, p) {
  if (k < 0) return 0;
  return Math.pow(1 - p, k) * p;
}
function normalPDF(x, mu, sigma) {
  var z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}
function exponentialPDF(x, lambda) { return x < 0 ? 0 : lambda * Math.exp(-lambda * x); }
function uniformPDF(x, a, b) { return (x >= a && x <= b) ? 1 / (b - a) : 0; }

function erf(x) {
  var sign = x < 0 ? -1 : 1; x = Math.abs(x);
  var a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;
  var t = 1/(1+p*x);
  var y = 1-(((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
  return sign*y;
}
function normalCDF(x, mu, sigma) { return 0.5*(1+erf((x-mu)/(sigma*Math.sqrt(2)))); }

function randNormal() {
  var u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function fmt(x, d) {
  if (d === undefined) d = 4;
  if (x === null || x === undefined || isNaN(x)) return "\u2014";
  return Number(x).toFixed(d);
}

/* Moments computed numerically from a discrete grid of {x, p} (or {x, w} weight for continuous via dx) */
function momentsFromGrid(xs, ps) {
  var EX = 0, EX2 = 0;
  for (var i = 0; i < xs.length; i++) { EX += xs[i] * ps[i]; EX2 += xs[i] * xs[i] * ps[i]; }
  var Var = EX2 - EX * EX;
  var mu3 = 0, mu4 = 0;
  for (var j = 0; j < xs.length; j++) {
    var d = xs[j] - EX;
    mu3 += Math.pow(d, 3) * ps[j];
    mu4 += Math.pow(d, 4) * ps[j];
  }
  var sd = Math.sqrt(Math.max(Var, 0));
  var skew = sd > 0 ? mu3 / Math.pow(sd, 3) : 0;
  var kurt = sd > 0 ? mu4 / Math.pow(sd, 4) - 3 : 0;
  return { EX: EX, EX2: EX2, Var: Var, SD: sd, skew: skew, kurt: kurt };
}

function sampleStats(arr) {
  var n = arr.length;
  var mean = arr.reduce(function (s, v) { return s + v; }, 0) / n;
  var m2 = 0, m3 = 0, m4 = 0;
  for (var i = 0; i < n; i++) {
    var d = arr[i] - mean;
    m2 += d * d; m3 += d * d * d; m4 += d * d * d * d;
  }
  m2 /= n; m3 /= n; m4 /= n;
  var sampleVar = n > 1 ? (arr.reduce(function (s, v) { return s + (v - mean) * (v - mean); }, 0) / (n - 1)) : 0;
  var sd = Math.sqrt(sampleVar);
  var skew = m2 > 0 ? m3 / Math.pow(m2, 1.5) : 0;
  var kurt = m2 > 0 ? m4 / (m2 * m2) - 3 : 0;
  return { n: n, mean: mean, variance: sampleVar, sd: sd, skew: skew, kurt: kurt };
}

function covariance(xs, ys) {
  var n = xs.length;
  var mx = xs.reduce(function (a, b) { return a + b; }, 0) / n;
  var my = ys.reduce(function (a, b) { return a + b; }, 0) / n;
  var s = 0;
  for (var i = 0; i < n; i++) s += (xs[i] - mx) * (ys[i] - my);
  return s / (n - 1);
}

function histogram(arr, nbins) {
  var min = Math.min.apply(null, arr), max = Math.max.apply(null, arr);
  if (min === max) { max = min + 1; }
  var width = (max - min) / nbins;
  var counts = new Array(nbins).fill(0);
  for (var i = 0; i < arr.length; i++) {
    var idx = Math.min(nbins - 1, Math.floor((arr[i] - min) / width));
    if (idx < 0) idx = 0;
    counts[idx]++;
  }
  var out = [];
  for (var b = 0; b < nbins; b++) {
    var center = min + width * (b + 0.5);
    out.push({ x: center, label: center.toFixed(2), count: counts[b], density: counts[b] / (arr.length * width) });
  }
  return out;
}

var COLORS = {
  ink: "#1E1B2E", inkSoft: "#5C5A72", accent: "#4F46E5", accent2: "#7C3AED",
  rose: "#E11D48", amber: "#D97706", panel: "#FFFFFF", paper: "#FAFAF9",
  line: "#E4E2ED", green: "#059669"
};

/* ============================================================
   SHARED UI PIECES
   ============================================================ */
function Slider(props) {
  return (
    <div className="st-slider">
      <div className="st-slider-label">
        <span>{props.label}</span>
        <span className="st-slider-val">{props.display !== undefined ? props.display : props.value}</span>
      </div>
      <input
        type="range" min={props.min} max={props.max} step={props.step}
        value={props.value}
        onChange={function (e) { props.onChange(parseFloat(e.target.value)); }}
      />
    </div>
  );
}

function StatCard(props) {
  return (
    <div className="st-statcard">
      <div className="st-statcard-label">{props.label}</div>
      <div className="st-statcard-value">{props.value}</div>
      {props.sub && <div className="st-statcard-sub">{props.sub}</div>}
    </div>
  );
}

function InfoBox(props) {
  return (
    <div className="st-infobox">
      <Info size={14} />
      <div>{props.children}</div>
    </div>
  );
}

/* ============================================================
   TAB 1: DISTRIBUTION EXPLORER
   ============================================================ */
var DIST_DEFS = {
  binomial: { label: "Binomial", kind: "discrete", params: { n: 20, p: 0.5 } },
  poisson: { label: "Poisson", kind: "discrete", params: { lambda: 4 } },
  geometric: { label: "Geometric", kind: "discrete", params: { p: 0.3 } },
  normal: { label: "Normal", kind: "continuous", params: { mu: 0, sigma: 1 } },
  exponential: { label: "Exponential", kind: "continuous", params: { lambda: 1 } },
  uniform: { label: "Uniform", kind: "continuous", params: { a: 0, b: 1 } }
};

var ANALYTIC = {
  binomial: function (p) {
    var n = p.n, pp = p.p;
    var mean = n*pp, v = n*pp*(1-pp);
    return { mean: mean, variance: v, skew: (1-2*pp)/Math.sqrt(v || 1e-9), kurt: (1-6*pp*(1-pp))/(v || 1e-9) };
  },
  poisson: function (p) {
    return { mean: p.lambda, variance: p.lambda, skew: 1/Math.sqrt(p.lambda||1e-9), kurt: 1/(p.lambda||1e-9) };
  },
  geometric: function (p) {
    var q = 1-p.p;
    return { mean: q/p.p, variance: q/(p.p*p.p), skew: (2-p.p)/Math.sqrt(q||1e-9), kurt: 6 + (p.p*p.p)/(q||1e-9) };
  },
  normal: function () { return { mean: null, variance: null, skew: 0, kurt: 0 }; },
  exponential: function () { return { mean: null, variance: null, skew: 2, kurt: 6 }; },
  uniform: function (p) { return { mean: (p.a+p.b)/2, variance: Math.pow(p.b-p.a,2)/12, skew: 0, kurt: -1.2 }; }
};
ANALYTIC.normal = function (p) { return { mean: p.mu, variance: p.sigma*p.sigma, skew: 0, kurt: 0 }; };
ANALYTIC.exponential = function (p) { return { mean: 1/p.lambda, variance: 1/(p.lambda*p.lambda), skew: 2, kurt: 6 }; };

function DistributionExplorer() {
  var _d = useState("binomial"); var distKey = _d[0], setDistKey = _d[1];
  var _params = useState({
    binomial: { n: 20, p: 0.5 }, poisson: { lambda: 4 }, geometric: { p: 0.3 },
    normal: { mu: 0, sigma: 1 }, exponential: { lambda: 1 }, uniform: { a: 0, b: 1 }
  });
  var allParams = _params[0], setAllParams = _params[1];
  var _cdf = useState(false); var showCDF = _cdf[0], setShowCDF = _cdf[1];

  var params = allParams[distKey];
  var def = DIST_DEFS[distKey];

  function setParam(key, val) {
    setAllParams(function (prev) {
      var next = Object.assign({}, prev);
      next[distKey] = Object.assign({}, prev[distKey]);
      next[distKey][key] = val;
      return next;
    });
  }

  var chartData = useMemo(function () {
    var xs = [], ps = [];
    var rows = [];
    if (distKey === "binomial") {
      var n = params.n;
      for (var k = 0; k <= n; k++) { xs.push(k); ps.push(binomialPMF(k, n, params.p)); }
    } else if (distKey === "poisson") {
      var upper = Math.max(20, Math.ceil(params.lambda * 4 + 10));
      for (var k2 = 0; k2 <= upper; k2++) { xs.push(k2); ps.push(poissonPMF(k2, params.lambda)); }
    } else if (distKey === "geometric") {
      var upper2 = Math.max(20, Math.ceil(6 / Math.max(params.p, 0.02)));
      for (var k3 = 0; k3 <= upper2; k3++) { xs.push(k3); ps.push(geometricPMF(k3, params.p)); }
    } else if (distKey === "normal") {
      var lo = params.mu - 4.5 * params.sigma, hi = params.mu + 4.5 * params.sigma;
      var steps = 160, dx = (hi - lo) / steps;
      for (var i = 0; i <= steps; i++) { var x = lo + i * dx; xs.push(x); ps.push(normalPDF(x, params.mu, params.sigma) * dx); }
    } else if (distKey === "exponential") {
      var hi2 = 6 / params.lambda; var steps2 = 160, dx2 = hi2 / steps2;
      for (var i2 = 0; i2 <= steps2; i2++) { var x2 = i2 * dx2; xs.push(x2); ps.push(exponentialPDF(x2, params.lambda) * dx2); }
    } else if (distKey === "uniform") {
      var span = params.b - params.a || 1;
      var lo3 = params.a - 0.2 * span, hi3 = params.b + 0.2 * span;
      var steps3 = 160, dx3 = (hi3 - lo3) / steps3;
      for (var i3 = 0; i3 <= steps3; i3++) { var x3 = lo3 + i3 * dx3; xs.push(x3); ps.push(uniformPDF(x3, params.a, params.b) * dx3); }
    }
    var cum = 0;
    for (var r = 0; r < xs.length; r++) {
      cum += ps[r];
      rows.push({ x: xs[r], label: (def.kind === "discrete" ? xs[r] : xs[r].toFixed(2)), p: ps[r], density: def.kind === "discrete" ? ps[r] : ps[r] / (xs.length > 1 ? (xs[1]-xs[0]) : 1), cdf: cum });
    }
    return { xs: xs, ps: ps, rows: rows };
  }, [distKey, JSON.stringify(params)]);

  var numeric = useMemo(function () { return momentsFromGrid(chartData.xs, chartData.ps); }, [chartData]);
  var analytic = ANALYTIC[distKey](params);

  return (
    <div className="st-panel-grid">
      <div className="st-controls-col">
        <div className="st-card">
          <div className="st-card-title">Choose a distribution</div>
          <div className="st-dist-tabs">
            {Object.keys(DIST_DEFS).map(function (k) {
              return (
                <button key={k} className={"st-dist-tab" + (k === distKey ? " st-dist-tab-on" : "")}
                  onClick={function () { setDistKey(k); }}>
                  {DIST_DEFS[k].label}
                </button>
              );
            })}
          </div>
          <div className="st-tag">{def.kind === "discrete" ? "Discrete \u2014 PMF" : "Continuous \u2014 PDF"}</div>
        </div>

        <div className="st-card">
          <div className="st-card-title">Parameters</div>
          {distKey === "binomial" && (<>
            <Slider label="n (trials)" min={1} max={100} step={1} value={params.n} onChange={function (v) { setParam("n", v); }} />
            <Slider label="p (success prob.)" min={0.01} max={0.99} step={0.01} value={params.p} display={fmt(params.p,2)} onChange={function (v) { setParam("p", v); }} />
          </>)}
          {distKey === "poisson" && (
            <Slider label="\u03bb (rate)" min={0.1} max={30} step={0.1} value={params.lambda} display={fmt(params.lambda,1)} onChange={function (v) { setParam("lambda", v); }} />
          )}
          {distKey === "geometric" && (
            <Slider label="p (success prob.)" min={0.02} max={0.95} step={0.01} value={params.p} display={fmt(params.p,2)} onChange={function (v) { setParam("p", v); }} />
          )}
          {distKey === "normal" && (<>
            <Slider label="\u03bc (mean)" min={-10} max={10} step={0.1} value={params.mu} display={fmt(params.mu,1)} onChange={function (v) { setParam("mu", v); }} />
            <Slider label="\u03c3 (std dev)" min={0.1} max={5} step={0.1} value={params.sigma} display={fmt(params.sigma,1)} onChange={function (v) { setParam("sigma", v); }} />
          </>)}
          {distKey === "exponential" && (
            <Slider label="\u03bb (rate)" min={0.1} max={5} step={0.05} value={params.lambda} display={fmt(params.lambda,2)} onChange={function (v) { setParam("lambda", v); }} />
          )}
          {distKey === "uniform" && (<>
            <Slider label="a (lower bound)" min={-10} max={9} step={0.5} value={params.a} onChange={function (v) { setParam("a", Math.min(v, params.b - 0.5)); }} />
            <Slider label="b (upper bound)" min={-9} max={10} step={0.5} value={params.b} onChange={function (v) { setParam("b", Math.max(v, params.a + 0.5)); }} />
          </>)}
          <label className="st-checkbox"><input type="checkbox" checked={showCDF} onChange={function(e){setShowCDF(e.target.checked);}} /> Overlay CDF</label>
        </div>

        <div className="st-card">
          <div className="st-card-title">Moments \u2014 numeric vs. theoretical</div>
          <table className="st-mtable">
            <thead><tr><th></th><th>Computed from curve</th><th>Closed-form formula</th></tr></thead>
            <tbody>
              <tr><td>Mean E[X]</td><td>{fmt(numeric.EX,3)}</td><td>{analytic.mean===null?"\u2014":fmt(analytic.mean,3)}</td></tr>
              <tr><td>Variance</td><td>{fmt(numeric.Var,3)}</td><td>{analytic.variance===null?"\u2014":fmt(analytic.variance,3)}</td></tr>
              <tr><td>Std. dev</td><td>{fmt(numeric.SD,3)}</td><td>{analytic.variance===null?"\u2014":fmt(Math.sqrt(analytic.variance),3)}</td></tr>
              <tr><td>Skewness</td><td>{fmt(numeric.skew,3)}</td><td>{fmt(analytic.skew,3)}</td></tr>
              <tr><td>Excess kurtosis</td><td>{fmt(numeric.kurt,3)}</td><td>{fmt(analytic.kurt,3)}</td></tr>
            </tbody>
          </table>
          <InfoBox>
            "Computed from curve" uses <span className="st-mono">E[X]=Σx·p(x)</span> (or the integral, for continuous) directly on
            the plotted points — the same Var(X)=E[X²]−(E[X])² identity from earlier. It should closely match the textbook formula.
          </InfoBox>
        </div>
      </div>

      <div className="st-chart-col">
        <div className="st-card st-chart-card">
          <div className="st-card-title">{def.label} {def.kind === "discrete" ? "PMF" : "PDF"}</div>
          <ResponsiveContainer width="100%" height={380}>
            <ComposedChart data={chartData.rows} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid stroke={COLORS.line} strokeDasharray="2 3" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 10, fill: COLORS.inkSoft }} interval={def.kind==="discrete" ? "preserveStartEnd" : Math.floor(chartData.rows.length/8)} />
              <YAxis yAxisId="left" tick={{ fontSize: 10, fill: COLORS.inkSoft }} width={50} tickFormatter={function(v){return fmt(v,3);}} />
              {showCDF && <YAxis yAxisId="right" orientation="right" domain={[0,1]} tick={{ fontSize: 10, fill: COLORS.accent2 }} width={36} />}
              <Tooltip formatter={function(v,name){ return [fmt(v,4), name]; }} labelFormatter={function(l){return "x = " + l;}} />
              {def.kind === "discrete"
                ? <Bar yAxisId="left" dataKey="p" name="P(X=x)" fill={COLORS.accent} radius={[3,3,0,0]} />
                : <Line yAxisId="left" type="monotone" dataKey="density" name="f(x)" stroke={COLORS.accent} strokeWidth={2} dot={false} />
              }
              {showCDF && <Line yAxisId="right" type="monotone" dataKey="cdf" name="CDF" stroke={COLORS.rose} strokeWidth={1.6} dot={false} strokeDasharray="4 2" />}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   TAB 2: EXPECTATION & MOMENTS CALCULATOR (custom distribution)
   ============================================================ */
function ExpectationCalculator() {
  var _rows = useState([
    { x: 1, p: 0.2 }, { x: 2, p: 0.3 }, { x: 3, p: 0.3 }, { x: 4, p: 0.2 }
  ]);
  var rows = _rows[0], setRows = _rows[1];

  function updateRow(i, field, val) {
    setRows(function (prev) {
      var next = prev.slice();
      next[i] = Object.assign({}, next[i]);
      next[i][field] = val;
      return next;
    });
  }
  function addRow() { setRows(function (prev) { return prev.concat([{ x: prev.length, p: 0 }]); }); }
  function removeRow(i) { setRows(function (prev) { return prev.filter(function (_, idx) { return idx !== i; }); }); }
  function normalize() {
    setRows(function (prev) {
      var sum = prev.reduce(function (s, r) { return s + (parseFloat(r.p) || 0); }, 0);
      if (sum <= 0) return prev;
      return prev.map(function (r) { return { x: r.x, p: (parseFloat(r.p) || 0) / sum }; });
    });
  }

  var xs = rows.map(function (r) { return parseFloat(r.x) || 0; });
  var ps = rows.map(function (r) { return parseFloat(r.p) || 0; });
  var sumP = ps.reduce(function (a, b) { return a + b; }, 0);
  var m = momentsFromGrid(xs, ps);

  var chartData = rows.map(function (r) { return { label: String(r.x), p: parseFloat(r.p) || 0 }; });

  return (
    <div className="st-panel-grid">
      <div className="st-controls-col">
        <div className="st-card">
          <div className="st-card-title">Your distribution</div>
          <table className="st-edit-table">
            <thead><tr><th>x</th><th>P(X=x)</th><th></th></tr></thead>
            <tbody>
              {rows.map(function (r, i) {
                return (
                  <tr key={i}>
                    <td><input type="number" value={r.x} onChange={function (e) { updateRow(i, "x", e.target.value); }} /></td>
                    <td><input type="number" step="0.01" value={r.p} onChange={function (e) { updateRow(i, "p", e.target.value); }} /></td>
                    <td><button className="st-icon-btn" onClick={function () { removeRow(i); }}><Trash2 size={14} /></button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="st-row-actions">
            <button className="st-btn" onClick={addRow}><Plus size={14} /> Add row</button>
            <button className="st-btn st-btn-secondary" onClick={normalize}>Normalize to sum=1</button>
          </div>
          <div className={"st-sum-check " + (Math.abs(sumP-1) < 0.001 ? "st-sum-ok" : "st-sum-bad")}>
            Σ P(x) = {fmt(sumP,4)} {Math.abs(sumP-1) < 0.001 ? "\u2713 valid" : "\u26A0 should equal 1"}
          </div>
        </div>

        <div className="st-card">
          <div className="st-card-title">Step-by-step calculation</div>
          <table className="st-mtable">
            <thead><tr><th>x</th><th>p(x)</th><th>x·p(x)</th><th>x²·p(x)</th></tr></thead>
            <tbody>
              {rows.map(function (r, i) {
                var x = xs[i], p = ps[i];
                return <tr key={i}><td>{x}</td><td>{fmt(p,3)}</td><td>{fmt(x*p,3)}</td><td>{fmt(x*x*p,3)}</td></tr>;
              })}
              <tr className="st-total-row"><td colSpan={2}>Sum</td><td>{fmt(m.EX,3)}</td><td>{fmt(m.EX2,3)}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="st-chart-col">
        <div className="st-card st-chart-card">
          <div className="st-card-title">PMF of your distribution</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid stroke={COLORS.line} strokeDasharray="2 3" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 11, fill: COLORS.inkSoft }} />
              <YAxis tick={{ fontSize: 10, fill: COLORS.inkSoft }} width={40} />
              <Tooltip formatter={function(v){return fmt(v,4);}} />
              <Bar dataKey="p" fill={COLORS.accent2} radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="st-stats-grid">
          <StatCard label="E[X]" value={fmt(m.EX,4)} sub="\u03A3 x\u00b7p(x)" />
          <StatCard label="E[X\u00b2]" value={fmt(m.EX2,4)} sub="\u03A3 x\u00b2\u00b7p(x)" />
          <StatCard label="Var(X)" value={fmt(m.Var,4)} sub="E[X\u00b2] \u2212 (E[X])\u00b2" />
          <StatCard label="SD(X)" value={fmt(m.SD,4)} sub="\u221aVar(X)" />
          <StatCard label="Skewness" value={fmt(m.skew,4)} sub="\u03BC\u2083 / \u03C3\u00b3" />
          <StatCard label="Excess kurtosis" value={fmt(m.kurt,4)} sub="\u03BC\u2084 / \u03C3\u2074 \u2212 3" />
        </div>

        <InfoBox>
          This is the exact derivation from before: <span className="st-mono">Var(X̄) = E[X̄²] − (E[X̄])²</span>,
          computed live as you edit the table. Try setting a symmetric distribution (skewness ≈ 0) vs.
          a lopsided one to see skewness respond.
        </InfoBox>
      </div>
    </div>
  );
}

/* ============================================================
   TAB 3: 2D DATA GENERATOR + REGRESSION
   ============================================================ */
var RELATIONS = {
  linear: { label: "Linear: y = a\u00b7x + b", params: { a: 2, b: 5 } },
  quadratic: { label: "Quadratic: y = a\u00b7x\u00b2 + b\u00b7x", params: { a: 0.5, b: 1 } },
  exponential: { label: "Exponential: y = a\u00b7e^(b\u00b7x)", params: { a: 2, b: 0.3 } },
  logarithmic: { label: "Logarithmic: y = a\u00b7ln(x) + b", params: { a: 5, b: 2 } },
  sinusoidal: { label: "Sinusoidal: y = a\u00b7sin(b\u00b7x)", params: { a: 5, b: 1 } },
  none: { label: "No relationship (independent noise)", params: { a: 0, b: 0 } }
};

function generateData(n, relation, a, b, noise, seedBump) {
  var xs = [], ys = [];
  for (var i = 0; i < n; i++) {
    var x = relation === "logarithmic" ? 0.2 + Math.random() * 10 : -5 + Math.random() * 10;
    var fx;
    if (relation === "linear") fx = a * x + b;
    else if (relation === "quadratic") fx = a * x * x + b * x;
    else if (relation === "exponential") fx = a * Math.exp(b * x * 0.3);
    else if (relation === "logarithmic") fx = a * Math.log(x) + b;
    else if (relation === "sinusoidal") fx = a * Math.sin(b * x);
    else fx = 0;
    var y = fx + randNormal() * noise + (relation === "none" ? randNormal() * 5 : 0);
    xs.push(x); ys.push(y);
  }
  return { xs: xs, ys: ys };
}

function ScatterRegression() {
  var _n = useState(80); var n = _n[0], setN = _n[1];
  var _rel = useState("linear"); var relation = _rel[0], setRelation = _rel[1];
  var _a = useState(2); var a = _a[0], setA = _a[1];
  var _b = useState(5); var b = _b[0], setB = _b[1];
  var _noise = useState(3); var noise = _noise[0], setNoise = _noise[1];
  var _seed = useState(0); var seed = _seed[0], setSeed = _seed[1];
  var _showFit = useState(true); var showFit = _showFit[0], setShowFit = _showFit[1];

  var data = useMemo(function () { return generateData(n, relation, a, b, noise, seed); }, [n, relation, a, b, noise, seed]);

  var stats = useMemo(function () {
    var sx = sampleStats(data.xs), sy = sampleStats(data.ys);
    var cov = covariance(data.xs, data.ys);
    var r = cov / (sx.sd * sy.sd || 1e-9);
    var b1 = cov / (sx.variance || 1e-9);
    var b0 = sy.mean - b1 * sx.mean;
    return { sx: sx, sy: sy, cov: cov, r: r, r2: r * r, b1: b1, b0: b0 };
  }, [data]);

  var scatterData = useMemo(function () {
    var pts = data.xs.map(function (x, i) { return { x: x, y: data.ys[i] }; });
    var minX = Math.min.apply(null, data.xs), maxX = Math.max.apply(null, data.xs);
    var fitLine = [
      { x: minX, y: stats.b1 * minX + stats.b0, fit: stats.b1 * minX + stats.b0 },
      { x: maxX, y: stats.b1 * maxX + stats.b0, fit: stats.b1 * maxX + stats.b0 }
    ];
    return { pts: pts, fitLine: fitLine };
  }, [data, stats]);

  function onChangeRelation(k) {
    setRelation(k);
    setA(RELATIONS[k].params.a);
    setB(RELATIONS[k].params.b);
  }

  return (
    <div className="st-panel-grid">
      <div className="st-controls-col">
        <div className="st-card">
          <div className="st-card-title">Generate data</div>
          <div className="st-select-label">Relationship</div>
          <select className="st-select" value={relation} onChange={function (e) { onChangeRelation(e.target.value); }}>
            {Object.keys(RELATIONS).map(function (k) { return <option key={k} value={k}>{RELATIONS[k].label}</option>; })}
          </select>
          <Slider label="Sample size (n)" min={10} max={400} step={10} value={n} onChange={setN} />
          {relation !== "none" && <>
            <Slider label="a (coefficient)" min={-10} max={10} step={0.1} value={a} display={fmt(a,1)} onChange={setA} />
            <Slider label="b (intercept/shift)" min={-10} max={10} step={0.1} value={b} display={fmt(b,1)} onChange={setB} />
          </>}
          <Slider label="Noise (std dev)" min={0} max={15} step={0.5} value={noise} display={fmt(noise,1)} onChange={setNoise} />
          <label className="st-checkbox"><input type="checkbox" checked={showFit} onChange={function(e){setShowFit(e.target.checked);}} /> Show OLS fit line</label>
          <button className="st-btn" onClick={function () { setSeed(seed + 1); }}><RefreshCw size={14} /> Regenerate</button>
        </div>

        <div className="st-card">
          <div className="st-card-title">Fitted OLS regression</div>
          <div className="st-formula">ŷ = {fmt(stats.b1,3)}·x + {fmt(stats.b0,3)}</div>
          <table className="st-mtable">
            <tbody>
              <tr><td>Slope (b\u2081)</td><td>{fmt(stats.b1,4)}</td></tr>
              <tr><td>Intercept (b\u2080)</td><td>{fmt(stats.b0,4)}</td></tr>
              <tr><td>Correlation (r)</td><td>{fmt(stats.r,4)}</td></tr>
              <tr><td>R²</td><td>{fmt(stats.r2,4)}</td></tr>
              <tr><td>Cov(X,Y)</td><td>{fmt(stats.cov,4)}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="st-chart-col">
        <div className="st-card st-chart-card">
          <div className="st-card-title">Scatter plot {showFit && "+ OLS fit"}</div>
          <ResponsiveContainer width="100%" height={360}>
            <ComposedChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid stroke={COLORS.line} strokeDasharray="2 3" />
              <XAxis type="number" dataKey="x" name="x" tick={{ fontSize: 10, fill: COLORS.inkSoft }} domain={["auto","auto"]} />
              <YAxis type="number" dataKey="y" name="y" tick={{ fontSize: 10, fill: COLORS.inkSoft }} domain={["auto","auto"]} width={44} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={function(v){return fmt(v,3);}} />
              <Scatter data={scatterData.pts} fill={COLORS.accent} fillOpacity={0.55} r={3} />
              {showFit && <Line data={scatterData.fitLine} dataKey="y" stroke={COLORS.rose} strokeWidth={2.4} dot={false} activeDot={false} isAnimationActive={false} />}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="st-stats-grid">
          <StatCard label="Mean X" value={fmt(stats.sx.mean,3)} />
          <StatCard label="Mean Y" value={fmt(stats.sy.mean,3)} />
          <StatCard label="SD X" value={fmt(stats.sx.sd,3)} />
          <StatCard label="SD Y" value={fmt(stats.sy.sd,3)} />
          <StatCard label="Skew X" value={fmt(stats.sx.skew,3)} />
          <StatCard label="Skew Y" value={fmt(stats.sy.skew,3)} />
          <StatCard label="Excess Kurt X" value={fmt(stats.sx.kurt,3)} />
          <StatCard label="Excess Kurt Y" value={fmt(stats.sy.kurt,3)} />
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   TAB 4: CENTRAL LIMIT THEOREM SIMULATOR
   ============================================================ */
var PARENTS = {
  uniform: { label: "Uniform(0,1)", mean: 0.5, variance: 1/12, draw: function () { return Math.random(); } },
  exponential: { label: "Exponential(rate=1) \u2014 skewed", mean: 1, variance: 1, draw: function () { return -Math.log(Math.random()); } },
  bernoulli: { label: "Bernoulli(p=0.1) \u2014 very skewed", mean: 0.1, variance: 0.09, draw: function () { return Math.random() < 0.1 ? 1 : 0; } }
};

function CLTSimulator() {
  var _parent = useState("exponential"); var parentKey = _parent[0], setParentKey = _parent[1];
  var _n = useState(5); var n = _n[0], setN = _n[1];
  var _reps = useState(2000); var reps = _reps[0], setReps = _reps[1];
  var _seed = useState(0); var seed = _seed[0], setSeed = _seed[1];

  var parent = PARENTS[parentKey];

  var sampleMeans = useMemo(function () {
    var out = [];
    for (var t = 0; t < reps; t++) {
      var s = 0;
      for (var i = 0; i < n; i++) s += parent.draw();
      out.push(s / n);
    }
    return out;
  }, [parentKey, n, reps, seed]);

  var stats = useMemo(function () { return sampleStats(sampleMeans); }, [sampleMeans]);
  var theoreticalSE = Math.sqrt(parent.variance / n);

  var hist = useMemo(function () { return histogram(sampleMeans, 30); }, [sampleMeans]);
  var chartData = useMemo(function () {
    return hist.map(function (h) {
      var theoDensity = normalPDF(h.x, parent.mean, theoreticalSE);
      return Object.assign({}, h, { theoretical: theoDensity });
    });
  }, [hist, parent, theoreticalSE]);

  return (
    <div className="st-panel-grid">
      <div className="st-controls-col">
        <div className="st-card">
          <div className="st-card-title">Draw samples from...</div>
          <select className="st-select" value={parentKey} onChange={function (e) { setParentKey(e.target.value); }}>
            {Object.keys(PARENTS).map(function (k) { return <option key={k} value={k}>{PARENTS[k].label}</option>; })}
          </select>
          <Slider label="Sample size per draw (n)" min={1} max={100} step={1} value={n} onChange={setN} />
          <Slider label="Number of repeated trials" min={200} max={5000} step={100} value={reps} onChange={setReps} />
          <button className="st-btn" onClick={function () { setSeed(seed + 1); }}><RefreshCw size={14} /> Re-simulate</button>
        </div>

        <div className="st-card">
          <div className="st-card-title">Var(X̄) = σ²/n, verified live</div>
          <table className="st-mtable">
            <tbody>
              <tr><td>Population mean μ</td><td>{fmt(parent.mean,4)}</td></tr>
              <tr><td>Population variance σ²</td><td>{fmt(parent.variance,4)}</td></tr>
              <tr><td>Theoretical SE = σ/√n</td><td>{fmt(theoreticalSE,4)}</td></tr>
              <tr><td>Mean of simulated X̄'s</td><td>{fmt(stats.mean,4)}</td></tr>
              <tr><td>SD of simulated X̄'s</td><td>{fmt(stats.sd,4)}</td></tr>
              <tr><td>Skewness of X̄ distribution</td><td>{fmt(stats.skew,4)}</td></tr>
            </tbody>
          </table>
          <InfoBox>
            As n grows, the histogram below should visibly narrow toward the theoretical curve and its skewness
            should drift toward 0 — even when the parent distribution (try Exponential or Bernoulli) is
            heavily skewed. That convergence <em>is</em> the Central Limit Theorem, and the narrowing rate is
            exactly <span className="st-mono">Var(X̄)=σ²/n</span> from before.
          </InfoBox>
        </div>
      </div>

      <div className="st-chart-col">
        <div className="st-card st-chart-card">
          <div className="st-card-title">Distribution of the sample mean X̄ ({reps} trials, n={n})</div>
          <ResponsiveContainer width="100%" height={380}>
            <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid stroke={COLORS.line} strokeDasharray="2 3" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 9, fill: COLORS.inkSoft }} interval={Math.floor(chartData.length/8)} />
              <YAxis tick={{ fontSize: 10, fill: COLORS.inkSoft }} width={40} />
              <Tooltip formatter={function(v){return fmt(v,4);}} />
              <Bar dataKey="density" name="Simulated density" fill={COLORS.accent} radius={[2,2,0,0]} />
              <Line type="monotone" dataKey="theoretical" name="Theoretical N(\u03bc, \u03c3\u00b2/n)" stroke={COLORS.rose} strokeWidth={2} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   ROOT APP
   ============================================================ */
var TABS = [
  { key: "dist", label: "Distribution Explorer", icon: Dice5, comp: DistributionExplorer },
  { key: "exp", label: "Expectation Calculator", icon: Sigma, comp: ExpectationCalculator },
  { key: "reg", label: "2D Data \u0026 Regression", icon: ScatterIcon, comp: ScatterRegression },
  { key: "clt", label: "CLT Simulator", icon: Repeat, comp: CLTSimulator }
];

export default function StatsToolkit() {
  var _tab = useState("dist"); var tab = _tab[0], setTab = _tab[1];
  var ActiveComp = TABS.filter(function (t) { return t.key === tab; })[0].comp;

  return (
    <div className="st-root">
      <style>{CSS}</style>
      <header className="st-header">
        <h1>Statistics Toolkit</h1>
        <p>Distributions, expectation & moments, regression, and the Central Limit Theorem — all interactive</p>
      </header>

      <nav className="st-tabnav">
        {TABS.map(function (t) {
          var Icon = t.icon;
          return (
            <button key={t.key} className={"st-tabbtn" + (tab === t.key ? " st-tabbtn-on" : "")} onClick={function () { setTab(t.key); }}>
              <Icon size={15} /> {t.label}
            </button>
          );
        })}
      </nav>

      <main className="st-main">
        <ActiveComp />
      </main>

      <footer className="st-footer">
        All computations run client-side in your browser (log-gamma based PMFs for numerical stability at large n/λ;
        Box-Muller for normal random draws). Moments are cross-checked two ways: numerically from the plotted curve, and
        via closed-form formulas, so you can see them agree.
      </footer>
    </div>
  );
}

var CSS = "\
.st-root {\
  width: 100%; min-height: 100vh; background: #FAFAF9; color: #1E1B2E;\
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\
  padding: 22px 26px 36px; box-sizing: border-box;\
}\
.st-header h1 { font-size: 23px; margin: 0 0 4px; font-weight: 700; }\
.st-header p { font-size: 13px; color: #5C5A72; margin: 0 0 16px; }\
.st-tabnav { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 18px; border-bottom: 1px solid #E4E2ED; padding-bottom: 12px; }\
.st-tabbtn {\
  display: flex; align-items: center; gap: 7px; padding: 8px 14px; border-radius: 8px;\
  border: 1px solid #E4E2ED; background: #fff; color: #5C5A72; font-size: 13px; cursor: pointer;\
  font-family: inherit; transition: all 0.12s;\
}\
.st-tabbtn:hover { background: #F4F3FA; }\
.st-tabbtn-on { background: #4F46E5; border-color: #4F46E5; color: #fff; font-weight: 600; }\
.st-panel-grid { display: grid; grid-template-columns: 300px 1fr; gap: 16px; align-items: start; }\
.st-controls-col { display: flex; flex-direction: column; gap: 14px; }\
.st-chart-col { display: flex; flex-direction: column; gap: 14px; }\
.st-card { background: #fff; border: 1px solid #E4E2ED; border-radius: 10px; padding: 16px; }\
.st-chart-card { padding-bottom: 6px; }\
.st-card-title { font-size: 13px; font-weight: 700; margin-bottom: 12px; color: #1E1B2E; }\
.st-dist-tabs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }\
.st-dist-tab {\
  border: 1px solid #E4E2ED; background: #FAFAF9; border-radius: 6px; padding: 5px 10px;\
  font-size: 12px; cursor: pointer; color: #5C5A72; font-family: inherit;\
}\
.st-dist-tab-on { background: #EEF2FF; border-color: #4F46E5; color: #4F46E5; font-weight: 600; }\
.st-tag { font-size: 11px; color: #8B87A0; font-style: italic; }\
.st-slider { margin-bottom: 14px; }\
.st-slider-label { display: flex; justify-content: space-between; font-size: 12.5px; margin-bottom: 4px; color: #3A3752; }\
.st-slider-val { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #4F46E5; font-weight: 600; }\
.st-slider input[type=range] { width: 100%; accent-color: #4F46E5; }\
.st-checkbox { display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: #3A3752; margin: 6px 0 10px; cursor: pointer; }\
.st-select-label { font-size: 11.5px; color: #8B87A0; margin-bottom: 4px; }\
.st-select {\
  width: 100%; padding: 7px 8px; border: 1px solid #E4E2ED; border-radius: 6px;\
  font-size: 12.5px; margin-bottom: 12px; background: #fff; color: #1E1B2E; font-family: inherit;\
}\
.st-btn {\
  display: flex; align-items: center; justify-content: center; gap: 6px; width: 100%;\
  padding: 9px; border-radius: 7px; border: none; background: #4F46E5; color: #fff;\
  font-size: 12.5px; font-weight: 600; cursor: pointer; margin-top: 4px; font-family: inherit;\
}\
.st-btn:hover { background: #4338CA; }\
.st-btn-secondary { background: #fff; color: #4F46E5; border: 1px solid #4F46E5; margin-top: 8px; }\
.st-btn-secondary:hover { background: #EEF2FF; }\
.st-mtable { width: 100%; border-collapse: collapse; font-size: 12.5px; }\
.st-mtable th, .st-mtable td { padding: 5px 6px; text-align: left; border-bottom: 1px solid #F0EFF7; }\
.st-mtable th { color: #8B87A0; font-weight: 600; font-size: 11px; }\
.st-mtable td:not(:first-child) { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #1E1B2E; }\
.st-total-row td { font-weight: 700; border-top: 2px solid #E4E2ED; }\
.st-infobox {\
  display: flex; gap: 8px; background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 8px;\
  padding: 10px 12px; font-size: 12px; color: #3730A3; line-height: 1.5; margin-top: 10px;\
}\
.st-infobox svg { flex-shrink: 0; margin-top: 2px; }\
.st-mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #F4F3FA; padding: 1px 4px; border-radius: 3px; }\
.st-stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }\
.st-statcard { background: #fff; border: 1px solid #E4E2ED; border-radius: 8px; padding: 10px 12px; }\
.st-statcard-label { font-size: 10.5px; color: #8B87A0; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 3px; }\
.st-statcard-value { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 16px; font-weight: 700; color: #1E1B2E; }\
.st-statcard-sub { font-size: 10px; color: #8B87A0; margin-top: 2px; font-family: ui-monospace, monospace; }\
.st-edit-table { width: 100%; border-collapse: collapse; margin-bottom: 10px; }\
.st-edit-table th { font-size: 11px; color: #8B87A0; text-align: left; padding-bottom: 4px; }\
.st-edit-table td { padding: 3px 4px 3px 0; }\
.st-edit-table input {\
  width: 100%; padding: 5px 6px; border: 1px solid #E4E2ED; border-radius: 5px;\
  font-size: 12.5px; font-family: ui-monospace, monospace; box-sizing: border-box;\
}\
.st-icon-btn { border: none; background: none; color: #B0413E; cursor: pointer; display: flex; padding: 4px; }\
.st-row-actions { display: flex; gap: 8px; }\
.st-row-actions .st-btn { margin-top: 0; }\
.st-sum-check { margin-top: 10px; font-size: 12px; font-family: ui-monospace, monospace; padding: 6px 8px; border-radius: 6px; }\
.st-sum-ok { background: #ECFDF5; color: #059669; }\
.st-sum-bad { background: #FEF2F2; color: #DC2626; }\
.st-formula {\
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 15px; font-weight: 700;\
  color: #4F46E5; background: #EEF2FF; padding: 8px 10px; border-radius: 6px; margin-bottom: 10px;\
}\
.st-footer { font-size: 10.5px; color: #8B87A0; line-height: 1.6; border-top: 1px solid #E4E2ED; padding-top: 12px; margin-top: 20px; }\
@media (max-width: 900px) {\
  .st-panel-grid { grid-template-columns: 1fr; }\
  .st-stats-grid { grid-template-columns: repeat(2, 1fr); }\
}\
";
