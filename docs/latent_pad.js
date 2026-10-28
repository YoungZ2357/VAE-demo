// Latent pad: decode a 2-D VAE latent live in the browser.
//
// One file, no build step, used two ways:
//   - notebook:    anywidget default export (data arrives through traitlets)
//   - static site: import { mount, urlSource } from "./latent_pad.js"
// Data format: see vae_demo/export.py.

const TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];
const PRIOR_COLS = 16, PRIOR_ROWS = 4, PRIOR_SEED = 7;
const MANIFOLD_N = 16;

// ------------------------------------------------------------------ data sources
function toArrayBuffer(v) {
  if (v instanceof ArrayBuffer) return v;
  if (ArrayBuffer.isView(v)) return v.buffer.slice(v.byteOffset, v.byteOffset + v.byteLength);
  throw new Error("expected binary buffer");
}

/** Bundle served as static files: `base` is the directory holding manifest.json. */
export function urlSource(base) {
  const root = new URL(base.endsWith("/") ? base : base + "/", document.baseURI);
  return {
    async getManifest() {
      const r = await fetch(new URL("manifest.json", root));
      if (!r.ok) throw new Error(`manifest.json: HTTP ${r.status}`);
      return r.json();
    },
    async getBuffer(path) {
      const r = await fetch(new URL(path, root));
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.arrayBuffer();
    },
  };
}

function traitletSource(model) {
  return {
    async getManifest() { return model.get("manifest"); },
    async getBuffer(path) {
      const v = model.get("buffers")[path];
      if (v == null) throw new Error(`missing buffer ${path}`);
      return toArrayBuffer(v);
    },
  };
}

// ------------------------------------------------------------------ numerics
function halfToFloat32(buf) {
  const h = new Uint16Array(buf), f = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const x = h[i], s = x & 0x8000 ? -1 : 1, e = (x >> 10) & 0x1f, m = x & 0x3ff;
    f[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity)
                                                 : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return f;
}

/** Same formula as vae_demo/conditioning.py:beta_features. */
export function betaFeatures(beta, cond) {
  if (cond.type !== "log_beta") throw new Error(`unknown cond type ${cond.type}`);
  return new Float32Array([(Math.log(beta) - cond.center) / cond.scale]);
}

/** MLP interpreted from the manifest's layer list. */
export class Decoder {
  constructor(spec, buffer) {
    this.spec = spec;
    this.params = spec.dtype === "float16" ? halfToFloat32(buffer) : new Float32Array(buffer);
    this.zDim = spec.input.z;
    this.cDim = spec.input.cond || 0;
    const last = spec.layers.filter((l) => l.type === "linear").pop();
    this.outDim = last.out;
  }

  /** z: Float32Array [n * zDim]; c: Float32Array [cDim] or null. Returns [n * outDim]. */
  forward(z, n, c = null) {
    const inDim = this.zDim + this.cDim;
    let x = new Float32Array(n * inDim);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < this.zDim; k++) x[i * inDim + k] = z[i * this.zDim + k];
      for (let k = 0; k < this.cDim; k++) x[i * inDim + this.zDim + k] = c[k];
    }
    const P = this.params;
    for (const L of this.spec.layers) {
      if (L.type === "linear") {
        const y = new Float32Array(n * L.out);
        for (let i = 0; i < n; i++) {
          const xi = i * L.in, yi = i * L.out;
          for (let j = 0; j < L.out; j++) {
            let s = P[L.b + j];
            const w = L.w + j * L.in;
            for (let k = 0; k < L.in; k++) s += P[w + k] * x[xi + k];
            y[yi + j] = s;
          }
        }
        x = y;
      } else if (L.type === "relu") {
        for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0;
      } else if (L.type === "sigmoid") {
        for (let i = 0; i < x.length; i++) x[i] = 1 / (1 + Math.exp(-x[i]));
      } else {
        throw new Error(`unknown layer type ${L.type}`);
      }
    }
    return x;
  }
}

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianSamples(n, seed) {
  const rnd = mulberry32(seed), out = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    const u = Math.max(rnd(), 1e-12), v = rnd(), r = Math.sqrt(-2 * Math.log(u));
    out[2 * i] = r * Math.cos(2 * Math.PI * v);
    out[2 * i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

const priorMass = (r) => 1 - Math.exp(-r * r / 2);  // P(|z| <= r) for a 2-D standard normal
const fmtBeta = (b) => (b >= 10 ? b.toFixed(0) : b >= 1 ? +b.toFixed(2) + "" : +b.toPrecision(2) + "");

// ------------------------------------------------------------------ DOM helpers
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}

function paintDigits(ctx, pixels, n, cols, size = 28) {
  const rows = Math.ceil(n / cols);
  const img = ctx.createImageData(cols * size, rows * size);
  for (let s = 0; s < n; s++) {
    const ox = (s % cols) * size, oy = Math.floor(s / cols) * size;
    for (let p = 0; p < size * size; p++) {
      const v = Math.round(pixels[s * size * size + p] * 255);
      const q = 4 * ((oy + Math.floor(p / size)) * cols * size + ox + (p % size));
      img.data[q] = img.data[q + 1] = img.data[q + 2] = v;
      img.data[q + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// ------------------------------------------------------------------ training curves
// One line per trained model, β order = slot order. Validated categorical slots (light / dark
// surface); seven ordinal steps of one hue are too close in lightness to tell apart.
const SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
const SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const CURVE_METRICS = [["recon", "reconstruction loss (nats)"], ["kl", "KL divergence (nats)"]];
const CW = 360, CH = 220, CM = { l: 40, r: 10, t: 10, b: 34 };
// Losses against β at one epoch. "nelbo" = recon + KL, the negative ELBO with the β weight removed:
// the same yardstick for every model, so its lowest point is the best bound on -log p(x).
const BETA_METRICS = [
  ["recon", "reconstruction loss (nats)", (r) => r],
  ["kl", "KL divergence (nats)", (r, k) => k],
  ["nelbo", "recon + KL = −ELBO (nats)", (r, k) => r + k],
];
const BW = 240, BH = 200, BM = { l: 36, r: 10, t: 10, b: 34 };
const SVG_NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, ...kids) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const k of kids) e.append(k);
  return e;
}

function niceTicks(lo, hi, n = 4) {
  const raw = (hi - lo) / n || 1, p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * p).find((s) => s >= raw);
  const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step, ticks = [];
  for (let v = a; v <= b + step / 2; v += step) ticks.push(+v.toFixed(10));
  return { lo: a, hi: b, ticks };
}

function isDarkSurface(color) {
  const g = document.createElement("canvas").getContext("2d");
  g.fillStyle = "#fff"; g.fillStyle = color || "#fff";  // normalises any CSS color to #rrggbb
  const hex = g.fillStyle;
  if (!hex.startsWith("#")) return false;
  const [r, gr, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return 0.2126 * r + 0.7152 * gr + 0.0722 * b < 128;
}

/** Recon and KL per epoch for every model that carries a history. onPick(model) on click. */
function createCurves(models, onPick) {
  const series = models.filter((m) => m.history && m.history.epoch.length)
    .map((m) => ({ m, label: m.beta == null ? m.name : `β = ${fmtBeta(m.beta)}`, on: true, color: "" }));
  if (!series.length) return null;
  const epochs = [...new Set(series.flatMap((s) => s.m.history.epoch))].sort((a, b) => a - b);
  const x0 = epochs[0], x1 = epochs[epochs.length - 1];
  const X = (e) => CM.l + ((e - x0) / (x1 - x0 || 1)) * (CW - CM.l - CM.r);
  // hover: epoch under the pointer in the training curves; bEpoch: epoch slider of the β charts
  const st = { split: "test", current: null, hover: null, bEpoch: x1, bHover: null };

  // ---- losses vs β (plain runs only: needs a β per model)
  const betaSeries = series.filter((s) => s.m.beta != null);
  const lb = betaSeries.map((s) => Math.log(s.m.beta));
  const lb0 = Math.min(...lb), lb1 = Math.max(...lb), lpad = 0.06 * (lb1 - lb0 || 1);
  const BX = (b) => BM.l + ((Math.log(b) - lb0 + lpad) / (lb1 - lb0 + 2 * lpad)) * (BW - BM.l - BM.r);
  const bcharts = betaSeries.length < 2 ? [] : BETA_METRICS.map(([key, title, f]) => ({
    key, title, f, y: null,
    svg: svg("svg", { viewBox: `0 0 ${BW} ${BH}`, class: "lp-chart", role: "img", "aria-label": `${title} against β` }),
  }));
  const bEpochNow = () => st.hover ?? st.bEpoch;
  const bCaption = h("span", { class: "lp-muted lp-bcap" });
  const bSlider = h("input", { type: "range", class: "lp-slider", min: "0", max: String(epochs.length - 1),
                               step: "1", value: String(epochs.length - 1), "aria-label": "epoch" });
  bSlider.addEventListener("input", () => { stopPlay(); st.bEpoch = epochs[+bSlider.value]; drawBeta(); });
  const playBtn = h("button", { type: "button", class: "lp-play", "aria-label": "play training",
                                onclick: () => (playTimer ? stopPlay() : startPlay()) }, "▶");
  let playTimer = 0;
  function startPlay() {
    if (st.bEpoch >= x1) st.bEpoch = x0;
    playBtn.textContent = "❚❚";
    playTimer = setInterval(() => {
      st.bEpoch = epochs[Math.min(epochs.length - 1, epochs.indexOf(st.bEpoch) + 1)];
      drawBeta();
      if (st.bEpoch >= x1) stopPlay();
    }, 160);
    drawBeta();
  }
  function stopPlay() {
    clearInterval(playTimer); playTimer = 0; playBtn.textContent = "▶";
  }

  const tip = h("div", { class: "lp-tip", role: "status" });
  const legend = h("div", { class: "lp-curve-legend" });
  const splitBtns = ["train", "test"].map((sp) =>
    h("button", { type: "button", "data-split": sp, onclick: () => { st.split = sp; draw(); } }, sp));
  const charts = CURVE_METRICS.map(([key, title]) => ({
    key, title, y: null,
    svg: svg("svg", { viewBox: `0 0 ${CW} ${CH}`, class: "lp-chart", role: "img",
                      "aria-label": `${title} per epoch, one line per β` }),
  }));
  const el = h("div", { class: "lp-curves" },
    h("div", { class: "lp-curves-head" },
      h("span", { class: "lp-label" }, "training curves — click a line to switch β"),
      h("div", { class: "lp-seg" }, ...splitBtns)),
    legend,
    h("div", { class: "lp-charts" },
      ...charts.map((c) => h("figure", { class: "lp-fig" }, h("figcaption", { class: "lp-label" }, c.title), c.svg))),
    ...(bcharts.length ? [
      h("div", { class: "lp-curves-head lp-bhead" },
        h("span", { class: "lp-label" }, "losses vs β — click a point to switch β"), bCaption),
      h("div", { class: "lp-bctl" }, playBtn, bSlider),
      h("div", { class: "lp-charts" }, ...bcharts.map((c) =>
        h("figure", { class: "lp-fig lp-fig-sm" }, h("figcaption", { class: "lp-label" }, c.title), c.svg))),
    ] : []),
    tip);

  const valueAt = (s, key, e) => {
    const i = s.m.history.epoch.indexOf(e);
    return i < 0 ? null : s.m.history[st.split][key][i];
  };
  const isCurrent = (s) => s.m === st.current;

  function drawChart(c) {
    const vis = series.filter((s) => s.on);
    const vals = vis.flatMap((s) => s.m.history[st.split][c.key]);
    const kids = [];
    if (!vals.length) { c.svg.replaceChildren(); return; }
    const yt = niceTicks(Math.min(...vals), Math.max(...vals));
    const Y = (v) => CH - CM.b - ((v - yt.lo) / (yt.hi - yt.lo || 1)) * (CH - CM.t - CM.b);
    c.y = Y;
    for (const v of yt.ticks) {
      kids.push(svg("line", { class: "lp-grid", x1: CM.l, x2: CW - CM.r, y1: Y(v), y2: Y(v) }),
        svg("text", { x: CM.l - 6, y: Y(v) + 4, "text-anchor": "end" }, String(v)));
    }
    for (const e of niceTicks(x0, x1, 5).ticks.filter((e) => e >= x0 && e <= x1)) {
      kids.push(svg("text", { x: X(e), y: CH - CM.b + 15, "text-anchor": "middle" }, String(e)));
    }
    kids.push(svg("line", { class: "lp-axis", x1: CM.l, x2: CW - CM.r, y1: CH - CM.b, y2: CH - CM.b }),
      svg("text", { x: (CM.l + CW - CM.r) / 2, y: CH - 4, "text-anchor": "middle" }, "epoch"));
    // current β drawn last so it sits on top
    const anyCurrent = vis.some(isCurrent);
    for (const s of [...vis].sort((a, b) => isCurrent(a) - isCurrent(b))) {
      const ys = s.m.history[st.split][c.key];
      const d = s.m.history.epoch.map((e, i) => `${i ? "L" : "M"}${X(e).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join("");
      kids.push(svg("path", { d, fill: "none", stroke: s.color, "stroke-linejoin": "round", "stroke-linecap": "round",
        "stroke-width": isCurrent(s) ? 3 : 2, opacity: anyCurrent && !isCurrent(s) ? 0.45 : 1 }));
    }
    if (st.hover != null) {
      kids.push(svg("line", { class: "lp-cross", x1: X(st.hover), x2: X(st.hover), y1: CM.t, y2: CH - CM.b }));
      for (const s of vis) {
        const v = valueAt(s, c.key, st.hover);
        if (v != null) kids.push(svg("circle", { class: "lp-dot", cx: X(st.hover), cy: Y(v), r: 4, fill: s.color }));
      }
    }
    c.svg.replaceChildren(...kids);
  }

  // value of f(recon, kl, β) for one model at epoch e
  const atEpoch = (s, f, e) => {
    const H = s.m.history, i = H.epoch.indexOf(e);
    return i < 0 ? null : f(H[st.split].recon[i], H[st.split].kl[i], s.m.beta);
  };

  function drawBetaChart(c) {
    const ep = bEpochNow();
    const pts = betaSeries.filter((s) => s.on)
      .map((s) => ({ s, v: atEpoch(s, c.f, ep), fin: atEpoch(s, c.f, x1) }))
      .filter((p) => p.v != null && p.fin != null);
    if (!pts.length) { c.svg.replaceChildren(); return; }
    const vals = pts.flatMap((p) => [p.v, p.fin]);
    const yt = niceTicks(Math.min(...vals), Math.max(...vals));
    const Y = (v) => BH - BM.b - ((v - yt.lo) / (yt.hi - yt.lo || 1)) * (BH - BM.t - BM.b);
    c.y = Y;
    const kids = [];
    for (const v of yt.ticks) {
      kids.push(svg("line", { class: "lp-grid", x1: BM.l, x2: BW - BM.r, y1: Y(v), y2: Y(v) }),
        svg("text", { x: BM.l - 6, y: Y(v) + 4, "text-anchor": "end" }, String(v)));
    }
    // decade labels; a small tick under every trained β
    for (const b of [0.001, 0.01, 0.1, 1, 10, 100, 1000]) {
      const x = BX(b);
      if (x >= BM.l && x <= BW - BM.r) kids.push(svg("text", { x, y: BH - BM.b + 15, "text-anchor": "middle" }, String(b)));
    }
    for (const s of betaSeries) {
      kids.push(svg("line", { class: "lp-axis", x1: BX(s.m.beta), x2: BX(s.m.beta), y1: BH - BM.b, y2: BH - BM.b + 4 }));
    }
    kids.push(svg("line", { class: "lp-axis", x1: BM.l, x2: BW - BM.r, y1: BH - BM.b, y2: BH - BM.b }),
      svg("text", { x: (BM.l + BW - BM.r) / 2, y: BH - 4, "text-anchor": "middle" }, "β (log scale)"));
    if (st.current && st.current.beta != null) {
      const x = BX(st.current.beta);
      kids.push(svg("line", { class: "lp-curmark", x1: x, x2: x, y1: BM.t, y2: BH - BM.b }));
    }
    if (st.bHover) {
      const x = BX(st.bHover.m.beta);
      kids.push(svg("line", { class: "lp-cross", x1: x, x2: x, y1: BM.t, y2: BH - BM.b }));
    }
    const path = (key, cls) => svg("path", { class: cls, fill: "none",
      d: pts.map((p, i) => `${i ? "L" : "M"}${BX(p.s.m.beta).toFixed(1)},${Y(p[key]).toFixed(1)}`).join("") });
    if (ep !== x1) kids.push(path("fin", "lp-ghost"));
    kids.push(path("v", "lp-bline"));
    for (const p of pts) {
      kids.push(svg("circle", { class: "lp-dot", cx: BX(p.s.m.beta), cy: Y(p.v), fill: p.s.color,
                                r: isCurrent(p.s) || p.s === st.bHover ? 6 : 4.5 }));
    }
    if (c.key === "nelbo") {
      // the tightest bound among the visible models
      const best = pts.reduce((a, p) => (p.v < a.v ? p : a));
      const x = BX(best.s.m.beta), y = Y(best.v);
      kids.push(svg("circle", { class: "lp-best", cx: x, cy: y, r: 9 }),
        svg("text", { class: "lp-best-label", x, y: y - 14,
                      "text-anchor": x < BW * 0.35 ? "start" : x > BW * 0.65 ? "end" : "middle" },
          `lowest: ${best.s.label} (${best.v.toFixed(1)})`));
    }
    c.svg.replaceChildren(...kids);
  }

  function drawBeta() {
    if (!bcharts.length) return;
    const ep = bEpochNow();
    bCaption.textContent = `epoch ${ep} of ${x1} · ${st.split}` + (ep === x1 ? " (final)" : " · dashed = final");
    bSlider.value = String(epochs.indexOf(st.bEpoch));
    for (const c of bcharts) drawBetaChart(c);
  }

  function epochTip() {
    const rows = series.filter((s) => s.on).map((s) => {
      const [r, k] = ["recon", "kl"].map((key) => valueAt(s, key, st.hover));
      return h("tr", { class: isCurrent(s) ? "lp-on" : "" },
        h("td", {}, h("i", { style: `background:${s.color}` }), s.label),
        h("td", {}, r == null ? "–" : r.toFixed(1)), h("td", {}, k == null ? "–" : k.toFixed(2)));
    });
    return [h("div", { class: "lp-label" }, `epoch ${st.hover} · ${st.split}`),
      h("table", {}, h("tr", {}, h("th", {}, ""), h("th", {}, "recon"), h("th", {}, "KL")), ...rows)];
  }

  function betaTip(s) {
    const ep = bEpochNow(), [r, k] = ["recon", "kl"].map((key) => valueAt(s, key, ep)), b = s.m.beta;
    if (r == null || k == null) return [h("div", { class: "lp-label" }, s.label)];
    const row = (name, v) => h("tr", {}, h("td", {}, name), h("td", {}, v));
    return [h("div", { class: "lp-label" }, h("i", { style: `background:${s.color}` }), ` ${s.label} · epoch ${ep} · ${st.split}`),
      h("table", {}, row("recon", r.toFixed(1)), row("KL", k.toFixed(2)),
        row("recon + KL (−ELBO)", (r + k).toFixed(1)),
        row("trained on recon + β·KL", (r + b * k).toFixed(1)))];
  }

  function hideTip() { tip.style.display = "none"; }
  function showTip(content, clientX, clientY) {
    tip.replaceChildren(...content);
    tip.style.display = "block";
    const box = el.getBoundingClientRect(), tw = tip.offsetWidth;
    let left = clientX - box.left + 14;
    if (left + tw > box.width) left = clientX - box.left - tw - 14;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${clientY - box.top + 14}px`;
  }

  function drawLegend() {
    legend.replaceChildren(...series.map((s) => h("button", {
      type: "button", class: (s.on ? "" : "lp-off ") + (isCurrent(s) ? "lp-on" : ""), "aria-pressed": String(s.on),
      title: "show / hide this β", onclick: () => { s.on = !s.on; draw(); },
    }, h("i", { style: `background:${s.color}` }), s.label)));
  }

  function draw() {
    const pal = isDarkSurface(getComputedStyle(el).getPropertyValue("--lp-pad-bg").trim()) ? SERIES_DARK : SERIES_LIGHT;
    series.forEach((s, i) => { s.color = pal[i % pal.length]; });
    for (const b of splitBtns) b.classList.toggle("lp-on", b.dataset.split === st.split);
    drawLegend();
    for (const c of charts) drawChart(c);
    drawBeta();
  }

  const epochAt = (c, e) => {
    const r = c.svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * CW;
    const t = x0 + ((px - CM.l) / (CW - CM.l - CM.r)) * (x1 - x0);
    return epochs.reduce((a, b) => (Math.abs(b - t) < Math.abs(a - t) ? b : a));
  };
  for (const c of charts) {
    c.svg.addEventListener("pointermove", (e) => {
      const ep = epochAt(c, e);
      if (ep !== st.hover) { st.hover = ep; for (const k of charts) drawChart(k); drawBeta(); }
      showTip(epochTip(), e.clientX, e.clientY);
    });
    c.svg.addEventListener("pointerleave", () => {
      st.hover = null; for (const k of charts) drawChart(k); drawBeta(); hideTip();
    });
    c.svg.addEventListener("click", (e) => {
      // nearest visible line to the pointer at the hovered epoch
      if (!onPick || !c.y) return;
      const r = c.svg.getBoundingClientRect(), py = ((e.clientY - r.top) / r.height) * CH, ep = epochAt(c, e);
      let best = null, bd = Infinity;
      for (const s of series.filter((q) => q.on)) {
        const v = valueAt(s, c.key, ep);
        if (v != null && Math.abs(c.y(v) - py) < bd) { bd = Math.abs(c.y(v) - py); best = s; }
      }
      if (best) onPick(best.m);
    });
  }

  const betaAt = (c, e) => {
    const r = c.svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * BW;
    return betaSeries.filter((s) => s.on)
      .reduce((a, s) => (!a || Math.abs(BX(s.m.beta) - px) < Math.abs(BX(a.m.beta) - px) ? s : a), null);
  };
  for (const c of bcharts) {
    c.svg.addEventListener("pointermove", (e) => {
      const s = betaAt(c, e);
      if (s !== st.bHover) { st.bHover = s; drawBeta(); }
      if (s) showTip(betaTip(s), e.clientX, e.clientY); else hideTip();
    });
    c.svg.addEventListener("pointerleave", () => { st.bHover = null; drawBeta(); hideTip(); });
    c.svg.addEventListener("click", (e) => {
      const s = betaAt(c, e);
      if (s && onPick) onPick(s.m);
    });
  }

  return {
    el, draw,
    setCurrent(m) { if (st.current !== m) { st.current = m; draw(); } },
    destroy: stopPlay,
  };
}

// ------------------------------------------------------------------ the widget
/**
 * Mount the pad into `el`. `source` provides getManifest() / getBuffer(path).
 * opts: {z, beta, background, onChange({z, beta, background})}
 * Returns a controller {setZ, setBeta, setBackground, decodeAt, destroy}.
 */
export async function mount(el, source, opts = {}) {
  const manifest = await source.getManifest();
  const lim = manifest.lim, betaSpec = manifest.beta;
  const discrete = betaSpec.mode === "discrete";
  const binarized = manifest.binarize && manifest.binarize !== "none";
  const state = {
    z: opts.z ? [...opts.z] : [0, 0],
    beta: opts.beta ?? 1,
    background: opts.background || "scatter",
    showPrior: false,
    hoverPrior: -1,
    pixels: opts.pixels || "mean",  // "mean": p(x|z) itself | "sample": x ~ Bernoulli(p)
  };
  if (discrete) state.beta = nearestValue(betaSpec.values, state.beta);
  else state.beta = Math.min(betaSpec.max, Math.max(betaSpec.min, state.beta));

  const cache = new Map();
  const load = (key, fn) => { if (!cache.has(key)) cache.set(key, fn()); return cache.get(key); };
  const priorZ = gaussianSamples(PRIOR_COLS * PRIOR_ROWS, PRIOR_SEED);
  // Sampled pixels use one fixed uniform draw u per pixel (x = 1 iff u < p), so a sample changes
  // smoothly while z is dragged instead of flickering; "resample" draws new u.
  const nPix = manifest.image[0] * manifest.image[1];
  let noiseSeed = 1, outU, priorU;
  function drawNoise() {
    const rnd = mulberry32(noiseSeed++);
    outU = Float32Array.from({ length: nPix }, rnd);
    priorU = Float32Array.from({ length: PRIOR_COLS * PRIOR_ROWS * nPix }, rnd);
  }
  drawNoise();
  const shown = (p, U) => {
    if (state.pixels !== "sample") return p;
    const x = new Float32Array(p.length);
    for (let i = 0; i < p.length; i++) x[i] = U[i] < p[i] ? 1 : 0;
    return x;
  };

  // ---- layout
  const pad = h("canvas", { class: "lp-pad", tabindex: "0",
    "aria-label": "Latent plane. Drag, or use arrow keys, to move z." });
  const out = h("canvas", { class: "lp-out", width: "28", height: "28" });
  const prior = h("canvas", { class: "lp-prior", width: String(PRIOR_COLS * 28),
                              height: String(PRIOR_ROWS * 28) });
  const zText = h("div", { class: "lp-z" });
  const massText = h("div", { class: "lp-muted" });
  const betaText = h("span", { class: "lp-beta-value" });
  const metricText = h("div", { class: "lp-muted lp-metrics" });
  const status = h("div", { class: "lp-status" });
  const slider = h("input", { type: "range", class: "lp-slider", "aria-label": "beta" });
  const ticks = h("div", { class: "lp-ticks" });
  const legend = h("div", { class: "lp-legend" },
    ...TAB10.map((c, d) => h("span", {}, h("i", { style: `background:${c}` }), String(d))));
  const bgButtons = ["scatter", "manifold", "none"].map((b) =>
    h("button", { type: "button", "data-bg": b, onclick: () => setBackground(b, true) },
      { scatter: "μ(x) scatter", manifold: "decoded grid", none: "none" }[b]));
  const pixelButtons = ["mean", "sample"].map((m) =>
    h("button", { type: "button", "data-pixels": m, onclick: () => setPixels(m, true) },
      { mean: "probability p", sample: "sampled x ~ Bern(p)" }[m]));
  const resampleBtn = h("button", { type: "button", class: "lp-btn", title: "draw new Bernoulli noise",
    onclick: () => { drawNoise(); if (cur) { drawOut(); renderPrior(); } } }, "resample");
  const priorToggle = h("input", { type: "checkbox" });
  priorToggle.addEventListener("change", () => { state.showPrior = priorToggle.checked; drawPad(); });

  const root = h("div", { class: "lp-root" },
    h("div", { class: "lp-main" },
      h("div", { class: "lp-left" }, pad, legend),
      h("div", { class: "lp-right" },
        h("div", { class: "lp-label" }, "decoder(z)"), out, zText, massText,
        h("div", { class: "lp-label lp-gap" }, "background"),
        h("div", { class: "lp-seg" }, ...bgButtons),
        h("label", { class: "lp-check" }, priorToggle, " mark prior samples on the plane"),
        h("div", { class: "lp-label lp-gap" }, "pixels (decoder(z) and prior samples)"),
        h("div", { class: "lp-row" }, h("div", { class: "lp-seg" }, ...pixelButtons), resampleBtn))),
    h("div", { class: "lp-beta" },
      h("div", { class: "lp-beta-head" }, h("span", { class: "lp-label" }, "β"), betaText, metricText),
      slider, ticks),
    h("div", { class: "lp-label lp-gap" },
      `prior samples: decoder(z), z ~ N(0, I), same ${PRIOR_COLS * PRIOR_ROWS} z for every β — click one to jump there`),
    prior, status);
  const curves = createCurves(manifest.models, (m) => { if (m.beta != null) setBeta(m.beta, true); });
  if (curves) root.append(curves.el);
  el.replaceChildren(root);

  // ---- β slider
  const tickBetas = discrete ? betaSpec.values
    : [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 100]
        .filter((b) => b >= betaSpec.min * 0.999 && b <= betaSpec.max * 1.001);
  const toSlider = (b) => discrete ? betaSpec.values.indexOf(b)
    : 1000 * Math.log(b / betaSpec.min) / Math.log(betaSpec.max / betaSpec.min);
  const fromSlider = (v) => discrete ? betaSpec.values[+v]
    : betaSpec.min * (betaSpec.max / betaSpec.min) ** (v / 1000);
  slider.min = 0;
  slider.max = discrete ? betaSpec.values.length - 1 : 1000;
  slider.step = 1;
  for (const b of tickBetas) {
    const t = h("button", { type: "button", onclick: () => setBeta(b, true) }, fmtBeta(b));
    // discrete ticks are evenly spaced (one per trained model); continuous ones sit on the log axis
    t.style.left = `${(100 * toSlider(b)) / slider.max}%`;
    ticks.append(t);
  }
  slider.addEventListener("input", () => setBeta(fromSlider(slider.value), true));

  // ---- model access
  function modelFor(beta) {
    return discrete ? manifest.models[betaSpec.values.indexOf(beta)] : manifest.models[0];
  }
  async function decoderFor(m) {
    return load("dec:" + m.name, async () => new Decoder(m.decoder, await source.getBuffer(m.decoder.path)));
  }
  function nearestEval(m, beta) {
    return m.evals.reduce((a, e) => Math.abs(Math.log(e.beta / beta)) < Math.abs(Math.log(a.beta / beta)) ? e : a);
  }
  async function latentsFor(m, ev) {
    const [mu, lab] = await Promise.all([
      load("buf:" + ev.mu, () => source.getBuffer(ev.mu).then((b) => new Float32Array(b))),
      load("buf:" + m.labels, () => source.getBuffer(m.labels).then((b) => new Uint8Array(b))),
    ]);
    return { mu, lab };
  }
  function metricsAt(m, beta) {
    if (discrete || m.evals.length === 1) return { ...m.evals[0], exact: true };
    const ev = [...m.evals].sort((a, b) => a.beta - b.beta);
    const t = Math.log(beta);
    let i = ev.findIndex((e) => e.beta >= beta);
    if (i <= 0) return { ...ev[Math.max(i, 0)], exact: i === 0 && ev[0].beta === beta };
    const a = ev[i - 1], b = ev[i], w = (t - Math.log(a.beta)) / (Math.log(b.beta) - Math.log(a.beta));
    return { recon: a.recon + w * (b.recon - a.recon), kl: a.kl + w * (b.kl - a.kl), exact: false };
  }

  // ---- current β assets
  let cur = null;          // {m, dec, c, lat, ev}
  let bg = null;           // offscreen canvas for the background layer
  let version = 0;

  async function refreshBeta() {
    const v = ++version, beta = state.beta, m = modelFor(beta);
    status.textContent = "";
    try {
      const dec = await decoderFor(m);
      const ev = nearestEval(m, beta);
      const lat = await latentsFor(m, ev);
      if (v !== version) return;  // a newer β arrived meanwhile
      cur = { m, dec, ev, lat, c: m.cond ? betaFeatures(beta, m.cond) : null };
      renderBackground();
      renderPrior();
      drawPad();
      drawOut();
      updateText();
    } catch (err) {
      status.textContent = String(err);
      throw err;
    }
  }

  // ---- drawing
  const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  const toPx = (z1, z2, S) => [((z1 + lim) / (2 * lim)) * S, ((lim - z2) / (2 * lim)) * S];
  const toZ = (x, y, S) => [(x / S) * 2 * lim - lim, lim - (y / S) * 2 * lim];

  function sizePad() {
    const dpr = window.devicePixelRatio || 1;
    const S = Math.round((pad.clientWidth || 400) * dpr);
    if (pad.width !== S) { pad.width = pad.height = S; renderBackground(); }
    return S;
  }

  function renderBackground() {
    if (!cur) return;
    const S = pad.width;
    bg = document.createElement("canvas");
    bg.width = bg.height = S;
    const g = bg.getContext("2d");
    if (state.background === "scatter") {
      const { mu, lab } = cur.lat, r = Math.max(1.2, S / 300);
      g.globalAlpha = 0.6;
      for (let i = 0; i < lab.length; i++) {
        const [x, y] = toPx(mu[2 * i], mu[2 * i + 1], S);
        g.fillStyle = TAB10[lab[i]];
        g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill();
      }
    } else if (state.background === "manifold") {
      const n = MANIFOLD_N, zs = new Float32Array(2 * n * n);
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
        // cell centres, row 0 = top = largest z2 (same orientation as the pad)
        zs[2 * (r * n + c)] = -lim + (2 * lim * (c + 0.5)) / n;
        zs[2 * (r * n + c) + 1] = lim - (2 * lim * (r + 0.5)) / n;
      }
      const tile = document.createElement("canvas");
      tile.width = tile.height = n * 28;
      paintDigits(tile.getContext("2d"), cur.dec.forward(zs, n * n, cur.c), n * n, n);
      g.imageSmoothingEnabled = true;
      g.drawImage(tile, 0, 0, S, S);
    }
  }

  function drawPad() {
    const S = sizePad(), g = pad.getContext("2d"), dpr = window.devicePixelRatio || 1;
    const dark = state.background === "manifold";
    g.fillStyle = dark ? "#000" : css("--lp-pad-bg") || "#fff";
    g.fillRect(0, 0, S, S);
    if (bg) g.drawImage(bg, 0, 0);
    const ink = dark ? "rgba(255,255,255,0.75)" : css("--lp-fg") || "#222";
    const faint = dark ? "rgba(255,255,255,0.18)" : css("--lp-grid") || "rgba(0,0,0,0.1)";

    // grid + axes
    g.lineWidth = dpr;
    g.strokeStyle = faint;
    for (let t = Math.ceil(-lim); t <= lim; t++) {
      const [x] = toPx(t, 0, S), [, y] = toPx(0, t, S);
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, S); g.moveTo(0, y); g.lineTo(S, y); g.stroke();
    }
    g.fillStyle = ink;
    g.font = `${11 * dpr}px system-ui, sans-serif`;
    g.globalAlpha = 0.7;
    for (let t = Math.ceil(-lim); t <= lim; t++) {
      if (t === 0) continue;
      const [x, y0] = toPx(t, 0, S);
      g.fillText(String(t), x + 2 * dpr, y0 - 3 * dpr);
    }
    g.fillText("z1 →", S - 34 * dpr, toPx(0, 0, S)[1] + 14 * dpr);
    g.fillText("z2 ↑", toPx(0, 0, S)[0] + 4 * dpr, 13 * dpr);
    g.globalAlpha = 1;

    // prior contours: radius r holds 1 - exp(-r²/2) of N(0, I)
    const [cx, cy] = toPx(0, 0, S), unit = S / (2 * lim);
    g.setLineDash([5 * dpr, 4 * dpr]);
    g.strokeStyle = ink;
    g.lineWidth = 1.2 * dpr;
    for (const r of [1, 2, 3]) {
      if (r > lim * 1.42) continue;
      g.beginPath(); g.arc(cx, cy, r * unit, 0, 2 * Math.PI); g.stroke();
      g.fillText(`${Math.round(100 * priorMass(r))}%`, cx + r * unit * 0.707 + 3 * dpr, cy - r * unit * 0.707 - 3 * dpr);
    }
    g.setLineDash([]);

    // prior sample markers
    if (state.showPrior || state.hoverPrior >= 0) {
      for (let i = 0; i < PRIOR_COLS * PRIOR_ROWS; i++) {
        if (!state.showPrior && i !== state.hoverPrior) continue;
        const [x, y] = toPx(priorZ[2 * i], priorZ[2 * i + 1], S), s = (i === state.hoverPrior ? 6 : 3.5) * dpr;
        g.strokeStyle = i === state.hoverPrior ? css("--lp-accent") || "#e8590c" : ink;
        g.lineWidth = (i === state.hoverPrior ? 2.5 : 1.5) * dpr;
        g.beginPath(); g.moveTo(x - s, y - s); g.lineTo(x + s, y + s);
        g.moveTo(x - s, y + s); g.lineTo(x + s, y - s); g.stroke();
      }
    }

    // the cursor
    const [x, y] = toPx(state.z[0], state.z[1], S), R = 9 * dpr;
    g.lineWidth = 4 * dpr; g.strokeStyle = dark ? "#000" : "#fff";
    g.beginPath(); g.arc(x, y, R, 0, 2 * Math.PI); g.stroke();
    g.lineWidth = 2 * dpr; g.strokeStyle = css("--lp-accent") || "#e8590c";
    g.beginPath(); g.arc(x, y, R, 0, 2 * Math.PI);
    g.moveTo(x - R * 1.8, y); g.lineTo(x - R * 0.5, y); g.moveTo(x + R * 0.5, y); g.lineTo(x + R * 1.8, y);
    g.moveTo(x, y - R * 1.8); g.lineTo(x, y - R * 0.5); g.moveTo(x, y + R * 0.5); g.lineTo(x, y + R * 1.8);
    g.stroke();
  }

  function drawOut() {
    if (!cur) return;
    paintDigits(out.getContext("2d"), shown(cur.dec.forward(new Float32Array(state.z), 1, cur.c), outU), 1, 1);
  }

  function renderPrior() {
    const n = PRIOR_COLS * PRIOR_ROWS;
    paintDigits(prior.getContext("2d"), shown(cur.dec.forward(priorZ, n, cur.c), priorU), n, PRIOR_COLS);
  }

  function updateText() {
    const [a, b] = state.z, r = Math.hypot(a, b);
    zText.textContent = `z = (${a.toFixed(2)}, ${b.toFixed(2)})`;
    massText.textContent = `|z| = ${r.toFixed(2)} · ${(100 * (1 - priorMass(r))).toFixed(1)}% of prior mass lies farther out`;
    betaText.textContent = fmtBeta(state.beta);
    if (cur) {
      const mt = metricsAt(cur.m, state.beta);
      metricText.textContent = `test recon ${mt.recon.toFixed(1)} nats · KL ${mt.kl.toFixed(2)} nats` +
        (mt.exact ? "" : " (interpolated)") + (binarized ? " · binarized MNIST" : " · grayscale MNIST");
    }
    if (curves) curves.setCurrent(modelFor(state.beta));
    slider.value = toSlider(state.beta);
    for (const t of ticks.children) t.classList.toggle("lp-on", discrete && t.textContent === fmtBeta(state.beta));
    for (const btn of bgButtons) btn.classList.toggle("lp-on", btn.dataset.bg === state.background);
    for (const btn of pixelButtons) btn.classList.toggle("lp-on", btn.dataset.pixels === state.pixels);
    resampleBtn.disabled = state.pixels !== "sample";
  }

  // ---- frame scheduling: pointer moves only mark dirty, one decode per animation frame
  let frame = 0, betaDirty = false;
  function schedule() {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      if (betaDirty) { betaDirty = false; refreshBeta(); }
      else { drawPad(); drawOut(); updateText(); }
    });
  }

  const notify = () => opts.onChange && opts.onChange({ z: [...state.z], beta: state.beta,
                                                         background: state.background, pixels: state.pixels });
  const clampZ = (v) => Math.max(-lim, Math.min(lim, v));

  function setZ(z, fromUser = false) {
    state.z = [clampZ(+z[0]), clampZ(+z[1])];
    schedule();
    if (fromUser) notify();
  }
  function setBeta(beta, fromUser = false) {
    beta = discrete ? nearestValue(betaSpec.values, +beta)
                    : Math.min(betaSpec.max, Math.max(betaSpec.min, +beta));
    if (beta === state.beta && cur) return;
    state.beta = beta;
    betaDirty = true;
    updateText();
    schedule();
    if (fromUser) notify();
  }
  function setPixels(m, fromUser = false) {
    if (m === state.pixels || !["mean", "sample"].includes(m)) return;
    state.pixels = m;
    if (cur) { drawOut(); renderPrior(); }
    updateText();
    if (fromUser) notify();
  }
  function setBackground(b, fromUser = false) {
    if (b === state.background) return;
    state.background = b;
    renderBackground();
    drawPad();
    updateText();
    if (fromUser) notify();
  }

  // ---- interaction
  let dragging = false;
  const zFromEvent = (e) => {
    const r = pad.getBoundingClientRect();
    return toZ(e.clientX - r.left, e.clientY - r.top, r.width);
  };
  pad.addEventListener("pointerdown", (e) => {
    dragging = true; pad.setPointerCapture(e.pointerId); pad.focus(); setZ(zFromEvent(e), true);
  });
  pad.addEventListener("pointermove", (e) => { if (dragging) setZ(zFromEvent(e), true); });
  const end = () => { dragging = false; };
  pad.addEventListener("pointerup", end);
  pad.addEventListener("pointercancel", end);
  pad.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 0.25 : 0.05;
    const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }[e.key];
    if (!d) return;
    e.preventDefault();
    setZ([state.z[0] + d[0], state.z[1] + d[1]], true);
  });
  const priorIndex = (e) => {
    const r = prior.getBoundingClientRect();
    const c = Math.floor(((e.clientX - r.left) / r.width) * PRIOR_COLS);
    const w = Math.floor(((e.clientY - r.top) / r.height) * PRIOR_ROWS);
    return c >= 0 && c < PRIOR_COLS && w >= 0 && w < PRIOR_ROWS ? w * PRIOR_COLS + c : -1;
  };
  prior.addEventListener("pointermove", (e) => {
    const i = priorIndex(e);
    if (i !== state.hoverPrior) { state.hoverPrior = i; drawPad(); }
  });
  prior.addEventListener("pointerleave", () => { state.hoverPrior = -1; drawPad(); });
  prior.addEventListener("click", (e) => {
    const i = priorIndex(e);
    if (i >= 0) setZ([priorZ[2 * i], priorZ[2 * i + 1]], true);
  });

  const ro = new ResizeObserver(() => { if (cur) drawPad(); });
  ro.observe(pad);
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const onScheme = () => { drawPad(); if (curves) curves.draw(); };
  mq.addEventListener("change", onScheme);

  updateText();
  await refreshBeta();

  return {
    setZ: (z) => setZ(z),
    setBeta: (b) => setBeta(b),
    setBackground: (b) => setBackground(b),
    setPixels: (m) => setPixels(m),
    /** For checking against the Python decoder: decodes z at the current β. */
    decodeAt: async (z, beta = state.beta) => {
      const m = modelFor(discrete ? nearestValue(betaSpec.values, beta) : state.beta);
      const dec = await decoderFor(m);
      return dec.forward(new Float32Array(z), 1, m.cond ? betaFeatures(beta, m.cond) : null);
    },
    get state() { return { ...state, z: [...state.z] }; },
    destroy() { if (curves) curves.destroy(); ro.disconnect(); mq.removeEventListener("change", onScheme); el.replaceChildren(); },
  };
}

function nearestValue(values, v) {
  return values.reduce((a, b) => (Math.abs(Math.log(b / v)) < Math.abs(Math.log(a / v)) ? b : a));
}

// ------------------------------------------------------------------ anywidget entry point
export default {
  async render({ model, el }) {
    // Dragging fires many changes per second; sync back to Python at most every 80 ms.
    let pending = null, timer = 0;
    const flush = () => {
      timer = 0;
      if (!pending) return;
      for (const [k, v] of Object.entries(pending)) model.set(k, v);
      model.save_changes();
      pending = null;
    };
    const ctl = await mount(el, traitletSource(model), {
      z: model.get("z"), beta: model.get("beta"), background: model.get("background"),
      pixels: model.get("pixels"),
      onChange(s) {
        pending = s;
        if (!timer) timer = setTimeout(flush, 80);
      },
    });
    const onZ = () => {
      const z = model.get("z"), s = ctl.state.z;
      if (z[0] !== s[0] || z[1] !== s[1]) ctl.setZ(z);
    };
    const onBeta = () => { if (model.get("beta") !== ctl.state.beta) ctl.setBeta(model.get("beta")); };
    const onBg = () => ctl.setBackground(model.get("background"));
    const onPixels = () => ctl.setPixels(model.get("pixels"));
    model.on("change:z", onZ);
    model.on("change:beta", onBeta);
    model.on("change:background", onBg);
    model.on("change:pixels", onPixels);
    window.__latentPad = ctl;  // handy for debugging in the browser console
    return () => {
      model.off("change:z", onZ);
      model.off("change:beta", onBeta);
      model.off("change:background", onBg);
      model.off("change:pixels", onPixels);
      ctl.destroy();
    };
  },
};
