// Small SVG charts of the server view: bars or lines over the one-second history, with a crosshair tooltip listing
// every series at the pointer. Colours come from CSS classes (`series-1`…`series-3`); values are inserted as text.

import { h } from './ui.js';

const NS = 'http://www.w3.org/2000/svg';
const W = 600;

function svg(tag, attrs, ...children) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, String(v));
  el.append(...children);
  return el;
}

const finite = (v) => typeof v === 'number' && Number.isFinite(v);

/** Colour key of a series, in the legend and the tooltip. */
const seriesKey = (s) => h('span', { class: `chart-key ${s.cls}${s.dashed ? ' is-dashed' : ''}`, 'aria-hidden': 'true' });

/** Path of a line with gaps where the value is missing; a lone value is drawn as a dot (round cap of a tiny step). */
function linePath(xs, ys) {
  let d = '';
  let pen = false;
  xs.forEach((x, i) => {
    if (!finite(ys[i])) {
      pen = false;
      return;
    }
    d += pen ? `L${x.toFixed(1)},${ys[i].toFixed(1)}` : `M${x.toFixed(1)},${ys[i].toFixed(1)}h0.01`;
    pen = true;
  });
  return d;
}

/** A "nice" upper bound for the y axis. */
function niceMax(v) {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v);
}

/**
 * Time chart over the history. `series`: [{label, value(point) → number|null, cls, dashed}]; `kind`: 'line' or 'bar'
 * (bars draw the first series only); `format(v)` for axis and tooltip; `max`: fixed y maximum (e.g. 100 for %);
 * `seconds`: visible window. Returns `{el, update(points)}`; the element keeps its identity across updates.
 */
export function timeChart({ title, series, kind = 'line', format = String, max, seconds = 300, height = 140 }) {
  const plot = svg('svg', { class: 'chart-svg', viewBox: `0 0 ${W} ${height}`, preserveAspectRatio: 'none', 'aria-hidden': 'true' });
  const yMax = h('span', { class: 'chart-y chart-y-max' });
  const yZero = h('span', { class: 'chart-y chart-y-zero' }, format(0));
  const tip = h('div', { class: 'chart-tip', hidden: true, role: 'presentation' });
  const empty = h('p', { class: 'chart-empty muted small', hidden: true }, 'No data yet.');
  const legend = series.length > 1
    ? h('ul', { class: 'chart-legend' }, series.map((s) => h('li', null, seriesKey(s), s.label)))
    : null;
  const area = h('div', { class: 'chart-area' }, plot, yMax, yZero, tip, empty);
  const el = h('figure', { class: 'chart' },
    h('figcaption', { class: 'chart-head' }, h('span', { class: 'chart-title' }, title), legend),
    area,
    h('div', { class: 'chart-x muted small' }, h('span', null, seconds >= 120 ? `−${seconds / 60} min` : `−${seconds} s`), h('span', null, 'now')));
  let points = [];
  let pointerX = null;
  let geom = null;

  function draw() {
    plot.replaceChildren();
    const last = points.length ? points[points.length - 1].at : 0;
    const start = last - seconds * 1000;
    const shown = points.filter((p) => p.at > start);
    const values = series.map((s) => shown.map((p) => s.value(p)));
    const top = max ?? niceMax(Math.max(0, ...values.flat().filter(finite)));
    const x = (at) => ((at - start) / (seconds * 1000)) * W;
    const y = (v) => height - 2 - (Math.max(0, v) / top) * (height - 4);
    yMax.textContent = format(top);
    empty.hidden = shown.length > 0;
    for (const f of [0.5, 1]) {
      plot.append(svg('line', { class: 'chart-rule', x1: 0, x2: W, y1: y(top * f), y2: y(top * f), 'vector-effect': 'non-scaling-stroke' }));
    }
    plot.append(svg('line', { class: 'chart-base', x1: 0, x2: W, y1: height - 1, y2: height - 1, 'vector-effect': 'non-scaling-stroke' }));
    const xs = shown.map((p) => x(p.at));
    if (kind === 'bar') {
      const bw = Math.max(1, W / seconds - 0.6);
      shown.forEach((p, i) => {
        const v = values[0][i];
        if (finite(v) && v > 0) {
          plot.append(svg('rect', { class: `chart-bar ${series[0].cls}`, x: (xs[i] - bw).toFixed(1), y: y(v).toFixed(1), width: bw.toFixed(1), height: (height - 1 - y(v)).toFixed(1) }));
        }
      });
    } else {
      series.forEach((s, i) => {
        plot.append(svg('path', { class: `chart-line ${s.cls}${s.dashed ? ' is-dashed' : ''}`, d: linePath(xs, values[i].map((v) => (finite(v) ? y(v) : null))), 'vector-effect': 'non-scaling-stroke' }));
      });
    }
    geom = { shown, xs, values };
    showTip();
  }

  function showTip() {
    if (pointerX == null || !geom || !geom.shown.length) {
      tip.hidden = true;
      return;
    }
    const target = pointerX * W;
    let best = 0;
    geom.xs.forEach((px, i) => {
      if (Math.abs(px - target) < Math.abs(geom.xs[best] - target)) best = i;
    });
    const px = geom.xs[best];
    plot.append(svg('line', { class: 'chart-cross', x1: px, x2: px, y1: 0, y2: height, 'vector-effect': 'non-scaling-stroke' }));
    tip.replaceChildren(h('div', { class: 'chart-tip-time' }, new Date(geom.shown[best].at).toLocaleTimeString()),
      series.map((s, i) => h('div', { class: 'chart-tip-row' }, seriesKey(s),
        h('strong', null, finite(geom.values[i][best]) ? format(geom.values[i][best]) : '—'), ` ${s.label}`)));
    tip.hidden = false;
    const frac = px / W;
    tip.style.left = `${frac * 100}%`;
    tip.classList.toggle('is-left', frac > 0.6);
  }

  area.addEventListener('pointermove', (ev) => {
    const r = plot.getBoundingClientRect();
    pointerX = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
    draw();
  });
  area.addEventListener('pointerleave', () => {
    pointerX = null;
    draw();
  });

  return {
    el,
    update(next) {
      points = next;
      draw();
    },
  };
}

/** Decorative trend line of a stat card (the value itself is shown as text next to it). */
export function sparkline(values, cls = 'series-1') {
  const n = values.length;
  const top = niceMax(Math.max(0, ...values.filter(finite)));
  const xs = values.map((_, i) => (n > 1 ? (i / (n - 1)) * 120 : 0));
  const ys = values.map((v) => (finite(v) ? 30 - (Math.max(0, v) / top) * 28 : null));
  return svg('svg', { class: 'spark', viewBox: '0 0 120 32', preserveAspectRatio: 'none', 'aria-hidden': 'true' },
    svg('path', { class: `chart-line ${cls}`, d: linePath(xs, ys), 'vector-effect': 'non-scaling-stroke' }));
}
