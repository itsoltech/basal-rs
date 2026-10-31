// Admin server view: live processing statistics, engine lanes, prefix caches, GPU and host telemetry, configuration
// and the most recent requests of this server process (`/ui/api/admin/stats`, `/ui/api/admin/requests`). Polls once
// a second; the history lives on the server (5 minutes), so reopening the view shows it again.

import { api, send, errorMessage } from './api.js';
import {
  h, button, pill, bar, fmtNum, fmtBytes, mono, toast, openDialog, confirmDialog, copyButton, tabs, table, callout,
  pageHeader, loading, kv,
} from './ui.js';
import { timeChart, sparkline } from './charts.js';

const POLL_MS = 1000;
const REQUESTS_POLL_MS = 2000;

const finite = (v) => typeof v === 'number' && Number.isFinite(v);

function fmtMs(v) {
  if (!finite(v)) return '—';
  return `${fmtNum(v, v < 10 ? 2 : v < 100 ? 1 : 0)} ms`;
}

function fmtPctValue(v, digits = 0) {
  return finite(v) ? `${fmtNum(v, digits)}%` : '—';
}

function fmtDuration(ms) {
  if (!finite(ms) || ms < 0) return '—';
  let s = Math.floor(ms / 1000);
  const d = Math.floor(s / 86400);
  s -= d * 86400;
  const hh = Math.floor(s / 3600);
  s -= hh * 3600;
  const mm = Math.floor(s / 60);
  s -= mm * 60;
  return [d && `${d}d`, (d || hh) && `${hh}h`, (d || hh || mm) && `${mm}m`, `${s}s`].filter(Boolean).join(' ');
}

const clock = (ms) => (finite(ms) ? new Date(ms).toLocaleTimeString() : '—');
const dateTime = (ms) => (finite(ms) ? new Date(ms).toLocaleString() : '—');
const sum = (xs) => xs.reduce((a, x) => a + (finite(x) ? x : 0), 0);

function card(title, ...children) {
  return h('section', { class: 'card' }, h('h2', { class: 'section-title' }, title), children);
}

function noData(text) {
  return h('p', { class: 'muted small' }, text);
}

function statCard(label, value, sub, spark) {
  return h('div', { class: 'stat-card' },
    h('span', { class: 'stat-label' }, label),
    h('span', { class: 'stat-big' }, value),
    sub ? h('span', { class: 'stat-sub' }, sub) : null,
    spark || null);
}

/** Prefix caches of all lanes of one model, summed (`lanes`: how many have one). `null` when its engine keeps none. */
function modelCaches(lanes) {
  const byModel = new Map();
  for (const l of lanes) {
    if (!byModel.has(l.model)) byModel.set(l.model, null);
    if (!l.cache) continue;
    const acc = byModel.get(l.model) || { lanes: 0 };
    acc.lanes += 1;
    for (const [k, v] of Object.entries(l.cache)) acc[k] = (acc[k] || 0) + v;
    byModel.set(l.model, acc);
  }
  return [...byModel];
}

function statusSummary(state) {
  const health = state.health;
  const s = state.snap;
  if (!health || typeof health.status !== 'string') return pill('unknown', 'muted');
  if (health.status !== 'ok') return pill(health.status);
  const busy = s && s.load.running + s.load.queued > 0;
  return pill(busy ? 'Serving' : 'Ready', busy ? 'info' : 'ok');
}

// ---------------------------------------------------------------------------------------------------------------
// Overview

function overview(state) {
  const header = h('div', { class: 'server-strip card' });
  const cards = h('div', { class: 'stat-cards' });
  const throughput = timeChart({ title: 'Throughput (requests/s)', kind: 'bar', format: (v) => fmtNum(v, 1), series: [{ label: 'requests/s', cls: 'series-1', value: (p) => p.requests }] });
  const latency = timeChart({
    title: 'Latency (ms)',
    format: (v) => fmtNum(v, v < 10 ? 1 : 0),
    series: [
      { label: 'p90', cls: 'series-1', value: (p) => p.p90 },
      { label: 'p99', cls: 'series-2', value: (p) => p.p99 },
      { label: 'mean of last 10 s', cls: 'series-3', dashed: true, value: (p) => p.mean10 },
    ],
  });
  const endpoints = h('div');
  const cache = h('div');
  const cacheChart = timeChart({ title: 'State cache memory', format: fmtBytes, series: [{ label: 'bytes', cls: 'series-1', value: (p) => p.state_cache_bytes }] });
  const gpus = h('div');
  const host = h('div');
  const el = h('div', { class: 'server-view' },
    header, cards,
    h('div', { class: 'chart-grid' }, h('section', { class: 'card' }, throughput.el), h('section', { class: 'card' }, latency.el)),
    card('Endpoints', endpoints),
    card('GPUs', gpus),
    h('div', { class: 'chart-grid' }, card('State cache', cache, cacheChart.el), card('Host', host)));

  function update() {
    const s = state.snap;
    const points = state.points;
    const models = (state.config && state.config.models) || [];
    const since = s.reset_at ? 'reset' : 'start';
    header.replaceChildren(
      h('div', { class: 'strip-item' }, h('span', { class: 'stat-label' }, models.length > 1 ? 'Models' : 'Model'),
        h('span', { class: 'strip-value' }, models.map((m) => m.name).join(', ') || '—')),
      h('div', { class: 'strip-item' }, h('span', { class: 'stat-label' }, 'Status'), statusSummary(state)),
      h('div', { class: 'strip-item' }, h('span', { class: 'stat-label' }, 'Version'), h('span', { class: 'strip-value' }, state.version.split(' ')[0])),
      h('div', { class: 'strip-item' }, h('span', { class: 'stat-label' }, 'Uptime'), h('span', { class: 'strip-value' }, fmtDuration(s.now - s.started_at))),
      h('div', { class: 'strip-item' }, h('span', { class: 'stat-label' }, 'Counting since'),
        h('span', { class: 'strip-value' }, s.reset_at ? `reset ${clock(s.reset_at)}` : `start ${clock(s.started_at)}`)));

    const recent = points.slice(-10);
    const rps = recent.length ? sum(recent.map((p) => p.requests)) / recent.length : null;
    const lastP90 = [...recent].reverse().find((p) => finite(p.p90));
    const gpu = s.telemetry.gpus;
    const util = gpu.filter((g) => finite(g.utilization_pct));
    const memKnown = gpu.filter((g) => finite(g.memory_used_bytes) && finite(g.memory_total_bytes));
    const spark = points.slice(-60);
    cards.replaceChildren(
      statCard('Requests', fmtNum(s.requests.total),
        h('span', { class: s.requests.failed ? 'err-text' : null }, `${fmtNum(s.requests.failed)} failed`)),
      statCard('Throughput', rps != null ? `${fmtNum(rps, 2)} req/s` : '—',
        `avg ${fmtNum(s.requests.per_second, 2)} req/s since ${since}`,
        sparkline(spark.map((p) => p.requests))),
      statCard('Latency p90', fmtMs(lastP90 && lastP90.p90), `${fmtMs(s.requests.latency_ms.p90)} since ${since}`,
        sparkline(spark.map((p) => p.p90))),
      statCard('In flight', fmtNum(s.load.running + s.load.queued), `running ${fmtNum(s.load.running)} · queued ${fmtNum(s.load.queued)}`),
      statCard('GPU', util.length ? fmtPctValue(sum(util.map((g) => g.utilization_pct)) / util.length) : '—',
        memKnown.length
          ? `VRAM ${fmtBytes(sum(memKnown.map((g) => g.memory_used_bytes)))} / ${fmtBytes(sum(memKnown.map((g) => g.memory_total_bytes)))}`
          : 'no readings',
        sparkline(spark.map((p) => (p.gpu_util.some(finite) ? sum(p.gpu_util) / p.gpu_util.filter(finite).length : null)))));

    const withMean = points.map((p, i) => {
      const w = points.slice(Math.max(0, i - 9), i + 1).filter((q) => finite(q.mean));
      return { ...p, mean10: w.length ? sum(w.map((q) => q.mean * q.requests)) / Math.max(1, sum(w.map((q) => q.requests))) : null };
    });
    throughput.update(points);
    latency.update(withMean);
    cacheChart.update(points);
    endpoints.replaceChildren(endpointTable(s.endpoints));
    cache.replaceChildren(h('div', null, cacheSummary(s.lanes)));
    gpus.replaceChildren(gpuTable(gpu));
    host.replaceChildren(hostKv(s.telemetry));
  }
  return { el, update };
}

function endpointTable(rows) {
  return table([
    { label: 'Endpoint', render: (e) => mono(e.endpoint) },
    { label: 'Requests', class: 'num', render: (e) => fmtNum(e.requests) },
    { label: 'Failed', class: 'num', render: (e) => h('span', { class: e.failed ? 'err-text' : null }, fmtNum(e.failed)) },
    { label: 'p50', class: 'num', render: (e) => fmtMs(e.latency_ms.p50) },
    { label: 'p90', class: 'num', render: (e) => fmtMs(e.latency_ms.p90) },
    { label: 'p99', class: 'num', render: (e) => fmtMs(e.latency_ms.p99) },
    { label: 'Mean', class: 'num', render: (e) => fmtMs(e.latency_ms.mean) },
    {
      label: 'HTTP status',
      render: (e) => (Object.keys(e.statuses).length
        ? h('span', { class: 'chips' }, Object.entries(e.statuses).map(([k, v]) => h('span', { class: `chip chip-static status-${k[0]}` }, `${k}: ${fmtNum(v)}`)))
        : h('span', { class: 'muted' }, '—')),
    },
  ], rows, { caption: 'Requests by endpoint' });
}

function cacheSummary(lanes) {
  const models = modelCaches(lanes);
  if (!models.length) return [noData('No engine lanes.')];
  return models.map(([model, c]) => {
    const title = models.length > 1 ? h('h3', { class: 'small-title' }, mono(model)) : null;
    if (!c) return [title, noData('This engine keeps no prefix cache (its architecture has no template prefixes or state cache).')];
    const lookups = c.state_hits + c.state_misses;
    const state = c.state_budget_bytes > 0
      ? [
        ['Budget', c.lanes > 1 ? `${fmtBytes(c.state_budget_bytes)} (${c.lanes} lanes, each its own cache)` : fmtBytes(c.state_budget_bytes)],
        ['Used', h('span', { class: 'progress' }, `${fmtBytes(c.state_bytes)} (${fmtPctValue((c.state_bytes / c.state_budget_bytes) * 100, 1)})`, bar(c.state_bytes / c.state_budget_bytes, { highlight: true }))],
        ['Entries', fmtNum(c.state_entries)],
        ['Hit rate', lookups ? fmtPctValue((c.state_hits / lookups) * 100, 1) : '—'],
        ['Hits / misses', `${fmtNum(c.state_hits)} / ${fmtNum(c.state_misses)}`],
        ['Inserts / evictions', `${fmtNum(c.state_inserts)} / ${fmtNum(c.state_evictions)}`],
      ]
      : [['State cache', h('span', { class: 'muted' }, 'off (state_cache_mb: 0)')]];
    return [title, kv([...state,
      ['Template prefixes', `${fmtNum(c.template_entries)} (${fmtBytes(c.template_bytes)})`],
      ['Template prefix hits', fmtNum(c.template_hits)]])];
  });
}

function gpuTable(gpus) {
  if (!gpus.length) return noData('No GPU readings yet.');
  return table([
    { label: 'Device', render: (g) => h('span', null, mono(g.label), g.name ? h('span', { class: 'muted small block' }, g.name) : null) },
    { label: 'Util', class: 'num', render: (g) => fmtPctValue(g.utilization_pct) },
    { label: 'Memory', class: 'num', render: (g) => (finite(g.memory_total_bytes) ? `${fmtBytes(g.memory_used_bytes)} / ${fmtBytes(g.memory_total_bytes)}` : '—') },
    { label: 'This server', class: 'num', render: (g) => fmtBytes(g.process_memory_bytes) },
    { label: 'Power', class: 'num', render: (g) => (finite(g.power_w) ? `${fmtNum(g.power_w, 0)} W` : '—') },
    { label: 'Temp', class: 'num', render: (g) => (finite(g.temperature_c) ? `${g.temperature_c} °C` : '—') },
  ], gpus, { caption: 'GPUs' });
}

function hostKv(t) {
  const x = t.host || {};
  return kv([
    ['CPU', x.cpu_pct != null ? `${fmtPctValue(x.cpu_pct, 1)} of ${fmtNum(x.cpus)} CPUs` : `${fmtNum(x.cpus)} CPUs`],
    ['Memory', finite(x.memory_total_bytes) ? `${fmtBytes(x.memory_used_bytes)} / ${fmtBytes(x.memory_total_bytes)}` : '—'],
    ['Load 1 / 5 / 15', x.load ? x.load.map((v) => fmtNum(v, 2)).join(' / ') : '—'],
    ['Process RSS', fmtBytes(x.process_rss_bytes)],
    ['Process CPU', x.process_cpu_pct != null ? `${fmtPctValue(x.process_cpu_pct, 1)} of one CPU` : '—'],
    ['Threads', fmtNum(x.process_threads)],
    ['Sampled', clock(t.at)],
  ]);
}

// ---------------------------------------------------------------------------------------------------------------
// Profile: engine lanes

function profile(state) {
  const lanes = h('div');
  const inflight = timeChart({
    title: 'Requests in flight',
    format: (v) => fmtNum(v, 0),
    series: [
      { label: 'running', cls: 'series-1', value: (p) => p.running },
      { label: 'queued', cls: 'series-2', value: (p) => p.queued },
    ],
  });
  const el = h('div', { class: 'server-view' },
    card('Engine lanes',
      h('p', { class: 'hint' }, 'One row per lane of every model instance (replica); the long lane only exists with long_tokens. Batch time is admission plus run_plans, including waits for the GPU gate.'),
      lanes),
    h('section', { class: 'card' }, inflight.el));
  function update() {
    lanes.replaceChildren(table([
      { label: 'Model', render: (l) => mono(l.model) },
      { label: 'Replica', class: 'num', render: (l) => String(l.replica) },
      { label: 'Lane', render: (l) => l.lane },
      { label: 'GPU', render: (l) => l.devices.join(', ') },
      { label: 'Batches', class: 'num', render: (l) => fmtNum(l.batches) },
      { label: 'Requests', class: 'num', render: (l) => fmtNum(l.requests) },
      { label: 'Avg batch', class: 'num', render: (l) => fmtNum(l.avg_batch_requests, 2) },
      { label: 'Avg batch time', class: 'num', render: (l) => fmtMs(l.avg_batch_ms) },
      { label: 'Avg queue', class: 'num', render: (l) => fmtMs(l.avg_queue_ms) },
      { label: 'Input tokens', class: 'num', render: (l) => fmtNum(l.input_tokens) },
      { label: 'Questions', class: 'num', render: (l) => fmtNum(l.questions) },
      { label: 'Waiting', class: 'num', render: (l) => fmtNum(l.waiting) },
      { label: 'Failed batches', class: 'num', render: (l) => h('span', { class: l.failed_batches ? 'err-text' : null }, fmtNum(l.failed_batches)) },
    ], state.snap.lanes, { caption: 'Engine lanes', empty: 'No engine lanes.' }));
    inflight.update(state.points);
  }
  return { el, update };
}

// ---------------------------------------------------------------------------------------------------------------
// Telemetry: GPUs and host

function telemetry(state) {
  const gpuCards = new Map();
  const gpuGrid = h('div', { class: 'health-grid' });
  const hostBox = h('div');
  const cpuChart = timeChart({ title: 'Host CPU', max: 100, format: (v) => `${fmtNum(v, 0)}%`, series: [{ label: 'CPU %', cls: 'series-1', value: (p) => p.cpu }] });
  const el = h('div', { class: 'server-view' }, gpuGrid,
    card('Host', hostBox, cpuChart.el));

  function gpuCard(index) {
    const info = h('div');
    const util = timeChart({ title: 'Utilization', max: 100, format: (v) => `${fmtNum(v, 0)}%`, series: [{ label: 'GPU %', cls: 'series-1', value: (p) => p.gpu_util[index] }] });
    const mem = timeChart({ title: 'Memory used', format: fmtBytes, series: [{ label: 'used', cls: 'series-1', value: (p) => p.gpu_memory[index] }] });
    const title = h('h2', { class: 'section-title' });
    return { el: h('section', { class: 'card' }, title, info, util.el, mem.el), title, info, util, mem };
  }

  function update() {
    const gpus = state.snap.telemetry.gpus;
    if (!gpus.length) gpuGrid.replaceChildren(card('GPUs', noData('No GPU readings yet.')));
    else if (gpuGrid.firstChild && !gpuCards.size) gpuGrid.replaceChildren();
    gpus.forEach((g, i) => {
      if (!gpuCards.has(i)) {
        const c = gpuCard(i);
        gpuCards.set(i, c);
        gpuGrid.append(c.el);
      }
      const c = gpuCards.get(i);
      c.title.replaceChildren(mono(g.label), g.name ? ` ${g.name}` : '');
      const w = (v, unit) => (finite(v) ? `${fmtNum(v, 0)} ${unit}` : '—');
      c.info.replaceChildren(h('div', null,
        g.error ? callout('warn', `No readings: ${g.error}`) : null,
        kv([
          ['Utilization', fmtPctValue(g.utilization_pct)],
          ['Memory bus', fmtPctValue(g.memory_bus_pct)],
          ['Memory', finite(g.memory_total_bytes) ? `${fmtBytes(g.memory_used_bytes)} / ${fmtBytes(g.memory_total_bytes)}${g.unified ? ' (shared with the host)' : ''}` : '—'],
          ['This server', fmtBytes(g.process_memory_bytes)],
          ['Power / limit', `${w(g.power_w, 'W')} / ${w(g.power_limit_w, 'W')}`],
          ['Temperature', finite(g.temperature_c) ? `${g.temperature_c} °C` : '—'],
          ['SM clock', w(g.sm_clock_mhz, 'MHz')],
          ['Source', g.source || '—'],
        ])));
      c.util.update(state.points);
      c.mem.update(state.points);
    });
    hostBox.replaceChildren(hostKv(state.snap.telemetry));
    cpuChart.update(state.points);
  }
  return { el, update };
}

// ---------------------------------------------------------------------------------------------------------------
// Configuration

function scalar(v) {
  return v === null || ['string', 'number', 'boolean'].includes(typeof v);
}

function show(v) {
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (v === null || v === '') return '—';
  return String(v);
}

function configuration(state) {
  const el = h('div', { class: 'server-view' });
  let rendered = null;
  function update() {
    const c = state.config;
    if (!c || rendered === c) return;
    rendered = c;
    const server = Object.entries(c).filter(([k, v]) => k !== 'models' && scalar(v));
    el.replaceChildren(
      card('Server', kv([
        ['Started', dateTime(state.snap.started_at)],
        ['Uptime', fmtDuration(state.snap.now - state.snap.started_at)],
        ['Devices', (state.snap.telemetry.gpus || []).map((g) => `${g.label}${g.name ? ` (${g.name})` : ''}`).join(', ') || '—'],
        ...server.map(([k, v]) => [k, show(v)]),
      ])),
      ...c.models.map((m) => {
        const engine = m.engine || {};
        const flat = Object.entries(engine).filter(([, v]) => scalar(v));
        const nested = Object.entries(engine).filter(([, v]) => !scalar(v));
        return card(m.name,
          kv([
            ['Instances (GPU)', m.instances.map((d) => d.join('+')).join(', ')],
            ['max_batch_tokens', fmtNum(m.max_batch_tokens)],
            ['schedule', m.schedule],
            ['long_tokens', m.long_tokens ? fmtNum(m.long_tokens) : 'off'],
            ...flat.map(([k, v]) => [k, show(v)]),
          ]),
          nested.map(([k, v]) => h('details', null, h('summary', null, k), h('pre', { class: 'code-block' }, JSON.stringify(v, null, 2)))));
      }),
      h('p', { class: 'hint' }, 'The engine description is the one written into reports (Backend::describe). GEMM tables are chosen at start and have no runtime counters.'));
  }
  return { el, update };
}

// ---------------------------------------------------------------------------------------------------------------
// Requests

function requestDrawer(summary) {
  const body = h('div', null, loading());
  const d = openDialog({ title: `Request #${summary.id} · ${summary.endpoint}`, body, wide: true });
  d.dialog.classList.add('dialog-drawer');
  api(`/ui/api/admin/requests/${summary.id}`).then((r) => {
    const metrics = () => kv([
      ['Time', dateTime(r.at)],
      ['Endpoint', mono(r.endpoint)],
      ['Model', r.model ? mono(r.model) : '—'],
      ['HTTP status', pill(r.status, r.status.startsWith('2') ? 'ok' : r.status.startsWith('4') ? 'warn' : 'err')],
      ['Latency', fmtMs(r.latency_ms)],
      ['Queue', fmtMs(r.queue_ms)],
      ['Compute (batch)', fmtMs(r.compute_ms)],
      ['Requests in its batch', fmtNum(r.batch_requests)],
      ['Lane', r.lane || '—'],
      ['Questions', fmtNum(r.questions)],
      ['Input tokens', fmtNum(r.input_tokens)],
    ]);
    const json = () => {
      if (!r.captured) {
        return callout('info', 'Request and response bodies are not kept on this server. An operator enables them with ',
          mono('ui.capture_requests: true'), ' (or ', mono('--ui-capture-requests'), '); then the last 200 requests stay in memory for administrators.');
      }
      const text = JSON.stringify({ request: r.request, response: r.response }, null, 2);
      return h('div', { class: 'code-wrap' }, h('div', { class: 'code-actions' }, copyButton(() => text, 'Copy JSON')),
        h('pre', { class: 'code-block' }, text));
    };
    body.replaceChildren(tabs([{ id: 'json', label: 'JSON', render: json }, { id: 'metrics', label: 'Metrics', render: metrics }], { label: 'Request details' }).el);
  }).catch((e) => body.replaceChildren(callout('error', errorMessage(e))));
}

function requests(state) {
  const note = h('div');
  const list = h('div', null, loading());
  const el = h('div', { class: 'server-view' }, note, h('section', { class: 'card' }, list));
  let busy = false;
  async function load() {
    if (busy) return;
    busy = true;
    try {
      const r = await api('/ui/api/admin/requests');
      note.replaceChildren(...(r.capture_requests ? [] : [callout('info', 'Only outcomes and timing are kept; request and response bodies need ', mono('ui.capture_requests'), '.')]));
      list.replaceChildren(table([
        { label: '#', class: 'num', render: (x) => button(String(x.id), () => requestDrawer(x), { kind: 'ghost', small: true, title: 'Open request details' }) },
        { label: 'Time', render: (x) => clock(x.at) },
        { label: 'Endpoint', render: (x) => mono(x.endpoint) },
        { label: 'Model', render: (x) => (x.model ? mono(x.model) : '—') },
        { label: 'Status', render: (x) => h('span', { class: `status-${x.status[0]}` }, x.status) },
        { label: 'Latency', class: 'num', render: (x) => fmtMs(x.latency_ms) },
        { label: 'Questions', class: 'num', render: (x) => fmtNum(x.questions) },
        { label: 'Input tokens', class: 'num', render: (x) => fmtNum(x.input_tokens) },
        { label: 'Batch', class: 'num', render: (x) => fmtNum(x.batch_requests) },
      ], r.data, { caption: 'Recent requests', empty: 'No requests since the start or the last reset.' }));
    } catch (e) {
      list.replaceChildren(callout('error', errorMessage(e)));
    } finally {
      busy = false;
    }
  }
  let last = 0;
  function update() {
    if (Date.now() - last >= REQUESTS_POLL_MS) {
      last = Date.now();
      load();
    }
  }
  return { el, update };
}

// ---------------------------------------------------------------------------------------------------------------
// View

export async function renderServer(root, ctx) {
  const state = { snap: null, points: [], config: null, health: null, version: ctx.config.version };
  let current = null;
  let paused = false;
  let disposed = false;
  let busy = false;
  const status = h('span');
  const updated = h('span', { class: 'muted small', 'aria-live': 'polite' });
  const content = h('div', null, loading());

  const view = (build) => () => {
    current = build(state);
    if (state.snap) current.update();
    return current.el;
  };
  const tabView = tabs([
    { id: 'overview', label: 'Overview', render: view(overview) },
    { id: 'profile', label: 'Profile', render: view(profile) },
    { id: 'telemetry', label: 'Telemetry', render: view(telemetry) },
    { id: 'configuration', label: 'Configuration', render: view(configuration) },
    { id: 'requests', label: 'Requests', render: view(requests) },
  ], { label: 'Server view' });

  async function poll() {
    if (busy || disposed || paused) return;
    busy = true;
    try {
      const prev = state.snap;
      const since = prev ? prev.seq : 0;
      const [snap, health] = await Promise.all([
        api(`/ui/api/admin/stats${since ? `?since=${since}` : ''}`),
        send('/health').then((r) => r.data).catch(() => null),
      ]);
      if (disposed) return;
      if (since && snap.reset_at !== prev.reset_at) {
        // reset elsewhere: the next poll starts over with the whole history
        state.snap = null;
        return;
      }
      if (snap.config) state.config = snap.config;
      state.points = since ? state.points.concat(snap.series).slice(-300) : snap.series;
      state.snap = snap;
      state.health = health;
      if (content.firstChild !== tabView.el) content.replaceChildren(tabView.el);
      if (current) current.update();
      status.replaceChildren(statusSummary(state));
      updated.textContent = `Updated ${new Date().toLocaleTimeString()}`;
    } catch (e) {
      updated.textContent = `Update failed: ${errorMessage(e)}`;
    } finally {
      busy = false;
    }
  }

  const pause = button('Pause', () => {
    paused = !paused;
    pause.textContent = paused ? 'Resume' : 'Pause';
    pause.setAttribute('aria-pressed', String(paused));
    if (!paused) poll();
  }, { small: true });
  pause.setAttribute('aria-pressed', 'false');
  const reset = button('Reset state', async () => {
    const ok = await confirmDialog({
      title: 'Reset statistics?',
      message: 'Counters, the 5-minute history and the recent requests of this server process start again from zero. Prometheus metrics are not affected.',
      confirmLabel: 'Reset',
      danger: true,
    });
    if (!ok) return;
    try {
      await api('/ui/api/admin/stats/reset', { method: 'POST' });
      state.snap = null;
      state.points = [];
      toast('Statistics reset.');
      poll();
    } catch (e) {
      toast(errorMessage(e), 'error');
    }
  }, { small: true, kind: 'danger-ghost' });

  root.append(pageHeader('Server', 'Live statistics of this server process: processing, engine lanes, caches, hardware and recent requests.',
    status, updated, pause, reset), content);
  await poll();
  const timer = setInterval(() => !document.hidden && poll(), POLL_MS);
  return () => {
    disposed = true;
    clearInterval(timer);
  };
}
