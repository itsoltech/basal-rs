// DOM helpers shared by all views. Server data is only ever inserted as text nodes or attribute values;
// nothing here assigns innerHTML.

/**
 * Create an element. `attrs`: `class`, `dataset`, `on<event>` handlers, DOM properties (`value`, `checked`,
 * `disabled`, `selected`), booleans as presence attributes; null/false values are skipped.
 * Children may be nodes, strings, numbers, arrays, null or false.
 */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'value' || k === 'checked' || k === 'disabled' || k === 'selected') el[k] = v;
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : String(c));
  }
  return el;
}

let idCounter = 0;
export const uid = (prefix = 'id') => `${prefix}-${++idCounter}`;

/** Label + control (+ optional hint) wired with for/id and aria-describedby. */
export function field(label, control, hint) {
  if (!control.id) control.id = uid('f');
  const wrap = h('div', { class: 'field' }, h('label', { for: control.id }, label), control);
  if (hint) {
    const hintEl = h('p', { class: 'hint', id: uid('hint') }, hint);
    control.setAttribute('aria-describedby', hintEl.id);
    wrap.append(hintEl);
  }
  return wrap;
}

export function checkbox(label, { checked = false, onchange, name, value } = {}) {
  const input = h('input', { type: 'checkbox', checked, onchange, name, value });
  return h('label', { class: 'check' }, input, h('span', null, label));
}

export function select(options, { value, onchange, id, label } = {}) {
  const el = h('select', { id, onchange, 'aria-label': label });
  for (const o of options) {
    const [v, text] = Array.isArray(o) ? o : [o, o];
    el.append(h('option', { value: v, selected: v === value }, text));
  }
  return el;
}

export function button(label, onclick, { kind = 'secondary', type = 'button', disabled, title, small } = {}) {
  const cls = `btn btn-${kind}${small ? ' btn-sm' : ''}`;
  return h('button', { type, class: cls, onclick, disabled, title }, label);
}

const PILL_KINDS = {
  ok: ['ok', 'active', 'completed', 'validated', 'processed', 'admin'],
  warn: ['validating', 'in_progress', 'finalizing', 'cancelling', 'degraded', 'queued', 'unvalidated', 'expired_soon'],
  err: ['failed', 'expired', 'revoked', 'device_lost', 'engine_stopped', 'unavailable', 'disabled', 'error'],
  muted: ['cancelled', 'user', 'text', 'none'],
};

/** Status pill; the colour comes from a known status vocabulary unless `kind` is given. */
export function pill(text, kind) {
  const k = kind || Object.keys(PILL_KINDS).find((key) => PILL_KINDS[key].includes(String(text))) || 'info';
  return h('span', { class: `pill pill-${k}` }, String(text).replaceAll('_', ' '));
}

/** Horizontal bar for a value in [0, 1]; decorative (the number is rendered as text next to it). */
export function bar(fraction, { highlight = false } = {}) {
  const fill = h('span', { class: 'bar-fill' });
  fill.style.width = `${Math.max(0, Math.min(1, Number(fraction) || 0)) * 100}%`;
  return h('span', { class: `bar${highlight ? ' bar-hl' : ''}`, 'aria-hidden': 'true' }, fill);
}

export function fmtPct(p, digits = 1) {
  return typeof p === 'number' && Number.isFinite(p) ? `${(p * 100).toFixed(digits)}%` : '—';
}

export function fmtNum(n, digits = 0) {
  if (typeof n !== 'number' || !Number.isFinite(n)) return '—';
  return n.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

export function fmtBytes(n) {
  if (typeof n !== 'number' || !Number.isFinite(n)) return '—';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${i ? n.toFixed(1) : n} ${units[i]}`;
}

/** Unix seconds → local date-time; null → em dash. */
export function fmtTime(secs) {
  if (secs == null) return '—';
  const d = new Date(secs * 1000);
  return h('time', { datetime: d.toISOString(), title: d.toISOString() }, d.toLocaleString());
}

export function mono(text) {
  return h('code', { class: 'mono' }, text);
}

// ---------------------------------------------------------------------------------------------------------------
// Toasts and dialogs

/** Transient message in the aria-live region. Errors stay longer. */
export function toast(message, kind = 'info') {
  const region = document.getElementById('toasts');
  const item = h('div', { class: `toast toast-${kind}`, role: kind === 'error' ? 'alert' : null }, message);
  const close = () => item.remove();
  item.append(h('button', { type: 'button', class: 'toast-close', 'aria-label': 'Dismiss', onclick: close }, '×'));
  region.append(item);
  setTimeout(close, kind === 'error' ? 10000 : 4000);
}

/**
 * Modal dialog built on <dialog> (focus trap and Esc handled by the browser). Returns `close()`.
 * The element is removed from the DOM when closed.
 */
export function openDialog({ title, body, actions = [], wide = false, onClose }) {
  const titleId = uid('dlg');
  const dlg = h(
    'dialog',
    { class: `dialog${wide ? ' dialog-wide' : ''}`, 'aria-labelledby': titleId },
    h('div', { class: 'dialog-head' }, h('h2', { id: titleId }, title),
      h('button', { type: 'button', class: 'icon-btn', 'aria-label': 'Close', onclick: () => dlg.close() }, '×')),
    h('div', { class: 'dialog-body' }, body),
    actions.length ? h('div', { class: 'dialog-actions' }, actions) : null,
  );
  dlg.addEventListener('close', () => {
    dlg.remove();
    if (onClose) onClose(dlg.returnValue);
  });
  document.body.append(dlg);
  dlg.showModal();
  return { dialog: dlg, close: (value) => dlg.close(value) };
}

/** Resolves true when confirmed. */
export function confirmDialog({ title, message, confirmLabel = 'Confirm', cancelLabel = 'Cancel', danger = false }) {
  return new Promise((resolve) => {
    let ok = false;
    const confirmBtn = button(confirmLabel, () => {
      ok = true;
      d.close();
    }, { kind: danger ? 'danger' : 'primary' });
    const d = openDialog({
      title,
      body: typeof message === 'string' ? h('p', null, message) : message,
      actions: [button(cancelLabel, () => d.close()), confirmBtn],
      onClose: () => resolve(ok),
    });
    confirmBtn.focus();
  });
}

/**
 * Dialog with a form. `onSubmit(form)` may throw; its message is shown inside the dialog and the dialog stays
 * open. Resolving closes the dialog.
 */
export function formDialog({ title, fields, submitLabel = 'Save', danger = false, wide = false }, onSubmit) {
  const error = h('p', { class: 'form-error', role: 'alert', hidden: true });
  const submit = h('button', { type: 'submit', class: `btn btn-${danger ? 'danger' : 'primary'}` }, submitLabel);
  const form = h('form', { class: 'form', novalidate: true }, fields, error,
    h('div', { class: 'dialog-actions' }, button('Cancel', () => d.close()), submit));
  const d = openDialog({ title, body: form, wide });
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    error.hidden = true;
    submit.disabled = true;
    try {
      await onSubmit(form);
      d.close();
    } catch (e) {
      error.textContent = e && e.message ? e.message : String(e);
      error.hidden = false;
    } finally {
      submit.disabled = false;
    }
  });
  const first = form.querySelector('input:not([type=hidden]):not([readonly]), select, textarea');
  if (first) first.focus();
  return d;
}

// ---------------------------------------------------------------------------------------------------------------
// Clipboard and downloads

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Clipboard API needs a secure context; fall back to a selection copy on plain-HTTP deployments.
    const ta = h('textarea', { class: 'offscreen', readonly: true, value: text });
    document.body.append(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    if (!ok) throw new Error('Copy failed; select the text and copy it manually.');
  }
}

export function copyButton(getText, label = 'Copy') {
  const btn = button(label, async () => {
    try {
      await copyText(typeof getText === 'function' ? getText() : getText);
      btn.textContent = 'Copied';
      setTimeout(() => (btn.textContent = label), 1500);
    } catch (e) {
      toast(e.message, 'error');
    }
  }, { small: true });
  return btn;
}

export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = h('a', { href: url, download: filename || 'download' });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------------------------------------------------------
// Code editors

/**
 * Make a textarea code-friendly: Tab inserts two spaces, Shift+Tab is left to the browser,
 * and Esc followed by Tab moves focus out (so keyboard users are never trapped).
 */
export function codeTextarea(ta) {
  let escaped = false;
  ta.classList.add('code');
  ta.spellcheck = false;
  ta.setAttribute('autocapitalize', 'off');
  ta.setAttribute('autocomplete', 'off');
  ta.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') {
      escaped = true;
      return;
    }
    if (ev.key === 'Tab' && !ev.shiftKey && !escaped && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
      ev.preventDefault();
      ta.setRangeText('  ', ta.selectionStart, ta.selectionEnd, 'end');
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    }
    escaped = false;
  });
  return ta;
}

/** Parse JSON and turn the engine's error into "message (line L, column C)". */
export function parseJson(text) {
  try {
    return { ok: true, value: JSON.parse(text) };
  } catch (e) {
    let msg = e.message.replace(/^JSON\.parse: /, '');
    const pos = /at position (\d+)/.exec(msg);
    if (pos && !/line \d+/.test(msg)) {
      const before = text.slice(0, Number(pos[1]));
      const line = before.split('\n').length;
      const col = Number(pos[1]) - before.lastIndexOf('\n');
      msg = `${msg.replace(/\s*\(?at position \d+.*$/, '')} (line ${line}, column ${col})`;
    }
    return { ok: false, error: msg };
  }
}

/**
 * JSON editor: monospaced textarea with a live validity indicator.
 * `accept(value)` may return an error string to reject well-formed JSON of the wrong shape.
 */
export function jsonEditor({ value = '', rows = 8, label, onInput, accept, emptyOk = false }) {
  const ta = codeTextarea(h('textarea', { rows, value, 'aria-label': label }));
  const status = h('p', { class: 'json-status', 'aria-live': 'polite', id: uid('js') });
  ta.setAttribute('aria-describedby', status.id);
  const check = () => {
    const text = ta.value;
    let result;
    if (!text.trim()) {
      result = emptyOk ? { ok: true, value: undefined } : { ok: false, error: 'Empty' };
    } else {
      result = parseJson(text);
      if (result.ok && accept) {
        const problem = accept(result.value);
        if (problem) result = { ok: false, error: problem };
      }
    }
    status.className = `json-status ${result.ok ? 'is-ok' : 'is-err'}`;
    status.textContent = result.ok ? (text.trim() ? '✓ Valid JSON' : '') : `✗ ${result.error}`;
    ta.setAttribute('aria-invalid', result.ok ? 'false' : 'true');
    return result;
  };
  ta.addEventListener('input', () => {
    const r = check();
    if (onInput) onInput(ta.value, r);
  });
  check();
  return {
    el: h('div', { class: 'json-editor' }, ta, status),
    textarea: ta,
    check,
    set(text) {
      ta.value = text;
      check();
    },
  };
}

/** Accessible tabs (roving tabindex, arrow keys). `tabs`: [{id, label, render: () => Node}]. */
export function tabs(items, { label = 'Tabs', initial } = {}) {
  const list = h('div', { class: 'tablist', role: 'tablist', 'aria-label': label });
  const panel = h('div', { class: 'tabpanel', role: 'tabpanel', tabindex: '0' });
  const buttons = new Map();
  let current = null;
  const show = (id) => {
    current = id;
    for (const [bid, btn] of buttons) {
      const on = bid === id;
      btn.setAttribute('aria-selected', String(on));
      btn.tabIndex = on ? 0 : -1;
      if (on) panel.setAttribute('aria-labelledby', btn.id);
    }
    panel.replaceChildren(items.find((t) => t.id === id).render());
  };
  items.forEach((t, i) => {
    const btn = h('button', { type: 'button', role: 'tab', class: 'tab', id: uid('tab'), onclick: () => show(t.id) }, t.label);
    btn.addEventListener('keydown', (ev) => {
      const delta = ev.key === 'ArrowRight' ? 1 : ev.key === 'ArrowLeft' ? -1 : 0;
      if (!delta) return;
      const next = items[(i + delta + items.length) % items.length];
      show(next.id);
      buttons.get(next.id).focus();
    });
    buttons.set(t.id, btn);
    list.append(btn);
  });
  show(initial || items[0].id);
  return { el: h('div', { class: 'tabs' }, list, panel), show, refresh: () => show(current) };
}

// ---------------------------------------------------------------------------------------------------------------
// Tables and states

/** `columns`: [{label, render(row) → Node|string, class}]. */
export function table(columns, rows, { empty = 'Nothing here yet.', caption } = {}) {
  const head = h('tr', null, columns.map((c) => h('th', { scope: 'col', class: c.class }, c.label)));
  const body = h('tbody');
  if (!rows.length) {
    body.append(h('tr', null, h('td', { colspan: columns.length, class: 'empty' }, empty)));
  }
  for (const row of rows) {
    body.append(h('tr', null, columns.map((c) => h('td', { class: c.class }, c.render(row)))));
  }
  return h('div', { class: 'table-wrap' },
    h('table', { class: 'table' }, caption ? h('caption', { class: 'sr-only' }, caption) : null, h('thead', null, head), body));
}

export function callout(kind, ...children) {
  return h('div', { class: `callout callout-${kind}`, role: kind === 'error' ? 'alert' : null }, children);
}

export function pageHeader(title, subtitle, ...actions) {
  return h('header', { class: 'page-head' },
    h('div', null, h('h1', null, title), subtitle ? h('p', { class: 'muted' }, subtitle) : null),
    actions.length ? h('div', { class: 'page-actions' }, actions) : null);
}

export function loading(text = 'Loading…') {
  return h('p', { class: 'loading', role: 'status' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), text);
}

/** Small key/value list. `pairs`: [[label, value]] (values: nodes or strings; null pairs skipped). */
export function kv(pairs) {
  return h('dl', { class: 'kv' }, pairs.filter(Boolean).map(([k, v]) => [h('dt', null, k), h('dd', null, v ?? '—')]));
}
