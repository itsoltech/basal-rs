// Admin views: users, all API keys and system health. Health is also shown without authentication.

import { api, send, enc, errorMessage, errorLines } from './api.js';
import {
  h, field, checkbox, select, button, pill, fmtTime, fmtNum, fmtBytes, mono, toast, confirmDialog, formDialog,
  table, callout, pageHeader, loading, kv,
} from './ui.js';
import { keyColumns, revokeKey, openCreateKeyDialog, MIN_PASSWORD } from './account.js';

// ---------------------------------------------------------------------------------------------------------------
// Users

/** Readable explanations of the 409 codes of user operations. */
const USER_CONFLICTS = {
  self: 'You cannot disable, delete or reset the password of your own account here.',
  user_not_disabled: 'Disable the user first and wait at least 30 seconds before deleting, so in-flight requests can finish.',
  user_has_data: 'The user still has Batch API files or batches. Delete them, or wait until retention removes them, then try again.',
};

function conflictMessage(e) {
  return (e.status === 409 && USER_CONFLICTS[e.code]) || errorMessage(e);
}

export async function renderUsers(root, ctx) {
  let includeDisabled = false;
  const body = h('div', null, loading());

  async function load() {
    try {
      const { data } = await api(`/ui/api/admin/users${includeDisabled ? '?all=true' : ''}`);
      body.replaceChildren(table(userColumns(ctx, load), data, { empty: 'No users.', caption: 'Users' }));
    } catch (e) {
      body.replaceChildren(callout('error', errorMessage(e)));
    }
  }

  root.append(
    pageHeader('Users', 'API clients and console accounts. A user ID is also the owner of its Batch API data.',
      button('Create user', () => createUser(load), { kind: 'primary' })),
    h('div', { class: 'toolbar' }, checkbox('Include disabled users', {
      onchange: (ev) => {
        includeDisabled = ev.target.checked;
        load();
      },
    })),
    body);
  await load();
}

function userColumns(ctx, reload) {
  const self = ctx.me.user.id;
  const act = async (path, okMsg, json) => {
    try {
      await api(path, { method: 'POST', json });
      toast(okMsg);
      reload();
    } catch (e) {
      toast(conflictMessage(e), 'error');
    }
  };
  return [
    { label: 'ID', render: (u) => [mono(u.id), u.id === self ? h('span', { class: 'muted small' }, ' (you)') : null] },
    { label: 'Name', render: (u) => u.name || '—' },
    { label: 'Role', render: (u) => pill(u.role) },
    { label: 'Status', render: (u) => (u.disabled_at == null ? pill('active') : h('span', null, pill('disabled'), ' ', h('span', { class: 'muted small' }, fmtTime(u.disabled_at)))) },
    { label: 'Password', render: (u) => (u.has_password ? 'Set' : h('span', { class: 'muted' }, 'None')) },
    { label: 'Created', render: (u) => fmtTime(u.created_at) },
    { label: 'Active keys', class: 'num', render: (u) => fmtNum(u.active_keys) },
    {
      label: 'Batch usage', render: (u) => (u.usage
        ? h('span', { class: 'small', title: `Reserved ${fmtBytes(u.usage.reserved_bytes)}, retained batches ${u.usage.retained_batches}, open upload sessions ${u.usage.open_sessions}` },
          `${fmtNum(u.usage.files)} files · ${fmtNum(u.usage.active_batches)} active batches · ${fmtBytes(u.usage.used_bytes)}`)
        : h('span', { class: 'muted small' }, 'No quota')),
    },
    {
      label: 'Actions', class: 'actions', render: (u) => h('div', { class: 'row-actions' },
        h('a', { class: 'btn btn-secondary btn-sm', href: `#/api-keys?${new URLSearchParams({ user: u.id })}` }, 'Keys'),
        // Your own password changes on the Account page, which asks for the current one.
        u.id === self ? null : button('Password', () => setPassword(u, reload), { small: true }),
        u.id === self ? null : u.disabled_at == null
          ? button('Disable', async () => {
            const ok = await confirmDialog({
              title: 'Disable user', confirmLabel: 'Disable', danger: true,
              message: `Disable "${u.id}"? All their API keys and console sessions stop working. Their data is kept and they can be enabled again.`,
            });
            if (ok) act(`/ui/api/admin/users/${enc(u.id)}/disable`, `User ${u.id} disabled.`);
          }, { small: true, kind: 'danger-ghost' })
          : button('Enable', () => act(`/ui/api/admin/users/${enc(u.id)}/enable`, `User ${u.id} enabled.`), { small: true }),
        u.id !== self && u.disabled_at != null ? button('Delete', () => deleteUser(u, reload), { small: true, kind: 'danger-ghost' }) : null),
    },
  ];
}

function createUser(reload) {
  const id = h('input', { type: 'text', required: true, autocomplete: 'off', spellcheck: 'false', class: 'mono' });
  const name = h('input', { type: 'text', autocomplete: 'off' });
  const role = select([['user', 'User'], ['admin', 'Administrator']], { value: 'user' });
  const password = h('input', { type: 'password', autocomplete: 'new-password' });
  formDialog({
    title: 'Create user',
    submitLabel: 'Create user',
    fields: [
      field('User ID', id, 'Permanent. It is also the Batch API owner name used for quota (batches.quota.owners).'),
      field('Name', name),
      field('Role', role, 'Administrators manage users and keys and see system health.'),
      field('Initial password (optional)', password, `Leave empty for an API-only user without console sign-in. At least ${MIN_PASSWORD} characters.`),
    ],
  }, async () => {
    if (!id.value.trim()) throw new Error('Enter a user ID.');
    if (password.value && password.value.length < MIN_PASSWORD) throw new Error(`The password must have at least ${MIN_PASSWORD} characters.`);
    await api('/ui/api/admin/users', {
      method: 'POST',
      json: { id: id.value.trim(), name: name.value.trim(), role: role.value, password: password.value || null },
    });
    toast(`User ${id.value.trim()} created.`);
    reload();
  });
}

function setPassword(u, reload) {
  const password = h('input', { type: 'password', autocomplete: 'new-password' });
  const repeat = h('input', { type: 'password', autocomplete: 'new-password' });
  formDialog({
    title: `Password for ${u.id}`,
    submitLabel: 'Save password',
    fields: [
      h('p', { class: 'hint' }, 'Saving ends all console sessions of this user. Leave both fields empty to remove console sign-in (API keys keep working).'),
      field('New password', password, `At least ${MIN_PASSWORD} characters.`),
      field('Repeat password', repeat),
    ],
  }, async () => {
    if (password.value !== repeat.value) throw new Error('The passwords do not match.');
    if (password.value && password.value.length < MIN_PASSWORD) throw new Error(`The password must have at least ${MIN_PASSWORD} characters.`);
    await api(`/ui/api/admin/users/${enc(u.id)}/password`, { method: 'POST', json: { password: password.value || null } });
    toast(password.value ? `Password of ${u.id} set.` : `Console sign-in of ${u.id} removed.`);
    reload();
  });
}

async function deleteUser(u, reload) {
  const ok = await confirmDialog({
    title: 'Delete user',
    confirmLabel: 'Delete permanently',
    danger: true,
    message: h('div', null,
      h('p', null, `Delete "${u.id}" permanently? The account, all its API keys and sessions are removed.`),
      h('p', { class: 'muted small' }, 'Users that still own Batch API files or batches cannot be deleted. The user must have been disabled for at least 30 seconds.')),
  });
  if (!ok) return;
  try {
    await api(`/ui/api/admin/users/${enc(u.id)}`, { method: 'DELETE' });
    toast(`User ${u.id} deleted.`);
    reload();
  } catch (e) {
    toast(conflictMessage(e), 'error');
  }
}

// ---------------------------------------------------------------------------------------------------------------
// All API keys

export async function renderAllKeys(root, ctx, params) {
  let user = params.get('user') || '';
  let includeRevoked = false;
  let users = [];
  const body = h('div', null, loading());
  const userSel = select([['', 'All users']], { value: user, label: 'Filter by user' });

  async function load() {
    const q = new URLSearchParams();
    if (user) q.set('user', user);
    if (includeRevoked) q.set('all', 'true');
    try {
      const { data } = await api(`/ui/api/admin/keys${q.toString() ? `?${q}` : ''}`);
      body.replaceChildren(table(keyColumns({
        withUser: true,
        onRevoke: (k) => revokeKey(k, `/ui/api/admin/keys/${enc(k.id)}/revoke`, load),
      }), data, { empty: 'No keys match.', caption: 'API keys' }));
    } catch (e) {
      body.replaceChildren(callout('error', errorMessage(e)));
    }
  }

  userSel.addEventListener('change', () => {
    user = userSel.value;
    history.replaceState(null, '', user ? `#/api-keys?${new URLSearchParams({ user })}` : '#/api-keys');
    load();
  });

  root.append(
    pageHeader('API keys', 'Keys of all users. Tokens are never shown again after creation.',
      button('Create key', () => {
        const active = users.filter((u) => u.disabled_at == null);
        if (!active.length) return toast('Create an active user first.', 'error');
        openCreateKeyDialog(ctx, { users: active, user: user || active[0].id, onCreated: load });
      }, { kind: 'primary' })),
    h('div', { class: 'toolbar' }, field('User', userSel),
      checkbox('Include revoked and expired keys', {
        onchange: (ev) => {
          includeRevoked = ev.target.checked;
          load();
        },
      })),
    body);

  try {
    users = (await api('/ui/api/admin/users?all=true')).data;
    for (const u of users) userSel.append(h('option', { value: u.id, selected: u.id === user }, `${u.id}${u.name ? ` (${u.name})` : ''}`));
  } catch (e) {
    toast(`Could not load users: ${errorMessage(e)}`, 'error');
  }
  await load();
}

// ---------------------------------------------------------------------------------------------------------------
// System health

/** Parse Prometheus text exposition into [{name, labels, value}] (comments and malformed lines skipped). */
function parseMetrics(text) {
  const out = [];
  const line = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)/;
  const label = /([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"/g;
  for (const raw of text.split('\n')) {
    if (!raw || raw.startsWith('#')) continue;
    const m = line.exec(raw);
    if (!m) continue;
    const labels = {};
    for (const l of (m[2] || '').matchAll(label)) labels[l[1]] = l[2].replace(/\\(.)/g, (_, c) => (c === 'n' ? '\n' : c));
    const value = Number(m[3]);
    if (!Number.isNaN(value)) out.push({ name: m[1], labels, value });
  }
  return out;
}

function summarizeMetrics(samples) {
  const sum = (name, filter = () => true) => samples.filter((s) => s.name === name && filter(s.labels)).reduce((a, s) => a + s.value, 0);
  const byClass = {};
  for (const s of samples.filter((x) => x.name === 'basal_http_requests_total')) {
    const st = s.labels.status || '';
    const cls = /^\d{3}$/.test(st) ? `${st[0]}xx` : st || 'other';
    byClass[cls] = (byClass[cls] || 0) + s.value;
  }
  const durSum = sum('basal_http_request_duration_seconds_sum');
  const durCount = sum('basal_http_request_duration_seconds_count');
  const replicas = {};
  for (const s of samples) {
    if (s.name !== 'basal_model_replicas' && s.name !== 'basal_model_replicas_available') continue;
    const m = (replicas[s.labels.model] ||= {});
    m[s.name === 'basal_model_replicas' ? 'total' : 'available'] = s.value;
  }
  return {
    byClass,
    inflight: sum('basal_http_inflight_requests'),
    maxInflight: samples.some((s) => s.name === 'basal_max_inflight_requests') ? sum('basal_max_inflight_requests') : null,
    inputTokens: sum('basal_input_tokens_total'),
    outputTokens: sum('basal_output_tokens_total'),
    questions: sum('basal_questions_total'),
    avgMs: durCount ? (durSum / durCount) * 1000 : null,
    replicas,
  };
}

function statusOf(r) {
  return r.data && typeof r.data.status === 'string' ? r.data.status : `HTTP ${r.status}`;
}

function serverCard(r, version) {
  if (!r.data || typeof r.data !== 'object') return callout('error', `/health: ${errorLines(r.data, r.text).join('; ') || `HTTP ${r.status}`}`);
  const d = r.data;
  const groups = [['Degraded models', d.degraded_models], ['Unavailable models', d.unavailable_models]].filter(([, v]) => Array.isArray(v) && v.length);
  return h('section', { class: 'card' },
    h('div', { class: 'section-row' }, h('h2', { class: 'section-title' }, 'Server'), pill(statusOf(r))),
    kv([['Version', version], ['Default model', d.default_model ? mono(d.default_model) : '—'],
      ['Models', (d.models || []).join(', ') || '—'], ['Running requests', fmtNum(d.running)], ['Queued requests', fmtNum(d.queued)],
      Array.isArray(d.lost_models) && d.lost_models.length ? ['Device lost', d.lost_models.join(', ')] : null]),
    groups.map(([title, list]) => h('div', { class: 'answer-section' }, h('h3', { class: 'small-title' }, title),
      list.map((m) => h('p', null, mono(m.model), ` — ${m.available} of ${m.replicas} replicas available; `,
        (m.unavailable || []).map((u) => `replica ${u.replica} (device ${u.device}): ${u.reason}`).join(', '))))));
}

function modelsCard(r) {
  if (!r.ok || !r.data || !Array.isArray(r.data.models)) return callout('error', `/v1/models: ${errorLines(r.data, r.text).join('; ') || `HTTP ${r.status}`}`);
  return h('section', { class: 'card card-wide' }, h('h2', { class: 'section-title' }, 'Models'),
    table([
      { label: 'Name', render: (m) => mono(m.name) },
      { label: 'Description', render: (m) => m.description || '—' },
      { label: 'Mode', render: (m) => m.mode || '—' },
      { label: 'Prompt format', class: 'num', render: (m) => (m.prompt_format != null ? String(m.prompt_format) : '—') },
      { label: 'Orders', render: (m) => (m.orders != null ? String(m.orders) : '—') },
      { label: 'Calibration', render: (m) => (m.calibration_warning ? h('span', { class: 'warn-text' }, '⚠ ', m.calibration_warning) : 'OK') },
    ], r.data.models, { empty: 'No models.', caption: 'Models' }));
}

function batchesCard(r) {
  const title = h('h2', { class: 'section-title' }, 'Batch API');
  if (r.status === 404) return h('section', { class: 'card' }, title, h('p', { class: 'muted' }, 'The Batch API is disabled on this server.'));
  if (!r.data || typeof r.data !== 'object') return h('section', { class: 'card' }, title, callout('error', errorLines(r.data, r.text).join('; ') || `HTTP ${r.status}`));
  const d = r.data;
  const yes = (v) => (v ? 'Yes' : 'No');
  const comp = (c, labelKey) => (c ? `${c[labelKey] || ''}${c.available ? '' : ` — unavailable${c.reason ? `: ${c.reason}` : ''}`}` : '—');
  return h('section', { class: 'card' },
    h('div', { class: 'section-row' }, title, pill(statusOf(r))),
    kv([['Profile', d.profile], ['Ready', yes(d.ready)], ['Accepting', yes(d.accepting)], ['Processing', yes(d.processing)],
      ['Storage', comp(d.storage, 'engine')], ['Artifacts', comp(d.artifacts, 'backend')],
      ['Execution profiles', Array.isArray(d.profiles) ? d.profiles.map((p) => `${p.model}${p.qualification && p.qualification.qualified ? '' : ' (not qualified)'}`).join(', ') || 'None' : '—'],
      d.backlog_error ? ['Backlog', h('span', { class: 'warn-text' }, d.backlog_error)] : null,
      Array.isArray(d.backlog) && !d.backlog.length ? ['Backlog', 'Empty'] : null]));
}

/** The Batch backlog in its own full-width card (a table is too wide for a grid column), or null when empty. */
function backlogCard(r) {
  const d = r.data && typeof r.data === 'object' ? r.data : {};
  if (!Array.isArray(d.backlog) || !d.backlog.length) return null;
  return h('section', { class: 'card card-wide' }, h('h2', { class: 'section-title' }, 'Batch backlog'),
    table([
      { label: 'Status', render: (b) => pill(b.status) },
      { label: 'Batches', class: 'num', render: (b) => fmtNum(b.batches) },
      { label: 'Queued records', class: 'num', render: (b) => fmtNum(b.queued) },
      { label: 'Model fingerprint', render: (b) => (b.fingerprint ? mono(String(b.fingerprint)) : '—') },
    ], d.backlog, { caption: 'Batch backlog' }));
}

function metricsCard(r) {
  const title = h('h2', { class: 'section-title' }, 'Metrics');
  if (!r.ok) return h('section', { class: 'card' }, title, callout('error', `/metrics: HTTP ${r.status}`));
  const m = summarizeMetrics(parseMetrics(r.text));
  const classes = Object.entries(m.byClass).sort();
  const total = classes.reduce((a, [, v]) => a + v, 0);
  return h('section', { class: 'card' }, title,
    h('div', { class: 'stats' },
      [['Requests', fmtNum(total)], ['In flight', `${fmtNum(m.inflight)}${m.maxInflight != null ? ` / ${fmtNum(m.maxInflight)}` : ''}`],
        ['Avg duration', m.avgMs != null ? `${fmtNum(m.avgMs, 1)} ms` : '—'], ['Questions', fmtNum(m.questions)],
        ['Input tokens', fmtNum(m.inputTokens)], ['Output tokens', fmtNum(m.outputTokens)]]
        .map(([k, v]) => h('div', { class: 'stat' }, h('span', { class: 'stat-label' }, k), h('span', { class: 'stat-value' }, v)))),
    h('h3', { class: 'small-title' }, 'Requests by status'),
    classes.length ? h('div', { class: 'chips' }, classes.map(([k, v]) => h('span', { class: `chip chip-static status-${k[0]}` }, `${k}: ${fmtNum(v)}`)))
      : h('p', { class: 'muted small' }, 'No requests yet.'),
    Object.keys(m.replicas).length ? [h('h3', { class: 'small-title' }, 'Replicas available'),
      kv(Object.entries(m.replicas).map(([model, x]) => [model, `${fmtNum(x.available)} / ${fmtNum(x.total)}`]))] : null,
    h('p', { class: 'hint' }, 'Totals since the server started. Averages cover all inference endpoints.'));
}

export async function renderHealth(root, ctx) {
  const { config } = ctx;
  const grid = h('div', { class: 'health-grid' }, loading());
  const updated = h('span', { class: 'muted small', 'aria-live': 'polite' });
  let timer = null;
  let busy = false;
  let disposed = false;

  async function refresh() {
    if (busy || disposed) return;
    busy = true;
    try {
      const safe = (p) => p.catch((e) => ({ ok: false, status: 0, data: null, text: errorMessage(e) }));
      const [health, models, batches, metrics] = await Promise.all([
        safe(send('/health')),
        safe(send('/v1/models')),
        config.batches ? safe(send('/health/batches')) : null,
        config.metrics ? safe(send('/metrics')) : null,
      ]);
      if (disposed) return;
      grid.replaceChildren(...[serverCard(health, config.version), batches && batchesCard(batches),
        metrics && metricsCard(metrics), batches && backlogCard(batches), modelsCard(models)].filter(Boolean));
      updated.textContent = `Updated ${new Date().toLocaleTimeString()}`;
    } finally {
      busy = false;
    }
  }

  const auto = checkbox('Auto-refresh every 5 s', {
    checked: true,
    onchange: (ev) => (ev.target.checked ? start() : stop()),
  });
  const start = () => {
    stop();
    timer = setInterval(() => !document.hidden && refresh(), 5000);
  };
  const stop = () => clearInterval(timer);

  root.append(pageHeader('System health', 'Readiness, models, Batch API and metrics of this server.',
    updated, auto, button('Refresh', refresh, { small: true })), grid);
  await refresh();
  start();
  return () => {
    disposed = true;
    stop();
  };
}
