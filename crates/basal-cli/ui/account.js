// Account views: the signed-in user's API keys and password. The key dialogs are shared with the admin views.

import { api, enc, errorMessage } from './api.js';
import {
  h, field, checkbox, select, button, pill, fmtTime, mono, toast, openDialog, confirmDialog, formDialog, copyButton,
  table, callout, pageHeader, loading, kv,
} from './ui.js';

export const MIN_PASSWORD = 12;
const EXPIRY = [['never', 'Never'], ['30d', '30 days'], ['90d', '90 days'], ['365d', '1 year'], ['custom', 'Custom…']];

/** Columns of a key table; `withUser` adds the owner column (admin view). */
export function keyColumns({ withUser = false, onRevoke }) {
  return [
    withUser ? { label: 'User', render: (k) => mono(k.user) } : null,
    { label: 'Name', render: (k) => k.name || '—' },
    { label: 'Key ID', render: (k) => mono(k.id) },
    { label: 'Scopes', render: (k) => h('span', { class: 'chips' }, (k.scopes || []).map((s) => h('span', { class: 'chip chip-static' }, s))) },
    { label: 'Status', render: (k) => pill(k.status) },
    { label: 'Created', render: (k) => fmtTime(k.created_at) },
    { label: 'Expires', render: (k) => (k.expires_at == null ? 'Never' : fmtTime(k.expires_at)) },
    { label: 'Last used', render: (k) => (k.last_used_at == null ? 'Never' : fmtTime(k.last_used_at)) },
    {
      label: 'Actions', class: 'actions', render: (k) => (k.status === 'active'
        ? button('Revoke', () => onRevoke(k), { small: true, kind: 'danger-ghost' })
        : h('span', { class: 'muted small' }, k.revoked_at ? 'Revoked' : '')),
    },
  ].filter(Boolean);
}

/** Confirm and revoke a key through `path` (own or admin route). */
export async function revokeKey(key, path, done) {
  const ok = await confirmDialog({
    title: 'Revoke API key',
    message: `Revoke "${key.name || key.id}"? Applications using it stop working immediately. This cannot be undone.`,
    confirmLabel: 'Revoke key',
    danger: true,
  });
  if (!ok) return;
  try {
    await api(path, { method: 'POST' });
    toast('Key revoked.');
    done();
  } catch (e) {
    toast(errorMessage(e), 'error');
  }
}

/**
 * Create-key dialog. For admins creating a key for someone, pass `users` (list for the user select) and the
 * admin endpoint is used; otherwise the key belongs to the signed-in user.
 */
export function openCreateKeyDialog(ctx, { users, user, onCreated }) {
  const forUser = Array.isArray(users);
  const name = h('input', { type: 'text', required: true, maxlength: 200, placeholder: 'e.g. CRM integration' });
  const userSel = forUser ? select(users.map((u) => [u.id, `${u.id}${u.name ? ` (${u.name})` : ''}`]), { value: user }) : null;
  const scopes = [
    ['inference', 'Inference: /v1/systemone, /v1/basal, /v1/decisions, /v1/models', true],
    ['batches', 'Batches: /v1/files and /v1/batches', true],
    ['mcp', `MCP: /mcp${ctx.config.mcp ? '' : ' (MCP is currently disabled on this server)'}`, false],
  ];
  if (forUser || ctx.isAdmin) scopes.push(['metrics', 'Metrics: /metrics (only when the server protects metrics)', false]);
  const scopeBoxes = scopes.map(([value, label, checked]) => checkbox(label, { checked, name: 'scope', value }));
  const expiry = select(EXPIRY, { value: '90d' });
  const custom = h('input', { type: 'text', placeholder: 'e.g. 14d, 12h', pattern: '^[0-9]+[dhms]$', hidden: true, 'aria-label': 'Custom expiry' });
  expiry.addEventListener('change', () => {
    custom.hidden = expiry.value !== 'custom';
    if (!custom.hidden) custom.focus();
  });

  formDialog({
    title: forUser ? 'Create API key for a user' : 'Create API key',
    submitLabel: 'Create key',
    fields: [
      userSel ? field('User', userSel) : null,
      field('Name', name, 'Helps you recognise the key later, e.g. the application or host using it.'),
      h('fieldset', { class: 'options' }, h('legend', null, 'Scopes'), scopeBoxes),
      field('Expires', expiry), custom,
    ],
  }, async (form) => {
    const chosen = [...form.querySelectorAll('input[name=scope]:checked')].map((i) => i.value);
    if (!name.value.trim()) throw new Error('Enter a name for the key.');
    if (!chosen.length) throw new Error('Select at least one scope.');
    let expiresIn = expiry.value === 'never' ? null : expiry.value;
    if (expiry.value === 'custom') {
      expiresIn = custom.value.trim();
      if (!/^[1-9][0-9]*[dhms]$/.test(expiresIn)) throw new Error('Custom expiry: a whole number followed by d, h, m or s (e.g. 14d).');
    }
    const body = { name: name.value.trim(), scopes: chosen, expires_in: expiresIn };
    const res = forUser
      ? await api('/ui/api/admin/keys', { method: 'POST', json: { user: userSel.value, ...body } })
      : await api('/ui/api/me/keys', { method: 'POST', json: body });
    showToken(res);
    if (onCreated) onCreated(res.key);
  });
}

/** Show a new token once. It lives only in this dialog's DOM and is dropped when the dialog closes. */
function showToken({ key, token }) {
  const out = h('input', { type: 'text', class: 'mono token', readonly: true, value: token, 'aria-label': 'New API key token' });
  const d = openDialog({
    title: 'API key created',
    body: [
      callout('warn', h('strong', null, 'Copy this token now. '), 'It is shown only once and cannot be recovered; the server stores only its hash.'),
      kv([['Key ID', mono(key.id)], ['User', mono(key.user)], ['Scopes', (key.scopes || []).join(', ')], ['Expires', key.expires_at == null ? 'Never' : fmtTime(key.expires_at)]]),
      h('div', { class: 'token-row' }, out, copyButton(() => out.value)),
      h('p', { class: 'hint' }, 'Send it as "Authorization: Bearer <token>".'),
    ],
    actions: [button('I have copied the token', () => d.close(), { kind: 'primary' })],
    onClose: () => {
      out.value = '';
    },
  });
  out.select();
}

// ---------------------------------------------------------------------------------------------------------------
// My keys

export async function renderMyKeys(root, ctx) {
  let showAll = false;
  const body = h('div', null, loading());

  async function load() {
    try {
      const { data } = await api(`/ui/api/me/keys${showAll ? '?all=true' : ''}`);
      body.replaceChildren(table(keyColumns({
        onRevoke: (k) => revokeKey(k, `/ui/api/me/keys/${enc(k.id)}/revoke`, load),
      }), data, { empty: showAll ? 'You have no API keys.' : 'You have no active API keys.', caption: 'My API keys' }));
    } catch (e) {
      body.replaceChildren(callout('error', errorMessage(e)));
    }
  }

  root.append(
    pageHeader('My API keys', 'Keys let applications call this server. All your keys share your batches and files.',
      button('Create key', () => openCreateKeyDialog(ctx, { onCreated: load }), { kind: 'primary' })),
    h('div', { class: 'toolbar' }, checkbox('Show revoked and expired keys', {
      onchange: (ev) => {
        showAll = ev.target.checked;
        load();
      },
    })),
    body);
  await load();
}

// ---------------------------------------------------------------------------------------------------------------
// Account

export function renderAccount(root, ctx) {
  const u = ctx.me.user;
  const current = h('input', { type: 'password', autocomplete: 'current-password', required: true });
  const next = h('input', { type: 'password', autocomplete: 'new-password', required: true, minlength: MIN_PASSWORD });
  const repeat = h('input', { type: 'password', autocomplete: 'new-password', required: true });
  const error = h('p', { class: 'form-error', role: 'alert', hidden: true });
  const submit = h('button', { type: 'submit', class: 'btn btn-primary' }, 'Change password');
  const form = h('form', { class: 'form narrow', novalidate: true },
    field('Current password', current),
    field('New password', next, `At least ${MIN_PASSWORD} characters. A long passphrase is best.`),
    field('Repeat new password', repeat),
    error, h('div', null, submit));

  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const fail = (msg) => {
      error.textContent = msg;
      error.hidden = false;
    };
    error.hidden = true;
    if (!current.value) return fail('Enter your current password.');
    if (next.value.length < MIN_PASSWORD) return fail(`The new password must have at least ${MIN_PASSWORD} characters.`);
    if (next.value !== repeat.value) return fail('The new passwords do not match.');
    submit.disabled = true;
    try {
      // A wrong current password is a 400 `wrong_password`; a 401 means the session ended.
      await api('/ui/api/me/password', {
        method: 'POST', json: { current_password: current.value, new_password: next.value },
      });
      form.reset();
      toast('Password changed. Your other sessions were signed out.');
    } catch (e) {
      fail(e.code === 'wrong_password' ? 'The current password is incorrect.' : errorMessage(e));
    } finally {
      submit.disabled = false;
    }
  });

  root.append(
    pageHeader('Account', null),
    h('section', { class: 'card' }, h('h2', { class: 'section-title' }, 'Profile'),
      kv([['User ID', mono(u.id)], ['Name', u.name || '—'], ['Role', pill(u.role)],
        ['Session scopes', (ctx.me.scopes || []).join(', ') || '—'], ['Created', fmtTime(u.created_at)]])),
    h('section', { class: 'card' }, h('h2', { class: 'section-title' }, 'Change password'), form));
}
