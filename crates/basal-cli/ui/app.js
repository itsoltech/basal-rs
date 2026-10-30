// Entry point: loads the server configuration, handles the session (login/logout), and routes between views
// with the URL hash (`#/view?param=value`).

import { api, send, onUnauthorized, errorLines, errorMessage } from './api.js';
import { h, field, toast, button, callout } from './ui.js';
import { renderPlayground } from './playground.js';
import { renderBatches } from './batches.js';
import { renderMyKeys, renderAccount } from './account.js';
import { renderUsers, renderAllKeys, renderHealth } from './admin.js';

const root = document.getElementById('root');

/** Shared state handed to every view. */
const ctx = {
  config: null,
  me: null,
  get auth() {
    return Boolean(this.config && this.config.auth);
  },
  get isAdmin() {
    return Boolean(this.me && this.me.user.role === 'admin');
  },
  navigate(view, params) {
    const query = params ? `?${new URLSearchParams(params)}` : '';
    location.hash = `#/${view}${query}`;
  },
};

const VIEWS = [
  { id: 'playground', label: 'Playground', group: 'Workspace', render: renderPlayground, visible: () => true },
  { id: 'batches', label: 'Batches & files', group: 'Workspace', render: renderBatches, visible: () => ctx.config.batches },
  { id: 'keys', label: 'My keys', group: 'Account', render: renderMyKeys, visible: () => ctx.auth },
  { id: 'account', label: 'Account', group: 'Account', render: renderAccount, visible: () => ctx.auth },
  { id: 'users', label: 'Users', group: 'Admin', render: renderUsers, visible: () => ctx.isAdmin },
  { id: 'api-keys', label: 'API keys', group: 'Admin', render: renderAllKeys, visible: () => ctx.isAdmin },
  { id: 'health', label: 'System health', group: 'Admin', render: renderHealth, visible: () => !ctx.auth || ctx.isAdmin },
];

let main = null;
let nav = null;
let cleanup = null;
let routeToken = 0;

function setRoot(...children) {
  root.className = '';
  root.removeAttribute('aria-busy');
  root.replaceChildren(...children);
}

// ---------------------------------------------------------------------------------------------------------------
// Start-up and session

async function boot() {
  try {
    ctx.config = await api('/ui/api/config');
  } catch (e) {
    return fatal(`Could not load the server configuration. ${errorMessage(e)}`);
  }
  if (!ctx.auth) return startShell();
  const r = await send('/ui/api/me', { quiet401: true });
  if (r.ok) {
    ctx.me = r.data;
    startShell();
  } else if (r.status === 401) {
    showLogin();
  } else {
    fatal(`Could not load the session. HTTP ${r.status}: ${errorLines(r.data, r.text).join('; ')}`);
  }
}

function fatal(message) {
  setRoot(h('div', { class: 'center-screen' },
    h('div', { class: 'card login-card' }, h('h1', null, 'basal'), callout('error', message),
      button('Retry', () => location.reload(), { kind: 'primary' }))));
}

onUnauthorized(() => {
  // A 401 on any call means the session ended (expiry, logout elsewhere, password change).
  if (!ctx.auth || !ctx.me) return;
  ctx.me = null;
  stopView();
  showLogin('Your session has ended. Sign in again.');
});

function showLogin(notice) {
  window.removeEventListener('hashchange', route);
  const userInput = h('input', { type: 'text', name: 'username', autocomplete: 'username', required: true, autocapitalize: 'off', spellcheck: 'false' });
  const passInput = h('input', { type: 'password', name: 'password', autocomplete: 'current-password', required: true });
  const error = h('p', { class: 'form-error', role: 'alert', hidden: !notice }, notice || '');
  const submit = h('button', { type: 'submit', class: 'btn btn-primary btn-block' }, 'Sign in');
  const form = h('form', { class: 'form', novalidate: true },
    field('User ID', userInput), field('Password', passInput), error, submit);
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if (!userInput.value.trim() || !passInput.value) {
      error.textContent = 'Enter your user ID and password.';
      error.hidden = false;
      return;
    }
    submit.disabled = true;
    submit.textContent = 'Signing in…';
    try {
      const r = await send('/ui/api/login', {
        method: 'POST', json: { user: userInput.value.trim(), password: passInput.value }, quiet401: true,
      });
      if (r.ok) {
        passInput.value = '';
        ctx.me = r.data;
        startShell();
        return;
      }
      error.textContent = r.status === 401
        ? 'Invalid user ID or password.'
        : r.status === 429
          ? `Too many failed attempts. Try again${r.headers.get('Retry-After') ? ` in ${r.headers.get('Retry-After')} s` : ' later'}.`
          : `Sign-in failed (HTTP ${r.status}). ${errorLines(r.data, r.text).join('; ')}`;
    } catch (e) {
      error.textContent = errorMessage(e);
    }
    error.hidden = false;
    passInput.value = '';
    passInput.focus();
    submit.disabled = false;
    submit.textContent = 'Sign in';
  });
  setRoot(h('main', { id: 'main', class: 'center-screen' },
    h('div', { class: 'card login-card' },
      h('div', { class: 'brand brand-lg' }, h('img', { src: './favicon.svg', alt: '', width: 32, height: 32 }), h('span', null, 'basal')),
      h('h1', { class: 'login-title' }, 'Sign in to the console'),
      form,
      h('p', { class: 'muted small' }, `Server ${ctx.config.version}`))));
  (userInput.value ? passInput : userInput).focus();
}

async function logout() {
  try {
    await send('/ui/api/logout', { method: 'POST', quiet401: true });
  } catch {
    /* the session is dropped locally either way */
  }
  ctx.me = null;
  stopView();
  showLogin();
}

// ---------------------------------------------------------------------------------------------------------------
// Shell and routing

function startShell() {
  const groups = new Map();
  for (const v of VIEWS.filter((v) => v.visible())) {
    const group = !ctx.auth && v.id === 'health' ? 'Server' : v.group;
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(v);
  }
  nav = h('nav', { class: 'nav', 'aria-label': 'Main' },
    [...groups].map(([group, views]) => h('div', { class: 'nav-group' },
      h('p', { class: 'nav-label' }, group),
      h('ul', null, views.map((v) => h('li', null, h('a', { href: `#/${v.id}`, dataset: { view: v.id } }, v.label)))))));

  const user = ctx.me
    ? h('div', { class: 'whoami' },
      h('span', { class: 'whoami-name' }, ctx.me.user.name || ctx.me.user.id),
      h('span', { class: 'muted small' }, `${ctx.me.user.id} · ${ctx.me.user.role}`),
      button('Sign out', logout, { small: true }))
    : h('div', { class: 'whoami' }, h('span', { class: 'muted small' }, 'Authentication is disabled on this server.'));

  main = h('main', { id: 'main', class: 'main', tabindex: '-1' });
  setRoot(h('div', { class: 'app' },
    h('aside', { class: 'sidebar' },
      h('a', { class: 'brand', href: '#/playground' }, h('img', { src: './favicon.svg', alt: '', width: 24, height: 24 }),
        h('span', null, 'basal'), h('span', { class: 'version' }, ctx.config.version)),
      nav, user),
    main));
  window.addEventListener('hashchange', route);
  route();
}

function stopView() {
  routeToken++;
  if (cleanup) cleanup();
  cleanup = null;
}

async function route(ev) {
  const [path, query] = location.hash.replace(/^#\/?/, '').split('?');
  const view = VIEWS.find((v) => v.id === path && v.visible()) || VIEWS[0];
  if (view.id !== path) {
    history.replaceState(null, '', `#/${view.id}`);
  }
  stopView();
  const token = routeToken;
  for (const a of nav.querySelectorAll('a')) {
    if (a.dataset.view === view.id) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  }
  document.title = `${view.label} · basal`;
  main.replaceChildren();
  if (ev) main.focus({ preventScroll: true });
  try {
    const dispose = await view.render(main, ctx, new URLSearchParams(query || ''));
    if (token !== routeToken) {
      if (dispose) dispose();
    } else {
      cleanup = dispose || null;
    }
  } catch (e) {
    if (token === routeToken) {
      main.replaceChildren(callout('error', `This view failed to load: ${errorMessage(e)}`));
      toast(errorMessage(e), 'error');
    }
  }
}

boot();
