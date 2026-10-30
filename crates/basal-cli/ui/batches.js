// Batches & files: the OpenAI-style Files/Batch API (docs/BATCHES.md) used with the console session.

import { api, fetchBlob, enc, errorMessage, ApiError } from './api.js';
import {
  h, field, checkbox, select, button, pill, bar, fmtTime, fmtNum, fmtBytes, mono, toast, openDialog, confirmDialog,
  formDialog, saveBlob, table, callout, pageHeader, loading, kv,
} from './ui.js';

const PAGE = 20;
const CANCELLABLE = new Set(['validating', 'in_progress', 'finalizing']);
const EXAMPLE_LINE = '{"custom_id":"ticket-1","method":"POST","url":"/v1/basal","body":{"state":"I want a refund.","questions":{"dept":{"type":"choice","criteria":{"returns":"Refunds","support":"Support"}}}}}';

function shortId(id) {
  const s = String(id);
  const cut = s.lastIndexOf('_') + 1;
  return h('code', { class: 'mono', title: s }, s.length > cut + 12 ? `${s.slice(0, cut + 6)}…${s.slice(-4)}` : s);
}

/** Error text, with the quota case explained. */
function explain(e, ctx) {
  if (e instanceof ApiError && e.status === 403 && e.code === 'quota_not_configured') {
    const who = ctx.me ? ` for user "${ctx.me.user.id}"` : '';
    return `Batch quota is not configured${who}. An administrator must add limits under batches.quota.owners and install them (basal batches install).`;
  }
  return errorMessage(e);
}

async function download(path, filename) {
  try {
    saveBlob(await fetchBlob(path), filename);
  } catch (e) {
    toast(errorMessage(e), 'error');
  }
}

/** Paginated list (has_more / last_id cursor) rendered into a table with a "Load more" button. */
function pagedList({ path, columns, empty, caption, onError }) {
  const items = [];
  let cursor = null;
  let hasMore = false;
  const el = h('div', null, loading());
  const more = button('Load more', () => load(false), { small: true });

  async function load(reset = true) {
    if (reset) {
      items.length = 0;
      cursor = null;
    }
    more.disabled = true;
    const q = new URLSearchParams({ limit: String(PAGE) });
    if (cursor) q.set('after', cursor);
    try {
      const page = await api(`${path()}${path().includes('?') ? '&' : '?'}${q}`);
      items.push(...page.data);
      cursor = page.last_id;
      hasMore = Boolean(page.has_more && page.last_id);
      el.replaceChildren(table(columns, items, { empty, caption }));
      if (hasMore) el.append(h('div', { class: 'load-more' }, more));
    } catch (e) {
      if (!onError(e)) el.replaceChildren(callout('error', errorMessage(e)));
    } finally {
      more.disabled = false;
    }
  }
  return { el, load };
}

export async function renderBatches(root, ctx) {
  let purpose = '';
  const blocked = h('div');

  // A missing quota makes every call fail the same way: show one explanation instead of two error boxes.
  const onError = (e) => {
    if (e instanceof ApiError && e.status === 403 && e.code === 'quota_not_configured') {
      blocked.replaceChildren(callout('warn', explain(e, ctx)));
      return true;
    }
    return false;
  };

  const batches = pagedList({
    path: () => '/v1/batches',
    empty: 'No batches yet. Upload a JSONL file and create a batch from it.',
    caption: 'Batches',
    onError,
    columns: [
      { label: 'Batch', render: (b) => h('button', { type: 'button', class: 'link', onclick: () => showBatch(b.id) }, shortId(b.id)) },
      { label: 'Status', render: (b) => pill(b.status) },
      { label: 'Model', render: (b) => (b.model ? mono(b.model) : '—') },
      { label: 'Progress', render: progress },
      { label: 'Created', render: (b) => fmtTime(b.created_at) },
      { label: 'Expires', render: (b) => fmtTime(b.expires_at) },
      {
        label: 'Actions', class: 'actions', render: (b) => h('div', { class: 'row-actions' },
          b.output_file_id ? button('Output', () => download(`/v1/files/${enc(b.output_file_id)}/content`, `${b.id}-output.jsonl`), { small: true }) : null,
          b.error_file_id ? button('Errors', () => download(`/v1/files/${enc(b.error_file_id)}/content`, `${b.id}-errors.jsonl`), { small: true }) : null,
          CANCELLABLE.has(b.status) ? button('Cancel', () => cancelBatch(b), { small: true, kind: 'danger-ghost' }) : null),
      },
    ],
  });

  const files = pagedList({
    path: () => `/v1/files${purpose ? `?purpose=${enc(purpose)}` : ''}`,
    empty: 'No files.',
    caption: 'Files',
    onError,
    columns: [
      { label: 'File', render: (f) => shortId(f.id) },
      { label: 'Name', render: (f) => f.filename },
      { label: 'Purpose', render: (f) => (f.purpose === 'batch' ? 'Input' : 'Output') },
      { label: 'Size', class: 'num', render: (f) => fmtBytes(f.bytes) },
      { label: 'Created', render: (f) => fmtTime(f.created_at) },
      { label: 'Expires', render: (f) => (f.expires_at == null ? 'Never' : fmtTime(f.expires_at)) },
      {
        label: 'Actions', class: 'actions', render: (f) => h('div', { class: 'row-actions' },
          button('Download', () => download(`/v1/files/${enc(f.id)}/content`, f.filename), { small: true }),
          f.purpose === 'batch' ? button('Create batch', () => createBatch(f), { small: true }) : null,
          button('Delete', () => deleteFile(f), { small: true, kind: 'danger-ghost' })),
      },
    ],
  });

  function progress(b) {
    const c = b.request_counts || { total: 0, completed: 0, failed: 0 };
    if (!c.total) return h('span', { class: 'muted small' }, b.status === 'validating' ? 'Validating…' : '—');
    const done = c.completed + c.failed;
    return h('div', { class: 'progress', title: `${c.completed} completed, ${c.failed} failed of ${c.total}` },
      bar(done / c.total),
      h('span', { class: 'small' }, `${fmtNum(done)} / ${fmtNum(c.total)}`, c.failed ? h('span', { class: 'err-text' }, ` · ${fmtNum(c.failed)} failed`) : null));
  }

  async function cancelBatch(b) {
    const ok = await confirmDialog({
      title: 'Cancel batch',
      message: `Cancel ${b.id}? Records already finished keep their results; the rest are not processed.`,
      confirmLabel: 'Cancel batch',
      cancelLabel: 'Keep running',
      danger: true,
    });
    if (!ok) return;
    try {
      await api(`/v1/batches/${enc(b.id)}/cancel`, { method: 'POST' });
      toast('Cancellation started.');
      batches.load();
    } catch (e) {
      toast(explain(e, ctx), 'error');
    }
  }

  async function deleteFile(f) {
    const ok = await confirmDialog({
      title: 'Delete file',
      message: `Delete "${f.filename}" (${f.id})? Files used as the input of an active batch cannot be deleted.`,
      confirmLabel: 'Delete file',
      danger: true,
    });
    if (!ok) return;
    try {
      await api(`/v1/files/${enc(f.id)}`, { method: 'DELETE' });
      toast('File deleted.');
      files.load();
    } catch (e) {
      toast(e.code === 'file_in_use' ? 'The file is the input of an active batch and cannot be deleted yet.' : explain(e, ctx), 'error');
    }
  }

  function createBatch(f) {
    formDialog({
      title: 'Create batch',
      submitLabel: 'Create batch',
      fields: [
        kv([['Input file', h('span', null, f.filename, ' ', shortId(f.id))], ['Endpoint', mono('/v1/basal')], ['Completion window', '24 hours']]),
        h('p', { class: 'hint' }, 'Every line must be a native /v1/basal request with a unique custom_id. Online traffic has priority; the batch expires 24 hours after creation, possibly with partial results.'),
      ],
    }, async () => {
      const headers = globalThis.crypto && crypto.randomUUID ? { 'Idempotency-Key': crypto.randomUUID() } : {};
      const b = await api('/v1/batches', {
        method: 'POST', headers, json: { input_file_id: f.id, endpoint: '/v1/basal', completion_window: '24h' },
      }).catch((e) => {
        throw new Error(explain(e, ctx));
      });
      toast(`Batch ${b.id} created.`);
      batches.load();
    });
  }

  async function showBatch(id) {
    let b;
    try {
      b = await api(`/v1/batches/${enc(id)}`);
    } catch (e) {
      return toast(explain(e, ctx), 'error');
    }
    const c = b.request_counts || {};
    const times = ['created_at', 'in_progress_at', 'finalizing_at', 'completed_at', 'failed_at', 'expired_at', 'cancelling_at', 'cancelled_at', 'expires_at']
      .filter((k) => b[k] != null).map((k) => [k.replace(/_at$/, '').replace('_', ' '), fmtTime(b[k])]);
    const fileRow = (label, fid, suffix) => (fid ? [label, h('span', null, shortId(fid), ' ',
      button('Download', () => download(`/v1/files/${enc(fid)}/content`, `${b.id}-${suffix}.jsonl`), { small: true }))] : null);
    const errors = b.errors && Array.isArray(b.errors.data) ? b.errors.data : [];
    openDialog({
      title: `Batch ${b.id}`,
      wide: true,
      body: [
        kv([['Status', pill(b.status)], ['Model', b.model || '—'], ['Endpoint', b.endpoint],
          ['Requests', `${fmtNum(c.completed)} completed, ${fmtNum(c.failed)} failed of ${fmtNum(c.total)}`],
          fileRow('Input file', b.input_file_id, 'input'), fileRow('Output file', b.output_file_id, 'output'), fileRow('Error file', b.error_file_id, 'errors'),
          ...times]),
        errors.length ? [h('h3', { class: 'small-title' }, 'Errors'), table([
          { label: 'Code', render: (e) => mono(e.code) },
          { label: 'Line', class: 'num', render: (e) => (e.line != null ? String(e.line) : '—') },
          { label: 'Message', render: (e) => e.message },
        ], errors, { caption: 'Batch errors' })] : null,
        b.metadata && Object.keys(b.metadata).length ? [h('h3', { class: 'small-title' }, 'Metadata'), kv(Object.entries(b.metadata))] : null,
        h('details', null, h('summary', null, 'Raw JSON'), h('pre', { class: 'code-block' }, JSON.stringify(b, null, 2))),
      ],
    });
  }

  // --- Upload ----------------------------------------------------------------------------------------------------

  const fileInput = h('input', { type: 'file', accept: '.jsonl,.json,.txt,application/jsonl,application/x-ndjson,text/plain', required: true });
  const thenCreate = checkbox('Create a batch from it right away', { checked: true });
  const uploadBtn = h('button', { type: 'submit', class: 'btn btn-primary' }, 'Upload');
  const uploadForm = h('form', { class: 'upload-form', novalidate: true }, field('JSONL input file', fileInput), thenCreate, uploadBtn);
  uploadForm.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const file = fileInput.files && fileInput.files[0];
    if (!file) return toast('Choose a file first.', 'error');
    const form = new FormData();
    form.append('purpose', 'batch');
    form.append('file', file, file.name);
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading…';
    try {
      const f = await api('/v1/files', { method: 'POST', body: form });
      toast(`Uploaded ${f.filename} (${fmtBytes(f.bytes)}).`);
      fileInput.value = '';
      files.load();
      if (thenCreate.querySelector('input').checked) createBatch(f);
    } catch (e) {
      toast(explain(e, ctx), 'error');
    } finally {
      uploadBtn.disabled = false;
      uploadBtn.textContent = 'Upload';
    }
  });

  const purposeSel = select([['', 'All files'], ['batch', 'Inputs'], ['batch_output', 'Outputs']], {
    value: purpose, label: 'Filter files', onchange: () => {
      purpose = purposeSel.value;
      files.load();
    },
  });

  root.append(
    pageHeader('Batches & files', 'Asynchronous jobs over JSONL files. Online requests always have priority.',
      button('Refresh', () => {
        blocked.replaceChildren();
        batches.load();
        files.load();
      }, { small: true })),
    blocked,
    h('section', { class: 'card' }, h('h2', { class: 'section-title' }, 'Upload input'), uploadForm,
      h('details', null, h('summary', null, 'Input format'),
        h('p', { class: 'hint' }, 'One JSON record per line; the body is a native /v1/basal request. The Playground “curl” tab can generate a line for the current request.'),
        h('pre', { class: 'code-block' }, EXAMPLE_LINE))),
    h('section', { class: 'card' }, h('h2', { class: 'section-title' }, 'Batches'), batches.el),
    h('section', { class: 'card' }, h('div', { class: 'section-row' }, h('h2', { class: 'section-title' }, 'Files'), purposeSel), files.el));
  await Promise.all([batches.load(), files.load()]);
}
