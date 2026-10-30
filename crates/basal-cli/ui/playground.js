// Playground: compose a decision request (state, instructions, typed questions), send it to /v1/systemone or
// /v1/basal and read the answers. The draft (never secrets) is kept in localStorage.

import { send, errorLines, errorMessage } from './api.js';
import {
  h, field, select, button, pill, bar, fmtPct, fmtNum, toast, openDialog, copyButton, jsonEditor,
  parseJson, tabs, callout, loading, uid,
} from './ui.js';
import { TEMPLATES } from './templates.js';

const STORAGE_KEY = 'basal-ui.playground.v1';
const ENDPOINTS = [
  ['/v1/systemone', '/v1/systemone (TypeSafe)'],
  ['/v1/basal', '/v1/basal (native)'],
];
const TYPES = {
  noul: { label: 'True / False', blurb: 'Probability that a statement is true. Optional descriptions of the true and false cases.' },
  choice: { label: 'Pick from list', blurb: 'Exactly one of 1–255 options, each a key with a description.' },
  score: { label: 'Scale with rubric', blurb: 'Ordered levels 0…n-1 (1–10), each described. Returns the expected level.' },
  multi: { label: 'Multi-select', blurb: 'Any subset of labels; each label is a yes/no decision against a threshold.' },
  act: { label: 'Action', blurb: 'Reads a true/false or list decision, then picks the action with the lowest expected cost.' },
};
const DEFAULT_COSTS = { approve: { true: 0, false: 10 }, reject: { true: 20, false: 0 }, review: { true: 2, false: 2 } };
const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const pretty = (v) => JSON.stringify(v, null, 2);

// ---------------------------------------------------------------------------------------------------------------
// Question form model. A question is edited either through a typed form or as raw JSON. `fromQuestion` returns
// null for questions the form cannot represent (e.g. non-string criteria), which then stay in JSON mode.
// Fields the form does not know are kept in `extra` and sent unchanged.

function blankForm(type) {
  return {
    type, instructions: '', options: [], levels: [], noulTrue: '', noulFalse: '',
    threshold: '', min: '', max: '', base: 'noul', costs: '', deferAction: '', maxError: '', extra: {},
  };
}

function defaultForm(type) {
  const f = blankForm(type);
  if (type === 'choice') f.options = [{ key: '', desc: '' }, { key: '', desc: '' }];
  if (type === 'multi') f.options = [{ key: '', desc: '' }, { key: '', desc: '' }, { key: '', desc: '' }];
  if (type === 'score') f.levels = ['', '', ''];
  if (type === 'act') {
    f.costs = pretty(DEFAULT_COSTS);
    f.deferAction = 'review';
  }
  return f;
}

function optionsFrom(c) {
  if (!isObject(c)) return null;
  const out = [];
  for (const [key, desc] of Object.entries(c)) {
    if (desc != null && typeof desc !== 'string') return null;
    out.push({ key, desc: desc ?? '' });
  }
  return out;
}

function noulFrom(c) {
  if (c == null) return { t: '', f: '' };
  if (!isObject(c) || Object.keys(c).some((k) => k !== 'true' && k !== 'false')) return null;
  if ([c.true, c.false].some((v) => v != null && typeof v !== 'string')) return null;
  return { t: c.true ?? '', f: c.false ?? '' };
}

function fromQuestion(q) {
  if (!isObject(q) || !TYPES[q.type]) return null;
  if (q.instructions != null && typeof q.instructions !== 'string') return null;
  const form = blankForm(q.type);
  form.instructions = q.instructions ?? '';
  const known = new Set(['type', 'instructions', 'criteria']);
  const numeric = (key, prop) => {
    known.add(key);
    if (q[key] == null) return true;
    if (typeof q[key] !== 'number') return false;
    form[prop] = String(q[key]);
    return true;
  };
  const c = q.criteria;
  if (q.type === 'noul' || (q.type === 'act' && (q.base ?? 'noul') === 'noul')) {
    const n = noulFrom(c);
    if (!n) return null;
    form.noulTrue = n.t;
    form.noulFalse = n.f;
  } else if (q.type === 'choice' || q.type === 'multi' || q.type === 'act') {
    const o = optionsFrom(c);
    if (!o) return null;
    form.options = o;
  } else if (q.type === 'score') {
    if (!Array.isArray(c) || c.some((v) => typeof v !== 'string')) return null;
    form.levels = [...c];
  }
  if (q.type === 'multi' && !(numeric('threshold', 'threshold') && numeric('min', 'min') && numeric('max', 'max'))) return null;
  if (q.type === 'act') {
    if (q.base != null && q.base !== 'noul' && q.base !== 'choice') return null;
    if (q.defer_action != null && typeof q.defer_action !== 'string') return null;
    form.base = q.base ?? 'noul';
    form.costs = q.costs === undefined ? '' : pretty(q.costs);
    form.deferAction = q.defer_action ?? '';
    for (const k of ['base', 'costs', 'defer_action']) known.add(k);
    if (!numeric('max_error', 'maxError')) return null;
  }
  for (const [k, v] of Object.entries(q)) if (!known.has(k)) form.extra[k] = v;
  return form;
}

/** Build the question JSON from a form; problems are reported with the question id. */
function toQuestion(form, id, errors) {
  const q = { type: form.type };
  const err = (msg) => errors.push(`Question "${id}": ${msg}`);
  if (form.instructions.trim()) q.instructions = form.instructions;
  const options = () => {
    if (!form.options.length) err('add at least one option.');
    if (form.options.length > 255) err('at most 255 options are allowed.');
    const crit = {};
    for (const o of form.options) {
      const key = o.key.trim();
      if (!key) {
        err('every option needs a key.');
        continue;
      }
      if (key in crit) err(`duplicate option key "${key}".`);
      // An empty choice description is sent as null ("described by its key", per the contract).
      crit[key] = o.desc.trim() ? o.desc : form.type === 'choice' ? null : key;
    }
    return crit;
  };
  const noul = () => {
    const crit = {};
    if (form.noulTrue.trim()) crit.true = form.noulTrue;
    if (form.noulFalse.trim()) crit.false = form.noulFalse;
    if (Object.keys(crit).length) q.criteria = crit;
  };
  const number = (key, text, label) => {
    if (!text.trim()) return;
    const n = Number(text);
    if (Number.isFinite(n)) q[key] = n;
    else err(`${label} must be a number.`);
  };
  switch (form.type) {
    case 'noul':
      noul();
      if (!q.instructions && !q.criteria) err('a true/false question needs instructions or criteria.');
      break;
    case 'choice':
      q.criteria = options();
      break;
    case 'multi':
      q.criteria = options();
      number('threshold', form.threshold, 'Threshold');
      number('min', form.min, 'Min');
      number('max', form.max, 'Max');
      break;
    case 'score':
      if (form.levels.length < 1 || form.levels.length > 10) err('a scale needs 1–10 levels.');
      form.levels.forEach((l, i) => !l.trim() && err(`level ${i} has no description.`));
      q.criteria = [...form.levels];
      break;
    case 'act': {
      q.base = form.base;
      if (form.base === 'noul') noul();
      else q.criteria = options();
      const costs = parseJson(form.costs);
      if (!form.costs.trim()) err('costs are required.');
      else if (!costs.ok) err(`costs: ${costs.error}`);
      else if (!isObject(costs.value)) err('costs must be a JSON object.');
      else q.costs = costs.value;
      if (form.deferAction.trim()) q.defer_action = form.deferAction.trim();
      number('max_error', form.maxError, 'Max error');
      break;
    }
  }
  for (const [k, v] of Object.entries(form.extra)) if (!(k in q)) q[k] = v;
  return q;
}

/** Convert between types, keeping the instructions and (for list types) the options. */
function convertForm(form, type) {
  const next = defaultForm(type);
  next.instructions = form.instructions;
  const listTypes = ['choice', 'multi'];
  const hasList = listTypes.includes(form.type) || (form.type === 'act' && form.base === 'choice');
  if ((listTypes.includes(type) || type === 'act') && hasList) {
    next.options = form.options.map((o) => ({ ...o }));
    if (type === 'act') next.base = 'choice';
  }
  if ((type === 'noul' || type === 'act') && (form.type === 'noul' || (form.type === 'act' && form.base === 'noul'))) {
    next.noulTrue = form.noulTrue;
    next.noulFalse = form.noulFalse;
  }
  return next;
}

// ---------------------------------------------------------------------------------------------------------------
// Draft (de)serialisation

function questionItem(id, q) {
  const form = fromQuestion(q);
  return form ? { key: uid('q'), id, raw: false, rawText: '', form } : { key: uid('q'), id, raw: true, rawText: pretty(q), form: null };
}

/** Load a request object into draft fields; throws if the form editors cannot hold it. */
function draftFromRequest(req, base) {
  if (!isObject(req)) throw new Error('The request must be a JSON object.');
  const { model, state, instructions, questions, ...extra } = req;
  if (instructions != null && typeof instructions !== 'string') {
    throw new Error('Non-text "instructions" can only be edited in raw request mode.');
  }
  if (questions != null && !isObject(questions)) throw new Error('"questions" must be an object of id → question.');
  const draft = { ...base, instructions: instructions ?? '', extra, questions: [] };
  if (typeof model === 'string') draft.model = model;
  if (typeof state === 'string' || state == null) {
    draft.stateMode = 'text';
    draft.stateText = state ?? '';
  } else {
    draft.stateMode = 'json';
    draft.stateText = pretty(state);
  }
  for (const [id, q] of Object.entries(questions || {})) draft.questions.push(questionItem(id, q));
  return draft;
}

function templateDraft(t, base) {
  return draftFromRequest({ state: t.state, instructions: t.instructions, questions: t.questions }, {
    ...base, rawMode: false, rawText: '',
  });
}

function loadDraft() {
  const base = { endpoint: ENDPOINTS[0][0], model: '', rawMode: false, rawText: '' };
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
    if (isObject(saved) && Array.isArray(saved.questions)) {
      return {
        ...base,
        endpoint: ENDPOINTS.some(([e]) => e === saved.endpoint) ? saved.endpoint : base.endpoint,
        model: String(saved.model || ''),
        stateMode: saved.stateMode === 'json' ? 'json' : 'text',
        stateText: String(saved.stateText ?? ''),
        instructions: String(saved.instructions ?? ''),
        extra: isObject(saved.extra) ? saved.extra : {},
        rawMode: Boolean(saved.rawMode),
        rawText: String(saved.rawText ?? ''),
        questions: saved.questions.filter(isObject).map((q) => ({
          key: uid('q'),
          id: String(q.id ?? ''),
          raw: Boolean(q.raw) || !isObject(q.form),
          rawText: String(q.rawText ?? ''),
          form: isObject(q.form) && TYPES[q.form.type] ? { ...blankForm(q.form.type), ...q.form } : null,
        })),
      };
    }
  } catch {
    /* corrupt or unavailable storage: start from a template */
  }
  return templateDraft(TEMPLATES[1], base);
}

// ---------------------------------------------------------------------------------------------------------------
// View

export async function renderPlayground(root, ctx) {
  let draft = loadDraft();
  let lastRun = null;
  let running = false;
  let saveTimer = null;
  let models = [];

  const save = () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      const { questions, ...rest } = draft;
      const stored = { ...rest, questions: questions.map(({ id, raw, rawText, form }) => ({ id, raw, rawText, form })) };
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
      } catch {
        /* storage full or disabled: the draft simply is not kept */
      }
    }, 300);
  };

  /** Build the body to send. Returns {body, errors}. */
  function buildRequest() {
    const errors = [];
    if (draft.rawMode) {
      const r = parseJson(draft.rawText);
      if (!r.ok) return { errors: [`Request JSON: ${r.error}`] };
      if (!isObject(r.value)) return { errors: ['The request must be a JSON object.'] };
      return { body: r.value, errors };
    }
    const body = {};
    if (draft.model) body.model = draft.model;
    if (draft.stateMode === 'text') {
      body.state = draft.stateText;
    } else {
      const r = parseJson(draft.stateText);
      if (!r.ok) errors.push(`State: ${r.error}`);
      else if (r.value === null || (typeof r.value !== 'object' && typeof r.value !== 'string')) {
        errors.push('State must be a JSON string, object or array.');
      } else body.state = r.value;
    }
    if (draft.instructions.trim()) body.instructions = draft.instructions;
    body.questions = {};
    if (!draft.questions.length) errors.push('Add at least one question.');
    for (const item of draft.questions) {
      const id = item.id.trim();
      if (!id) {
        errors.push('Every question needs an ID.');
        continue;
      }
      if (id in body.questions) errors.push(`Question ID "${id}" is used more than once.`);
      if (item.raw) {
        const r = parseJson(item.rawText);
        if (!r.ok) errors.push(`Question "${id}": ${r.error}`);
        else if (!isObject(r.value)) errors.push(`Question "${id}" must be a JSON object.`);
        else body.questions[id] = r.value;
      } else {
        body.questions[id] = toQuestion(item.form, id, errors);
      }
    }
    for (const [k, v] of Object.entries(draft.extra || {})) if (!(k in body)) body[k] = v;
    return { body, errors };
  }

  // --- Toolbar ---------------------------------------------------------------------------------------------------

  const modelSelect = select([], { label: 'Model' });
  const fillModels = () => {
    const names = models.map((m) => m.name);
    if (draft.model && !names.includes(draft.model)) names.unshift(draft.model);
    if (!draft.model && names.length) draft.model = names[0];
    modelSelect.replaceChildren(...names.map((n) => h('option', { value: n, selected: n === draft.model }, n)));
    if (!names.length) modelSelect.append(h('option', { value: '' }, 'No models'));
  };
  modelSelect.addEventListener('change', () => {
    draft.model = modelSelect.value;
    if (draft.rawMode) {
      const r = parseJson(draft.rawText);
      if (r.ok && isObject(r.value)) {
        r.value.model = draft.model;
        draft.rawText = pretty(r.value);
        renderEditor();
      }
    }
    save();
  });
  const endpointSelect = select(ENDPOINTS, {
    value: draft.endpoint, label: 'Endpoint', onchange: () => {
      draft.endpoint = endpointSelect.value;
      save();
    },
  });
  const rawToggle = h('input', { type: 'checkbox', checked: draft.rawMode, role: 'switch' });
  rawToggle.addEventListener('change', () => {
    try {
      if (rawToggle.checked) {
        const { body, errors } = buildRequest();
        if (errors.length) throw new Error(`Fix the form first: ${errors[0]}`);
        draft.rawText = pretty(body);
        draft.rawMode = true;
      } else {
        const r = parseJson(draft.rawText);
        if (!r.ok) throw new Error(`The request JSON is invalid: ${r.error}`);
        draft = draftFromRequest(r.value, { ...draft, rawMode: false });
        fillModels();
      }
    } catch (e) {
      rawToggle.checked = !rawToggle.checked;
      toast(e.message, 'error');
      return;
    }
    save();
    renderEditor();
  });

  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  const runBtn = h('button', { type: 'button', class: 'btn btn-primary', onclick: () => run(), 'aria-keyshortcuts': isMac ? 'Meta+Enter' : 'Control+Enter' },
    'Run request', h('kbd', { class: 'kbd' }, isMac ? '⌘ ↵' : 'Ctrl ↵'));
  const runStatus = h('span', { class: 'run-status muted', 'aria-live': 'polite' });

  const toolbar = h('div', { class: 'pg-toolbar' },
    field('Model', modelSelect), field('Endpoint', endpointSelect),
    h('label', { class: 'check switch' }, rawToggle, h('span', null, 'Raw request')),
    h('div', { class: 'spacer' }), runBtn);

  const templates = h('div', { class: 'pg-templates', role: 'group', 'aria-label': 'Example templates' },
    h('span', { class: 'muted small' }, 'Examples:'),
    TEMPLATES.map((t) => h('button', {
      type: 'button', class: 'chip', title: t.description, onclick: () => {
        draft = templateDraft(t, draft);
        rawToggle.checked = false;
        save();
        renderEditor();
        toast(`Loaded "${t.name}".`);
      },
    }, t.name)));

  // --- Editor ----------------------------------------------------------------------------------------------------

  const editor = h('section', { class: 'card pg-editor', 'aria-label': 'Request editor' });
  const questionList = h('div', { class: 'question-list' });

  function renderEditor() {
    if (draft.rawMode) {
      const ed = jsonEditor({
        value: draft.rawText, rows: 28, label: 'Request body (JSON)',
        accept: (v) => (isObject(v) ? null : 'The request must be a JSON object.'),
        onInput: (text) => {
          draft.rawText = text;
          save();
        },
      });
      editor.replaceChildren(h('h2', { class: 'section-title' }, 'Request body'),
        h('p', { class: 'hint' }, 'Edit the whole JSON body. Turn off "Raw request" to return to the form; Tab inserts spaces, Esc then Tab leaves the editor.'),
        ed.el);
      return;
    }
    editor.replaceChildren(stateSection(), instructionsSection(), questionsSection());
  }

  function stateSection() {
    const wrap = h('div', { class: 'state-body' });
    const draw = () => {
      if (draft.stateMode === 'text') {
        const ta = h('textarea', { rows: 6, value: draft.stateText, 'aria-label': 'State (text)', oninput: () => {
          draft.stateText = ta.value;
          save();
        } });
        wrap.replaceChildren(ta);
      } else {
        const ed = jsonEditor({
          value: draft.stateText, rows: 10, label: 'State (JSON)',
          accept: (v) => (v === null || (typeof v !== 'object' && typeof v !== 'string') ? 'Use a JSON string, object or array.' : null),
          onInput: (text) => {
            draft.stateText = text;
            save();
          },
        });
        wrap.replaceChildren(ed.el);
      }
    };
    const modeBtn = (mode, label) => h('button', {
      type: 'button', class: 'seg', 'aria-pressed': String(draft.stateMode === mode), onclick: (ev) => {
        if (draft.stateMode === mode) return;
        if (mode === 'json') {
          const r = parseJson(draft.stateText);
          if (!r.ok || r.value === null || typeof r.value !== 'object') draft.stateText = JSON.stringify(draft.stateText);
        } else {
          const r = parseJson(draft.stateText);
          if (r.ok && typeof r.value === 'string') draft.stateText = r.value;
        }
        draft.stateMode = mode;
        for (const b of ev.currentTarget.parentElement.children) b.setAttribute('aria-pressed', String(b === ev.currentTarget));
        save();
        draw();
      },
    }, label);
    draw();
    return h('div', { class: 'pg-section' },
      h('div', { class: 'section-row' }, h('h2', { class: 'section-title' }, 'State / context'),
        h('div', { class: 'segmented', role: 'group', 'aria-label': 'State format' }, modeBtn('text', 'Text'), modeBtn('json', 'JSON'))),
      wrap);
  }

  function instructionsSection() {
    const ta = h('textarea', { rows: 2, value: draft.instructions, placeholder: 'Optional instructions shared by all questions', oninput: () => {
      draft.instructions = ta.value;
      save();
    } });
    return h('div', { class: 'pg-section' }, field('Instructions', ta));
  }

  function questionsSection() {
    questionList.replaceChildren(...draft.questions.map((item) => {
      item.el = questionCard(item);
      return item.el;
    }));
    markIds();
    return h('div', { class: 'pg-section' },
      h('div', { class: 'section-row' }, h('h2', { class: 'section-title' }, `Questions (${draft.questions.length})`),
        button('+ Add question', openBuilder, { small: true })),
      questionList,
      draft.questions.length ? null : h('p', { class: 'empty' }, 'No questions yet. Add one or load an example.'),
      Object.keys(draft.extra || {}).length
        ? h('p', { class: 'hint' }, `Also sent with the request: ${Object.keys(draft.extra).join(', ')} (edit in raw request mode).`) : null);
  }

  /** Flag empty and duplicate IDs on the inputs. */
  function markIds() {
    const counts = new Map();
    for (const q of draft.questions) counts.set(q.id.trim(), (counts.get(q.id.trim()) || 0) + 1);
    for (const q of draft.questions) {
      const input = q.el && q.el.querySelector('.q-id');
      if (!input) continue;
      const bad = !q.id.trim() || counts.get(q.id.trim()) > 1;
      input.setAttribute('aria-invalid', String(bad));
      input.title = bad ? (q.id.trim() ? 'This ID is used more than once' : 'ID is required') : '';
    }
  }

  function rerender(item, focus) {
    const el = questionCard(item);
    item.el.replaceWith(el);
    item.el = el;
    markIds();
    save();
    if (focus) {
      const target = el.querySelector(focus);
      if (target) target.focus();
    }
  }

  function questionCard(item) {
    const idInput = h('input', { type: 'text', class: 'q-id mono', value: item.id, spellcheck: 'false', 'aria-label': 'Question ID' });
    idInput.addEventListener('input', () => {
      item.id = idInput.value;
      markIds();
      save();
    });
    const index = () => draft.questions.indexOf(item);
    const remove = button('×', () => {
      draft.questions.splice(index(), 1);
      renderEditor();
      save();
    }, { kind: 'ghost', small: true, title: `Remove question ${item.id}` });
    remove.setAttribute('aria-label', `Remove question ${item.id || ''}`.trim());
    const modeBtn = button(item.raw ? 'Form' : 'JSON', () => {
      if (item.raw) {
        const r = parseJson(item.rawText);
        if (!r.ok) return toast(`Fix the JSON first: ${r.error}`, 'error');
        const form = fromQuestion(r.value);
        if (!form) return toast('This question uses values the form cannot edit (for example non-text criteria). Keep editing it as JSON.', 'error');
        item.form = form;
        item.raw = false;
      } else {
        const errors = [];
        item.rawText = pretty(toQuestion(item.form, item.id, errors));
        item.raw = true;
      }
      rerender(item, '.q-mode');
    }, { small: true, title: item.raw ? 'Edit with the form' : 'Edit as JSON' });
    modeBtn.classList.add('q-mode');
    modeBtn.setAttribute('aria-pressed', String(item.raw));

    const head = h('div', { class: 'q-head' }, idInput);
    if (!item.raw) {
      const typeSel = select(Object.entries(TYPES).map(([t, d]) => [t, `${d.label} · ${t}`]), { value: item.form.type, label: 'Question type' });
      typeSel.addEventListener('change', () => {
        item.form = convertForm(item.form, typeSel.value);
        rerender(item, 'select');
      });
      head.append(typeSel);
    } else {
      head.append(h('span', { class: 'muted small' }, 'Edited as JSON'));
    }
    head.append(modeBtn, remove);

    let body;
    if (item.raw) {
      body = jsonEditor({
        value: item.rawText, rows: 10, label: `Question ${item.id} (JSON)`,
        accept: (v) => (isObject(v) && typeof v.type === 'string' ? null : 'A question is an object with a "type".'),
        onInput: (text) => {
          item.rawText = text;
          save();
        },
      }).el;
    } else {
      body = formBody(item);
    }
    return h('article', { class: 'q-card', 'aria-label': `Question ${item.id}` }, head, body);
  }

  function formBody(item) {
    const f = item.form;
    const input = (prop, attrs = {}) => {
      const el = h(attrs.rows ? 'textarea' : 'input', { type: attrs.rows ? null : attrs.type || 'text', value: f[prop], ...attrs });
      el.addEventListener('input', () => {
        f[prop] = el.value;
        save();
      });
      return el;
    };
    const parts = [field('Instructions', input('instructions', { rows: 2, class: 'q-instructions', placeholder: 'What should the model decide?' }))];
    const noul = () => h('div', { class: 'grid-2' },
      field('When true (optional)', input('noulTrue', { placeholder: 'Description of the true case' })),
      field('When false (optional)', input('noulFalse', { placeholder: 'Description of the false case' })));
    if (f.type === 'noul') parts.push(noul());
    if (f.type === 'choice' || f.type === 'multi') parts.push(optionsEditor(item));
    if (f.type === 'score') parts.push(levelsEditor(item));
    if (f.type === 'multi') {
      parts.push(h('div', { class: 'grid-3' },
        field('Threshold', input('threshold', { type: 'number', step: '0.05', min: '0', max: '1', placeholder: '0.5' })),
        field('Min selected', input('min', { type: 'number', step: '1', min: '0', placeholder: 'none' })),
        field('Max selected', input('max', { type: 'number', step: '1', min: '0', placeholder: 'none' }))));
    }
    if (f.type === 'act') {
      const baseSel = select([['noul', 'True / False'], ['choice', 'Pick from list']], { value: f.base, label: 'Reads' });
      baseSel.addEventListener('change', () => {
        f.base = baseSel.value;
        if (f.base === 'choice' && !f.options.length) f.options = [{ key: '', desc: '' }, { key: '', desc: '' }];
        rerender(item, '.act-base');
      });
      baseSel.classList.add('act-base');
      parts.push(field('Decision read before the action', baseSel));
      parts.push(f.base === 'noul' ? noul() : optionsEditor(item));
      const costs = jsonEditor({
        value: f.costs, rows: 6, label: 'Costs',
        accept: (v) => (isObject(v) ? null : 'Costs must be a JSON object.'),
        onInput: (text) => {
          f.costs = text;
          save();
        },
      });
      costs.textarea.id = uid('costs');
      parts.push(h('div', { class: 'field' }, h('label', { for: costs.textarea.id }, 'Costs'), costs.el,
        h('p', { class: 'hint' }, 'Full form: action → {outcome: cost}, outcomes being true/false or option keys. Short form: {"wrong": n, "defer": n}.')));
      parts.push(h('div', { class: 'grid-2' },
        field('Defer action (optional)', input('deferAction', { placeholder: 'e.g. review' })),
        field('Max error (optional)', input('maxError', { type: 'number', step: '0.01', min: '0', max: '1', placeholder: 'e.g. 0.05' }))));
    }
    if (Object.keys(f.extra).length) {
      parts.push(h('p', { class: 'hint' }, `Also sent: ${Object.keys(f.extra).join(', ')} (switch to JSON to edit).`));
    }
    return h('div', { class: 'q-body' }, parts);
  }

  function optionsEditor(item) {
    const f = item.form;
    const rows = f.options.map((o, i) => {
      const key = h('input', { type: 'text', class: 'mono', value: o.key, placeholder: 'key', 'aria-label': `Option ${i + 1} key`, dataset: { focus: `opt-${i}` } });
      key.addEventListener('input', () => {
        o.key = key.value;
        save();
      });
      const desc = h('input', { type: 'text', value: o.desc, placeholder: 'Description', 'aria-label': `Option ${i + 1} description` });
      desc.addEventListener('input', () => {
        o.desc = desc.value;
        save();
      });
      const del = button('×', () => {
        f.options.splice(i, 1);
        rerender(item, '.add-option');
      }, { kind: 'ghost', small: true });
      del.setAttribute('aria-label', `Remove option ${i + 1}`);
      return h('div', { class: 'opt-row' }, key, desc, del);
    });
    const add = button('+ Add option', () => {
      f.options.push({ key: '', desc: '' });
      rerender(item, `[data-focus="opt-${f.options.length - 1}"]`);
    }, { small: true, disabled: f.options.length >= 255 });
    add.classList.add('add-option');
    return h('fieldset', { class: 'options' },
      h('legend', null, `${f.type === 'multi' ? 'Labels' : 'Options'} `, h('span', { class: 'muted small' }, `${f.options.length} / 255`)),
      rows, add);
  }

  function levelsEditor(item) {
    const f = item.form;
    const move = (i, d) => {
      [f.levels[i], f.levels[i + d]] = [f.levels[i + d], f.levels[i]];
      rerender(item, `[data-focus="lvl-${i + d}"]`);
    };
    const rows = f.levels.map((text, i) => {
      const input = h('input', { type: 'text', value: text, placeholder: `Description of level ${i}`, 'aria-label': `Level ${i} description`, dataset: { focus: `lvl-${i}` } });
      input.addEventListener('input', () => {
        f.levels[i] = input.value;
        save();
      });
      const ctl = (label, aria, onclick, disabled) => {
        const b = button(label, onclick, { kind: 'ghost', small: true, disabled });
        b.setAttribute('aria-label', aria);
        return b;
      };
      return h('div', { class: 'lvl-row' }, h('span', { class: 'lvl-index', 'aria-hidden': 'true' }, String(i)), input,
        ctl('↑', `Move level ${i} up`, () => move(i, -1), i === 0),
        ctl('↓', `Move level ${i} down`, () => move(i, 1), i === f.levels.length - 1),
        ctl('×', `Remove level ${i}`, () => {
          f.levels.splice(i, 1);
          rerender(item, '.add-level');
        }));
    });
    const add = button('+ Add level', () => {
      f.levels.push('');
      rerender(item, `[data-focus="lvl-${f.levels.length - 1}"]`);
    }, { small: true, disabled: f.levels.length >= 10 });
    add.classList.add('add-level');
    return h('fieldset', { class: 'options' },
      h('legend', null, 'Levels (lowest first) ', h('span', { class: 'muted small' }, `${f.levels.length} / 10`)), rows, add);
  }

  function openBuilder() {
    const taken = new Set(draft.questions.map((q) => q.id.trim()));
    const nextId = () => {
      let n = draft.questions.length + 1;
      while (taken.has(`q${n}`)) n++;
      return `q${n}`;
    };
    const d = openDialog({
      title: 'Add question',
      wide: true,
      body: h('div', { class: 'type-grid' }, Object.entries(TYPES).map(([type, t]) => h('button', {
        type: 'button', class: 'type-card', onclick: () => {
          const item = { key: uid('q'), id: nextId(), raw: false, rawText: '', form: defaultForm(type) };
          draft.questions.push(item);
          d.close();
          renderEditor();
          save();
          const target = item.el && item.el.querySelector('.q-instructions');
          if (target) target.focus();
        },
      }, h('span', { class: 'type-card-title' }, t.label), h('code', { class: 'mono small' }, type), h('span', { class: 'type-card-blurb' }, t.blurb)))),
    });
  }

  // --- Results ---------------------------------------------------------------------------------------------------

  const results = h('section', { class: 'card pg-results', 'aria-label': 'Results' });
  const resultTabs = tabs([
    { id: 'result', label: 'Result', render: renderResult },
    { id: 'request', label: 'Request', render: () => codeBlock(lastRun && lastRun.requestText, 'Nothing has been sent yet.') },
    { id: 'response', label: 'Response', render: renderResponseRaw },
    { id: 'curl', label: 'curl', render: renderCurl },
  ], { label: 'Result views' });
  results.append(runStatus, resultTabs.el);

  function codeBlock(text, empty) {
    if (!text) return h('p', { class: 'empty' }, empty);
    return h('div', { class: 'code-wrap' }, h('div', { class: 'code-actions' }, copyButton(text)), h('pre', { class: 'code-block' }, text));
  }

  function renderResponseRaw() {
    if (!lastRun || lastRun.status == null) return h('p', { class: 'empty' }, 'No response yet.');
    const body = lastRun.data != null ? pretty(lastRun.data) : lastRun.text;
    return h('div', null,
      h('p', { class: 'muted small' }, `HTTP ${lastRun.status} ${lastRun.statusText || ''}`),
      codeBlock(body, '(empty body)'));
  }

  function renderCurl() {
    const { body, errors } = buildRequest();
    if (errors.length) return callout('warn', 'Fix the request first: ', errors[0]);
    const json = pretty(body);
    const lines = [`curl -sS '${location.origin}${draft.endpoint}'`, "  -H 'Content-Type: application/json'"];
    if (ctx.auth) lines.push('  -H "Authorization: Bearer $BASAL_API_KEY"');
    lines.push("  --data-binary @- <<'JSON'");
    const curl = `${lines.join(' \\\n')}\n${json}\nJSON`;
    const batchLine = JSON.stringify({ custom_id: 'request-1', method: 'POST', url: '/v1/basal', body });
    return h('div', null,
      codeBlock(curl),
      ctx.auth ? h('p', { class: 'hint' }, 'Set BASAL_API_KEY to one of your API keys (My keys).') : null,
      h('h3', { class: 'section-title' }, 'Batch input line'),
      h('p', { class: 'hint' }, 'One JSONL record for the Batch API. Batch records always use the native /v1/basal contract; give every line a unique custom_id.'),
      codeBlock(batchLine));
  }

  function renderResult() {
    if (running) return loading('Running…');
    if (!lastRun) {
      return h('div', { class: 'empty-state' }, h('p', null, 'Run a request to see the answers here.'),
        h('p', { class: 'muted small' }, `Shortcut: ${isMac ? '⌘' : 'Ctrl'} + Enter`));
    }
    if (lastRun.validation) {
      return callout('warn', h('strong', null, 'The request is not ready:'), h('ul', null, lastRun.validation.map((e) => h('li', null, e))));
    }
    if (lastRun.networkError) return callout('error', lastRun.networkError);
    if (!lastRun.ok) {
      const lines = errorLines(lastRun.data, lastRun.text);
      return callout('error', h('strong', null, `HTTP ${lastRun.status} ${lastRun.statusText || ''}`),
        lines.length > 1 ? h('ul', null, lines.map((l) => h('li', null, l))) : h('p', null, lines[0] || '(empty body)'));
    }
    const data = lastRun.data;
    if (!isObject(data) || !isObject(data.answers)) return codeBlock(lastRun.text, '(empty body)');
    const usage = isObject(data.usage) ? data.usage : {};
    const questions = isObject(lastRun.body && lastRun.body.questions) ? lastRun.body.questions : {};
    const stats = [
      ['Model', data.model], ['Input tokens', fmtNum(usage.input_tokens)], ['Questions', fmtNum(usage.questions)],
      ['Branches', fmtNum(usage.branches)], ['Server latency', usage.latency_ms != null ? `${fmtNum(usage.latency_ms, 1)} ms` : null],
    ].filter(([, v]) => v != null && v !== '—');
    return h('div', { class: 'answers' },
      h('div', { class: 'stats' }, stats.map(([k, v]) => h('div', { class: 'stat' }, h('span', { class: 'stat-label' }, k), h('span', { class: 'stat-value' }, v)))),
      isObject(usage.truncated) ? callout('warn', `The state was truncated: ${fmtNum(usage.truncated.dropped_tokens)} tokens dropped.`) : null,
      Object.entries(data.answers).map(([id, ans]) => answerCard(id, ans, questions[id])),
      genericFields(data, new Set(['model', 'answers', 'usage'])));
  }

  async function run() {
    if (running) return;
    const { body, errors } = buildRequest();
    if (errors.length) {
      lastRun = { validation: errors };
      resultTabs.show('result');
      return;
    }
    running = true;
    runBtn.disabled = true;
    results.setAttribute('aria-busy', 'true');
    runStatus.textContent = 'Running…';
    resultTabs.refresh();
    const requestText = pretty(body);
    const t0 = performance.now();
    try {
      const r = await send(draft.endpoint, { method: 'POST', json: body });
      lastRun = { ...r, body, requestText, elapsed: performance.now() - t0 };
    } catch (e) {
      lastRun = { body, requestText, networkError: errorMessage(e), elapsed: performance.now() - t0 };
    } finally {
      running = false;
      runBtn.disabled = false;
      results.removeAttribute('aria-busy');
    }
    const parts = [lastRun.status ? `HTTP ${lastRun.status}` : 'Failed', `${fmtNum(lastRun.elapsed, 0)} ms`];
    if (lastRun.headers) {
      const q = lastRun.headers.get('x-basal-queue-ms');
      const c = lastRun.headers.get('x-basal-compute-ms');
      if (q) parts.push(`queue ${fmtNum(Number(q), 1)} ms`);
      if (c) parts.push(`compute ${fmtNum(Number(c), 1)} ms`);
    }
    runStatus.textContent = parts.join(' · ');
    runStatus.classList.toggle('is-err', !lastRun.ok);
    resultTabs.refresh();
  }

  const onKey = (ev) => {
    if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey) && !document.querySelector('dialog[open]')) {
      ev.preventDefault();
      run();
    }
  };
  document.addEventListener('keydown', onKey);

  root.append(h('h1', { class: 'sr-only' }, 'Playground'), toolbar, templates,
    h('div', { class: 'pg-grid' }, editor, results));
  renderEditor();
  fillModels();

  send('/v1/models').then((r) => {
    if (r.ok && isObject(r.data) && Array.isArray(r.data.models)) {
      models = r.data.models.filter((m) => isObject(m) && typeof m.name === 'string');
      fillModels();
      save();
    } else if (r.status !== 401) {
      toast(`Could not load models (HTTP ${r.status}).`, 'error');
    }
  }).catch((e) => toast(errorMessage(e), 'error'));

  return () => {
    document.removeEventListener('keydown', onKey);
    clearTimeout(saveTimer);
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Answer rendering. Known fields get a readable layout; everything else is listed generically so new or
// unknown fields never break the view.

function probRow(label, p, chosen, title) {
  return h('div', { class: `prob-row${chosen ? ' is-chosen' : ''}` },
    h('span', { class: 'prob-label', title: title || label }, label), bar(p, { highlight: chosen }), h('span', { class: 'prob-val' }, fmtPct(p)));
}

function text(v) {
  return typeof v === 'string' ? v : JSON.stringify(v);
}

function answerCard(id, ans, question) {
  if (!isObject(ans)) return h('article', { class: 'answer' }, h('h3', null, id), h('pre', { class: 'code-block' }, pretty(ans)));
  const type = ans.type;
  const handled = new Set(['type', 'calibration']);
  const crit = question && question.criteria;
  const parts = [];
  const chosen = new Set();
  const headline = (label, value, sub) => h('div', { class: 'headline' },
    h('span', { class: 'headline-label' }, label), h('span', { class: 'headline-value' }, value), sub ? h('span', { class: 'headline-sub' }, sub) : null);

  if (type === 'choice' && 'choice' in ans) {
    handled.add('choice');
    chosen.add(String(ans.choice));
    const desc = isObject(crit) && typeof crit[ans.choice] === 'string' ? crit[ans.choice] : null;
    parts.push(headline('Choice', String(ans.choice), desc));
  } else if (type === 'noul' && typeof ans.noul === 'number') {
    handled.add('noul');
    chosen.add(ans.noul >= 0.5 ? 'true' : 'false');
    parts.push(headline('P(true)', fmtPct(ans.noul), ans.noul >= 0.5 ? 'Likely true' : 'Likely false'));
    if (!isObject(ans.probabilities)) parts.push(probRow('true', ans.noul, ans.noul >= 0.5));
  } else if (type === 'score' && typeof ans.score === 'number') {
    handled.add('score');
    const probs = isObject(ans.probabilities) ? Object.entries(ans.probabilities) : [];
    const top = probs.reduce((best, cur) => (cur[1] > best[1] ? cur : best), probs[0] || [null, -1])[0];
    if (top != null) chosen.add(top);
    const legend = isObject(ans.legend) && top != null && ans.legend[top] != null ? text(ans.legend[top]) : null;
    parts.push(headline('Score', `${fmtNum(ans.score, 2)}${probs.length ? ` / ${probs.length - 1}` : ''}`,
      legend ? `Most likely level ${top}: ${legend}` : null));
  } else if (type === 'multi' && Array.isArray(ans.selected)) {
    handled.add('selected');
    ans.selected.forEach((s) => chosen.add(String(s)));
    parts.push(headline('Selected', ans.selected.length ? h('span', { class: 'chips' }, ans.selected.map((s) => h('span', { class: 'chip chip-static' }, String(s)))) : 'None'));
  } else if (type === 'act' && 'action' in ans) {
    handled.add('action').add('answer').add('deferred').add('refused');
    if (ans.answer != null) chosen.add(String(ans.answer));
    const notes = [ans.answer != null ? `Read answer: ${ans.answer}` : null, ans.deferred ? 'deferred' : null,
      ans.refused ? `refused: ${ans.refused}` : null].filter(Boolean).join(' · ');
    parts.push(headline('Action', String(ans.action), notes || null));
  }

  for (const [key, label] of [['confidence', 'Confidence'], ['set_confidence', 'Set confidence']]) {
    if (typeof ans[key] === 'number') {
      handled.add(key);
      parts.push(h('div', { class: 'prob-row prob-conf' }, h('span', { class: 'prob-label' }, label), bar(ans[key]), h('span', { class: 'prob-val' }, fmtPct(ans[key]))));
    }
  }

  if (isObject(ans.probabilities)) {
    handled.add('probabilities');
    const legend = isObject(ans.legend) ? ans.legend : null;
    if (legend) handled.add('legend');
    parts.push(h('div', { class: 'answer-section' }, h('h4', null, 'Probabilities'),
      Object.entries(ans.probabilities).map(([k, p]) => {
        const label = legend && legend[k] != null ? `${k} · ${text(legend[k])}` : k;
        return probRow(label, p, chosen.has(k));
      })));
  }

  if (isObject(ans.expected_costs)) {
    handled.add('expected_costs');
    const entries = Object.entries(ans.expected_costs);
    const maxCost = Math.max(...entries.map(([, v]) => (typeof v === 'number' ? v : 0)), 0) || 1;
    parts.push(h('div', { class: 'answer-section' }, h('h4', null, 'Expected costs (lower is better)'),
      entries.map(([k, v]) => h('div', { class: `prob-row${k === ans.action ? ' is-chosen' : ''}` },
        h('span', { class: 'prob-label' }, k), bar(v / maxCost, { highlight: k === ans.action }), h('span', { class: 'prob-val' }, fmtNum(v, 3))))));
  }

  return h('article', { class: 'answer', 'aria-label': `Answer ${id}` },
    h('div', { class: 'answer-head' }, h('h3', null, id), type ? pill(type, 'info') : null,
      ans.calibration ? pill(`calibration: ${ans.calibration}`, ans.calibration === 'validated' ? 'ok' : 'muted') : null),
    parts, genericFields(ans, handled));
}

/** Remaining fields: scalars inline, structures in a collapsible JSON block. */
function genericFields(obj, skip) {
  const rest = Object.entries(obj).filter(([k]) => !skip.has(k));
  if (!rest.length) return null;
  const scalars = rest.filter(([, v]) => v === null || typeof v !== 'object');
  const nested = rest.filter(([, v]) => v !== null && typeof v === 'object');
  return h('div', { class: 'answer-extra' },
    scalars.length ? h('dl', { class: 'kv kv-inline' }, scalars.map(([k, v]) => [h('dt', null, k), h('dd', null, typeof v === 'number' ? fmtNum(v, 4) : String(v))])) : null,
    nested.map(([k, v]) => h('details', null, h('summary', null, k), h('pre', { class: 'code-block' }, pretty(v)))));
}
