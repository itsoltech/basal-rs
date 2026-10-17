import type {ActAnswer, Answer, ChoiceAnswer, DecisionResponse, EvidenceSpan, JsonValue, MultiAnswer, NoulAnswer, Question, ScoreAnswer, ToolAnswer, ToolDefinition, MultiObjective, SelectionResult, DecisionRequest} from './index.js';

/** Required result metadata on the typed path; missing spans become an empty array. */
export type TypedChoiceAnswer<K extends string = string> = ChoiceAnswer<K> & {evidence: EvidenceSpan[]};
export type TypedNoulAnswer = NoulAnswer & {evidence: EvidenceSpan[]};
export type TypedScoreAnswer = ScoreAnswer & {evidence: EvidenceSpan[]};
export type TypedMultiAnswer<K extends string = string> = Omit<MultiAnswer<K>, 'evidence'>;
export type TypedActAnswer<K extends string = string, A extends string = string> = ActAnswer<K,A> & {evidence: EvidenceSpan[]};

export interface EvidenceOptions { evidence?: boolean }
export interface ChoiceOptions extends EvidenceOptions { option_keys?: 'show' | 'hide' }
export interface MultiOptions<K extends string = string> {
  threshold?: number; min?: number; max?: number; objective?: MultiObjective;
  exclusive?: readonly K[]; fallback?: K; fallback_threshold?: number;
}
export interface ToolOptions extends MultiOptions {
  extra?: readonly ("answer_directly" | "ask_user" | "none")[];
  rank?: number; multi?: boolean; max_error?: number;
}
export interface ActOptions<K extends string = string, A extends string = string> extends EvidenceOptions {
  criteria?: Record<K, string>;
  defer_action?: A;
  max_error?: number;
}
export interface DecisionOptions extends Pick<DecisionRequest,'model'|'facts'|'lang'|'media'|'detail'> { signal?: AbortSignal }

function text(value: string, name: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${name} must be a nonempty string`);
  return value;
}
function criteria(values: Record<string, string>, min: number, max = Infinity): Record<string, string> {
  if (!values || typeof values !== 'object' || Array.isArray(values)) throw new TypeError('criteria must be an object');
  const entries = Object.entries(values);
  if (entries.length < min || entries.length > max) throw new TypeError('invalid number of criteria');
  for (const [key,value] of entries) { text(key,'criterion key'); text(value,'criterion description'); }
  return Object.fromEntries(entries);
}
function number(value: number, name: string, probability = false): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || (probability && (value < 0 || value > 1))) {
    throw new TypeError(`invalid ${name}`);
  }
  return value;
}
function evidence(options: EvidenceOptions): Question {
  if (options.evidence !== undefined && typeof options.evidence !== 'boolean') throw new TypeError('evidence must be boolean');
  return options.evidence ? {evidence:true} : {};
}

/** Immutable snapshot of a typed question. Use factories or the concrete constructors. */
export abstract class QuestionSpec<A extends Answer = Answer> {
  private readonly wire: Question;
  protected constructor(kind: A['type'], instructions: string, fields: Question) {
    this.wire = structuredClone({...fields,type:kind,instructions:text(instructions,'instructions')});
  }
  toJSON(): Question { return structuredClone(this.wire); }
  /** Validate and retrieve an answer with this question's result type. */
  read(response: DecisionResponse, id: string): A {
    const answer = response.answers?.[id];
    if (!answer || answer.type !== this.wire.type || !answer.probabilities || typeof answer.probabilities !== 'object' || Array.isArray(answer.probabilities)) {
      throw new TypeError('Basal returned a missing or mismatched answer');
    }
    const fields: Record<string,Record<string,'number'|'string'|'object'|'boolean'>> = {
      choice:{choice:'string',confidence:'number'}, noul:{noul:'number',confidence:'number'},
      score:{score:'number',legend:'object',confidence:'number'},
      multi:{selected:'object',set_confidence:'number'},
      tool:answer.type==='tool'&&answer.multi ? {tools:'object',call:'boolean',set_confidence:'number'} : {call:'boolean',confidence:'number'},
      act:{action:'string',answer:'string',expected_costs:'object',confidence:'number',calibration:'string'},
    };
    const data = answer as unknown as Record<string,unknown>;
    for (const [field,kind] of Object.entries(fields[answer.type]!)) {
      if (typeof data[field] !== kind || data[field] === null || (kind === 'number' && !Number.isFinite(data[field]))) {
        throw new TypeError('Basal returned an invalid answer field');
      }
    }
    for (const value of Object.values(answer.probabilities)) number(value,'probability',true);
    const expectedKeys = this.wire.type === 'noul' || (this.wire.type === 'act' && this.wire.criteria === undefined)
      ? ['true','false'] : Array.isArray(this.wire.criteria)
        ? this.wire.criteria.map((_,index)=>String(index)) : Object.keys(this.wire.criteria ?? {});
    if (expectedKeys.some(key=>!Object.hasOwn(answer.probabilities,key))) throw new TypeError('missing outcome probability');
    if (answer.evidence !== undefined) {
      if (!Array.isArray(answer.evidence)) throw new TypeError('invalid evidence');
      for (const span of answer.evidence) {
        if (!span || typeof span.text!=='string' || !Number.isSafeInteger(span.start) || !Number.isSafeInteger(span.end)
            || span.start<0 || span.end<0) throw new TypeError('invalid evidence span');
        number(span.probability,'evidence probability',true);
      }
    }

    if (answer.type === 'choice' && !Object.hasOwn(this.wire.criteria ?? {},answer.choice)) throw new TypeError('unknown choice');
    if (answer.type === 'multi' && (!Array.isArray(answer.selected) || answer.selected.some(k=>typeof k!=='string'||!Object.hasOwn(this.wire.criteria ?? {},k)))) throw new TypeError('invalid selected labels');
    if (answer.type === 'score' && (Array.isArray(answer.legend) || Object.values(answer.legend).some(v=>typeof v!=='string'))) throw new TypeError('invalid legend');
    if (answer.type === 'act') {
      if (!['validated','unvalidated','text'].includes(answer.calibration) || !Object.hasOwn(this.wire.costs ?? {},answer.action) || !expectedKeys.includes(answer.answer)) throw new TypeError('invalid action');
      if (Array.isArray(answer.expected_costs) || Object.keys(this.wire.costs ?? {}).some(key=>!Object.hasOwn(answer.expected_costs,key))) throw new TypeError('invalid expected costs');
      for (const value of Object.values(answer.expected_costs)) number(value,'expected cost');
    }
    if (answer.type === 'multi' || (answer.type === 'tool' && answer.multi)) {
      validateSelection(answer, this.wire.objective ?? 'threshold');
    }
    if (answer.type === 'tool') {
      if ((answer.multi ?? false) !== (this.wire.multi ?? false)) throw new TypeError('mismatched tool mode');
      if (answer.multi) {
        if (!Array.isArray(answer.tools)||answer.tools.some(k=>typeof k!=='string'||!Object.hasOwn(answer.probabilities,k))) throw new TypeError('invalid selected tools');
      } else {
        if (answer.tool === null ? answer.deferred !== true : typeof answer.tool !== 'string'||!Object.hasOwn(answer.probabilities,answer.tool)) throw new TypeError('invalid selected tool');
        if (answer.deferred!==undefined&&typeof answer.deferred!=='boolean') throw new TypeError('invalid deferred');
        for(const key of ['refused','max_error'] as const) if(answer[key]!==undefined&&typeof answer[key]!=='string')throw new TypeError('invalid tool metadata');
      }
      if (answer.ranking!==undefined) {
        if(!Array.isArray(answer.ranking))throw new TypeError('invalid ranking');
        for(const entry of answer.ranking){if(!entry||typeof entry.tool!=='string'||!Object.hasOwn(answer.probabilities,entry.tool))throw new TypeError('invalid ranking entry');number(entry.probability,'ranking probability',true);}
      }
    }
    if(answer.type==='act'&&answer.deferred!==undefined&&typeof answer.deferred!=='boolean')throw new TypeError('invalid deferred');
    return {...answer,...(answer.type==='multi'||answer.type==='tool'?{}:{evidence:answer.evidence ?? []})} as A;
  }
}
export class ChoiceQuestion<K extends string = string> extends QuestionSpec<TypedChoiceAnswer<K>> {
  constructor(instructions: string, values: Record<K,string>, options: ChoiceOptions = {}) {
    if (options.option_keys !== undefined && !['show','hide'].includes(options.option_keys)) throw new TypeError('invalid option_keys');
    super('choice',instructions,{criteria:criteria(values,2,255),...evidence(options),...(options.option_keys==='hide'?{option_keys:'hide'}:{})});
  }
}
export class NoulQuestion extends QuestionSpec<TypedNoulAnswer> {
  constructor(instructions: string, options: EvidenceOptions = {}) { super('noul',instructions,evidence(options)); }
}
export class ScoreQuestion extends QuestionSpec<TypedScoreAnswer> {
  constructor(instructions: string, levels: readonly string[], options: EvidenceOptions = {}) {
    if (!Array.isArray(levels) || levels.length<2 || levels.length>255) throw new TypeError('score requires 2–255 ordered levels');
    super('score',instructions,{criteria:levels.map(v=>text(v,'level')),...evidence(options)});
  }
}
function selectionOptions(options: MultiOptions, keys?: readonly string[]): Question {
  const fields: Question={};
  const objective=options.objective ?? 'threshold';
  if(!['threshold','exact','iou','f1'].includes(objective))throw new TypeError('invalid objective');
  if(objective!=='threshold')fields.objective=objective;
  if(options.threshold!==undefined){if(objective!=='threshold')throw new TypeError('threshold applies only to objective threshold');fields.threshold=number(options.threshold,'threshold',true);}
  for(const key of ['min','max'] as const){const v=options[key];if(v!==undefined){if(!Number.isSafeInteger(v)||v<0||(keys&&v>keys.length))throw new TypeError(`invalid ${key}`);fields[key]=v;}}
  if(options.min!==undefined&&options.max!==undefined&&options.min>options.max)throw new TypeError('min exceeds max');
  if(options.exclusive!==undefined){if(!Array.isArray(options.exclusive))throw new TypeError('exclusive must be an array');fields.exclusive=options.exclusive.map(k=>{text(k,'exclusive key');if(keys&&!keys.includes(k))throw new TypeError('unknown exclusive key');return k;});}
  if(options.fallback!==undefined){fields.fallback=text(options.fallback,'fallback');if(keys&&!keys.includes(options.fallback))throw new TypeError('unknown fallback');}
  if(options.fallback_threshold!==undefined){if(options.fallback===undefined)throw new TypeError('fallback_threshold requires fallback');fields.fallback_threshold=number(options.fallback_threshold,'fallback_threshold',true);}
  return fields;
}
function validateSelection(answer: SelectionResult, objective: MultiObjective): void {
  number(answer.set_confidence,'set_confidence',true);
  if(objective==='threshold')number(answer.threshold!,'threshold',true);
  else {
    if(answer.objective!==objective)throw new TypeError('mismatched objective');
    if(answer.expected_score!==null&&answer.expected_score!==undefined)number(answer.expected_score,'expected_score',true);
  }
}
export class MultiQuestion<K extends string = string> extends QuestionSpec<TypedMultiAnswer<K>> {
  constructor(instructions: string, values: Record<K,string>, options: MultiOptions<K> = {}) {
    if ('evidence' in options) throw new TypeError('multi does not support evidence');
    super('multi',instructions,{criteria:criteria(values,1),...selectionOptions(options,Object.keys(values))});
  }
}
export class ToolQuestion extends QuestionSpec<ToolAnswer> {
  constructor(instructions: string, tools: readonly ToolDefinition[], options: ToolOptions = {}) {
    if(!Array.isArray(tools)||tools.some(t=>!t||typeof t!=='object'||Array.isArray(t)))throw new TypeError('tools must contain definition objects');
    if(options.extra!==undefined&&(!Array.isArray(options.extra)||options.extra.some(k=>!['answer_directly','ask_user','none'].includes(k))))throw new TypeError('invalid extra tool option');
    if(options.multi!==undefined&&typeof options.multi!=='boolean')throw new TypeError('invalid multi');
    const fields: Question={tools,...(options.extra===undefined?{}:{extra:options.extra})};
    if(options.rank!==undefined){if(!Number.isSafeInteger(options.rank)||options.rank<1)throw new TypeError('rank must be a positive integer');fields.rank=options.rank;}
    if(options.max_error!==undefined)fields.max_error=number(options.max_error,'max_error',true);
    const selection=selectionOptions(options);
    if(!options.multi&&Object.keys(selection).length)throw new TypeError('selection options require multi');
    if(options.multi){fields.multi=true;Object.assign(fields,selection);}
    super('tool',instructions,fields);
  }
}
export class ActQuestion<K extends string = 'true'|'false', A extends string = string> extends QuestionSpec<TypedActAnswer<K,A>> {
  constructor(instructions: string, costs: Record<A,Record<NoInfer<K>,number>>, options: ActOptions<K,NoInfer<A>> = {}) {
    const optionsMap=options.criteria===undefined?undefined:criteria(options.criteria,2,255);
    const outcomes=Object.keys(optionsMap ?? {'true':'Yes','false':'No'});
    if (!costs || typeof costs!=='object'||Array.isArray(costs)||!Object.keys(costs).length) throw new TypeError('act requires a cost matrix');
    const matrix: Record<string,Record<string,number>>={};
    for(const [action,row]of Object.entries(costs) as [string,Record<string,number>][]) {
      text(action,'action');
      if(!row||typeof row!=='object'||Array.isArray(row)||Object.keys(row).length!==outcomes.length||outcomes.some(k=>!Object.hasOwn(row,k)))throw new TypeError('each cost row must cover every outcome exactly');
      Object.defineProperty(matrix,action,{value:Object.fromEntries(Object.entries(row).map(([k,v])=>[k,number(v,'cost')])),enumerable:true});
    }
    if(options.defer_action!==undefined&&!Object.hasOwn(matrix,options.defer_action))throw new TypeError('unknown defer_action');
    super('act',instructions,{costs:matrix,...evidence(options),...(optionsMap?{criteria:optionsMap}:{}),
      ...(options.defer_action===undefined?{}:{defer_action:options.defer_action}),
      ...(options.max_error===undefined?{}:{max_error:number(options.max_error,'max_error',true)})});
  }
}
export function choice<const K extends string>(instructions: string, values: Record<K,string>, options?: ChoiceOptions): ChoiceQuestion<K> { return new ChoiceQuestion(instructions,values,options); }
export function noul(instructions: string, options?: EvidenceOptions): NoulQuestion { return new NoulQuestion(instructions,options); }
export function score(instructions: string, levels: readonly string[], options?: EvidenceOptions): ScoreQuestion { return new ScoreQuestion(instructions,levels,options); }
export function multi<const K extends string>(instructions: string, values: Record<K,string>, options?: MultiOptions<K>): MultiQuestion<K> { return new MultiQuestion(instructions,values,options); }
export function act<const K extends string = 'true'|'false',const A extends string = string>(instructions: string, costs: Record<A,Record<NoInfer<K>,number>>, options?: ActOptions<K,NoInfer<A>>): ActQuestion<K,A> { return new ActQuestion<K,A>(instructions,costs,options); }
export function tool(instructions: string, tools: readonly ToolDefinition[], options?: ToolOptions): ToolQuestion {return new ToolQuestion(instructions,tools,options);}
export type QuestionSet = Record<string,QuestionSpec>;
export type TypedDecisionResponse<Q extends QuestionSet> = Omit<DecisionResponse,'answers'> & {
  answers:{[K in keyof Q]: Q[K] extends QuestionSpec<infer A> ? A : never};
  answer<K extends keyof Q & (string | number)>(key: K): Q[K] extends QuestionSpec<infer A> ? A : never;
  raw: DecisionResponse;
};
export function buildRequest(state: JsonValue, questions: QuestionSet, options: DecisionOptions): DecisionRequest {
  if(state===null||state===undefined)throw new TypeError('state must not be null/undefined');
  if(!questions||!Object.keys(questions).length)throw new TypeError('questions must not be empty');
  if(Object.getOwnPropertySymbols(questions).length)throw new TypeError('question keys must be strings or numbers');
  const entries=Object.entries(questions).map(([id,q])=>{text(id,'question ID');if(!(q instanceof QuestionSpec))throw new TypeError('expected typed question');return [id,q.toJSON()] as const;});
  if(options.model!==undefined)text(options.model,'model');
  if(options.facts!==undefined&&!['off','auto'].includes(options.facts))throw new TypeError('invalid facts');
  if(options.lang!==undefined&&!['pl','en'].includes(options.lang))throw new TypeError('invalid lang');
  if(options.detail!==undefined&&!['default','high'].includes(options.detail))throw new TypeError('invalid detail');
  if(options.media!==undefined&&(!Array.isArray(options.media)||options.media.length<1||options.media.length>8||options.media.some(m=>!m||!['image','audio'].includes(m.type)||typeof m.data!=='string'||!m.data.trim())))throw new TypeError('invalid media');
  return {state,questions:Object.fromEntries(entries),...(options.lang===undefined?{}:{lang:options.lang}),...(options.detail===undefined?{}:{detail:options.detail}),...(options.media===undefined?{}:{media:structuredClone(options.media)}),...(options.model===undefined?{}:{model:options.model}),...(options.facts===undefined?{}:{facts:options.facts})};
}
