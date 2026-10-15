import type {ActAnswer, Answer, ChoiceAnswer, DecisionResponse, EvidenceSpan, JsonValue, MultiAnswer, NoulAnswer, Question, ScoreAnswer} from './index.js';

/** Required result metadata on the typed path; missing spans become an empty array. */
export type TypedChoiceAnswer<K extends string = string> = ChoiceAnswer<K> & {evidence: EvidenceSpan[]};
export type TypedNoulAnswer = NoulAnswer & {evidence: EvidenceSpan[]};
export type TypedScoreAnswer = ScoreAnswer & {evidence: EvidenceSpan[]};
export type TypedMultiAnswer<K extends string = string> = Omit<MultiAnswer<K>, 'evidence'>;
export type TypedActAnswer<K extends string = string, A extends string = string> = ActAnswer<K,A> & {evidence: EvidenceSpan[]};

export interface EvidenceOptions { evidence?: boolean }
export interface ChoiceOptions extends EvidenceOptions { option_keys?: 'show' | 'hide' }
export interface MultiOptions { threshold?: number; min?: number; max?: number }
export interface ActOptions<K extends string = string, A extends string = string> extends EvidenceOptions {
  criteria?: Record<K, string>;
  defer_action?: A;
  max_error?: number;
}
export interface DecisionOptions { model?: string; facts?: 'off' | 'auto'; signal?: AbortSignal }

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
    const fields: Record<string,Record<string,'number'|'string'|'object'>> = {
      choice:{choice:'string',confidence:'number'}, noul:{noul:'number',confidence:'number'},
      score:{score:'number',legend:'object',confidence:'number'},
      multi:{selected:'object',threshold:'number',set_confidence:'number'},
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
      if (!['validated','unvalidated'].includes(answer.calibration) || !Object.hasOwn(this.wire.costs ?? {},answer.action) || !expectedKeys.includes(answer.answer)) throw new TypeError('invalid action');
      if (Array.isArray(answer.expected_costs) || Object.keys(this.wire.costs ?? {}).some(key=>!Object.hasOwn(answer.expected_costs,key))) throw new TypeError('invalid expected costs');
      for (const value of Object.values(answer.expected_costs)) number(value,'expected cost');
    }
    return {...answer,...(answer.type==='multi'?{}:{evidence:answer.evidence ?? []})} as A;
  }
}
export class ChoiceQuestion<K extends string = string> extends QuestionSpec<TypedChoiceAnswer<K>> {
  constructor(instructions: string, values: Record<K,string>, options: ChoiceOptions = {}) {
    if (options.option_keys !== undefined && !['show','hide'].includes(options.option_keys)) throw new TypeError('invalid option_keys');
    super('choice',instructions,{criteria:criteria(values,2,10),...evidence(options),...(options.option_keys==='hide'?{option_keys:'hide'}:{})});
  }
}
export class NoulQuestion extends QuestionSpec<TypedNoulAnswer> {
  constructor(instructions: string, options: EvidenceOptions = {}) { super('noul',instructions,evidence(options)); }
}
export class ScoreQuestion extends QuestionSpec<TypedScoreAnswer> {
  constructor(instructions: string, levels: readonly string[], options: EvidenceOptions = {}) {
    if (!Array.isArray(levels) || levels.length<2 || levels.length>10) throw new TypeError('score requires 2–10 ordered levels');
    super('score',instructions,{criteria:levels.map(v=>text(v,'level')),...evidence(options)});
  }
}
export class MultiQuestion<K extends string = string> extends QuestionSpec<TypedMultiAnswer<K>> {
  constructor(instructions: string, values: Record<K,string>, options: MultiOptions = {}) {
    if ('evidence' in options) throw new TypeError('multi does not support evidence');
    const fields: Question={criteria:criteria(values,1)};
    if (options.threshold !== undefined) fields.threshold=number(options.threshold,'threshold',true);
    for (const key of ['min','max'] as const) {
      const value=options[key];
      if(value!==undefined){if(!Number.isInteger(value)||value<0||value>Object.keys(values).length)throw new TypeError(`invalid ${key}`);fields[key]=value;}
    }
    if(options.min!==undefined&&options.max!==undefined&&options.min>options.max)throw new TypeError('min exceeds max');
    super('multi',instructions,fields);
  }
}
export class ActQuestion<K extends string = 'true'|'false', A extends string = string> extends QuestionSpec<TypedActAnswer<K,A>> {
  constructor(instructions: string, costs: Record<A,Record<NoInfer<K>,number>>, options: ActOptions<K,NoInfer<A>> = {}) {
    const optionsMap=options.criteria===undefined?undefined:criteria(options.criteria,2,10);
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
export function multi<const K extends string>(instructions: string, values: Record<K,string>, options?: MultiOptions): MultiQuestion<K> { return new MultiQuestion(instructions,values,options); }
export function act<const K extends string = 'true'|'false',const A extends string = string>(instructions: string, costs: Record<A,Record<NoInfer<K>,number>>, options?: ActOptions<K,NoInfer<A>>): ActQuestion<K,A> { return new ActQuestion<K,A>(instructions,costs,options); }
export type QuestionSet = Record<string,QuestionSpec>;
export type TypedDecisionResponse<Q extends QuestionSet> = Omit<DecisionResponse,'answers'> & {
  answers:{[K in keyof Q]: Q[K] extends QuestionSpec<infer A> ? A : never};
  answer<K extends keyof Q & (string | number)>(key: K): Q[K] extends QuestionSpec<infer A> ? A : never;
  raw: DecisionResponse;
};
export function buildRequest(state: JsonValue, questions: QuestionSet, options: DecisionOptions): {state:JsonValue;questions:Record<string,Question>;model?:string;facts?:'off'|'auto'} {
  if(state===null||state===undefined)throw new TypeError('state must not be null/undefined');
  if(!questions||!Object.keys(questions).length)throw new TypeError('questions must not be empty');
  if(Object.getOwnPropertySymbols(questions).length)throw new TypeError('question keys must be strings or numbers');
  const entries=Object.entries(questions).map(([id,q])=>{text(id,'question ID');if(!(q instanceof QuestionSpec))throw new TypeError('expected typed question');return [id,q.toJSON()] as const;});
  if(options.model!==undefined)text(options.model,'model');
  if(options.facts!==undefined&&!['off','auto'].includes(options.facts))throw new TypeError('invalid facts');
  return {state,questions:Object.fromEntries(entries),...(options.model===undefined?{}:{model:options.model}),...(options.facts===undefined?{}:{facts:options.facts})};
}
