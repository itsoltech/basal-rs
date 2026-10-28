import {QuestionSpec} from './questions.js';
import type {AnswerAdditions, DecisionResponse, JsonValue, Question, MultiObjective} from './index.js';

/** Task automation is unknown (null) without a threshold certified for that task/error target. */
export type Automation = Record<string, boolean | null>;
export interface ClassifyAnswer<K extends string = string> extends AnswerAdditions {
  type: 'classify'; experimental: true; scores: Record<K,number>;
  label?: K; top_label?: K; confidence?: number; automated?: Automation;
  labels?: K[]; set_confidence?: number;
}
export interface ClaimResult {
  verdict: 'supported'|'contradicted'|'not_stated'|'not_supported';
  probabilities: Record<string,number>; confidence: number; automated: Automation;
}
export interface VerifyAnswer extends AnswerAdditions {type:'verify';experimental:true;claims:Record<string,ClaimResult>;all_supported:boolean;not_supported:string[]}
/** Inclusive start/exclusive end count Unicode code points, not JavaScript UTF-16 units. */
export interface CompressionUnit {start:number;end:number;probability:number}
export interface CompressAnswer extends AnswerAdditions {
  type:'compress';experimental:true;compressed:string;kept:number[];units:CompressionUnit[];
  unit:'sentence'|'line'|'paragraph';original_chars:number;compressed_chars:number;ratio:number;
}
export interface ExtractField {
  value:JsonValue;confidence:number;source:'choice'|'span';calibration:'validated'|'per decision';
  probabilities?:Record<string,number>;present_probability?:number;text?:string;start?:number;end?:number;
  line_probability?:number;span_probability?:number;error?:string;
}
export interface ExtractAnswer extends AnswerAdditions {type:'extract';experimental:true;data:Record<string,JsonValue>;fields:Record<string,ExtractField>}
export type TaskAnswer = ClassifyAnswer | VerifyAnswer | CompressAnswer | ExtractAnswer;
export interface ClassifyOptions {
  multi_label?:boolean;none?:boolean;max_error?:number|readonly number[];option_keys?:'show'|'hide';
  threshold?:number;min?:number;max?:number;objective?:MultiObjective;
}
export interface VerifyOptions {binary?:boolean;max_error?:number|readonly number[]}
export interface CompressOptions {unit?:'sentence'|'line'|'paragraph';keep_ratio?:number;threshold?:number;min_keep?:number}

function object(v:unknown):Record<string,unknown> {
  if(!v||typeof v!=='object'||Array.isArray(v))throw new TypeError('expected task object');
  return v as Record<string,unknown>;
}
function probability(v:unknown):void {if(typeof v!=='number'||!Number.isFinite(v)||v<0||v>1)throw new TypeError('invalid probability');}
function probabilities(v:unknown):void {for(const n of Object.values(object(v)))probability(n);}
function automation(v:unknown):void {if(Object.values(object(v)).some(n=>n!==null&&typeof n!=='boolean'))throw new TypeError('invalid automation');}
function strings(v:unknown):asserts v is string[] {if(!Array.isArray(v)||v.some(k=>typeof k!=='string'))throw new TypeError('expected string list');}
function offset(v:unknown):asserts v is number {if(!Number.isSafeInteger(v)||(v as number)<0)throw new TypeError('invalid Unicode code point offset/count');}
function span(v:Record<string,unknown>):void {offset(v.start);offset(v.end);if(v.end<v.start)throw new TypeError('reversed source span');}
function risks(v:number|readonly number[]|undefined):void {
  if(v===undefined)return;const a=Array.isArray(v)?v:[v];
  if(a.length>64||a.some(n=>typeof n!=='number'||!Number.isFinite(n)||n<=0||n>=1))throw new TypeError('invalid max_error');
}
function instructions(v:string):string {if(typeof v!=='string')throw new TypeError('instructions must be a string');return v;}

abstract class TaskQuestion<A extends TaskAnswer> extends QuestionSpec<A> {
  protected constructor(kind:A['type'], text:string, fields:Question) {super(kind,instructions(text),fields);}
  override read(response:DecisionResponse,id:string):A {
    const d=object(response.answers?.[id]);
    if(d.type!==this.toJSON().type||d.experimental!==true)throw new TypeError('Basal returned a missing or mismatched task');
    if(d.type==='classify') {
      probabilities(d.scores);const scores=object(d.scores);
      if(this.toJSON().multi_label){strings(d.labels);if(d.labels.some(k=>!Object.hasOwn(scores,k)))throw new TypeError('unknown label');probability(d.set_confidence);}
      else{if(typeof d.label!=='string'||!Object.hasOwn(scores,d.label)||d.top_label!==d.label)throw new TypeError('invalid label');probability(d.confidence);automation(d.automated);}
    } else if(d.type==='verify') {
      const claims=object(d.claims);
      for(const raw of Object.values(claims)){const c=object(raw);const p=object(c.probabilities);probabilities(p);if(typeof c.verdict!=='string'||!['supported','contradicted','not_stated','not_supported'].includes(c.verdict)||!Object.hasOwn(p,c.verdict))throw new TypeError('invalid verdict');probability(c.confidence);automation(c.automated);}
      strings(d.not_supported);if(typeof d.all_supported!=='boolean'||d.not_supported.some(k=>!Object.hasOwn(claims,k)))throw new TypeError('invalid verification summary');
    } else if(d.type==='compress') {
      if(typeof d.compressed!=='string'||!['sentence','line','paragraph'].includes(d.unit as string)||!Array.isArray(d.units)||!Array.isArray(d.kept))throw new TypeError('invalid compression');
      for(const u of d.units){const v=object(u);span(v);probability(v.probability);}
      for(const k of d.kept){offset(k);if(k>=d.units.length)throw new TypeError('invalid kept unit');}
      offset(d.original_chars);offset(d.compressed_chars);if(typeof d.ratio!=='number'||!Number.isFinite(d.ratio)||d.ratio<0)throw new TypeError('invalid ratio');
    } else {
      object(d.data);for(const raw of Object.values(object(d.fields))){const f=object(raw);if(!Object.hasOwn(f,'value')||!['choice','span'].includes(f.source as string)||!['validated','per decision'].includes(f.calibration as string))throw new TypeError('invalid extraction field');probability(f.confidence);if(f.probabilities!==undefined)probabilities(f.probabilities);for(const k of ['present_probability','line_probability','span_probability'])if(f[k]!==undefined)probability(f[k]);if(f.text!==undefined){if(typeof f.text!=='string')throw new TypeError('invalid source text');span(f);}}
    }
    return structuredClone(d) as unknown as A;
  }
}
export class ClassifyQuestion<K extends string = string> extends TaskQuestion<ClassifyAnswer<K>> {
  constructor(text:string,labels:Record<K,JsonValue>|readonly JsonValue[],options:ClassifyOptions={}) {
    if(!labels||typeof labels!=='object'||Object.keys(labels).length<1||Object.keys(labels).length>256)throw new TypeError('classify requires 1..256 labels');
    risks(options.max_error);
    for(const k of ['multi_label','none'] as const)if(options[k]!==undefined&&typeof options[k]!=='boolean')throw new TypeError(`invalid ${k}`);
    if(options.option_keys!==undefined&&!['show','hide'].includes(options.option_keys))throw new TypeError('invalid option_keys');
    if(options.threshold!==undefined){probability(options.threshold);if(options.objective!==undefined&&options.objective!=='threshold')throw new TypeError('threshold requires threshold objective');}
    for(const k of ['min','max'] as const)if(options[k]!==undefined)offset(options[k]);
    if(options.min!==undefined&&options.max!==undefined&&options.min>options.max)throw new TypeError('min exceeds max');
    if(options.objective!==undefined&&!['threshold','exact','iou','f1'].includes(options.objective))throw new TypeError('invalid objective');
    if(!options.multi_label&&['threshold','min','max','objective'].some(k=>Object.hasOwn(options,k)))throw new TypeError('selection options require multi_label');
    super('classify',text,{labels:structuredClone(labels) as Record<string,JsonValue>|JsonValue[],...options});
  }
}
export class VerifyQuestion extends TaskQuestion<VerifyAnswer> {
  constructor(text:string,claims:Record<string,JsonValue>|readonly JsonValue[],options:VerifyOptions={}) {
    if(!claims||typeof claims!=='object'||!Object.keys(claims).length)throw new TypeError('verify requires claims');
    risks(options.max_error);if(options.binary!==undefined&&typeof options.binary!=='boolean')throw new TypeError('invalid binary');
    super('verify',text,{claims:structuredClone(claims) as Record<string,JsonValue>|JsonValue[],...options});
  }
}
export class CompressQuestion extends TaskQuestion<CompressAnswer> {
  constructor(text='',options:CompressOptions={}) {
    if(options.unit!==undefined&&!['sentence','line','paragraph'].includes(options.unit))throw new TypeError('invalid unit');
    if(options.keep_ratio!==undefined){probability(options.keep_ratio);if(options.keep_ratio===0)throw new TypeError('keep_ratio must be positive');}
    if(options.threshold!==undefined)probability(options.threshold);
    if(options.min_keep!==undefined)offset(options.min_keep);
    super('compress',text,options);
  }
}
export class ExtractQuestion extends TaskQuestion<ExtractAnswer> {
  constructor(text:string,schema:Record<string,JsonValue>) {object(schema);super('extract',text,{schema});}
}
export function classify<const K extends string>(text:string,labels:Record<K,JsonValue>,options?:ClassifyOptions):ClassifyQuestion<K>;
export function classify(text:string,labels:readonly JsonValue[],options?:ClassifyOptions):ClassifyQuestion;
export function classify(text:string,labels:Record<string,JsonValue>|readonly JsonValue[],options?:ClassifyOptions):ClassifyQuestion{return new ClassifyQuestion(text,labels,options);}
export function verify(text:string,claims:Record<string,JsonValue>|readonly JsonValue[],options?:VerifyOptions):VerifyQuestion{return new VerifyQuestion(text,claims,options);}
export function compress(text='',options?:CompressOptions):CompressQuestion{return new CompressQuestion(text,options);}
export function extract(text:string,schema:Record<string,JsonValue>):ExtractQuestion{return new ExtractQuestion(text,schema);}
