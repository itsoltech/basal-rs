/** Manual live GPU SDK comparison. Build against the locally packed npm package. */
import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {createRequire} from 'node:module';
import * as esm from '@itsoltech/basal-client';
import type {JsonValue, QuestionSet, DecisionOptions, ToolDefinition} from '@itsoltech/basal-client';
const index=process.argv.indexOf('--measurement-artifacts');
if(index<0||!process.argv[index+1])throw new Error('--measurement-artifacts required');
const root=process.argv[index+1]!;
const api: typeof esm=process.argv.includes('--cjs')?createRequire(import.meta.url)('@itsoltech/basal-client'):esm;
const sdk=process.argv.includes('--cjs')?'typescript-cjs':'typescript-esm';
const client=new api.BasalClient({baseUrl:process.env.BASAL_URL??'http://127.0.0.1:19439',timeoutMs:300000});
// Dynamic input is an archived measurement corpus, not an untrusted network schema.
const cases=readFileSync(join(root,'inputs/cases.jsonl'),'utf8').trim().split('\n').map(line=>JSON.parse(line));
console.log(JSON.stringify({sdk,models:await client.models()}));
const diagnostic={n:api.noul('Is this a diagnostic?')};
try {await client.decide('diagnostic',diagnostic,{model:'__sdk_unknown_model__'});throw new Error('unknown model accepted');}
catch(error) {if(!(error instanceof api.BasalHTTPError)||error.statusCode!==422)throw error;console.log(JSON.stringify({sdk,check:'unknown-model',status:error.statusCode}));}
let rejected=false;
try {await client.decide('diagnostic',diagnostic,{media:[]});}
catch(error) {if(!(error instanceof Error)||!error.message.includes('media'))throw error;rejected=true;}
if(!rejected)throw new Error('empty media accepted');
console.log(JSON.stringify({sdk,check:'empty-media',status:'local-rejection'}));
for(const c of cases) {
  if(c.media&&process.env.BASAL_MEDIA!=='1')continue;
  const questions: QuestionSet={};
  for(const [key,value]of Object.entries(c.questions)) {
    const q=value as any;
    switch(q.type) {
      case 'choice':questions[key]=api.choice(q.instructions,q.criteria,q);break;
      case 'noul':questions[key]=api.noul(q.instructions,q);break;
      case 'score':questions[key]=api.score(q.instructions,q.criteria,q);break;
      case 'multi':questions[key]=api.multi(q.instructions,q.criteria,q);break;
      case 'act':questions[key]=api.act(q.instructions,q.costs,q);break;
      case 'tool':questions[key]=api.tool(q.instructions,q.tools,q);break;
      case 'classify':questions[key]=api.classify(q.instructions??'',q.labels,q);break;
      case 'verify':questions[key]=api.verify(q.instructions??'',q.claims,q);break;
      case 'compress':questions[key]=api.compress(q.instructions??'',q);break;
      case 'extract':questions[key]=api.extract(q.instructions??'',q.schema);break;
      default:throw new Error('unsupported diagnostic case');
    }
  }
  const options:DecisionOptions={model:process.env.BASAL_MODEL,lang:c.lang,detail:c.detail};
  if(c.media)options.media=c.media.map((m:any)=>{
    const bytes=readFileSync(join(root,m.file));
    const data=m.encoding==='bytes'?bytes:m.encoding==='data-url'?`data:${m.mime};base64,${bytes.toString('base64')}`:bytes.toString('base64');
    return (m.type==='image'?api.image:api.audio)(data);
  });
  const response=await client.decide(c.state as JsonValue,questions,options);
  console.log(JSON.stringify({sdk,case:c.id,answers:response.answers,raw:response.raw}));
  if(c.id==='named-api') {
    // Compile and exercise exact named answer inference in addition to runtime-keyed maps.
    const named=await client.decide(c.state,{
      route:api.tool(c.questions.route.instructions,c.questions.route.tools as ToolDefinition[],c.questions.route),
      labels:api.multi(c.questions.labels.instructions,c.questions.labels.criteria,c.questions.labels),
    },options);
    const call:boolean=named.answers.route.call;
    const selected:string[]=named.answers.labels.selected;
    console.log(JSON.stringify({sdk:sdk+'-named',case:c.id,answers:named.answers,raw:named.raw,inference:{call,selected}}));
  }
}
