import importlib.util,sys,types,pathlib,json,copy
import argparse
p=argparse.ArgumentParser(description='Generate host-only task expansion/assembly observations with supplied primitive payloads; no model execution.')
p.add_argument('--upstream',type=pathlib.Path,required=True)
p.add_argument('--measurement-artifacts',type=pathlib.Path,required=True)
a=p.parse_args()
up=a.upstream;pkg=types.ModuleType('basal');pkg.__path__=[str(up/'basal')];sys.modules['basal']=pkg
for name in ['prompt','tasks']:
 spec=importlib.util.spec_from_file_location('basal.'+name,up/'basal'/f'{name}.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
t=sys.modules['basal.tasks'];rows=[]
def answers(request):
 result={}
 for key,q in request['questions'].items():
  typ=q['type']
  if typ=='noul':result[key]={'type':'noul','noul':0.1234565 if '::u::0' in key else .75,'confidence':.75,'probabilities':{'true':.75,'false':.25}}
  elif typ=='multi':
   p={k:(.8 if i==0 else .1) for i,k in enumerate(q['criteria'])};result[key]={'type':'multi','selected':[next(iter(p))],'probabilities':p,'set_confidence':.8}
  else:
   options=q['criteria'];keys=list(options);winner=next((k for k,v in options.items() if v=='ACME Ltd'),next((k for k,v in options.items() if 'ACME' in str(v)),keys[0]))
   pr={k:.9 if k==winner else .1/max(1,len(keys)-1) for k in keys};result[key]={'type':'choice','choice':winner,'probabilities':pr,'confidence':pr[winner]}
 return {'model':'host-only','answers':result,'usage':{'questions':len(result),'input_tokens':0,'branches':len(result)}}
def add(name,state,questions,lang='en'):
 body={'model':'host-only','state':state,'lang':lang,'questions':questions};expanded,plans=t.expand(body,255);first=answers(expanded);second_req=t.second_round(first,body,plans,255);second=answers(second_req) if second_req else None
 row={'id':name,'request':body,'first':first,'calibration':{'task_thresholds':{'classify':{'0.05':.8},'verify':{'0.05':.9}}}}
 if second:row['second']=second
 rows.append(row)
for labels in [{'a':'Payment','b':'Weather'},['a','b'],[{'name':'a','rubric':'Payment'},{'name':'b','description':'Weather'}]]:
 for multi in [False,True]:add('classify-'+str(len(rows)),'Charged twice.',{'q':{'type':'classify','labels':labels,'multi_label':multi,'none':True}})
for lang in ['en','pl']:
 for binary in [True,False]:
  for claims in [['The total is 20.',{'total':20}],{'a':'Total 20.','b':{'total':20}}]:add('verify-'+str(len(rows)),{'document':'Kwota 20 😀'},{'v':{'type':'verify','claims':claims,'binary':binary}},lang)
texts=['Dr. Smith paid 1.2 EUR. Mr. Doe left. Next thing!', 'Sp. z o.o.\nNIP: 123\n\nKwota: 1 234,56 zł. Dalej… „Tak!”','😀 Start. Ącki? Next!\n- Bullet\n\nLast.','A. B. Example e.g. Money\n\tŻółć? End','A\r\nB\r\n\r\nC','First.\x1cSecond.\x1dNext!','Total: 12345','---\n---']
for text in texts:
 for unit in ['sentence','line','paragraph']:
  for ratio in [None,.3]:
   q={'type':'compress','unit':unit,'instructions':'Where is {unit} and {q}?','min_keep':1}
   if ratio:q['keep_ratio']=ratio
   add('compress-'+str(len(rows)),text,{'c':q})
for state in ['Seller: ACME Ltd\nTotal: 1 234,56 PLN','Seller: ACME Ltd\nTotal: 1,234.56 EUR','😀 Seller: ACME Ltd\nTotal: -12.50 EUR','ACME Ltd','---\nACME Ltd']:
 schema={'type':'object','properties':{'seller':{'type':'string','description':'seller name'},'total':{'type':'number'},'nested':{'type':'object','properties':{'paid':{'type':'boolean'},'mode':{'enum':['cash','card',None]},'tags':{'type':'array','items':{'enum':['a','b']}}}}}}
 add('extract-'+str(len(rows)),state,{'e':{'type':'extract','schema':schema},'v':{'type':'verify','claims':['Paid?']}})

for instructions in [{'flags':[True,None,"O'Reilly",'"quoted"',"a\u00a0b\u200dc"]},[True,False,None,1.0],123,False]:
 for task in ['classify','verify','compress','extract']:
  q={'type':task,'instructions':instructions}
  if task=='classify':q['labels']={'a':'A','b':'B'}
  if task=='verify':q['claims']=['A']
  if task=='extract':q['schema']={'type':'object','properties':{'x':{'enum':[True,False,7,'abc'], 'description':{'hint':True}}}}
  add('structured-'+str(len(rows)),'A. Next. Mówi 😀',{'q':q})
for total in ['١٢٣,٤٥','１２３.４５','12345678901234567890123456','-९८७६','+12.00']:
 add('numeric-'+str(len(rows)),'Total: '+total,{'e':{'type':'extract','schema':{'type':'object','properties':{'x':{'type':'number'},'i':{'type':'integer'}}}}})
root=a.measurement_artifacts
root.mkdir(parents=True,exist_ok=True)
with (root/'host-inputs.jsonl').open('x') as output:
 for row in rows:output.write(json.dumps(row,ensure_ascii=False)+'\n')
print('Host cases',len(rows))
