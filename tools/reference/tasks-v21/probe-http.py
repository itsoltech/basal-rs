#!/usr/bin/env python3
"""Finite task endpoint/admission/cancellation observations on a separately supervised local server.

The server must have metrics enabled and max_inflight=4. Intentional 529 responses here
exercise admission, and are separate from capacity/performance measurements. No CI gate.
"""
import argparse
import concurrent.futures
import http.client
import json
from pathlib import Path
import socket
import struct
import threading
import time
from urllib.parse import urlsplit


def bits(v):
    if isinstance(v, bool) or v is None or isinstance(v, str):return v
    if isinstance(v, (float,int)):return ('number',struct.pack('!d',float(v)).hex())
    if isinstance(v, list):return [bits(x) for x in v]
    return {k:bits(x) for k,x in v.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--url',required=True)
    p.add_argument('--model',required=True)
    a=p.parse_args();url=urlsplit(a.url)
    if url.scheme!='http' or url.path not in ('','/') or not url.hostname:p.error('plain HTTP base URL required')
    a.out.mkdir(parents=True,exist_ok=False)
    cases=[json.loads(x) for x in (a.measurement_artifacts/'inputs/cases.jsonl').read_text().splitlines() if x.strip()]
    case=next(c for c in cases if c['id']=='tasks-en');body={k:v for k,v in case.items() if k!='id'};body['model']=a.model
    observations=[];issues=[]
    def request(path,body=None):
        conn=http.client.HTTPConnection(url.hostname,url.port or 80,timeout=60);started=time.perf_counter()
        try:
            conn.request('GET' if body is None else 'POST',path,body=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
            response=conn.getresponse();raw=response.read();return {'status':response.status,'elapsed_ms':(time.perf_counter()-started)*1000,'body':json.loads(raw) if body is not None else raw.decode()}
        finally:conn.close()
    def save(tag,value):observations.append({'tag':tag,**value});(a.out/'observations.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2)+'\n')
    baseline=request('/v1/basal',body);save('baseline',baseline)
    typed=request('/v1/systemone',body);save('typesafe',typed)
    if baseline['status']!=200 or typed['status']!=200 or bits(baseline['body']['answers'])!=bits(typed['body']['answers']):issues.append('endpoint answers differ')
    invalid=[{'type':'classify','labels':[]},{'type':'extract','schema':{'type':'array'}},{'type':'compress','keep_ratio':-1},{'type':'verify','claims':[]}]
    for i,q in enumerate(invalid):
        value=request('/v1/basal',dict(body,questions={'q':q}));save('invalid-'+str(i),value)
        if value['status'] not in (400,422):issues.append('invalid task accepted '+str(i))
    barrier=threading.Barrier(16)
    def one(i):barrier.wait();return request('/v1/basal',body)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:burst=list(pool.map(one,range(16)))
    for i,value in enumerate(burst):save('admission-'+str(i),value)
    if not any(x['status']==529 for x in burst) or any(x['status'] not in (200,529) for x in burst):issues.append('unexpected admission statuses')
    for value in burst:
        if value['status']==200 and bits(value['body']['answers'])!=bits(baseline['body']['answers']):issues.append('burst answer differs')
    data=json.dumps(body,ensure_ascii=False).encode();wire=(f'POST /v1/basal HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n').encode()+data
    for _ in range(8):
        with socket.create_connection((url.hostname,url.port or 80),timeout=5) as conn:conn.sendall(wire)
    until=time.monotonic()+60
    while time.monotonic()<until:
        metrics=request('/metrics');text=metrics['body']
        busy=[x for x in text.splitlines() if x.startswith(('basal_replica_inflight_requests{','basal_model_inflight_requests{')) and float(x.rsplit(' ',1)[1])!=0]
        if metrics['status']==200 and 'basal_replica_inflight_requests{' in text and not busy:break
        time.sleep(.1)
    else:issues.append('reservations did not drain')
    save('drained-metrics',metrics)
    for i in range(4):
        value=request('/v1/basal',body);save('after-cancel-'+str(i),value)
        if value['status']!=200 or bits(value['body']['answers'])!=bits(baseline['body']['answers']):issues.append('request after cancellation differs')
    result={'status':'FAIL' if issues else 'PASS','model':a.model,'issues':issues,'admission_statuses':{str(code):sum(x['status']==code for x in burst) for code in (200,529)},'cancelled_connections':8,'scope':'Single replica; observed post-disconnect drain and complete answer invariance. Does not prove all races or multi-GPU task routing.'}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if issues:raise SystemExit(1)


if __name__=='__main__':main()
