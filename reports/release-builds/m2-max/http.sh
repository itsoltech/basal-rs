#!/bin/bash
# HTTP, basal-1.5-mini, release vs dist binary, ABBA: sequential and 32 clients
cd .
O=.cache/dist; PY=.baseline/upstream-1.5/.venv/bin/python
k=0
for p in release dist dist release; do
  k=$((k + 1)); sleep 30
  ./target/$p/basal serve --model .models/basal-1.5-mini --addr 127.0.0.1:8130 > $O/http-serve-$k.log 2>&1 & P=$!
  for i in $(seq 120); do curl --fail-with-body --silent http://127.0.0.1:8130/health > /dev/null && break; sleep 1; done
  $PY tools/bench/loadtest.py --url http://127.0.0.1:8130/v1/systemone --model basal-1.5-mini \
    --reference reports/reference-basal-1.5-mini-fp32 --n-seq 88 --n-conc 400 --concurrency 32 --out $O/http-$k-$p.json > /dev/null 2>&1
  kill -TERM $P; wait $P
  python3 -c "
import json; d=json.load(open('$O/http-$k-$p.json'))
print('$k $p', [(ph.get('name') or ph.get('phase'), round(ph.get('req_s',0),2), round(ph.get('p50_ms',0),1)) for ph in d['phases']] if 'phases' in d else list(d.keys()))"
done
echo HTTP-DONE
