#!/bin/bash
# dist (LTO fat, codegen-units 1) vs release profile on the M2 Max: bitwise exports, ABBA single decision
cd .
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:$PATH"
O=.cache/dist
for m in basal-1.5-mini basal-1.5-4.5B; do
  R=reports/reference-$m-fp32
  for p in release dist; do
    [ -d $O/export-$m-$p ] || ./target/$p/basal export --model .models/$m --batching tree --inputs $R --out $O/export-$m-$p > $O/export-$m-$p.log 2>&1
  done
  printf "%s release vs dist: " $m; ./target/release/basal compare --a $O/export-$m-release --b $O/export-$m-dist --out $O/cmp-$m.json 2>&1 | tail -1
done
for m in basal-1.5-mini basal-1.5-4.5B; do
  R=reports/reference-$m-fp32
  python3 tools/bench/ab.py --out $O/ab-bench-$m --rounds 3 \
    --a "sleep 30; ./target/release/basal bench --model .models/$m --reference $R" \
    --b "sleep 30; ./target/dist/basal bench --model .models/$m --reference $R" > /dev/null
  python3 -c "import json; d=json.load(open('$O/ab-bench-$m/ab.json')); print('$m', json.dumps({k:d[k] for k in ['ratio_b_over_a','b_faster_items','lat2_median_ms','dec_s']}))"
done
echo DIST-DONE
