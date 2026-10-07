#!/bin/bash
# basal-1.5-max load time with the weights read through the page cache (BASAL_LOAD_NOCACHE=0) or past it (1, F_NOCACHE),
# ABBA, each run stopped once the model is loaded. Run from the repository root with the intermediate build that had
# this switch (the final loader reads through the page cache and has no switch).
O=.cache/loadab; mkdir -p $O
k=0
for v in 0 1 1 0; do
  k=$((k + 1)); sleep 30
  BASAL_LOAD_NOCACHE=$v ./target/release/basal bench --model .models/basal-1.5-max --reference reports/reference-1.5-max-fp32 --out $O/x-$k.json > $O/load-$k-$v.log 2>&1 &
  P=$!
  until rg -q "loaded on the GPU" $O/load-$k-$v.log; do sleep 1; done
  kill $P; wait $P 2>/dev/null
  echo "nocache=$v $(head -1 $O/load-$k-$v.log)"
done
echo LOADAB-DONE
