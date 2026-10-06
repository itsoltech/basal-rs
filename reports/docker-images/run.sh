#!/bin/bash
# The published images ghcr.io/itsoltech/basal-rs:{latest,latest-sm80,latest-sm90} on the RTX 6000 Ada (compute
# capability 8.9). Run on the Docker host from the repository root (mounted at /work in the containers); the load test
# client runs in the development container `basal-dev`, the servers share its network namespace.
#   1. out of the box: fresh volume, basal-1.5-mini from Hugging Face, GEMM table generated at start (latest)
#   2. latest-sm90 on this GPU (8.9 < 9.0: the kernels cannot load)
#   3. agreement: exports of basal-1.5-max and basal-1.5-mini with latest and latest-sm80, compared with the upstream
#      FP32 references and with each other
#   4. single decision (basal-bench methodology), latest vs latest-sm80 in ABBA order (tools/bench/ab.py)
#   5. HTTP, one-question requests, 1 and 32 clients, basal-1.5-max, ABBA
set -e
O=reports/docker-images
IMG=ghcr.io/itsoltech/basal-rs
MAXT=reports/rust-cuda-1.5-max/gemm-equiv/gemm-algos-f16-invariant-groups.json
MINIT=.cache/gemm/basal-1.5-mini-f16.json
RUN="docker run --rm --gpus all -v $PWD:/work -w /work --entrypoint basal"
up() { for i in $(seq 900); do docker exec basal-dev curl --fail-with-body --silent http://127.0.0.1:8110/health > /dev/null && return 0; sleep 1; done; return 1; }

for t in latest latest-sm80 latest-sm90; do docker pull -q $IMG:$t; done > /dev/null
for t in latest latest-sm80 latest-sm90; do echo "$t $(docker inspect -f '{{index .RepoDigests 0}}' $IMG:$t)"; done > $O/images.txt

# 1. out of the box
docker volume rm basal-oob > /dev/null 2>&1 || true
printf 'models:\n  - repo: Remek/basal-1.5-mini\n    revision: 1978d0705ce09cd2d7d8d3e87b304468121e7255\n' > $O/serve-oob.yml
t0=$(date +%s)
docker run -d --name basal-oob --gpus all --network container:basal-dev -e BASAL_ADDR=127.0.0.1:8110 \
  -v $PWD/$O/serve-oob.yml:/config/serve.yml:ro -v basal-oob:/data $IMG:latest > /dev/null
up
echo "ready after $(( $(date +%s) - t0 )) s" > $O/oob.txt
docker exec basal-dev curl --fail-with-body --silent http://127.0.0.1:8110/v1/models >> $O/oob.txt
docker stop basal-oob > /dev/null; echo "exit $(docker inspect -f '{{.State.ExitCode}}' basal-oob)" >> $O/oob.txt
docker logs basal-oob > $O/oob.log 2>&1; docker rm basal-oob > /dev/null; docker volume rm basal-oob > /dev/null

# 2. sm90 image on an 8.9 GPU
$RUN $IMG:latest-sm90 export --model .models/basal-1.5-mini --gemm-table $MINIT --inputs reports/reference-1.5-max-fp32 \
  --out /tmp/x > $O/sm90-on-ada.log 2>&1 || echo "exit $?" >> $O/sm90-on-ada.log

# 3. agreement
for t in latest latest-sm80; do
  for spec in basal-1.5-max:$MAXT:reports/reference-1.5-max-fp32 basal-1.5-mini:$MINIT:reports/reference-basal-1.5-mini-fp32; do
    IFS=: read m table ref <<< "$spec"
    $RUN $IMG:$t export --model .models/$m --gemm-table $table --inputs $ref --out $O/export-$m-$t > /dev/null 2>&1
    $RUN $IMG:$t compare --a $ref --b $O/export-$m-$t --out $O/compare-upstream-fp32-vs-$m-$t.json | tail -1
  done
done
for m in basal-1.5-max basal-1.5-mini; do
  $RUN $IMG:latest compare --a $O/export-$m-latest --b $O/export-$m-latest-sm80 --out $O/compare-$m-latest-vs-sm80.json | tail -1
done

# 4. single decision, ABBA
for spec in basal-1.5-max:$MAXT basal-1.5-mini:$MINIT; do
  IFS=: read m table <<< "$spec"
  python3 tools/bench/ab.py --out $O/ab-bench-$m --rounds 2 \
    --a "$RUN $IMG:latest bench --model .models/$m --gemm-table $table --reference reports/reference-1.5-max-fp32" \
    --b "$RUN $IMG:latest-sm80 bench --model .models/$m --gemm-table $table --reference reports/reference-1.5-max-fp32" \
    | tail -4
done

# 5. HTTP, basal-1.5-max, ABBA
printf 'models:\n  - path: /work/.models/basal-1.5-max\n    gemm_table: /work/%s\n' $MAXT > $O/serve-max.yml
k=0
for t in latest latest-sm80 latest-sm80 latest; do
  k=$((k + 1))
  docker run -d --name basal-load --gpus all --network container:basal-dev -e BASAL_ADDR=127.0.0.1:8110 -v $PWD:/work \
    -v $PWD/$O/serve-max.yml:/config/serve.yml:ro $IMG:$t > /dev/null
  up
  docker exec basal-dev bash -lc "cd /work && .baseline/upstream/.venv/bin/python tools/bench/loadtest.py --gpu \
    --model basal-1.5-max --reference reports/reference-1.5-max-fp32 --n-seq 44 --n-conc 300 --concurrency 32 \
    --url http://127.0.0.1:8110/v1/systemone --out $O/load-$k-$t.json" > /dev/null 2>&1
  docker stop basal-load > /dev/null; docker rm basal-load > /dev/null
done
echo IMAGES-DONE
