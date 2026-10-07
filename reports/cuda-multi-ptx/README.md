# CUDA: PTX dla kilku architektur w jednej binarce

Data: 2026-10-07, RTX 6000 Ada (compute capability 8.9), limit mocy 250 W na
czas testu. Binarka z `CUDA_COMPUTE_CAPS` domyślnym (80, 89, 90), która przy
starcie wybiera PTX dla 8.9, wobec binarki z jednym PTX dla 8.9 (commit
`854c475`, eksporty z [metal-m2-max-tree/cuda-unchanged](../metal-m2-max-tree/cuda-unchanged/README.md)).

Eksporty pojedynczo i w drzewie dla basal-1.5-mini, 1.5-4.5B i 1.5-max, te
same tabele GEMM (`cmp-*.json`): wszystkie logity bitowo równe (różnica 0,0),
44/44. PTX dla 8.0 na tej karcie sprawdzono wcześniej obrazem `-sm80`
(bitowo te same wyniki i ta sama wydajność, [docker-images](../docker-images/README.md));
kart A100 i H100 nie testowano.
