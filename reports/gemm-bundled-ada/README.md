# Tabela GEMM z binarki na RTX 6000 Ada

Data: 2026-10-08. Serwer testowy z RTX 6000 Ada (limit 200 W, sterownik 580),
build CUDA z tabelami z `crates/basal-cli/gemm-tables/`, basal-1.5-4.5B f16,
skrypt [check.sh](check.sh) w kontenerze z `tools/cuda/Dockerfile`.

Start `basal serve` z pustym `BASAL_HOME`: serwer bierze tabelę
`nvidia-rtx-6000-ada-generation--basal-1.5-4.5B--f16--cublaslt120901.json` z
binarki, cuBLASLt przyjmuje wszystkie jej algorytmy, tabela trafia do cache
([serve.log](serve.log)). Gotowy po 11 s, z czego 10,3 s to wczytanie wag;
wygenerowanie tej tabeli na tej karcie trwało 327 s
([gemm-invariant-v2](../gemm-invariant-v2/README.md)).

Z tą tabelą ([check.log](check.log)):

- eksport 44 przykładów basal-bench pojedynczo i w drzewie: różnica 0,0
  ([compare-single-tree.json](compare-single-tree.json));
- wobec FP32 upstream 44/44 decyzji, max |Δ logit| 0,0366
  ([compare-fp32-tree.json](compare-fp32-tree.json)); wszystkie liczby
  porównania są te same co w `gemm-invariant-v2/rtx6000ada/compare-fp32-vs-new-4.5B.json`
  (tabela wygenerowana na tej karcie, build sprzed fuzji residual+RMSNorm);
- `basal gemm-share` na tym cache nie ma czego wysłać.

Tabele H100 PCIe z binarki nie zostały sprawdzone na H100 w tym buildzie;
pochodzą z [gemm-invariant-v2](../gemm-invariant-v2/README.md), gdzie
zmierzono ich niezależność od partii i zgodność z FP32.
