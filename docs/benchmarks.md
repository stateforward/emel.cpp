# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1590.067 | 6551.321 | 0.243x |
| `batch/splitter_seq` | 1494.233 | 2769.658 | 0.540x |
| `batch/splitter_simple` | 764.492 | 2407.667 | 0.318x |
| `buffer/allocator_alloc_graph` | 17.333 | 56.792 | 0.305x |
| `buffer/allocator_full` | 39.583 | 263.475 | 0.150x |
| `buffer/allocator_reserve_n` | 20.758 | 450.979 | 0.046x |
| `jinja/parser_long` | 32224.275 | 50599.421 | 0.637x |
| `jinja/parser_short` | 404.867 | 506.729 | 0.799x |
| `jinja/renderer_long` | 94664.079 | 232153.237 | 0.408x |
| `jinja/renderer_short` | 1574.596 | 3978.121 | 0.396x |
| `memory/coordinator_recurrent_full` | 3873.150 | 5626.817 | 0.688x |
| `tokenizer/preprocessor_bpe_long` | 16275.712 | 16453.733 | 0.989x |
| `tokenizer/preprocessor_bpe_short` | 509.725 | 692.804 | 0.736x |
| `tokenizer/preprocessor_plamo2_long` | 3142.508 | 4691.025 | 0.670x |
| `tokenizer/preprocessor_plamo2_short` | 2429.562 | 3608.113 | 0.673x |
| `tokenizer/preprocessor_rwkv_long` | 3149.004 | 4657.842 | 0.676x |
| `tokenizer/preprocessor_rwkv_short` | 2500.412 | 3560.512 | 0.702x |
| `tokenizer/preprocessor_spm_long` | 3164.762 | 4422.837 | 0.716x |
| `tokenizer/preprocessor_spm_short` | 2489.713 | 3470.771 | 0.717x |
| `tokenizer/preprocessor_ugm_long` | 3222.725 | 4466.550 | 0.722x |
| `tokenizer/preprocessor_ugm_short` | 2468.867 | 3528.483 | 0.700x |
| `tokenizer/preprocessor_wpm_long` | 3217.846 | 4422.783 | 0.728x |
| `tokenizer/preprocessor_wpm_short` | 2435.762 | 3464.592 | 0.703x |
