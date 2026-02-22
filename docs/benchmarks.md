# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1631.754 | 6351.083 | 0.257x |
| `batch/splitter_seq` | 1436.879 | 2645.471 | 0.543x |
| `batch/splitter_simple` | 753.696 | 2433.262 | 0.310x |
| `buffer/allocator_alloc_graph` | 17.104 | 56.333 | 0.304x |
| `buffer/allocator_full` | 37.492 | 252.875 | 0.148x |
| `buffer/allocator_reserve_n` | 20.154 | 420.925 | 0.048x |
| `jinja/parser_long` | 31264.683 | 49676.042 | 0.629x |
| `jinja/parser_short` | 400.996 | 502.142 | 0.799x |
| `jinja/renderer_long` | 92530.250 | 230795.192 | 0.401x |
| `jinja/renderer_short` | 1409.454 | 3881.404 | 0.363x |
| `memory/coordinator_recurrent_full` | 3699.621 | 5572.546 | 0.664x |
| `tokenizer/preprocessor_bpe_long` | 15668.188 | 16857.483 | 0.929x |
| `tokenizer/preprocessor_bpe_short` | 476.679 | 724.396 | 0.658x |
| `tokenizer/preprocessor_spm_long` | 3058.825 | 4750.458 | 0.644x |
| `tokenizer/preprocessor_spm_short` | 2363.933 | 3636.988 | 0.650x |
| `tokenizer/preprocessor_ugm_long` | 3160.283 | 4711.521 | 0.671x |
| `tokenizer/preprocessor_ugm_short` | 2385.463 | 3630.792 | 0.657x |
| `tokenizer/preprocessor_wpm_long` | 3044.938 | 4724.696 | 0.644x |
| `tokenizer/preprocessor_wpm_short` | 2337.904 | 3604.729 | 0.649x |
