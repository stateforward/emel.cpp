# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1646.325 | 6760.867 | 0.244x |
| `batch/splitter_seq` | 1375.383 | 3947.471 | 0.348x |
| `batch/splitter_simple` | 798.883 | 5614.829 | 0.142x |
| `buffer/allocator_alloc_graph` | 17.979 | 55.375 | 0.325x |
| `buffer/allocator_full` | 40.812 | 260.942 | 0.156x |
| `buffer/allocator_reserve_n` | 21.596 | 430.408 | 0.050x |
| `jinja/parser_long` | 30898.646 | 51140.142 | 0.604x |
| `jinja/parser_short` | 390.904 | 497.171 | 0.786x |
| `jinja/renderer_long` | 92005.421 | 229056.804 | 0.402x |
| `jinja/renderer_short` | 1435.567 | 3816.592 | 0.376x |
| `memory/coordinator_recurrent_full` | 3593.079 | 8472.942 | 0.424x |
| `tokenizer/preprocessor_bpe_long` | 16330.625 | 16613.804 | 0.983x |
| `tokenizer/preprocessor_bpe_short` | 478.100 | 741.296 | 0.645x |
