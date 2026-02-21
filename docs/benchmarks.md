# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1666.533 | 6358.221 | 0.262x |
| `batch/splitter_seq` | 1453.046 | 2704.738 | 0.537x |
| `batch/splitter_simple` | 828.279 | 2332.900 | 0.355x |
| `buffer/allocator_alloc_graph` | 17.342 | 52.050 | 0.333x |
| `buffer/allocator_full` | 38.896 | 252.404 | 0.154x |
| `buffer/allocator_reserve_n` | 20.700 | 435.033 | 0.048x |
| `jinja/parser_long` | 31278.742 | 50935.963 | 0.614x |
| `jinja/parser_short` | 400.958 | 528.079 | 0.759x |
| `jinja/renderer_long` | 92063.429 | 229345.550 | 0.401x |
| `jinja/renderer_short` | 1406.263 | 3805.404 | 0.370x |
| `memory/coordinator_recurrent_full` | 3877.404 | 5666.467 | 0.684x |
