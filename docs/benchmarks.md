# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
intertwined. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1566.118 | 6407.417 | 0.244x |
| `batch/splitter_seq` | 1478.985 | 2674.557 | 0.553x |
| `batch/splitter_simple` | 780.130 | 2329.701 | 0.335x |
| `buffer/allocator_alloc_graph` | 17.466 | 54.495 | 0.321x |
| `buffer/allocator_full` | 38.734 | 248.193 | 0.156x |
| `buffer/allocator_reserve_n` | 20.168 | 434.495 | 0.046x |
| `jinja/parser_long` | 30985.106 | 49460.950 | 0.626x |
| `jinja/parser_short` | 389.338 | 499.469 | 0.780x |
| `jinja/renderer_long` | 90407.199 | 224831.546 | 0.402x |
| `jinja/renderer_short` | 1408.404 | 3755.293 | 0.375x |
| `memory/coordinator_recurrent_full` | 3834.430 | 5579.448 | 0.687x |
