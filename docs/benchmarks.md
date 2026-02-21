# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
intertwined. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1549.500 | 6532.379 | 0.237x |
| `batch/splitter_seq` | 1467.775 | 2697.758 | 0.544x |
| `batch/splitter_simple` | 739.300 | 2319.042 | 0.319x |
| `buffer/allocator_alloc_graph` | 17.308 | 54.829 | 0.316x |
| `buffer/allocator_full` | 39.208 | 255.554 | 0.153x |
| `buffer/allocator_reserve_n` | 20.696 | 434.938 | 0.048x |
| `jinja/parser_long` | 30993.783 | 50291.800 | 0.616x |
| `jinja/parser_short` | 389.946 | 493.871 | 0.790x |
| `jinja/renderer_long` | 94130.154 | 227203.433 | 0.414x |
| `jinja/renderer_short` | 1396.850 | 3808.033 | 0.367x |
| `memory/coordinator_recurrent_full` | 3825.479 | 5688.446 | 0.672x |
