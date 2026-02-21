# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
intertwined. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1612.731 | 6375.223 | 0.253x |
| `batch/splitter_seq` | 1494.051 | 2637.307 | 0.567x |
| `batch/splitter_simple` | 779.220 | 2259.085 | 0.345x |
| `buffer/allocator_alloc_graph` | 17.263 | 52.344 | 0.330x |
| `buffer/allocator_full` | 39.801 | 239.172 | 0.166x |
| `buffer/allocator_reserve_n` | 20.497 | 426.365 | 0.048x |
| `memory/coordinator_recurrent_full` | 3821.260 | 5509.109 | 0.694x |
