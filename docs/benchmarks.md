# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
intertwined. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1884.083 | 6167.943 | 0.305x |
| `batch/splitter_seq` | 1745.865 | 2582.198 | 0.676x |
| `batch/splitter_simple` | 804.447 | 2174.350 | 0.370x |
| `buffer/allocator_alloc_graph` | 17.637 | 54.470 | 0.324x |
| `buffer/allocator_full` | 39.142 | 252.186 | 0.155x |
| `buffer/allocator_reserve_n` | 20.856 | 445.627 | 0.047x |
| `jinja/parser_long` | 30136.921 | 49115.570 | 0.614x |
| `jinja/parser_short` | 387.295 | 484.637 | 0.799x |
| `memory/coordinator_recurrent_full` | 3768.017 | 5383.530 | 0.700x |
