# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
intertwined. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1855.848 | 6468.462 | 0.287x |
| `batch/splitter_seq` | 1689.102 | 2733.218 | 0.618x |
| `batch/splitter_simple` | 1001.087 | 2314.423 | 0.433x |
| `buffer/allocator_alloc_graph` | 17.082 | 53.830 | 0.317x |
| `buffer/allocator_full` | 38.901 | 249.018 | 0.156x |
| `buffer/allocator_reserve_n` | 20.415 | 433.773 | 0.047x |
| `jinja/parser_long` | 35795.242 | 50689.574 | 0.706x |
| `jinja/parser_short` | 406.129 | 525.902 | 0.772x |
| `jinja/renderer_long` | 91463.533 | 224436.605 | 0.408x |
| `jinja/renderer_short` | 1415.785 | 3799.992 | 0.373x |
| `memory/coordinator_recurrent_full` | 4163.629 | 5571.910 | 0.747x |
