# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
intertwined. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1701.508 | 6533.645 | 0.260x |
| `batch/splitter_seq` | 1458.249 | 2749.791 | 0.530x |
| `batch/splitter_simple` | 949.929 | 2355.584 | 0.403x |
| `buffer/allocator_alloc_graph` | 18.317 | 55.225 | 0.332x |
| `buffer/allocator_full` | 39.505 | 258.423 | 0.153x |
| `buffer/allocator_reserve_n` | 20.461 | 439.468 | 0.047x |
| `jinja/parser_long` | 30877.825 | 50481.628 | 0.612x |
| `jinja/parser_short` | 389.587 | 507.760 | 0.767x |
| `jinja/renderer_long` | 90752.743 | 225734.513 | 0.402x |
| `jinja/renderer_short` | 1424.506 | 3899.438 | 0.365x |
| `memory/coordinator_recurrent_full` | 4041.750 | 5664.168 | 0.714x |
