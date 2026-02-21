# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
intertwined. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1863.059 | 6245.366 | 0.298x |
| `batch/splitter_seq` | 1746.334 | 2747.782 | 0.636x |
| `batch/splitter_simple` | 807.071 | 2380.369 | 0.339x |
| `buffer/allocator_alloc_graph` | 17.284 | 53.085 | 0.326x |
| `buffer/allocator_full` | 39.908 | 252.686 | 0.158x |
| `buffer/allocator_reserve_n` | 20.470 | 452.188 | 0.045x |
| `memory/coordinator_recurrent_full` | 3852.476 | 6002.800 | 0.642x |
