# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1719.135 | 6177.673 | 0.278x |
| `batch/splitter_seq` | 1447.235 | 2587.691 | 0.559x |
| `batch/splitter_simple` | 762.933 | 2170.691 | 0.351x |
| `buffer/allocator_alloc_graph` | 16.752 | 12.770 | 1.312x |
| `buffer/allocator_full` | 35.639 | 78.312 | 0.455x |
| `buffer/allocator_reserve_n` | 20.139 | 148.171 | 0.136x |
