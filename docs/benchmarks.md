# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1927.551 | 6330.960 | 0.304x |
| `batch/splitter_seq` | 1505.408 | 2496.272 | 0.603x |
| `batch/splitter_simple` | 835.505 | 2120.028 | 0.394x |
| `buffer/allocator_alloc_graph` | 15.915 | 12.732 | 1.250x |
| `buffer/allocator_full` | 36.215 | 73.595 | 0.492x |
| `buffer/allocator_reserve_n` | 19.538 | 142.770 | 0.137x |
