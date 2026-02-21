# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `#` |  |  |  |
| `#` |  |  |  |
| `batch/splitter_equal` | 1622.685 | 6390.384 | 0.254x |
| `batch/splitter_seq` | 1551.720 | 2648.650 | 0.586x |
| `batch/splitter_simple` | 795.522 | 2279.945 | 0.349x |
| `buffer/allocator_alloc_graph` | 17.379 | 55.270 | 0.314x |
| `buffer/allocator_full` | 39.379 | 246.239 | 0.160x |
| `buffer/allocator_reserve_n` | 20.415 | 448.910 | 0.045x |
| `memory/coordinator_recurrent_full` | 3967.959 | 5571.804 | 0.712x |
