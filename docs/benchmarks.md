# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1738.563 | 6175.283 | 0.282x |
| `batch/splitter_seq` | 1456.650 | 2580.850 | 0.564x |
| `batch/splitter_simple` | 793.081 | 2182.625 | 0.363x |
| `buffer/allocator_alloc_graph` | 17.278 | 11.867 | 1.456x |
| `buffer/allocator_full` | 33.836 | 76.478 | 0.442x |
| `buffer/allocator_reserve_n` | 18.413 | 150.385 | 0.122x |
