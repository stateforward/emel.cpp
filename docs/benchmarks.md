# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `buffer/allocator_alloc_graph` | 17.579 | 11.892 | 1.478x |
| `buffer/allocator_full` | 36.845 | 76.450 | 0.482x |
| `buffer/allocator_reserve_n` | 18.436 | 141.390 | 0.130x |
