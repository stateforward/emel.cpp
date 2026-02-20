# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/sanitizer_decode` | 1144.114 | 6400.777 | 0.179x |
| `batch/splitter_equal` | 1682.211 | 5996.362 | 0.281x |
| `batch/splitter_seq` | 1433.652 | 2591.535 | 0.553x |
| `batch/splitter_simple` | 778.503 | 2150.086 | 0.362x |
| `buffer/allocator_alloc_graph` | 15.867 | 12.251 | 1.295x |
| `buffer/allocator_full` | 36.969 | 78.695 | 0.470x |
| `buffer/allocator_reserve_n` | 18.492 | 149.090 | 0.124x |
