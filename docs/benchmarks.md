# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
intertwined. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1592.440 | 6484.798 | 0.246x |
| `batch/splitter_seq` | 1451.627 | 2720.506 | 0.534x |
| `batch/splitter_simple` | 800.218 | 2349.349 | 0.341x |
| `buffer/allocator_alloc_graph` | 16.767 | 53.678 | 0.312x |
| `buffer/allocator_full` | 39.332 | 251.407 | 0.156x |
| `buffer/allocator_reserve_n` | 20.590 | 433.747 | 0.047x |
| `memory/coordinator_recurrent_full` | 3800.401 | 5600.444 | 0.679x |
