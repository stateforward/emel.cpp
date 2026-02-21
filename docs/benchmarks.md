# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1621.550 | 6470.413 | 0.251x |
| `batch/splitter_seq` | 1430.258 | 2666.900 | 0.536x |
| `batch/splitter_simple` | 788.733 | 2349.996 | 0.336x |
| `buffer/allocator_alloc_graph` | 17.279 | 53.546 | 0.323x |
| `buffer/allocator_full` | 38.854 | 255.446 | 0.152x |
| `buffer/allocator_reserve_n` | 20.429 | 445.683 | 0.046x |
| `jinja/parser_long` | 31418.179 | 50684.708 | 0.620x |
| `jinja/parser_short` | 393.692 | 496.675 | 0.793x |
| `jinja/renderer_long` | 91428.421 | 225905.117 | 0.405x |
| `jinja/renderer_short` | 1433.275 | 3821.408 | 0.375x |
| `memory/coordinator_recurrent_full` | 3655.917 | 5560.779 | 0.657x |
| `tokenizer/preprocessor_bpe_long` | 15969.133 | 16435.283 | 0.972x |
| `tokenizer/preprocessor_bpe_short` | 475.017 | 695.300 | 0.683x |
