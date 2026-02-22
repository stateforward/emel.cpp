# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1728.079 | 6453.962 | 0.268x |
| `batch/splitter_seq` | 1558.554 | 2653.787 | 0.587x |
| `batch/splitter_simple` | 731.362 | 2304.171 | 0.317x |
| `buffer/allocator_alloc_graph` | 17.571 | 56.804 | 0.309x |
| `buffer/allocator_full` | 40.513 | 263.467 | 0.154x |
| `buffer/allocator_reserve_n` | 21.079 | 451.738 | 0.047x |
| `jinja/parser_long` | 30677.896 | 49257.442 | 0.623x |
| `jinja/parser_short` | 395.800 | 499.596 | 0.792x |
| `jinja/renderer_long` | 92039.608 | 225672.625 | 0.408x |
| `jinja/renderer_short` | 1443.375 | 3848.754 | 0.375x |
| `memory/coordinator_recurrent_full` | 3634.921 | 5636.025 | 0.645x |
| `tokenizer/preprocessor_bpe_long` | 15782.367 | 16053.500 | 0.983x |
| `tokenizer/preprocessor_bpe_short` | 498.363 | 726.817 | 0.686x |
| `tokenizer/preprocessor_spm_long` | 3033.567 | 4630.917 | 0.655x |
| `tokenizer/preprocessor_spm_short` | 2357.088 | 3541.925 | 0.665x |
| `tokenizer/preprocessor_ugm_long` | 3154.358 | 4700.567 | 0.671x |
| `tokenizer/preprocessor_ugm_short` | 2344.287 | 3531.192 | 0.664x |
