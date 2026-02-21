# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1537.300 | 6376.304 | 0.241x |
| `batch/splitter_seq` | 1689.346 | 2605.417 | 0.648x |
| `batch/splitter_simple` | 760.000 | 2290.971 | 0.332x |
| `buffer/allocator_alloc_graph` | 17.067 | 55.596 | 0.307x |
| `buffer/allocator_full` | 39.717 | 249.071 | 0.159x |
| `buffer/allocator_reserve_n` | 20.421 | 427.475 | 0.048x |
| `jinja/parser_long` | 31251.229 | 49294.158 | 0.634x |
| `jinja/parser_short` | 398.696 | 501.533 | 0.795x |
| `jinja/renderer_long` | 90781.492 | 220542.663 | 0.412x |
| `jinja/renderer_short` | 1422.304 | 3777.592 | 0.377x |
| `memory/coordinator_recurrent_full` | 3848.525 | 5483.962 | 0.702x |
| `tokenizer/preprocessor_bpe_long` | 16304.067 | 16237.467 | 1.004x |
| `tokenizer/preprocessor_bpe_short` | 516.013 | 701.867 | 0.735x |
| `tokenizer/preprocessor_ugm_long` | 3199.125 | 4484.833 | 0.713x |
| `tokenizer/preprocessor_ugm_short` | 2452.967 | 3440.771 | 0.713x |
