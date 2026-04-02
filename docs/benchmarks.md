# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

## Current Generation Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `benchmark_config: iterations=1000 runs=3 warmup_iterations=100 warmup_runs=1 generation_iterations=1 generation_runs=3 generation_warmup_iterations=0 generation_warmup_runs=0`
- `reference_impl: source=cmake_fetch_latest ref=c30e012253dd9e322c8e3424f808a5c74ecc46bf`
- `generation_architecture: lfm2`
- `generation_formatter_contract: source=tokenizer.chat_template support=supported_contract shape=structured_chat_messages_v1 roles=system,user tools=none add_generation_prompt=true enable_thinking=false keep_past_thinking=false bos=<|startoftext|>`
- `generation_flash_evidence: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 flash_dispatch_calls=174 optimized_flash_dispatch_calls=174 shared_flash_dispatch_calls=0 emel_decode_calls=0 emel_logits_calls=0 reference_decode_calls=0 reference_logits_calls=0`
- Current compare row: `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 emel.cpp 537234834.000 ns/op, llama.cpp 299485916.000 ns/op, ratio=1.794x`

- The compare table below keeps additive generation rows for all maintained supported fixtures; this evidence block stays tied to the current maintained publication case.

## Current Quantized Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `generation_runtime_contract: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 native_quantized=8 approved_dense_f32_by_contract=6 disallowed_fallback=0 explicit_no_claim=0`
- `generation_quantized_evidence: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 native_q8_0_dispatch_calls=0 packed_q8_0_dispatch_calls=0 optimized_q2_dispatch_calls=0 shared_q2_dispatch_calls=0 optimized_q3_dispatch_calls=0 shared_q3_dispatch_calls=0 optimized_q4_dispatch_calls=2378 shared_q4_dispatch_calls=0 optimized_q6_dispatch_calls=291 shared_q6_dispatch_calls=0`

- Contract summary: the maintained canonical Liquid workload stayed on the approved runtime contract with explicit dense-f32-by-contract stages, native quantized dispatch on the maintained path, and no disallowed fallback or explicit no-claim branch on the supported path.

## Generation Stage Probes

- These are single-run request probes that split full request latency into conditioning, prefill, first decode, steady decode, and residual unattributed overhead.
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 emel_total_ns=531806959 emel_conditioning_ns=34583 emel_prefill_ns=549330167 emel_first_decode_ns=21630041 emel_steady_decode_ns=0 emel_unattributed_ns=0 reference_total_ns=315533584 reference_conditioning_ns=43792 reference_prefill_ns=272842875 reference_first_decode_ns=23771125 reference_steady_decode_ns=0 reference_unattributed_ns=18875792`
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10 emel_total_ns=727808000 emel_conditioning_ns=34875 emel_prefill_ns=559134208 emel_first_decode_ns=21156000 emel_steady_decode_ns=193099041 emel_unattributed_ns=0 reference_total_ns=544001417 reference_conditioning_ns=48666 reference_prefill_ns=270250083 reference_first_decode_ns=24095083 reference_steady_decode_ns=214884668 reference_unattributed_ns=34722917`
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100 emel_total_ns=2707609417 emel_conditioning_ns=33750 emel_prefill_ns=542596791 emel_first_decode_ns=21695542 emel_steady_decode_ns=2152592871 emel_unattributed_ns=0 reference_total_ns=2696373292 reference_conditioning_ns=56209 reference_prefill_ns=273146542 reference_first_decode_ns=23749750 reference_steady_decode_ns=2414900044 reference_unattributed_ns=0`
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000 emel_total_ns=24159561000 emel_conditioning_ns=35625 emel_prefill_ns=583159792 emel_first_decode_ns=22381084 emel_steady_decode_ns=23216809589 emel_unattributed_ns=337174910 reference_total_ns=25773145083 reference_conditioning_ns=45375 reference_prefill_ns=274091917 reference_first_decode_ns=24195334 reference_steady_decode_ns=25428668004 reference_unattributed_ns=46144453`

## Preserved ARM Flash Baseline

- Preserved baseline artifact: `snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt`
- `source_commit=3a5a4ee692912429a6d666bb709ec5934ef5655f`
- `baseline_ref=ecbcb7ea9d3303097519723b264a8b5f1e977028`
- `case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`
- `baseline_emel_ns=6995375.000`
- `baseline_reference_ns=5146125.000`
- `baseline_ratio=1.359x`
- Note: this preserved ARM flash baseline remains tied to the archived Llama canonical slice and is not directly compared against the current maintained publication because the benchmark case identity changed explicitly.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 1986.917 | 7326.375 | 0.271x |
| `batch/planner_seq` | 2164.750 | 2993.583 | 0.723x |
| `batch/planner_simple` | 1069.042 | 2303.167 | 0.464x |
| `flash_attention/aarch64/op_flash_attn_ext_decode_like` | 15525.250 | 17809.375 | 0.872x |
| `gbnf/rule_parser_basic` | 472.875 | 277.083 | 1.707x |
| `gbnf/rule_parser_complex` | 3413.042 | 1583.708 | 2.155x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1` | 537234834.000 | 299485916.000 | 1.794x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10` | 731305709.000 | 519616500.000 | 1.407x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100` | 2718986750.000 | 2731346333.000 | 0.995x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000` | 23529726125.000 | 25770462000.000 | 0.913x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1` | 99618000.000 | 102432166.000 | 0.973x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_10` | 197667666.000 | 198620875.000 | 0.995x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_100` | 1145100667.000 | 1230846042.000 | 0.930x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1000` | 14651746792.000 | 17098876042.000 | 0.857x |
| `kernel/aarch64/op_add` | 103.750 | 5082.125 | 0.020x |
| `kernel/aarch64/op_cos` | 1707.083 | 6020.375 | 0.284x |
| `kernel/aarch64/op_div` | 104.708 | 4041.500 | 0.026x |
| `kernel/aarch64/op_dup` | 98.583 | 4143.250 | 0.024x |
| `kernel/aarch64/op_log` | 1850.625 | 5732.584 | 0.323x |
| `kernel/aarch64/op_mul` | 104.542 | 4908.167 | 0.021x |
| `kernel/aarch64/op_mul_mat` | 4862.583 | 10232.000 | 0.475x |
| `kernel/aarch64/op_sin` | 1294.125 | 6213.791 | 0.208x |
| `kernel/aarch64/op_soft_max` | 1623.166 | 5190.000 | 0.313x |
| `kernel/aarch64/op_sqr` | 98.541 | 3877.250 | 0.025x |
| `kernel/aarch64/op_sqrt` | 141.375 | 4005.708 | 0.035x |
| `kernel/aarch64/op_sub` | 103.417 | 5044.417 | 0.021x |
| `kernel/aarch64/op_unary_exp` | 1361.708 | 5331.208 | 0.255x |
| `kernel/aarch64/op_unary_neg` | 111.958 | 4225.958 | 0.026x |
| `kernel/aarch64/op_unary_relu` | 124.792 | 3984.750 | 0.031x |
| `logits/sampler_raw/vocab_128000` | 17851.958 | 18268.708 | 0.977x |
| `logits/sampler_raw/vocab_256000` | 37764.042 | 38444.209 | 0.982x |
| `logits/sampler_raw/vocab_32000` | 5534.459 | 4786.333 | 1.156x |
| `logits/sampler_sml/vocab_128000` | 14873.417 | 16201.667 | 0.918x |
| `logits/sampler_sml/vocab_256000` | 27610.833 | 33147.958 | 0.833x |
| `logits/sampler_sml/vocab_32000` | 3558.959 | 3805.459 | 0.935x |
| `logits/validator_raw/vocab_128000` | 90147.125 | 91376.333 | 0.987x |
| `logits/validator_raw/vocab_256000` | 180985.291 | 179261.334 | 1.010x |
| `logits/validator_raw/vocab_32000` | 24064.917 | 23542.250 | 1.022x |
| `logits/validator_sml/vocab_128000` | 98390.917 | 98089.958 | 1.003x |
| `logits/validator_sml/vocab_256000` | 197095.041 | 199089.708 | 0.990x |
| `logits/validator_sml/vocab_32000` | 24723.750 | 24515.542 | 1.008x |
| `memory/hybrid_full` | 437.417 | 35434.625 | 0.012x |
| `memory/kv_full` | 126.166 | 34408.958 | 0.004x |
| `memory/recurrent_full` | 136.375 | 4487.250 | 0.030x |
| `text/encoders/bpe_long` | 63.375 | 64.875 | 0.977x |
| `text/encoders/bpe_short` | 65.584 | 60.583 | 1.083x |
| `text/encoders/fallback_long` | 2408.542 | 2397.458 | 1.005x |
| `text/encoders/fallback_short` | 62.375 | 61.750 | 1.010x |
| `text/encoders/plamo2_long` | 7746.084 | 7756.917 | 0.999x |
| `text/encoders/plamo2_short` | 204.167 | 206.500 | 0.989x |
| `text/encoders/rwkv_long` | 826385.375 | 829429.333 | 0.996x |
| `text/encoders/rwkv_short` | 55478.208 | 56231.667 | 0.987x |
| `text/encoders/spm_long` | 3582023.750 | 3616470.625 | 0.990x |
| `text/encoders/spm_short` | 1359.375 | 1328.333 | 1.023x |
| `text/encoders/ugm_long` | 1368120.416 | 1366758.583 | 1.001x |
| `text/encoders/ugm_short` | 754.375 | 719.625 | 1.048x |
| `text/encoders/wpm_long` | 30331.292 | 30229.125 | 1.003x |
| `text/encoders/wpm_short` | 550.542 | 556.042 | 0.990x |
| `text/jinja/formatter_long` | 59.667 | 226794.709 | 0.000x |
| `text/jinja/formatter_short` | 15.792 | 3806.084 | 0.004x |
| `text/jinja/parser_long` | 64787.959 | 50228.917 | 1.290x |
| `text/jinja/parser_short` | 940.167 | 502.000 | 1.873x |
| `tokenizer/full_bpe_long` | 13194.167 | 14142.750 | 0.933x |
| `tokenizer/full_bpe_short` | 319.750 | 306.458 | 1.043x |
| `tokenizer/full_plamo2_long` | 12590.500 | 12331.167 | 1.021x |
| `tokenizer/full_plamo2_short` | 1943.542 | 1895.625 | 1.025x |
| `tokenizer/full_rwkv_long` | 832662.959 | 833921.667 | 0.998x |
| `tokenizer/full_rwkv_short` | 55263.250 | 55366.875 | 0.998x |
| `tokenizer/full_spm_long` | 3591620.417 | 3606636.958 | 0.996x |
| `tokenizer/full_spm_short` | 1516.167 | 1518.833 | 0.998x |
| `tokenizer/full_ugm_long` | 1368147.042 | 1384219.459 | 0.988x |
| `tokenizer/full_ugm_short` | 2439.375 | 2511.167 | 0.971x |
| `tokenizer/full_wpm_long` | 32574.875 | 32410.875 | 1.005x |
| `tokenizer/full_wpm_short` | 2220.750 | 2299.375 | 0.966x |
| `tokenizer/preprocessor_bpe_long` | 3351.084 | 5316.875 | 0.630x |
| `tokenizer/preprocessor_bpe_short` | 141.875 | 1770.625 | 0.080x |
| `tokenizer/preprocessor_plamo2_long` | 3920.042 | 7002.166 | 0.560x |
| `tokenizer/preprocessor_plamo2_short` | 2567.208 | 5285.875 | 0.486x |
| `tokenizer/preprocessor_rwkv_long` | 4154.375 | 7212.833 | 0.576x |
| `tokenizer/preprocessor_rwkv_short` | 2429.791 | 4906.750 | 0.495x |
| `tokenizer/preprocessor_spm_long` | 4222.667 | 7035.750 | 0.600x |
| `tokenizer/preprocessor_spm_short` | 2475.583 | 5549.209 | 0.446x |
| `tokenizer/preprocessor_ugm_long` | 4372.959 | 7337.792 | 0.596x |
| `tokenizer/preprocessor_ugm_short` | 2501.167 | 4744.625 | 0.527x |
| `tokenizer/preprocessor_wpm_long` | 4045.750 | 6955.750 | 0.582x |
| `tokenizer/preprocessor_wpm_short` | 2477.125 | 5594.458 | 0.443x |
