#include "bench_runner_registry.hpp"

#include <array>

namespace emel::bench {

namespace {

constexpr test_case make_test_case(const append_case_fn emel_fn,
                                   const append_case_fn reference_fn,
                                   const std::string_view suite,
                                   const bool tokenizer_case = false) {
  return test_case{
    .append_emel = emel_fn,
    .append_reference = reference_fn,
    .suite = suite,
    .tokenizer_case = tokenizer_case,
  };
}

const std::array<test_case, 36> & all_runner_cases() {
  static const std::array<test_case, 36> cases = {{
    make_test_case(append_emel_batch_planner_cases,
                   append_reference_batch_planner_cases,
                   "batch_planner"),
    make_test_case(append_emel_memory_kv_cases, append_reference_memory_kv_cases, "memory_kv"),
    make_test_case(append_emel_memory_recurrent_cases,
                   append_reference_memory_recurrent_cases,
                   "memory_recurrent"),
    make_test_case(append_emel_memory_hybrid_cases,
                   append_reference_memory_hybrid_cases,
                   "memory_hybrid"),
    make_test_case(append_emel_jinja_parser_cases,
                   append_reference_jinja_parser_cases,
                   "jinja_parser"),
    make_test_case(append_emel_jinja_formatter_cases,
                   append_reference_jinja_formatter_cases,
                   "jinja_formatter"),
    make_test_case(append_emel_gbnf_rule_parser_cases,
                   append_reference_gbnf_rule_parser_cases,
                   "gbnf_rule_parser"),
    make_test_case(append_emel_generation_cases, append_reference_generation_cases, "generation"),
    make_test_case(append_emel_sortformer_diarization_cases,
                   append_reference_sortformer_diarization_cases,
                   "diarization_sortformer"),
    make_test_case(append_emel_flash_attention_cases,
                   append_reference_flash_attention_cases,
                   "flash_attention"),
    make_test_case(append_emel_speech_codec_mimi_cases,
                   append_reference_speech_codec_mimi_cases,
                   "speech_codec_mimi"),
    make_test_case(append_emel_speech_lm_moshi_cases,
                   append_reference_speech_lm_moshi_cases,
                   "speech_lm_moshi"),
    make_test_case(append_emel_logits_validator_cases,
                   append_reference_logits_validator_cases,
                   "logits_validator"),
    make_test_case(append_emel_logits_sampler_cases,
                   append_reference_logits_sampler_cases,
                   "logits_sampler"),
    make_test_case(append_emel_kernel_x86_64_cases,
                   append_reference_kernel_x86_64_cases,
                   "kernel_x86_64"),
    make_test_case(append_emel_kernel_aarch64_cases,
                   append_reference_kernel_aarch64_cases,
                   "kernel_aarch64"),
    make_test_case(append_emel_sm_any_cases, append_reference_sm_any_cases, "sm_any"),
    make_test_case(append_emel_sm_scheduler_cases,
                   append_reference_sm_scheduler_cases,
                   "sm_scheduler"),
    make_test_case(append_emel_graph_processor_cases,
                   append_reference_graph_processor_cases,
                   "graph_processor"),
    make_test_case(append_emel_decode_wavefront_cases,
                   append_reference_decode_wavefront_cases,
                   "decode_wavefront"),
    make_test_case(append_emel_parallel_matmul_cases,
                   append_reference_parallel_matmul_cases,
                   "parallel_matmul"),
    make_test_case(append_emel_weight_streaming_cases,
                   append_reference_weight_streaming_cases,
                   "weight_streaming"),
    make_test_case(append_emel_tokenizer_preprocessor_bpe_cases,
                   append_reference_tokenizer_preprocessor_bpe_cases,
                   "tokenizer_preprocessor_bpe"),
    make_test_case(append_emel_tokenizer_preprocessor_spm_cases,
                   append_reference_tokenizer_preprocessor_spm_cases,
                   "tokenizer_preprocessor_spm"),
    make_test_case(append_emel_tokenizer_preprocessor_ugm_cases,
                   append_reference_tokenizer_preprocessor_ugm_cases,
                   "tokenizer_preprocessor_ugm"),
    make_test_case(append_emel_tokenizer_preprocessor_wpm_cases,
                   append_reference_tokenizer_preprocessor_wpm_cases,
                   "tokenizer_preprocessor_wpm"),
    make_test_case(append_emel_tokenizer_preprocessor_rwkv_cases,
                   append_reference_tokenizer_preprocessor_rwkv_cases,
                   "tokenizer_preprocessor_rwkv"),
    make_test_case(append_emel_tokenizer_preprocessor_plamo2_cases,
                   append_reference_tokenizer_preprocessor_plamo2_cases,
                   "tokenizer_preprocessor_plamo2"),
    make_test_case(append_emel_encoder_bpe_cases, append_reference_encoder_bpe_cases, "encoder_bpe"),
    make_test_case(append_emel_encoder_spm_cases, append_reference_encoder_spm_cases, "encoder_spm"),
    make_test_case(append_emel_encoder_wpm_cases, append_reference_encoder_wpm_cases, "encoder_wpm"),
    make_test_case(append_emel_encoder_ugm_cases, append_reference_encoder_ugm_cases, "encoder_ugm"),
    make_test_case(append_emel_encoder_rwkv_cases,
                   append_reference_encoder_rwkv_cases,
                   "encoder_rwkv"),
    make_test_case(append_emel_encoder_plamo2_cases,
                   append_reference_encoder_plamo2_cases,
                   "encoder_plamo2"),
    make_test_case(append_emel_encoder_fallback_cases,
                   append_reference_encoder_fallback_cases,
                   "encoder_fallback"),
    make_test_case(append_emel_tokenizer_cases,
                   append_reference_tokenizer_cases,
                   "tokenizer",
                   true),
  }};
  return cases;
}

const std::array<test_case, 2> & all_kernel_runner_cases() {
  static const std::array<test_case, 2> cases = {{
    make_test_case(append_emel_kernel_x86_64_cases,
                   append_reference_kernel_x86_64_cases,
                   "kernel_x86_64"),
    make_test_case(append_emel_kernel_aarch64_cases,
                   append_reference_kernel_aarch64_cases,
                   "kernel_aarch64"),
  }};
  return cases;
}

}  // namespace

std::span<const test_case> default_runner_cases() noexcept {
  const auto & cases = all_runner_cases();
  return {cases.data(), cases.size()};
}

std::span<const test_case> kernel_runner_cases() noexcept {
  const auto & cases = all_kernel_runner_cases();
  return {cases.data(), cases.size()};
}

std::size_t registered_runner_count() noexcept {
  return all_runner_cases().size();
}

std::string_view registered_runner_suite_at(const std::size_t index) noexcept {
  const auto & cases = all_runner_cases();
  if (index >= cases.size()) {
    return {};
  }
  return cases[index].suite;
}

const test_case * find_registered_runner(const std::string_view suite) noexcept {
  for (const auto & tc : all_runner_cases()) {
    if (tc.suite == suite) {
      return &tc;
    }
  }
  return nullptr;
}

}  // namespace emel::bench
