#include "bench_cases.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace emel::bench {

#ifndef EMEL_BENCH_ENABLE_BATCH_PLANNER
void append_emel_batch_planner_cases(std::vector<result> &, const config &) {}
void append_reference_batch_planner_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_MEMORY_KV
void append_emel_memory_kv_cases(std::vector<result> &, const config &) {}
void append_reference_memory_kv_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_MEMORY_RECURRENT
void append_emel_memory_recurrent_cases(std::vector<result> &, const config &) {}
void append_reference_memory_recurrent_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_MEMORY_HYBRID
void append_emel_memory_hybrid_cases(std::vector<result> &, const config &) {}
void append_reference_memory_hybrid_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_JINJA_PARSER
void append_emel_jinja_parser_cases(std::vector<result> &, const config &) {}
void append_reference_jinja_parser_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_JINJA_FORMATTER
void append_emel_jinja_formatter_cases(std::vector<result> &, const config &) {}
void append_reference_jinja_formatter_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_GBNF_RULE_PARSER
void append_emel_gbnf_rule_parser_cases(std::vector<result> &, const config &) {}
void append_reference_gbnf_rule_parser_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_GENERATION
void set_generation_lane_mode(generation_lane_mode) noexcept {}
generation_lane_mode generation_lane_mode_current() noexcept {
  return generation_lane_mode::emel;
}

std::string_view generation_formatter_contract() noexcept {
  return {};
}

std::string_view generation_architecture_contract() noexcept {
  return {};
}

bool generation_flash_evidence_ready() noexcept {
  return false;
}

std::uint64_t generation_flash_evidence_dispatch_calls() noexcept {
  return 0u;
}

std::uint64_t generation_flash_evidence_optimized_dispatch_calls() noexcept {
  return 0u;
}

std::uint64_t generation_flash_evidence_shared_dispatch_calls() noexcept {
  return 0u;
}

std::uint32_t generation_runtime_contract_native_quantized_stage_count() noexcept {
  return 0u;
}

std::uint32_t generation_runtime_contract_approved_dense_f32_stage_count() noexcept {
  return 0u;
}

std::uint32_t generation_runtime_contract_disallowed_fallback_stage_count() noexcept {
  return 0u;
}

std::uint32_t generation_runtime_contract_explicit_no_claim_stage_count() noexcept {
  return 0u;
}

std::uint64_t generation_quantized_evidence_native_q8_0_dispatch_calls() noexcept {
  return 0u;
}

std::uint64_t generation_quantized_evidence_packed_q8_0_dispatch_calls() noexcept {
  return 0u;
}

std::uint64_t generation_quantized_evidence_optimized_q2_dispatch_calls() noexcept {
  return 0u;
}

std::uint64_t generation_quantized_evidence_shared_q2_dispatch_calls() noexcept {
  return 0u;
}

std::uint64_t generation_quantized_evidence_optimized_q3_dispatch_calls() noexcept {
  return 0u;
}

std::uint64_t generation_quantized_evidence_shared_q3_dispatch_calls() noexcept {
  return 0u;
}

std::uint64_t generation_quantized_evidence_optimized_q4_dispatch_calls() noexcept {
  return 0u;
}

std::uint64_t generation_quantized_evidence_shared_q4_dispatch_calls() noexcept {
  return 0u;
}

std::uint64_t generation_quantized_evidence_optimized_q6_dispatch_calls() noexcept {
  return 0u;
}

std::uint64_t generation_quantized_evidence_shared_q6_dispatch_calls() noexcept {
  return 0u;
}

std::int32_t generation_flash_evidence_emel_decode_calls() noexcept {
  return 0;
}

std::int32_t generation_flash_evidence_emel_logits_calls() noexcept {
  return 0;
}

std::int32_t generation_flash_evidence_reference_decode_calls() noexcept {
  return 0;
}

std::int32_t generation_flash_evidence_reference_logits_calls() noexcept {
  return 0;
}

std::size_t generation_stage_probe_count() noexcept {
  return 0u;
}

generation_stage_probe generation_stage_probe_at(std::size_t) noexcept {
  return {};
}

void append_emel_generation_cases(std::vector<result> &, const config &) {}
void append_reference_generation_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_DIARIZATION_SORTFORMER
void append_emel_sortformer_diarization_cases(std::vector<result> &, const config &) {}
void append_reference_sortformer_diarization_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_FLASH_ATTENTION
void append_emel_flash_attention_cases(std::vector<result> &, const config &) {}
void append_reference_flash_attention_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_LOGITS_VALIDATOR
void append_emel_logits_validator_cases(std::vector<result> &, const config &) {}
void append_reference_logits_validator_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_LOGITS_SAMPLER
void append_emel_logits_sampler_cases(std::vector<result> &, const config &) {}
void append_reference_logits_sampler_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_KERNEL_X86_64
void append_emel_kernel_x86_64_cases(std::vector<result> &, const config &) {}
void append_reference_kernel_x86_64_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_KERNEL_AARCH64
void append_emel_kernel_aarch64_cases(std::vector<result> &, const config &) {}
void append_reference_kernel_aarch64_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_SM_ANY
void append_emel_sm_any_cases(std::vector<result> &, const config &) {}
void append_reference_sm_any_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_TOKENIZER_PREPROCESSOR_BPE
void append_emel_tokenizer_preprocessor_bpe_cases(std::vector<result> &, const config &) {}
void append_reference_tokenizer_preprocessor_bpe_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_TOKENIZER_PREPROCESSOR_SPM
void append_emel_tokenizer_preprocessor_spm_cases(std::vector<result> &, const config &) {}
void append_reference_tokenizer_preprocessor_spm_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_TOKENIZER_PREPROCESSOR_UGM
void append_emel_tokenizer_preprocessor_ugm_cases(std::vector<result> &, const config &) {}
void append_reference_tokenizer_preprocessor_ugm_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_TOKENIZER_PREPROCESSOR_WPM
void append_emel_tokenizer_preprocessor_wpm_cases(std::vector<result> &, const config &) {}
void append_reference_tokenizer_preprocessor_wpm_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_TOKENIZER_PREPROCESSOR_RWKV
void append_emel_tokenizer_preprocessor_rwkv_cases(std::vector<result> &, const config &) {}
void append_reference_tokenizer_preprocessor_rwkv_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_TOKENIZER_PREPROCESSOR_PLAMO2
void append_emel_tokenizer_preprocessor_plamo2_cases(std::vector<result> &, const config &) {}
void append_reference_tokenizer_preprocessor_plamo2_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_ENCODER_BPE
void append_emel_encoder_bpe_cases(std::vector<result> &, const config &) {}
void append_reference_encoder_bpe_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_ENCODER_SPM
void append_emel_encoder_spm_cases(std::vector<result> &, const config &) {}
void append_reference_encoder_spm_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_ENCODER_WPM
void append_emel_encoder_wpm_cases(std::vector<result> &, const config &) {}
void append_reference_encoder_wpm_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_ENCODER_UGM
void append_emel_encoder_ugm_cases(std::vector<result> &, const config &) {}
void append_reference_encoder_ugm_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_ENCODER_RWKV
void append_emel_encoder_rwkv_cases(std::vector<result> &, const config &) {}
void append_reference_encoder_rwkv_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_ENCODER_PLAMO2
void append_emel_encoder_plamo2_cases(std::vector<result> &, const config &) {}
void append_reference_encoder_plamo2_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_ENCODER_FALLBACK
void append_emel_encoder_fallback_cases(std::vector<result> &, const config &) {}
void append_reference_encoder_fallback_cases(std::vector<result> &, const config &) {}
#endif

#ifndef EMEL_BENCH_ENABLE_TOKENIZER
void append_emel_tokenizer_cases(std::vector<result> &, const config &) {}
void append_reference_tokenizer_cases(std::vector<result> &, const config &) {}
#endif

}  // namespace emel::bench
