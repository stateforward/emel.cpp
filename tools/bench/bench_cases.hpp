#pragma once

#include "bench_common.hpp"

#include <string_view>
#include <vector>

namespace emel::bench {

inline constexpr std::string_view k_generation_case_name =
  "generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1";

using append_case_fn = void (*)(std::vector<result> & results, const config & cfg);

struct test_case {
  append_case_fn append_emel = nullptr;
  append_case_fn append_reference = nullptr;
  bool tokenizer_case = false;
};

inline void append_test_case(std::vector<result> & results,
                             const config & cfg,
                             const test_case & tc,
                             const bool reference) {
  const append_case_fn fn = reference ? tc.append_reference : tc.append_emel;
  if (fn != nullptr) {
    fn(results, cfg);
  }
}

void append_emel_batch_planner_cases(std::vector<result> & results, const config & cfg);
void append_reference_batch_planner_cases(std::vector<result> & results, const config & cfg);
void append_emel_memory_kv_cases(std::vector<result> & results, const config & cfg);
void append_reference_memory_kv_cases(std::vector<result> & results, const config & cfg);
void append_emel_memory_recurrent_cases(std::vector<result> & results, const config & cfg);
void append_reference_memory_recurrent_cases(std::vector<result> & results, const config & cfg);
void append_emel_memory_hybrid_cases(std::vector<result> & results, const config & cfg);
void append_reference_memory_hybrid_cases(std::vector<result> & results, const config & cfg);
void append_emel_jinja_parser_cases(std::vector<result> & results, const config & cfg);
void append_reference_jinja_parser_cases(std::vector<result> & results, const config & cfg);
void append_emel_jinja_formatter_cases(std::vector<result> & results, const config & cfg);
void append_reference_jinja_formatter_cases(std::vector<result> & results, const config & cfg);
void append_emel_gbnf_rule_parser_cases(std::vector<result> & results, const config & cfg);
void append_reference_gbnf_rule_parser_cases(std::vector<result> & results, const config & cfg);
void append_emel_generation_cases(std::vector<result> & results, const config & cfg);
void append_reference_generation_cases(std::vector<result> & results, const config & cfg);
void append_emel_logits_validator_cases(std::vector<result> & results, const config & cfg);
void append_reference_logits_validator_cases(std::vector<result> & results, const config & cfg);
void append_emel_logits_sampler_cases(std::vector<result> & results, const config & cfg);
void append_reference_logits_sampler_cases(std::vector<result> & results, const config & cfg);
void append_emel_kernel_x86_64_cases(std::vector<result> & results, const config & cfg);
void append_reference_kernel_x86_64_cases(std::vector<result> & results, const config & cfg);
void append_emel_kernel_aarch64_cases(std::vector<result> & results, const config & cfg);
void append_reference_kernel_aarch64_cases(std::vector<result> & results, const config & cfg);
void append_emel_sm_any_cases(std::vector<result> & results, const config & cfg);
void append_reference_sm_any_cases(std::vector<result> & results, const config & cfg);
void append_emel_tokenizer_preprocessor_bpe_cases(std::vector<result> & results,
                                                  const config & cfg);
void append_reference_tokenizer_preprocessor_bpe_cases(std::vector<result> & results,
                                                       const config & cfg);
void append_emel_tokenizer_preprocessor_spm_cases(std::vector<result> & results,
                                                  const config & cfg);
void append_reference_tokenizer_preprocessor_spm_cases(std::vector<result> & results,
                                                       const config & cfg);
void append_emel_tokenizer_preprocessor_ugm_cases(std::vector<result> & results,
                                                  const config & cfg);
void append_reference_tokenizer_preprocessor_ugm_cases(std::vector<result> & results,
                                                       const config & cfg);
void append_emel_tokenizer_preprocessor_wpm_cases(std::vector<result> & results,
                                                  const config & cfg);
void append_reference_tokenizer_preprocessor_wpm_cases(std::vector<result> & results,
                                                       const config & cfg);
void append_emel_tokenizer_preprocessor_rwkv_cases(std::vector<result> & results,
                                                   const config & cfg);
void append_reference_tokenizer_preprocessor_rwkv_cases(std::vector<result> & results,
                                                        const config & cfg);
void append_emel_tokenizer_preprocessor_plamo2_cases(std::vector<result> & results,
                                                     const config & cfg);
void append_reference_tokenizer_preprocessor_plamo2_cases(std::vector<result> & results,
                                                          const config & cfg);
void append_emel_encoder_bpe_cases(std::vector<result> & results, const config & cfg);
void append_reference_encoder_bpe_cases(std::vector<result> & results, const config & cfg);
void append_emel_encoder_spm_cases(std::vector<result> & results, const config & cfg);
void append_reference_encoder_spm_cases(std::vector<result> & results, const config & cfg);
void append_emel_encoder_wpm_cases(std::vector<result> & results, const config & cfg);
void append_reference_encoder_wpm_cases(std::vector<result> & results, const config & cfg);
void append_emel_encoder_ugm_cases(std::vector<result> & results, const config & cfg);
void append_reference_encoder_ugm_cases(std::vector<result> & results, const config & cfg);
void append_emel_encoder_rwkv_cases(std::vector<result> & results, const config & cfg);
void append_reference_encoder_rwkv_cases(std::vector<result> & results, const config & cfg);
void append_emel_encoder_plamo2_cases(std::vector<result> & results, const config & cfg);
void append_reference_encoder_plamo2_cases(std::vector<result> & results, const config & cfg);
void append_emel_encoder_fallback_cases(std::vector<result> & results, const config & cfg);
void append_reference_encoder_fallback_cases(std::vector<result> & results, const config & cfg);
void append_emel_tokenizer_cases(std::vector<result> & results, const config & cfg);
void append_reference_tokenizer_cases(std::vector<result> & results, const config & cfg);

}  // namespace emel::bench
