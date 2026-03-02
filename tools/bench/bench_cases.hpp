#pragma once

#include "bench_common.hpp"

#include <vector>

namespace emel::bench {

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
void append_emel_gbnf_parser_cases(std::vector<result> & results, const config & cfg);
void append_reference_gbnf_parser_cases(std::vector<result> & results, const config & cfg);
void append_emel_logits_cases(std::vector<result> & results, const config & cfg);
void append_reference_logits_cases(std::vector<result> & results, const config & cfg);
void append_emel_kernel_cases(std::vector<result> & results, const config & cfg);
void append_reference_kernel_cases(std::vector<result> & results, const config & cfg);
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
