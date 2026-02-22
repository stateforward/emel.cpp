#pragma once

#include "bench_common.hpp"

#include <vector>

namespace emel::bench {

void append_emel_buffer_allocator_cases(std::vector<result> & results, const config & cfg);
void append_reference_buffer_allocator_cases(std::vector<result> & results, const config & cfg);
void append_emel_batch_splitter_cases(std::vector<result> & results, const config & cfg);
void append_reference_batch_splitter_cases(std::vector<result> & results, const config & cfg);
void append_emel_memory_coordinator_recurrent_cases(std::vector<result> & results,
                                                    const config & cfg);
void append_reference_memory_coordinator_recurrent_cases(std::vector<result> & results,
                                                         const config & cfg);
void append_emel_jinja_parser_cases(std::vector<result> & results, const config & cfg);
void append_reference_jinja_parser_cases(std::vector<result> & results, const config & cfg);
void append_emel_jinja_renderer_cases(std::vector<result> & results, const config & cfg);
void append_reference_jinja_renderer_cases(std::vector<result> & results, const config & cfg);
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
void append_emel_tokenizer_cases(std::vector<result> & results, const config & cfg);
void append_reference_tokenizer_cases(std::vector<result> & results, const config & cfg);

}  // namespace emel::bench
