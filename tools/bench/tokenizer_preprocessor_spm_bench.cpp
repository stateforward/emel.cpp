#include "bench_cases.hpp"
#include "tokenizer_preprocessor_bench_common.hpp"

#include "emel/tokenizer/preprocessor/spm/sm.hpp"

namespace emel::bench {

void append_emel_tokenizer_preprocessor_spm_cases(std::vector<result> & results,
                                                  const config & cfg) {
  tokenizer_preprocessor::append_emel_special_preprocessor_cases<
      emel::tokenizer::preprocessor::spm::sm>(
      results,
      cfg,
      "tokenizer/preprocessor_spm_short",
      "tokenizer/preprocessor_spm_long",
      emel::model::data::tokenizer_model::SPM,
      false,
      "spm");
}

void append_reference_tokenizer_preprocessor_spm_cases(std::vector<result> & results,
                                                       const config & cfg) {
  tokenizer_preprocessor::append_reference_special_preprocessor_cases(
      results,
      cfg,
      "tokenizer/preprocessor_spm_short",
      "tokenizer/preprocessor_spm_long",
      emel::model::data::tokenizer_model::SPM,
      false,
      "spm");
}

}  // namespace emel::bench
