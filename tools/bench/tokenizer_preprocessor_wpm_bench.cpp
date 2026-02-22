#include "bench_cases.hpp"
#include "tokenizer_preprocessor_bench_common.hpp"

#include "emel/tokenizer/preprocessor/wpm/sm.hpp"

namespace emel::bench {

void append_emel_tokenizer_preprocessor_wpm_cases(std::vector<result> & results,
                                                  const config & cfg) {
  tokenizer_preprocessor::append_emel_special_preprocessor_cases<
      emel::tokenizer::preprocessor::wpm::sm>(
      results,
      cfg,
      "tokenizer/preprocessor_wpm_short",
      "tokenizer/preprocessor_wpm_long",
      emel::model::data::tokenizer_model::WPM,
      false,
      "wpm");
}

void append_reference_tokenizer_preprocessor_wpm_cases(std::vector<result> & results,
                                                       const config & cfg) {
  tokenizer_preprocessor::append_reference_special_preprocessor_cases(
      results,
      cfg,
      "tokenizer/preprocessor_wpm_short",
      "tokenizer/preprocessor_wpm_long",
      emel::model::data::tokenizer_model::WPM,
      false,
      "wpm");
}

}  // namespace emel::bench
