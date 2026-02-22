#include "bench_cases.hpp"
#include "tokenizer_preprocessor_bench_common.hpp"

#include "emel/tokenizer/preprocessor/ugm/sm.hpp"

namespace emel::bench {

void append_emel_tokenizer_preprocessor_ugm_cases(std::vector<result> & results,
                                                  const config & cfg) {
  tokenizer_preprocessor::append_emel_special_preprocessor_cases<
      emel::tokenizer::preprocessor::ugm::sm>(
      results,
      cfg,
      "tokenizer/preprocessor_ugm_short",
      "tokenizer/preprocessor_ugm_long",
      emel::model::data::tokenizer_model::UGM,
      true,
      "ugm");
}

void append_reference_tokenizer_preprocessor_ugm_cases(std::vector<result> & results,
                                                       const config & cfg) {
  tokenizer_preprocessor::append_reference_special_preprocessor_cases(
      results,
      cfg,
      "tokenizer/preprocessor_ugm_short",
      "tokenizer/preprocessor_ugm_long",
      emel::model::data::tokenizer_model::UGM,
      true,
      "ugm");
}

}  // namespace emel::bench
