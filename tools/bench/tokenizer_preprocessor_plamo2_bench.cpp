#include "bench_cases.hpp"
#include "tokenizer_preprocessor_bench_common.hpp"

#include "emel/tokenizer/preprocessor/plamo2/sm.hpp"

namespace emel::bench {

void append_emel_tokenizer_preprocessor_plamo2_cases(std::vector<result> & results,
                                                     const config & cfg) {
  tokenizer_preprocessor::append_emel_special_preprocessor_cases<
      emel::tokenizer::preprocessor::plamo2::sm>(
      results,
      cfg,
      "tokenizer/preprocessor_plamo2_short",
      "tokenizer/preprocessor_plamo2_long",
      emel::model::data::tokenizer_model::PLAMO2,
      false,
      "plamo2");
}

void append_reference_tokenizer_preprocessor_plamo2_cases(std::vector<result> & results,
                                                          const config & cfg) {
  tokenizer_preprocessor::append_reference_special_preprocessor_cases(
      results,
      cfg,
      "tokenizer/preprocessor_plamo2_short",
      "tokenizer/preprocessor_plamo2_long",
      emel::model::data::tokenizer_model::PLAMO2,
      false,
      "plamo2");
}

}  // namespace emel::bench
