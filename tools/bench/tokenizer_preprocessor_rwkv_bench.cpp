#include "bench_cases.hpp"
#include "tokenizer_preprocessor_bench_common.hpp"

#include "emel/tokenizer/preprocessor/rwkv/sm.hpp"

namespace emel::bench {

void append_emel_tokenizer_preprocessor_rwkv_cases(std::vector<result> & results,
                                                   const config & cfg) {
  tokenizer_preprocessor::append_emel_special_preprocessor_cases<
      emel::tokenizer::preprocessor::rwkv::sm>(
      results,
      cfg,
      "tokenizer/preprocessor_rwkv_short",
      "tokenizer/preprocessor_rwkv_long",
      emel::model::data::tokenizer_model::RWKV,
      false,
      "rwkv");
}

void append_reference_tokenizer_preprocessor_rwkv_cases(std::vector<result> & results,
                                                        const config & cfg) {
  tokenizer_preprocessor::append_reference_special_preprocessor_cases(
      results,
      cfg,
      "tokenizer/preprocessor_rwkv_short",
      "tokenizer/preprocessor_rwkv_long",
      emel::model::data::tokenizer_model::RWKV,
      false,
      "rwkv");
}

}  // namespace emel::bench
