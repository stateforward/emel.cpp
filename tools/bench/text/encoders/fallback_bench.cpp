#include "bench_cases.hpp"
#include "bench_common.hpp"

#include <memory>
#include <vector>

#include "emel/model/data.hpp"
#include "emel/text/encoders/fallback/sm.hpp"

namespace {

std::unique_ptr<emel::model::data::vocab> make_fallback_vocab() {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->tokenizer_model_id = emel::model::data::tokenizer_model::UNKNOWN;
  emel::bench::encoder_bench::add_all_raw_byte_tokens(*vocab);
  return vocab;
}

}  // namespace

namespace emel::bench {

void append_emel_encoder_fallback_cases(std::vector<result> & results, const config & cfg) {
  encoder_bench::append_emel_encoder_cases<emel::text::encoders::fallback::sm>(
    results,
    cfg,
    "text/encoders/fallback_short",
    "text/encoders/fallback_long",
    make_fallback_vocab,
    false);
}

void append_reference_encoder_fallback_cases(std::vector<result> & results, const config & cfg) {
  // Reference encoder benchmarks reuse the EMEL path until llama.cpp parity is wired.
  append_emel_encoder_fallback_cases(results, cfg);
}

}  // namespace emel::bench
