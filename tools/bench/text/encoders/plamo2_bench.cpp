#include "bench_cases.hpp"
#include "bench_common.hpp"

#include <memory>
#include <vector>

#include "emel/model/data.hpp"
#include "emel/text/encoders/plamo2/sm.hpp"

namespace {

std::unique_ptr<emel::model::data::vocab> make_plamo2_vocab() {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->tokenizer_model_id = emel::model::data::tokenizer_model::PLAMO2;
  // Keep id=0 non-byte so parsed byte ids are all non-zero in the table-complete check.
  (void)emel::bench::encoder_bench::add_token(*vocab, "", 0.0f, 1);
  emel::bench::encoder_bench::add_all_plamo2_byte_tokens(*vocab);
  return vocab;
}

}  // namespace

namespace emel::bench {

void append_emel_encoder_plamo2_cases(std::vector<result> & results, const config & cfg) {
  encoder_bench::append_emel_encoder_cases<emel::text::encoders::plamo2::sm>(
    results,
    cfg,
    "text/encoders/plamo2_short",
    "text/encoders/plamo2_long",
    make_plamo2_vocab,
    false);
}

void append_reference_encoder_plamo2_cases(std::vector<result> & results, const config & cfg) {
  // Reference encoder benchmarks reuse the EMEL path until llama.cpp parity is wired.
  append_emel_encoder_plamo2_cases(results, cfg);
}

}  // namespace emel::bench
