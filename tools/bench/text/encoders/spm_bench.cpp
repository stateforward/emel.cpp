#include "bench_cases.hpp"
#include "bench_common.hpp"

#include <memory>
#include <vector>

#include "emel/model/data.hpp"
#include "emel/text/encoders/spm/sm.hpp"

namespace {

std::unique_ptr<emel::model::data::vocab> make_spm_vocab() {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->tokenizer_model_id = emel::model::data::tokenizer_model::SPM;
  vocab->add_space_prefix = true;
  emel::bench::encoder_bench::add_all_plamo2_byte_tokens(*vocab);
  (void)emel::bench::encoder_bench::add_token(*vocab, "\xE2\x96\x81" "hello", 0.5f, 1);
  (void)emel::bench::encoder_bench::add_token(*vocab, "\xE2\x96\x81" "world", 0.5f, 1);
  (void)emel::bench::encoder_bench::add_token(*vocab, "\xE2\x96\x81", 0.1f, 1);
  return vocab;
}

}  // namespace

namespace emel::bench {

void append_emel_encoder_spm_cases(std::vector<result> & results, const config & cfg) {
  encoder_bench::append_emel_encoder_cases<emel::text::encoders::spm::sm>(
    results, cfg, "text/encoders/spm_short", "text/encoders/spm_long", make_spm_vocab, false);
}

void append_reference_encoder_spm_cases(std::vector<result> & results, const config & cfg) {
  // Reference encoder benchmarks reuse the EMEL path until llama.cpp parity is wired.
  append_emel_encoder_spm_cases(results, cfg);
}

}  // namespace emel::bench
