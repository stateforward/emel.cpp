#include "bench_cases.hpp"
#include "bench_common.hpp"

#include <memory>
#include <vector>

#include "emel/model/data.hpp"
#include "emel/text/encoders/bpe/sm.hpp"

namespace {

std::unique_ptr<emel::model::data::vocab> make_bpe_vocab() {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  vocab->tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  vocab->tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  vocab->ignore_merges = true;
  emel::bench::encoder_bench::add_all_byte_tokens(*vocab);
  (void)emel::bench::encoder_bench::add_token(*vocab, "hello", 0.5f, 1);
  (void)emel::bench::encoder_bench::add_token(*vocab, " world", 0.5f, 1);
  return vocab;
}

}  // namespace

namespace emel::bench {

void append_emel_encoder_bpe_cases(std::vector<result> & results, const config & cfg) {
  encoder_bench::append_emel_encoder_cases_with_text<emel::text::encoders::bpe::sm>(
    results,
    cfg,
    "text/encoders/bpe_short",
    "text/encoders/bpe_long",
    make_bpe_vocab,
    true,
    "hello",
    " world");
}

void append_reference_encoder_bpe_cases(std::vector<result> & results, const config & cfg) {
  // Reference encoder benchmarks reuse the EMEL path until llama.cpp parity is wired.
  append_emel_encoder_bpe_cases(results, cfg);
}

}  // namespace emel::bench
