#include <array>
#include <cstddef>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/tokenizer/preprocessor/fallback/sm.hpp"
#include "emel/tokenizer/preprocessor/types.hpp"

namespace {

// Reference: ggml-org/llama.cpp @ 35715657cb2fa6eb302a2c1933ed10dbd0d8bc75
emel::model::data::vocab make_vocab_with_specials() {
  emel::model::data::vocab vocab = {};
  vocab.n_tokens = 2;
  vocab.entries[0].text_offset = 0;
  vocab.entries[0].text_length = 1;
  vocab.entries[0].type = 4;
  vocab.entries[1].text_offset = 2;
  vocab.entries[1].text_length = 3;
  vocab.entries[1].type = 3;
  vocab.token_storage[0] = 'A';
  vocab.token_storage[2] = 'B';
  vocab.token_storage[3] = 'B';
  vocab.token_storage[4] = 'B';
  return vocab;
}

emel::model::data::vocab make_bpe_vocab() {
  emel::model::data::vocab vocab = {};
  vocab.n_tokens = 0;
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  vocab.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  return vocab;
}

}  // namespace

TEST_CASE("tokenizer_preprocessor_fallback_identity_even_for_bpe_vocab") {
  emel::model::data::vocab vocab = make_bpe_vocab();

  std::array<emel::tokenizer::preprocessor::fragment,
             emel::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::tokenizer::preprocessor::fallback::sm machine{};
  emel::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view("hello world");
  ev.parse_special = false;
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = fragments.size();
  ev.fragment_count_out = &count;
  ev.error_out = &err;

  CHECK(machine.process_event(ev));
  CHECK(err == EMEL_OK);
  CHECK(count == 1);
  CHECK(fragments[0].kind ==
        emel::tokenizer::preprocessor::fragment_kind::raw_text);
  CHECK(fragments[0].text == std::string_view("hello world"));
}

TEST_CASE("tokenizer_preprocessor_fallback_parse_special_true") {
  emel::model::data::vocab vocab = make_vocab_with_specials();

  std::array<emel::tokenizer::preprocessor::fragment,
             emel::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::tokenizer::preprocessor::fallback::sm machine{};
  emel::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view("ABBB");
  ev.parse_special = true;
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = fragments.size();
  ev.fragment_count_out = &count;
  ev.error_out = &err;

  CHECK(machine.process_event(ev));
  CHECK(err == EMEL_OK);
  REQUIRE(count == 2);
  CHECK(fragments[0].kind ==
        emel::tokenizer::preprocessor::fragment_kind::token);
  CHECK(fragments[0].token == 0);
  CHECK(fragments[1].kind ==
        emel::tokenizer::preprocessor::fragment_kind::token);
  CHECK(fragments[1].token == 1);
}

TEST_CASE("tokenizer_preprocessor_fallback_parse_special_false") {
  emel::model::data::vocab vocab = make_vocab_with_specials();

  std::array<emel::tokenizer::preprocessor::fragment,
             emel::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::tokenizer::preprocessor::fallback::sm machine{};
  emel::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view("ABBB");
  ev.parse_special = false;
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = fragments.size();
  ev.fragment_count_out = &count;
  ev.error_out = &err;

  CHECK(machine.process_event(ev));
  CHECK(err == EMEL_OK);
  REQUIRE(count == 2);
  CHECK(fragments[0].kind ==
        emel::tokenizer::preprocessor::fragment_kind::token);
  CHECK(fragments[0].token == 0);
  CHECK(fragments[1].kind ==
        emel::tokenizer::preprocessor::fragment_kind::raw_text);
  CHECK(fragments[1].text == std::string_view("BBB"));
}
