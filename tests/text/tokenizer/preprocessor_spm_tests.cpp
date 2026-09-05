#include <array>
#include <cstddef>
#include <cstring>
#include <span>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/tokenizer/preprocessor/spm/sm.hpp"
#include "emel/text/tokenizer/preprocessor/types.hpp"

namespace {

emel::model::data::vocab & make_spm_vocab_with_specials() {
  static emel::model::data::vocab vocab = {};
  vocab = {};
  vocab.n_tokens = 2;
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::SPM;
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

}  // namespace

TEST_CASE("tokenizer_preprocessor_spm_valid_request") {
  static emel::model::data::vocab vocab = {};
  vocab = {};
  vocab.n_tokens = 0;
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::SPM;

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = emel::text::tokenizer::preprocessor::error_code(emel::text::tokenizer::preprocessor::error::none);

  emel::text::tokenizer::preprocessor::spm::sm machine{};
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("hello"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count,
      err);

  CHECK(machine.process_event(ev));
  CHECK(err == emel::text::tokenizer::preprocessor::error_code(emel::text::tokenizer::preprocessor::error::none));
  CHECK(count == 1);
  CHECK(fragments[0].kind ==
        emel::text::tokenizer::preprocessor::fragment_kind::raw_text);
  CHECK(fragments[0].text == std::string_view("hello"));
}

TEST_CASE("tokenizer_preprocessor_spm_parse_special_true") {
  auto & vocab = make_spm_vocab_with_specials();

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = emel::text::tokenizer::preprocessor::error_code(emel::text::tokenizer::preprocessor::error::none);

  emel::text::tokenizer::preprocessor::spm::sm machine{};
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("ABBB"), true,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count,
      err);

  CHECK(machine.process_event(ev));
  CHECK(err == emel::text::tokenizer::preprocessor::error_code(emel::text::tokenizer::preprocessor::error::none));
  REQUIRE(count == 2);
  CHECK(fragments[0].kind ==
        emel::text::tokenizer::preprocessor::fragment_kind::token);
  CHECK(fragments[0].token == 0);
  CHECK(fragments[1].kind ==
        emel::text::tokenizer::preprocessor::fragment_kind::token);
  CHECK(fragments[1].token == 1);
}

TEST_CASE("tokenizer_preprocessor_spm_parse_special_false") {
  auto & vocab = make_spm_vocab_with_specials();

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = emel::text::tokenizer::preprocessor::error_code(emel::text::tokenizer::preprocessor::error::none);

  emel::text::tokenizer::preprocessor::spm::sm machine{};
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("ABBB"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count,
      err);

  CHECK(machine.process_event(ev));
  CHECK(err == emel::text::tokenizer::preprocessor::error_code(emel::text::tokenizer::preprocessor::error::none));
  REQUIRE(count == 2);
  CHECK(fragments[0].kind ==
        emel::text::tokenizer::preprocessor::fragment_kind::token);
  CHECK(fragments[0].token == 0);
  CHECK(fragments[1].kind ==
        emel::text::tokenizer::preprocessor::fragment_kind::raw_text);
  CHECK(fragments[1].text == std::string_view("BBB"));
}

TEST_CASE("tokenizer_preprocessor_spm_needle_inserts_dummy_prefix") {
  auto & vocab = make_spm_vocab_with_specials();
  vocab.tokenizer_pre_id = emel::model::data::tokenizer_pre::NEEDLE;
  vocab.add_space_prefix = true;
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = -1;
  emel::text::tokenizer::preprocessor::spm::sm machine{};
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, "ABBB", true,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments),
      count, err);

  REQUIRE(machine.process_event(ev));
  REQUIRE(count == 3);
  CHECK(fragments[0].kind ==
        emel::text::tokenizer::preprocessor::fragment_kind::raw_text);
  CHECK(fragments[0].text == " ");
  CHECK(fragments[1].token == 0);
  CHECK(fragments[2].token == 1);
}

TEST_CASE("tokenizer_preprocessor_spm_standard_route_does_not_insert_prefix") {
  auto & vocab = make_spm_vocab_with_specials();
  vocab.add_space_prefix = true;
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = -1;
  emel::text::tokenizer::preprocessor::spm::sm machine{};
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, "ABBB", true,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments),
      count, err);

  REQUIRE(machine.process_event(ev));
  REQUIRE(count == 2);
  CHECK(fragments[0].kind ==
        emel::text::tokenizer::preprocessor::fragment_kind::token);
  CHECK(fragments[0].token == 0);
}

TEST_CASE("tokenizer_preprocessor_spm_needle_prefix_rejects_insufficient_capacity") {
  auto & vocab = make_spm_vocab_with_specials();
  vocab.tokenizer_pre_id = emel::model::data::tokenizer_pre::NEEDLE;
  vocab.add_space_prefix = true;
  std::array<emel::text::tokenizer::preprocessor::fragment, 2> fragments = {};
  size_t count = 99;
  int32_t err = -1;
  emel::text::tokenizer::preprocessor::spm::sm machine{};
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, "ABBB", true,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments),
      count, err);

  CHECK_FALSE(machine.process_event(ev));
  CHECK(count == 0);
  CHECK(err == emel::text::tokenizer::preprocessor::error_code(
                     emel::text::tokenizer::preprocessor::error::invalid_request));
}

TEST_CASE("tokenizer_preprocessor_spm_empty_input_and_recovery") {
  auto & vocab = make_spm_vocab_with_specials();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  emel::text::tokenizer::preprocessor::spm::sm machine{};

  size_t count = 99;
  bool preprocessed = false;
  int32_t err = -1;
  emel::text::tokenizer::preprocessor::event::preprocess empty(
      vocab, "", true,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count, err);
  empty.preprocessed_out = &preprocessed;
  CHECK(machine.process_event(empty));
  CHECK(count == 0);
  CHECK(preprocessed);

  std::array<emel::text::tokenizer::preprocessor::fragment, 1> one = {};
  count = 99;
  err = -1;
  emel::text::tokenizer::preprocessor::event::preprocess invalid(
      vocab, "x", false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(one.data(), 0), count, err);
  CHECK_FALSE(machine.process_event(invalid));
  CHECK(count == 0);

  count = 0;
  err = -1;
  emel::text::tokenizer::preprocessor::event::preprocess recovered(
      vocab, "hello", false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count, err);
  CHECK(machine.process_event(recovered));
  REQUIRE(count == 1);
  CHECK(fragments[0].text == "hello");
}

TEST_CASE("tokenizer_preprocessor_spm_rejects_oversized_fragment_span") {
  auto & vocab = make_spm_vocab_with_specials();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments + 1>
      fragments = {};
  size_t count = 99;
  int32_t err = -1;
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, "hello", false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count, err);
  emel::text::tokenizer::preprocessor::spm::sm machine{};

  CHECK_FALSE(machine.process_event(ev));
  CHECK(count == 0);
  CHECK(err == emel::text::tokenizer::preprocessor::error_code(
                   emel::text::tokenizer::preprocessor::error::invalid_request));
}
