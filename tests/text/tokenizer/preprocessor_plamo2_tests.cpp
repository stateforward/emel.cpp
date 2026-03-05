#include <array>
#include <cstddef>
#include <cstring>
#include <span>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
#include "emel/text/tokenizer/preprocessor/plamo2/sm.hpp"
#include "emel/text/tokenizer/preprocessor/types.hpp"

namespace {

emel::model::data::vocab & make_plamo2_vocab_with_specials() {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.n_tokens = 2;
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::PLAMO2;
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

TEST_CASE("tokenizer_preprocessor_plamo2_valid_request") {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.n_tokens = 0;
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::PLAMO2;

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = emel::text::tokenizer::preprocessor::error_code(emel::text::tokenizer::preprocessor::error::none);

  emel::text::tokenizer::preprocessor::plamo2::sm machine{};
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

TEST_CASE("tokenizer_preprocessor_plamo2_parse_special_true") {
  auto & vocab = make_plamo2_vocab_with_specials();

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = emel::text::tokenizer::preprocessor::error_code(emel::text::tokenizer::preprocessor::error::none);

  emel::text::tokenizer::preprocessor::plamo2::sm machine{};
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

TEST_CASE("tokenizer_preprocessor_plamo2_parse_special_false") {
  auto & vocab = make_plamo2_vocab_with_specials();

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = emel::text::tokenizer::preprocessor::error_code(emel::text::tokenizer::preprocessor::error::none);

  emel::text::tokenizer::preprocessor::plamo2::sm machine{};
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

TEST_CASE("tokenizer_preprocessor_plamo2_phase_result_guards") {
  using emel::text::tokenizer::preprocessor::error;
  using emel::text::tokenizer::preprocessor::event::preprocess;
  using emel::text::tokenizer::preprocessor::event::preprocess_ctx;
  using emel::text::tokenizer::preprocessor::event::preprocess_runtime;

  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::PLAMO2;

  std::array<emel::text::tokenizer::preprocessor::fragment, 1> fragments = {};
  size_t count = 0;
  int32_t err = 0;
  preprocess request(vocab, std::string_view("x"), false,
                     std::span<emel::text::tokenizer::preprocessor::fragment>(fragments),
                     count, err);
  preprocess_ctx ctx{};
  preprocess_runtime runtime_ev{request, ctx};
  emel::text::tokenizer::preprocessor::action::context sm_ctx{};

  ctx.phase_error = error::none;
  CHECK(emel::text::tokenizer::preprocessor::plamo2::guard::build_specials_ok{}(
      runtime_ev, sm_ctx));
  CHECK(emel::text::tokenizer::preprocessor::plamo2::guard::partition_ok{}(
      runtime_ev, sm_ctx));

  ctx.phase_error = error::invalid_request;
  CHECK(emel::text::tokenizer::preprocessor::plamo2::guard::build_specials_invalid_request_error{}(
      runtime_ev, sm_ctx));
  CHECK(emel::text::tokenizer::preprocessor::plamo2::guard::partition_invalid_request_error{}(
      runtime_ev, sm_ctx));

  ctx.phase_error = error::backend_error;
  CHECK(emel::text::tokenizer::preprocessor::plamo2::guard::build_specials_backend_error{}(
      runtime_ev, sm_ctx));
  CHECK(emel::text::tokenizer::preprocessor::plamo2::guard::partition_backend_error{}(
      runtime_ev, sm_ctx));

  ctx.phase_error = static_cast<error>(0xFF);
  CHECK(emel::text::tokenizer::preprocessor::plamo2::guard::build_specials_unknown_error{}(
      runtime_ev, sm_ctx));
  CHECK(emel::text::tokenizer::preprocessor::plamo2::guard::partition_unknown_error{}(
      runtime_ev, sm_ctx));
}
