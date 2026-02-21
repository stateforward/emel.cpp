#include <array>
#include <cstddef>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/tokenizer/preprocessor/any.hpp"
#include "emel/tokenizer/preprocessor/actions.hpp"
#include "emel/tokenizer/preprocessor/detail.hpp"

namespace {

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
  vocab.lstrip_flags[0] = 0x01;
  vocab.rstrip_flags[0] = 0x01;
  return vocab;
}

TEST_CASE("tokenizer_preprocessor_any_valid_request") {
  emel::model::data::vocab vocab = {};
  vocab.n_tokens = 0;

  std::array<emel::tokenizer::preprocessor::fragment,
             emel::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::tokenizer::preprocessor::any machine(
      emel::tokenizer::preprocessor::preprocessor_kind::fallback);
  emel::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view("hello");
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
  CHECK(fragments[0].text == std::string_view("hello"));
}

TEST_CASE("tokenizer_preprocessor_any_invalid_request") {
  std::array<emel::tokenizer::preprocessor::fragment,
             emel::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::tokenizer::preprocessor::any machine(
      emel::tokenizer::preprocessor::preprocessor_kind::fallback);
  emel::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = nullptr;
  ev.text = std::string_view("hello");
  ev.parse_special = false;
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = fragments.size();
  ev.fragment_count_out = &count;
  ev.error_out = &err;

  CHECK_FALSE(machine.process_event(ev));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(count == 0);
}

TEST_CASE("tokenizer_preprocessor_build_special_tokens") {
  emel::model::data::vocab vocab = make_vocab_with_specials();
  emel::tokenizer::preprocessor::special_token_cache cache = {};

  CHECK(emel::tokenizer::preprocessor::detail::build_special_tokens(cache,
                                                                    vocab));
  CHECK(cache.count == 2);
  CHECK(cache.tokens[0].text.size() >= cache.tokens[1].text.size());
}

TEST_CASE("tokenizer_preprocessor_partition_with_specials_skips_control") {
  emel::model::data::vocab vocab = make_vocab_with_specials();
  emel::tokenizer::preprocessor::special_token_cache cache = {};
  CHECK(emel::tokenizer::preprocessor::detail::build_special_tokens(cache,
                                                                    vocab));

  std::array<emel::tokenizer::preprocessor::fragment,
             emel::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  const std::string_view text = "xxAyyBBBzz";

  CHECK(emel::tokenizer::preprocessor::detail::partition_with_specials(
      text, cache, false, fragments.data(), fragments.size(), &count));
  CHECK(count == 3);
  CHECK(fragments[0].text == std::string_view("xx"));
  CHECK(fragments[1].kind ==
        emel::tokenizer::preprocessor::fragment_kind::token);
  CHECK(fragments[2].text == std::string_view("yyBBBzz"));
}

TEST_CASE("tokenizer_preprocessor_partition_with_specials_parse_control") {
  emel::model::data::vocab vocab = make_vocab_with_specials();
  emel::tokenizer::preprocessor::special_token_cache cache = {};
  CHECK(emel::tokenizer::preprocessor::detail::build_special_tokens(cache,
                                                                    vocab));

  std::array<emel::tokenizer::preprocessor::fragment,
             emel::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  const std::string_view text = "BBB";

  CHECK(emel::tokenizer::preprocessor::detail::partition_with_specials(
      text, cache, true, fragments.data(), fragments.size(), &count));
  CHECK(count == 1);
  CHECK(fragments[0].kind ==
        emel::tokenizer::preprocessor::fragment_kind::token);
}

TEST_CASE("tokenizer_preprocessor_actions_success") {
  emel::model::data::vocab vocab = make_vocab_with_specials();
  std::array<emel::tokenizer::preprocessor::fragment,
             emel::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view("A");
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = fragments.size();
  ev.fragment_count_out = &count;
  ev.error_out = &err;

  emel::tokenizer::preprocessor::action::context ctx = {};
  struct emel::tokenizer::preprocessor::action::begin_preprocess begin_preprocess{};
  struct emel::tokenizer::preprocessor::action::build_specials build_specials{};
  struct emel::tokenizer::preprocessor::action::partition_specials partition_specials{};
  struct emel::tokenizer::preprocessor::action::mark_done mark_done{};
  begin_preprocess(ev, ctx);
  build_specials(ctx);
  partition_specials(ctx);
  mark_done(ctx);

  CHECK(ctx.last_error == EMEL_OK);
  CHECK(ctx.fragment_count == 1);
}

TEST_CASE("tokenizer_preprocessor_actions_errors") {
  emel::tokenizer::preprocessor::action::context ctx = {};
  emel::tokenizer::preprocessor::event::preprocess ev = {};

  struct emel::tokenizer::preprocessor::action::reject_invalid reject_invalid{};
  reject_invalid(ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.last_error = EMEL_OK;
  ctx.phase_error = EMEL_OK;
  struct emel::tokenizer::preprocessor::action::ensure_last_error ensure_last_error{};
  ensure_last_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  struct emel::tokenizer::preprocessor::action::on_unexpected handler {};
  handler(ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tokenizer_preprocessor_on_unexpected_sets_error") {
  emel::tokenizer::preprocessor::action::context ctx = {};
  emel::tokenizer::preprocessor::event::preprocess ev = {};

  struct emel::tokenizer::preprocessor::action::on_unexpected handler {};
  handler(ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

}  // namespace
