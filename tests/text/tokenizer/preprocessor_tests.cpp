#include <array>
#include <cstddef>
#include <cstring>
#include <span>
#include <string>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/tokenizer/preprocessor/any.hpp"
#include "emel/text/tokenizer/preprocessor/actions.hpp"
#include "emel/text/tokenizer/preprocessor/bpe/sm.hpp"
#include "emel/text/tokenizer/preprocessor/detail.hpp"

namespace {

emel::model::data::vocab & make_vocab_with_specials() {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
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

emel::model::data::vocab & make_bpe_vocab() {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.n_tokens = 0;
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  vocab.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  return vocab;
}

emel::model::data::vocab & make_bpe_vocab_with_specials() {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.n_tokens = 2;
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  vocab.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
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

TEST_CASE("tokenizer_preprocessor_any_valid_request") {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.n_tokens = 0;

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::text::tokenizer::preprocessor::any machine(
      emel::text::tokenizer::preprocessor::preprocessor_kind::fallback);
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("hello"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count,
      err);

  CHECK(machine.process_event(ev));
  CHECK(err == EMEL_OK);
  CHECK(count == 1);
  CHECK(fragments[0].kind ==
        emel::text::tokenizer::preprocessor::fragment_kind::raw_text);
  CHECK(fragments[0].text == std::string_view("hello"));
}

TEST_CASE("tokenizer_preprocessor_any_invalid_request") {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.n_tokens = 0;

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::text::tokenizer::preprocessor::any machine(
      emel::text::tokenizer::preprocessor::preprocessor_kind::fallback);
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("hello"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments.data(),
                                                                static_cast<size_t>(0)),
      count, err);

  CHECK_FALSE(machine.process_event(ev));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(count == 0);
}

TEST_CASE("tokenizer_preprocessor_build_special_tokens") {
  auto & vocab = make_vocab_with_specials();
  emel::text::tokenizer::preprocessor::special_token_cache cache = {};

  CHECK(emel::text::tokenizer::preprocessor::detail::build_special_tokens(cache,
                                                                    vocab));
  CHECK(cache.count == 2);
  CHECK(cache.tokens[0].text.size() >= cache.tokens[1].text.size());
}

TEST_CASE("tokenizer_preprocessor_build_special_tokens_reuse_empty") {
  auto & vocab = make_bpe_vocab();
  emel::text::tokenizer::preprocessor::special_token_cache cache = {};
  cache.vocab = &vocab;
  cache.count = 0;
  cache.tokens[0].token = 123;

  CHECK(emel::text::tokenizer::preprocessor::detail::build_special_tokens(cache,
                                                                    vocab));
  CHECK(cache.tokens[0].token == 123);
}

TEST_CASE("tokenizer_preprocessor_build_special_tokens_skips_non_special") {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.n_tokens = 1;
  vocab.entries[0].text_offset = 0;
  vocab.entries[0].text_length = 1;
  vocab.entries[0].type = 0;
  vocab.token_storage[0] = 'Z';

  emel::text::tokenizer::preprocessor::special_token_cache cache = {};
  CHECK(emel::text::tokenizer::preprocessor::detail::build_special_tokens(cache,
                                                                    vocab));
  CHECK(cache.count == 0);
}

TEST_CASE("tokenizer_preprocessor_build_special_tokens_empty_text") {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.n_tokens = 1;
  vocab.entries[0].text_offset = 0;
  vocab.entries[0].text_length = 0;
  vocab.entries[0].type = 4;

  emel::text::tokenizer::preprocessor::special_token_cache cache = {};
  CHECK(emel::text::tokenizer::preprocessor::detail::build_special_tokens(cache,
                                                                    vocab));
  CHECK(cache.count == 0);
}

TEST_CASE("tokenizer_preprocessor_build_special_tokens_overflow") {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.n_tokens = emel::text::tokenizer::preprocessor::k_max_special_tokens + 1;
  for (uint32_t i = 0; i < vocab.n_tokens; ++i) {
    vocab.entries[i].text_offset = 0;
    vocab.entries[i].text_length = 1;
    vocab.entries[i].type = 4;
  }
  vocab.token_storage[0] = 'A';

  emel::text::tokenizer::preprocessor::special_token_cache cache = {};
  CHECK_FALSE(emel::text::tokenizer::preprocessor::detail::build_special_tokens(cache,
                                                                          vocab));
}

TEST_CASE("tokenizer_preprocessor_partition_with_specials_invalid_args") {
  auto & vocab = make_bpe_vocab();
  emel::text::tokenizer::preprocessor::special_token_cache cache = {};
  cache.vocab = &vocab;
  cache.count = 0;
  size_t count = 0;
  std::array<emel::text::tokenizer::preprocessor::fragment, 1> one_fragment = {};
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments + 1>
      too_many_fragments = {};

  CHECK_FALSE(emel::text::tokenizer::preprocessor::detail::partition_with_specials(
      std::string_view("hi"), cache, false,
      std::span<emel::text::tokenizer::preprocessor::fragment>{}, count));
  CHECK_FALSE(emel::text::tokenizer::preprocessor::detail::partition_with_specials(
      std::string_view("hi"), cache, false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(
          one_fragment.data(), static_cast<size_t>(0)),
      count));
  CHECK_FALSE(emel::text::tokenizer::preprocessor::detail::partition_with_specials(
      std::string_view("hi"), cache, false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(too_many_fragments),
      count));
}

TEST_CASE("tokenizer_preprocessor_partition_with_specials_empty_token_text") {
  emel::text::tokenizer::preprocessor::special_token_cache cache = {};
  cache.count = 1;
  cache.tokens[0].text = std::string_view();
  cache.tokens[0].token = 1;

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;

  CHECK(emel::text::tokenizer::preprocessor::detail::partition_with_specials(
      std::string_view("hi"), cache, false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count));
  CHECK(count == 1);
}

TEST_CASE("tokenizer_preprocessor_detail_push_helpers") {
  emel::text::tokenizer::preprocessor::fragment fragments[1] = {};
  size_t count = 0;

  CHECK(emel::text::tokenizer::preprocessor::detail::push_raw_fragment(
      fragments, 1, count, std::string_view()));
  CHECK_FALSE(emel::text::tokenizer::preprocessor::detail::push_raw_fragment(
      fragments, 0, count, std::string_view("x")));

  count = 0;
  CHECK_FALSE(emel::text::tokenizer::preprocessor::detail::push_token_fragment(
      fragments, 1, count, -1));
  CHECK(emel::text::tokenizer::preprocessor::detail::push_token_fragment(
      fragments, 1, count, 1));
  CHECK_FALSE(emel::text::tokenizer::preprocessor::detail::push_token_fragment(
      fragments, 1, count, 2));
}

TEST_CASE("tokenizer_preprocessor_detail_flag_set_out_of_range") {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.n_tokens = 0;
  CHECK_FALSE(emel::text::tokenizer::preprocessor::detail::has_lstrip(vocab, 1));
  CHECK_FALSE(emel::text::tokenizer::preprocessor::detail::is_special_type(vocab, 1));
}

TEST_CASE("tokenizer_preprocessor_partition_with_specials_empty_cache") {
  auto & vocab = make_bpe_vocab();
  emel::text::tokenizer::preprocessor::special_token_cache cache = {};
  cache.vocab = &vocab;
  cache.count = 0;

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;

  CHECK(emel::text::tokenizer::preprocessor::detail::partition_with_specials(
      std::string_view("hi"), cache, false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count));
  CHECK(count == 1);
  CHECK(fragments[0].kind ==
        emel::text::tokenizer::preprocessor::fragment_kind::raw_text);
}

TEST_CASE("tokenizer_preprocessor_partition_with_specials_skips_control") {
  auto & vocab = make_vocab_with_specials();
  emel::text::tokenizer::preprocessor::special_token_cache cache = {};
  CHECK(emel::text::tokenizer::preprocessor::detail::build_special_tokens(cache,
                                                                    vocab));

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  const std::string_view text = "xxAyyBBBzz";

  CHECK(emel::text::tokenizer::preprocessor::detail::partition_with_specials(
      text, cache, false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count));
  CHECK(count == 3);
  CHECK(fragments[0].text == std::string_view("xx"));
  CHECK(fragments[1].kind ==
        emel::text::tokenizer::preprocessor::fragment_kind::token);
  CHECK(fragments[2].text == std::string_view("yyBBBzz"));
}

TEST_CASE("tokenizer_preprocessor_partition_with_specials_parse_control") {
  auto & vocab = make_vocab_with_specials();
  emel::text::tokenizer::preprocessor::special_token_cache cache = {};
  CHECK(emel::text::tokenizer::preprocessor::detail::build_special_tokens(cache,
                                                                    vocab));

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  const std::string_view text = "BBB";

  CHECK(emel::text::tokenizer::preprocessor::detail::partition_with_specials(
      text, cache, true,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count));
  CHECK(count == 1);
  CHECK(fragments[0].kind ==
        emel::text::tokenizer::preprocessor::fragment_kind::token);
}

TEST_CASE("tokenizer_preprocessor_actions_success") {
  auto & vocab = make_vocab_with_specials();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("A"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count,
      err);

  emel::text::tokenizer::preprocessor::event::preprocess_ctx runtime_ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess_runtime runtime_ev{
      ev, runtime_ctx};
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::begin_preprocess begin_preprocess{};
  struct emel::text::tokenizer::preprocessor::action::build_specials build_specials{};
  struct emel::text::tokenizer::preprocessor::action::partition_non_bpe partition_non_bpe{};
  struct emel::text::tokenizer::preprocessor::action::mark_done mark_done{};
  begin_preprocess(runtime_ev, ctx);
  build_specials(runtime_ev, ctx);
  partition_non_bpe(runtime_ev, ctx);
  mark_done(runtime_ev, ctx);

  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::none);
  CHECK(runtime_ctx.fragment_count == 1);
}

TEST_CASE("tokenizer_preprocessor_partition_bpe_no_specials") {
  auto & vocab = make_bpe_vocab();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("hello"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count,
      err);

  emel::text::tokenizer::preprocessor::event::preprocess_ctx runtime_ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess_runtime runtime_ev{
      ev, runtime_ctx};
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::begin_preprocess begin_preprocess{};
  struct emel::text::tokenizer::preprocessor::action::partition_bpe_no_specials
      partition_bpe_no_specials{};

  begin_preprocess(runtime_ev, ctx);
  partition_bpe_no_specials(runtime_ev, ctx);
  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::none);
  CHECK(runtime_ctx.fragment_count == 1);
  CHECK(fragments[0].kind ==
        emel::text::tokenizer::preprocessor::fragment_kind::raw_text);
}

TEST_CASE("tokenizer_preprocessor_partition_bpe_no_specials_large_input") {
  auto & vocab = make_bpe_vocab();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  std::string text;
  const size_t word_count =
      emel::text::tokenizer::preprocessor::k_max_fragments + 1;
  text.reserve(word_count * 2);
  for (size_t idx = 0; idx < word_count; ++idx) {
    if (idx > 0) {
      text += ' ';
    }
    text += 'a';
  }

  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view(text), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count,
      err);

  emel::text::tokenizer::preprocessor::event::preprocess_ctx runtime_ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess_runtime runtime_ev{
      ev, runtime_ctx};
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::begin_preprocess begin_preprocess{};
  struct emel::text::tokenizer::preprocessor::action::partition_bpe_no_specials
      partition_bpe_no_specials{};

  begin_preprocess(runtime_ev, ctx);
  partition_bpe_no_specials(runtime_ev, ctx);
  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::invalid_request);
  CHECK_FALSE(runtime_ctx.preprocessed);
  CHECK(runtime_ctx.fragment_count == 0);
}

TEST_CASE("tokenizer_preprocessor_partition_bpe_no_specials_invalid") {
  auto & vocab = make_bpe_vocab();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("hi"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments.data(),
                                                                static_cast<size_t>(0)),
      count, err);
  emel::text::tokenizer::preprocessor::event::preprocess_ctx runtime_ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess_runtime runtime_ev{
      ev, runtime_ctx};
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::partition_bpe_no_specials
      partition_bpe_no_specials{};
  partition_bpe_no_specials(runtime_ev, ctx);
  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::invalid_request);
}

TEST_CASE("tokenizer_preprocessor_partition_bpe_with_specials") {
  auto & vocab = make_bpe_vocab_with_specials();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("A hi"), true,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count,
      err);

  emel::text::tokenizer::preprocessor::event::preprocess_ctx runtime_ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess_runtime runtime_ev{
      ev, runtime_ctx};
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::begin_preprocess begin_preprocess{};
  struct emel::text::tokenizer::preprocessor::action::build_specials build_specials{};
  struct emel::text::tokenizer::preprocessor::action::partition_bpe_with_specials
      partition_bpe_with_specials{};

  begin_preprocess(runtime_ev, ctx);
  build_specials(runtime_ev, ctx);
  partition_bpe_with_specials(runtime_ev, ctx);
  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::none);
  CHECK(runtime_ctx.fragment_count >= 1);
  CHECK(fragments[0].kind ==
        emel::text::tokenizer::preprocessor::fragment_kind::token);
}

TEST_CASE("tokenizer_preprocessor_partition_bpe_with_specials_invalid") {
  auto & vocab = make_bpe_vocab_with_specials();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("A"), true,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments.data(),
                                                                static_cast<size_t>(0)),
      count, err);

  emel::text::tokenizer::preprocessor::event::preprocess_ctx runtime_ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess_runtime runtime_ev{
      ev, runtime_ctx};
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::partition_bpe_with_specials
      partition_bpe_with_specials{};
  partition_bpe_with_specials(runtime_ev, ctx);
  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::invalid_request);
}

TEST_CASE("tokenizer_preprocessor_bpe_regex_split") {
  auto & vocab = make_bpe_vocab();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::text::tokenizer::preprocessor::bpe::sm machine;
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("hello world"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count,
      err);

  CHECK(machine.process_event(ev));
  CHECK(err == EMEL_OK);
  CHECK(count == 2);
  CHECK(fragments[0].text == std::string_view("hello"));
  const char encoded_word[] = "\xC4\xA0""world";
  CHECK(fragments[1].text ==
        std::string_view(encoded_word, sizeof(encoded_word) - 1));
}

TEST_CASE("tokenizer_preprocessor_bpe_machine_does_not_branch_on_model_metadata") {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.n_tokens = 0;
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::SPM;
  vocab.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;

  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::text::tokenizer::preprocessor::bpe::sm machine;
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("hello world"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count,
      err);

  CHECK(machine.process_event(ev));
  CHECK(err == EMEL_OK);
  CHECK(count == 2);
  CHECK(fragments[0].text == std::string_view("hello"));
  const char encoded_word[] = "\xC4\xA0""world";
  CHECK(fragments[1].text ==
        std::string_view(encoded_word, sizeof(encoded_word) - 1));
}

TEST_CASE("tokenizer_preprocessor_bpe_capacity_overflow") {
  auto & vocab = make_bpe_vocab();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::text::tokenizer::preprocessor::bpe::sm machine;
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("hello world"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments.data(),
                                                                static_cast<size_t>(1)),
      count, err);

  CHECK_FALSE(machine.process_event(ev));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(count == 0);
}

TEST_CASE("tokenizer_preprocessor_actions_errors") {
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  auto & vocab = make_bpe_vocab();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view(), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count,
      err);
  emel::text::tokenizer::preprocessor::event::preprocess_ctx runtime_ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess_runtime runtime_ev{
      ev, runtime_ctx};

  struct emel::text::tokenizer::preprocessor::action::reject_invalid reject_invalid{};
  reject_invalid(runtime_ev, ctx);
  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::invalid_request);

  runtime_ctx.err = emel::text::tokenizer::preprocessor::error::none;
  runtime_ctx.phase_error = emel::text::tokenizer::preprocessor::error::none;
  struct emel::text::tokenizer::preprocessor::action::ensure_last_error ensure_last_error{};
  ensure_last_error(runtime_ev, ctx);
  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::backend_error);

  struct emel::text::tokenizer::preprocessor::action::on_unexpected handler {};
  handler(runtime_ev, ctx);
  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::invalid_request);
}

TEST_CASE("tokenizer_preprocessor_on_unexpected_sets_error") {
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  auto & vocab = make_bpe_vocab();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view(), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count,
      err);
  emel::text::tokenizer::preprocessor::event::preprocess_ctx runtime_ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess_runtime runtime_ev{
      ev, runtime_ctx};

  struct emel::text::tokenizer::preprocessor::action::on_unexpected handler {};
  handler(runtime_ev, ctx);
  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::invalid_request);
}

TEST_CASE("tokenizer_preprocessor_build_specials_invalid_vocab") {
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.n_tokens = emel::text::tokenizer::preprocessor::k_max_special_tokens + 1;
  for (uint32_t i = 0; i < vocab.n_tokens; ++i) {
    vocab.entries[i].text_offset = 0;
    vocab.entries[i].text_length = 1;
    vocab.entries[i].type = 4;
  }
  vocab.token_storage[0] = 'A';
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("x"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments), count,
      err);
  emel::text::tokenizer::preprocessor::event::preprocess_ctx runtime_ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess_runtime runtime_ev{
      ev, runtime_ctx};
  struct emel::text::tokenizer::preprocessor::action::build_specials build_specials{};
  build_specials(runtime_ev, ctx);
  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::invalid_request);
}

TEST_CASE("tokenizer_preprocessor_partition_invalid_request") {
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  auto & vocab = make_bpe_vocab();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("hi"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments.data(),
                                                                static_cast<size_t>(0)),
      count, err);
  emel::text::tokenizer::preprocessor::event::preprocess_ctx runtime_ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess_runtime runtime_ev{
      ev, runtime_ctx};
  struct emel::text::tokenizer::preprocessor::action::partition_non_bpe partition_non_bpe{};
  partition_non_bpe(runtime_ev, ctx);
  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::invalid_request);
}

TEST_CASE("tokenizer_preprocessor_partition_non_bpe_failure") {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::SPM;
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("hi"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments.data(),
                                                                static_cast<size_t>(0)),
      count, err);

  emel::text::tokenizer::preprocessor::event::preprocess_ctx runtime_ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess_runtime runtime_ev{
      ev, runtime_ctx};
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::partition_non_bpe partition_non_bpe{};
  partition_non_bpe(runtime_ev, ctx);
  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::invalid_request);
}

TEST_CASE("tokenizer_preprocessor_partition_bpe_failure") {
  auto & vocab = make_bpe_vocab();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;
  emel::text::tokenizer::preprocessor::event::preprocess ev(
      vocab, std::string_view("hi"), false,
      std::span<emel::text::tokenizer::preprocessor::fragment>(fragments.data(),
                                                                static_cast<size_t>(0)),
      count, err);

  emel::text::tokenizer::preprocessor::event::preprocess_ctx runtime_ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess_runtime runtime_ev{
      ev, runtime_ctx};
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::partition_bpe_no_specials partition_bpe_no_specials{};
  partition_bpe_no_specials(runtime_ev, ctx);
  CHECK(runtime_ctx.err == emel::text::tokenizer::preprocessor::error::invalid_request);
}

}  // namespace
