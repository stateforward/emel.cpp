#include <array>
#include <cstddef>
#include <cstring>
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
  emel::text::tokenizer::preprocessor::event::preprocess ev = {};
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
        emel::text::tokenizer::preprocessor::fragment_kind::raw_text);
  CHECK(fragments[0].text == std::string_view("hello"));
}

TEST_CASE("tokenizer_preprocessor_any_invalid_request") {
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::text::tokenizer::preprocessor::any machine(
      emel::text::tokenizer::preprocessor::preprocessor_kind::fallback);
  emel::text::tokenizer::preprocessor::event::preprocess ev = {};
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

  CHECK_FALSE(emel::text::tokenizer::preprocessor::detail::partition_with_specials(
      std::string_view("hi"), cache, false, nullptr, 1, &count));
  CHECK_FALSE(emel::text::tokenizer::preprocessor::detail::partition_with_specials(
      std::string_view("hi"), cache, false, nullptr, 1, nullptr));
  CHECK_FALSE(emel::text::tokenizer::preprocessor::detail::partition_with_specials(
      std::string_view("hi"), cache, false, reinterpret_cast<emel::text::tokenizer::preprocessor::fragment *>(0x1),
      0, &count));
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
      std::string_view("hi"), cache, false, fragments.data(),
      fragments.size(), &count));
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
      std::string_view("hi"), cache, false, fragments.data(),
      fragments.size(), &count));
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
      text, cache, false, fragments.data(), fragments.size(), &count));
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
      text, cache, true, fragments.data(), fragments.size(), &count));
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

  emel::text::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view("A");
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = fragments.size();
  ev.fragment_count_out = &count;
  ev.error_out = &err;

  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::begin_preprocess begin_preprocess{};
  struct emel::text::tokenizer::preprocessor::action::build_specials build_specials{};
  struct emel::text::tokenizer::preprocessor::action::partition_non_bpe partition_non_bpe{};
  struct emel::text::tokenizer::preprocessor::action::mark_done mark_done{};
  begin_preprocess(ev, ctx);
  build_specials(ctx);
  partition_non_bpe(ctx);
  mark_done(ctx);

  CHECK(ctx.last_error == EMEL_OK);
  CHECK(ctx.fragment_count == 1);
}

TEST_CASE("tokenizer_preprocessor_partition_bpe_no_specials") {
  auto & vocab = make_bpe_vocab();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::text::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view("hello");
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = fragments.size();
  ev.fragment_count_out = &count;
  ev.error_out = &err;

  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::begin_preprocess begin_preprocess{};
  struct emel::text::tokenizer::preprocessor::action::partition_bpe_no_specials
      partition_bpe_no_specials{};

  begin_preprocess(ev, ctx);
  partition_bpe_no_specials(ctx);
  CHECK(ctx.last_error == EMEL_OK);
  CHECK(ctx.fragment_count == 1);
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

  emel::text::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view(text);
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = fragments.size();
  ev.fragment_count_out = &count;
  ev.error_out = &err;

  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::begin_preprocess begin_preprocess{};
  struct emel::text::tokenizer::preprocessor::action::partition_bpe_no_specials
      partition_bpe_no_specials{};

  begin_preprocess(ev, ctx);
  partition_bpe_no_specials(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK_FALSE(ctx.preprocessed);
  CHECK(ctx.fragment_count == 0);
}

TEST_CASE("tokenizer_preprocessor_partition_bpe_no_specials_invalid") {
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::partition_bpe_no_specials
      partition_bpe_no_specials{};
  partition_bpe_no_specials(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tokenizer_preprocessor_partition_bpe_with_specials") {
  auto & vocab = make_bpe_vocab_with_specials();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::text::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view("A hi");
  ev.parse_special = true;
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = fragments.size();
  ev.fragment_count_out = &count;
  ev.error_out = &err;

  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::begin_preprocess begin_preprocess{};
  struct emel::text::tokenizer::preprocessor::action::build_specials build_specials{};
  struct emel::text::tokenizer::preprocessor::action::partition_bpe_with_specials
      partition_bpe_with_specials{};

  begin_preprocess(ev, ctx);
  build_specials(ctx);
  partition_bpe_with_specials(ctx);
  CHECK(ctx.last_error == EMEL_OK);
  CHECK(ctx.fragment_count >= 1);
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

  emel::text::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view("A");
  ev.parse_special = true;
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = 0;
  ev.fragment_count_out = &count;
  ev.error_out = &err;

  emel::text::tokenizer::preprocessor::action::context ctx = {};
  ctx.vocab = &vocab;
  ctx.text = ev.text;
  ctx.fragments_out = ev.fragments_out;
  ctx.parse_special = ev.parse_special;
  ctx.fragment_capacity = ev.fragment_capacity;

  struct emel::text::tokenizer::preprocessor::action::partition_bpe_with_specials
      partition_bpe_with_specials{};
  partition_bpe_with_specials(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tokenizer_preprocessor_bpe_regex_split") {
  auto & vocab = make_bpe_vocab();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t count = 0;
  int32_t err = EMEL_OK;

  emel::text::tokenizer::preprocessor::bpe::sm machine;
  emel::text::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view("hello world");
  ev.parse_special = false;
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = fragments.size();
  ev.fragment_count_out = &count;
  ev.error_out = &err;

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
  emel::text::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view("hello world");
  ev.parse_special = false;
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = 1;
  ev.fragment_count_out = &count;
  ev.error_out = &err;

  CHECK_FALSE(machine.process_event(ev));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(count == 0);
}

TEST_CASE("tokenizer_preprocessor_actions_errors") {
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess ev = {};

  struct emel::text::tokenizer::preprocessor::action::reject_invalid reject_invalid{};
  reject_invalid(ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.last_error = EMEL_OK;
  ctx.phase_error = EMEL_OK;
  struct emel::text::tokenizer::preprocessor::action::ensure_last_error ensure_last_error{};
  ensure_last_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  struct emel::text::tokenizer::preprocessor::action::on_unexpected handler {};
  handler(ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tokenizer_preprocessor_on_unexpected_sets_error") {
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  emel::text::tokenizer::preprocessor::event::preprocess ev = {};

  struct emel::text::tokenizer::preprocessor::action::on_unexpected handler {};
  handler(ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tokenizer_preprocessor_build_specials_invalid_vocab") {
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::build_specials build_specials{};
  build_specials(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tokenizer_preprocessor_partition_invalid_request") {
  emel::text::tokenizer::preprocessor::action::context ctx = {};
  struct emel::text::tokenizer::preprocessor::action::partition_non_bpe partition_non_bpe{};
  partition_non_bpe(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tokenizer_preprocessor_partition_non_bpe_failure") {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::SPM;
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  emel::text::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view("hi");
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = 0;

  emel::text::tokenizer::preprocessor::action::context ctx = {};
  ctx.vocab = &vocab;
  ctx.text = ev.text;
  ctx.fragments_out = ev.fragments_out;
  ctx.fragment_capacity = ev.fragment_capacity;
  struct emel::text::tokenizer::preprocessor::action::partition_non_bpe partition_non_bpe{};
  partition_non_bpe(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tokenizer_preprocessor_partition_bpe_failure") {
  auto & vocab = make_bpe_vocab();
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  emel::text::tokenizer::preprocessor::event::preprocess ev = {};
  ev.vocab = &vocab;
  ev.text = std::string_view("hi");
  ev.fragments_out = fragments.data();
  ev.fragment_capacity = 0;

  emel::text::tokenizer::preprocessor::action::context ctx = {};
  ctx.vocab = &vocab;
  ctx.text = ev.text;
  ctx.fragments_out = ev.fragments_out;
  ctx.fragment_capacity = ev.fragment_capacity;
  struct emel::text::tokenizer::preprocessor::action::partition_bpe_no_specials partition_bpe_no_specials{};
  partition_bpe_no_specials(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

}  // namespace
