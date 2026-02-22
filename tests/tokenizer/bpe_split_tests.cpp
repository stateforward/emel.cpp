#include <array>
#include <string>

#include <doctest/doctest.h>

#include "emel/model/data.hpp"
#include "emel/tokenizer/bpe/split.hpp"

namespace {

emel::model::data::vocab make_vocab(
    const emel::model::data::tokenizer_pre pre) {
  emel::model::data::vocab vocab = {};
  vocab.tokenizer_pre_id = pre;
  return vocab;
}

}  // namespace

TEST_CASE("tokenizer_bpe_split_empty") {
  auto vocab = make_vocab(emel::model::data::tokenizer_pre::GPT2);
  emel::tokenizer::bpe::detail::split_scratch scratch = {};
  emel::tokenizer::bpe::detail::split_view view = {};

  const bool ok = emel::tokenizer::bpe::detail::split_and_encode_append(
      std::string_view{}, vocab, scratch, view);
  CHECK(ok);
  CHECK(view.count == 0);
}

TEST_CASE("tokenizer_bpe_split_gpt2_basic") {
  auto vocab = make_vocab(emel::model::data::tokenizer_pre::GPT2);
  emel::tokenizer::bpe::detail::split_scratch scratch = {};
  emel::tokenizer::bpe::detail::split_view view = {};

  const bool ok = emel::tokenizer::bpe::detail::split_and_encode_append(
      "hello world", vocab, scratch, view);
  CHECK(ok);
  CHECK(view.count == 2);
  CHECK(view.words[0] == std::string_view("hello"));
  const char encoded_word[] = "\xC4\xA0" "world";
  CHECK(view.words[1] == std::string_view(encoded_word, sizeof(encoded_word) - 1));
}

TEST_CASE("tokenizer_bpe_split_gpt2_branches") {
  auto vocab = make_vocab(emel::model::data::tokenizer_pre::GPT2);
  emel::tokenizer::bpe::detail::split_scratch scratch = {};
  emel::tokenizer::bpe::detail::split_view view = {};

  const std::string text = "I'm 123!!  café\n";
  const bool ok = emel::tokenizer::bpe::detail::split_and_encode_append(
      text, vocab, scratch, view);
  CHECK(ok);
  CHECK(view.count > 0);
}

TEST_CASE("tokenizer_bpe_split_llama3_branches") {
  auto vocab = make_vocab(emel::model::data::tokenizer_pre::LLAMA3);
  emel::tokenizer::bpe::detail::split_scratch scratch = {};
  emel::tokenizer::bpe::detail::split_view view = {};

  const std::string text = "Hello\nWORLD 1234 99";
  const bool ok = emel::tokenizer::bpe::detail::split_and_encode_append(
      text, vocab, scratch, view);
  CHECK(ok);
  CHECK(view.count > 0);
}

TEST_CASE("tokenizer_bpe_split_fallback_multi_regex") {
  auto vocab = make_vocab(emel::model::data::tokenizer_pre::FALCON);
  emel::tokenizer::bpe::detail::split_scratch scratch = {};
  emel::tokenizer::bpe::detail::split_view view = {};

  const bool ok = emel::tokenizer::bpe::detail::split_and_encode_append(
      "hello!!! 123", vocab, scratch, view);
  CHECK(ok);
  CHECK(view.count > 0);
}

TEST_CASE("tokenizer_bpe_split_fallback_single_regex") {
  auto vocab = make_vocab(emel::model::data::tokenizer_pre::PORO);
  emel::tokenizer::bpe::detail::split_scratch scratch = {};
  emel::tokenizer::bpe::detail::split_view view = {};

  const bool ok = emel::tokenizer::bpe::detail::split_and_encode_append(
      "hello world", vocab, scratch, view);
  CHECK(ok);
  CHECK(view.count > 0);
}

TEST_CASE("tokenizer_bpe_split_accepts_long_text") {
  auto vocab = make_vocab(emel::model::data::tokenizer_pre::GPT2);
  emel::tokenizer::bpe::detail::split_scratch scratch = {};
  emel::tokenizer::bpe::detail::split_view view = {};

  const std::string text(
      emel::tokenizer::bpe::detail::k_max_bpe_bytes + 1, 'a');
  const bool ok = emel::tokenizer::bpe::detail::split_and_encode_append(
      text, vocab, scratch, view);
  CHECK_FALSE(ok);
  CHECK(view.count == 0);
}

TEST_CASE("tokenizer_bpe_encode_utf8_branches") {
  char out[4] = {};
  CHECK(emel::tokenizer::bpe::detail::encode_utf8(0x41u, nullptr, 0) == 0);
  CHECK(emel::tokenizer::bpe::detail::encode_utf8(0x41u, out, 0) == 0);
  CHECK(emel::tokenizer::bpe::detail::encode_utf8(0x41u, out, 1) == 1);
  CHECK(emel::tokenizer::bpe::detail::encode_utf8(0x7FFu, out, 1) == 0);
  CHECK(emel::tokenizer::bpe::detail::encode_utf8(0x7FFu, out, 2) == 2);
  CHECK(emel::tokenizer::bpe::detail::encode_utf8(0xFFFFu, out, 2) == 0);
  CHECK(emel::tokenizer::bpe::detail::encode_utf8(0xFFFFu, out, 3) == 3);
  CHECK(emel::tokenizer::bpe::detail::encode_utf8(0x10FFFFu, out, 3) == 0);
  CHECK(emel::tokenizer::bpe::detail::encode_utf8(0x10FFFFu, out, 4) == 4);
  CHECK(emel::tokenizer::bpe::detail::encode_utf8(0x110000u, out, 4) == 0);
}

TEST_CASE("tokenizer_bpe_decode_utf8_to_cpts_branches") {
  emel::tokenizer::bpe::detail::split_scratch scratch = {};
  CHECK(emel::tokenizer::bpe::detail::decode_utf8_to_cpts("A", scratch));
  CHECK(scratch.cpt_count == 1);

  const std::string two = "\xC3\xA9";
  CHECK(emel::tokenizer::bpe::detail::decode_utf8_to_cpts(two, scratch));
  CHECK(scratch.cpt_count == 1);

  const std::string three = "\xE2\x82\xAC";
  CHECK(emel::tokenizer::bpe::detail::decode_utf8_to_cpts(three, scratch));
  CHECK(scratch.cpt_count == 1);

  const std::string four = "\xF0\x9F\x98\x80";
  CHECK(emel::tokenizer::bpe::detail::decode_utf8_to_cpts(four, scratch));
  CHECK(scratch.cpt_count == 1);
}

TEST_CASE("tokenizer_bpe_push_offset_branches") {
  size_t out[2] = {};
  size_t out_count = 0;
  CHECK(emel::tokenizer::bpe::detail::push_offset(0, out, 2, out_count));
  CHECK(out_count == 0);
  CHECK(emel::tokenizer::bpe::detail::push_offset(1, out, 1, out_count));
  CHECK(out_count == 1);
  CHECK_FALSE(emel::tokenizer::bpe::detail::push_offset(1, out, 1, out_count));
}

TEST_CASE("tokenizer_bpe_split_gpt2_error_paths") {
  std::array<uint32_t, 2> cpts = {{'\'', 's'}};
  size_t offsets_in_bad[1] = {3};
  size_t offsets_out[4] = {};
  size_t out_count = 0;
  CHECK_FALSE(emel::tokenizer::bpe::detail::split_gpt2(
      cpts.data(), cpts.size(), offsets_in_bad, 1, offsets_out, 4, out_count));

  size_t offsets_in_ok[1] = {2};
  CHECK_FALSE(emel::tokenizer::bpe::detail::split_gpt2(
      cpts.data(), cpts.size(), offsets_in_ok, 1, offsets_out, 0, out_count));

  std::array<uint32_t, 3> cpts_re = {{'\'', 'r', 'e'}};
  size_t offsets_in_re[1] = {3};
  CHECK_FALSE(emel::tokenizer::bpe::detail::split_gpt2(
      cpts_re.data(), cpts_re.size(), offsets_in_re, 1, offsets_out, 0, out_count));
}

TEST_CASE("tokenizer_bpe_split_llama3_error_paths") {
  std::array<uint32_t, 3> cpts = {{'\'', 'R', 'E'}};
  size_t offsets_in_bad[1] = {4};
  size_t offsets_out[4] = {};
  size_t out_count = 0;
  CHECK_FALSE(emel::tokenizer::bpe::detail::split_llama3(
      cpts.data(), cpts.size(), offsets_in_bad, 1, offsets_out, 4, out_count));

  size_t offsets_in_ok[1] = {3};
  CHECK_FALSE(emel::tokenizer::bpe::detail::split_llama3(
      cpts.data(), cpts.size(), offsets_in_ok, 1, offsets_out, 0, out_count));

  std::array<uint32_t, 1> punct = {{'!'}};
  size_t offsets_in_punct[1] = {1};
  CHECK_FALSE(emel::tokenizer::bpe::detail::split_llama3(
      punct.data(), punct.size(), offsets_in_punct, 1, offsets_out, 0, out_count));
}

TEST_CASE("tokenizer_bpe_encode_bpe_segment_errors") {
  emel::tokenizer::bpe::detail::split_scratch scratch = {};
  std::array<uint32_t, 1> bad_cpt = {{0x110000u}};
  CHECK_FALSE(emel::tokenizer::bpe::detail::encode_bpe_segment(
      bad_cpt.data(), 0, bad_cpt.size(), scratch));

  scratch.reset();
  scratch.encoded_size = scratch.encoded.size();
  std::array<uint32_t, 1> ok_cpt = {{'a'}};
  CHECK_FALSE(emel::tokenizer::bpe::detail::encode_bpe_segment(
      ok_cpt.data(), 0, ok_cpt.size(), scratch));

  scratch.reset();
  scratch.word_count = scratch.words.size();
  CHECK_FALSE(emel::tokenizer::bpe::detail::encode_bpe_segment(
      ok_cpt.data(), 0, ok_cpt.size(), scratch));
}

TEST_CASE("tokenizer_bpe_split_fallback_overflow") {
  emel::tokenizer::bpe::detail::regex_list regex = {};
  regex.exprs[0] = "\\p{L}+";
  regex.count = 1;
  emel::tokenizer::bpe::detail::split_scratch scratch = {};
  scratch.encoded_size = scratch.encoded.size();

  CHECK_FALSE(emel::tokenizer::bpe::detail::split_and_encode_fallback(
      "hello", regex, scratch));
}
