#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "emel/text/unicode.hpp"

TEST_CASE("unicode_len_utf8_classifies_leading_bytes") {
  CHECK(emel::text::unicode_len_utf8(static_cast<char>(0x41)) == 1);
  CHECK(emel::text::unicode_len_utf8(static_cast<char>(0xC2)) == 2);
  CHECK(emel::text::unicode_len_utf8(static_cast<char>(0xE2)) == 3);
  CHECK(emel::text::unicode_len_utf8(static_cast<char>(0xF0)) == 4);
}

TEST_CASE("unicode_cpt_to_utf8_roundtrips_known_codepoints") {
  const uint32_t ascii = 0x24;
  const uint32_t two_byte = 0x00A2;
  const uint32_t three_byte = 0x20AC;
  const uint32_t four_byte = 0x1F4A9;

  const std::string a = emel::text::unicode_cpt_to_utf8(ascii);
  const std::string b = emel::text::unicode_cpt_to_utf8(two_byte);
  const std::string c = emel::text::unicode_cpt_to_utf8(three_byte);
  const std::string d = emel::text::unicode_cpt_to_utf8(four_byte);

  CHECK(a.size() == 1);
  CHECK(b.size() == 2);
  CHECK(c.size() == 3);
  CHECK(d.size() == 4);

  size_t offset = 0;
  CHECK(emel::text::unicode_cpt_from_utf8(a, offset) == ascii);
  offset = 0;
  CHECK(emel::text::unicode_cpt_from_utf8(b, offset) == two_byte);
  offset = 0;
  CHECK(emel::text::unicode_cpt_from_utf8(c, offset) == three_byte);
  offset = 0;
  CHECK(emel::text::unicode_cpt_from_utf8(d, offset) == four_byte);
}

TEST_CASE("unicode_cpt_to_utf8_rejects_invalid_codepoints") {
  CHECK_THROWS_AS(emel::text::unicode_cpt_to_utf8(0x110000u), std::invalid_argument);
}

TEST_CASE("unicode_cpt_from_utf8_rejects_invalid_sequences") {
  size_t offset = 0;
  CHECK_THROWS_AS(emel::text::unicode_cpt_from_utf8("\x80", offset), std::invalid_argument);
  offset = 0;
  CHECK_THROWS_AS(emel::text::unicode_cpt_from_utf8("\xC2\x20", offset), std::invalid_argument);
  offset = 0;
  CHECK_THROWS_AS(emel::text::unicode_cpt_from_utf8("\xE2\x28\xA1", offset), std::invalid_argument);
  offset = 0;
  CHECK_THROWS_AS(emel::text::unicode_cpt_from_utf8("\xF0\x28\x8C\x28", offset), std::invalid_argument);
}

TEST_CASE("unicode_cpts_from_utf8_replaces_invalid_bytes") {
  const std::string input = std::string("A") + std::string("\xFF", 1) + "B";
  const auto cpts = emel::text::unicode_cpts_from_utf8(input);
  REQUIRE(cpts.size() == 3);
  CHECK(cpts[0] == 0x41);
  CHECK(cpts[1] == 0xFFFD);
  CHECK(cpts[2] == 0x42);
}

TEST_CASE("unicode_flags_from_utf8_empty_is_undefined") {
  const auto flags = emel::text::unicode_cpt_flags_from_utf8("");
  CHECK((flags.as_uint() & emel::text::unicode_cpt_flags::UNDEFINED) != 0);
}

TEST_CASE("unicode_flags_from_cpt_out_of_range_is_undefined") {
  const auto flags = emel::text::unicode_cpt_flags_from_cpt(0x110000u);
  CHECK((flags.as_uint() & emel::text::unicode_cpt_flags::UNDEFINED) != 0);
}

TEST_CASE("unicode_byte_roundtrip_mapping") {
  const uint8_t value = 65;
  const std::string utf8 = emel::text::unicode_byte_to_utf8(value);
  CHECK(emel::text::unicode_utf8_to_byte(utf8) == value);
}

TEST_CASE("unicode_tolower_converts_ascii") {
  CHECK(emel::text::unicode_tolower(0x41) == 0x61);
  CHECK(emel::text::unicode_tolower(0x61) == 0x61);
}

TEST_CASE("unicode_cpt_is_han_detects_cjk_ranges") {
  CHECK(emel::text::unicode_cpt_is_han(0x4E00));
  CHECK_FALSE(emel::text::unicode_cpt_is_han(0x0041));
}

TEST_CASE("unicode_regex_split_custom_paths") {
  const std::string ascii_text = "hello 123";
  const std::vector<size_t> ascii_offsets = {0u, ascii_text.size()};

  const auto gpt2 = emel::text::unicode_regex_split_custom(
    ascii_text,
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
    ascii_offsets);
  CHECK(!gpt2.empty());

  const auto llama3 = emel::text::unicode_regex_split_custom(
    ascii_text,
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    ascii_offsets);
  CHECK(!llama3.empty());

  const std::string han_text = "汉字";
  const std::vector<size_t> han_offsets = {0u, emel::text::unicode_cpts_from_utf8(han_text).size()};
  const auto kimi = emel::text::unicode_regex_split_custom(han_text, "\\p{han}+", han_offsets);
  CHECK(!kimi.empty());

  const std::string digits = "1234567";
  const std::vector<size_t> digit_offsets = {0u, digits.size()};
  const auto afmoe = emel::text::unicode_regex_split_custom(digits, "\\p{AFMoE_digits}", digit_offsets);
  CHECK(!afmoe.empty());
}

TEST_CASE("unicode_regex_split_runs_default_pipeline") {
  const std::string text = "hello, world!";
  const std::vector<std::string> exprs = {
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
  };
  const auto words = emel::text::unicode_regex_split(text, exprs);
  CHECK(!words.empty());
}

TEST_CASE("unicode_flags_category_and_normalize_nfd") {
  emel::text::unicode_cpt_flags flags(emel::text::unicode_cpt_flags::NUMBER |
                                      emel::text::unicode_cpt_flags::WHITESPACE);
  CHECK((flags.category_flag() & emel::text::unicode_cpt_flags::NUMBER) != 0);

  const std::vector<uint32_t> cpts = {0x00C1u, 0x0041u};
  const auto normalized = emel::text::unicode_cpts_normalize_nfd(cpts);
  CHECK(normalized.size() == cpts.size());
}

TEST_CASE("unicode_cpts_to_utf8_roundtrip") {
  const std::vector<uint32_t> cpts = {0x41u, 0x00A2u, 0x20ACu};
  const std::string utf8 = emel::text::unicode_cpts_to_utf8(cpts);
  CHECK(!utf8.empty());
  const auto roundtrip = emel::text::unicode_cpts_from_utf8(utf8);
  CHECK(roundtrip.size() == cpts.size());
}

TEST_CASE("unicode_regex_split_category_patterns") {
  const std::string text = "abc 123.";
  const std::vector<std::string> exprs = {
    "\\p{L}+",
    "\\p{N}+",
    "[\\.!]+",
  };
  const auto words = emel::text::unicode_regex_split(text, exprs);
  CHECK(!words.empty());
}

TEST_CASE("unicode_regex_split_custom_kimi_k2_branches") {
  const std::string text = "汉字abc12345DEF";
  const auto cpts = emel::text::unicode_cpts_from_utf8(text);
  const std::vector<size_t> offsets = {0u, cpts.size()};
  const auto result = emel::text::unicode_regex_split_custom_kimi_k2(text, offsets);
  CHECK(!result.empty());
}

TEST_CASE("unicode_regex_split_custom_afmoe_digits") {
  const std::string text = "1234567";
  const std::vector<size_t> offsets = {0u, text.size()};
  const auto result = emel::text::unicode_regex_split_custom_afmoe(text, offsets);
  CHECK(!result.empty());
}

TEST_CASE("unicode_regex_split_custom_gpt2_branches") {
  const std::string text = " 're  123 !!\t";
  const auto cpts = emel::text::unicode_cpts_from_utf8(text);
  const std::vector<size_t> offsets = {cpts.size()};
  const auto result = emel::text::unicode_regex_split_custom_gpt2(text, offsets);
  CHECK(!result.empty());
}

TEST_CASE("unicode_regex_split_custom_llama3_branches") {
  const std::string text = "!hello 'RE 1234!!!\r\n";
  const auto cpts = emel::text::unicode_cpts_from_utf8(text);
  const std::vector<size_t> offsets = {cpts.size()};
  const auto result = emel::text::unicode_regex_split_custom_llama3(text, offsets);
  CHECK(!result.empty());
}

TEST_CASE("unicode_regex_split_custom_kimi_k2_extra") {
  const std::string text = "汉字!abc's ?1234\r\n";
  const auto cpts = emel::text::unicode_cpts_from_utf8(text);
  const std::vector<size_t> offsets = {cpts.size()};
  const auto result = emel::text::unicode_regex_split_custom_kimi_k2(text, offsets);
  CHECK(!result.empty());
}

TEST_CASE("unicode_regex_split_custom_afmoe_multiple_lengths") {
  const std::string text = "123456";
  const std::vector<size_t> offsets = {0u, text.size()};
  const auto result = emel::text::unicode_regex_split_custom_afmoe(text, offsets);
  CHECK(!result.empty());
}

TEST_CASE("unicode_regex_split_collapsed_and_wregex_paths") {
  const std::string text = std::string("a") + std::string("\xC2\xA0", 2) + "b汉";
  const std::vector<std::string> collapsed_exprs = {
    "\\p{L}+",
    "\\p{N}+",
  };
  const auto collapsed = emel::text::unicode_regex_split(text, collapsed_exprs);
  CHECK(!collapsed.empty());

  const std::vector<std::string> wregex_exprs = {
    "\\s+",
  };
  const auto wregex = emel::text::unicode_regex_split(text, wregex_exprs);
  CHECK(!wregex.empty());
}
