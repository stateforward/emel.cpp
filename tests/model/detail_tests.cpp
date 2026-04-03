#include "doctest/doctest.h"

#include <array>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "emel/gguf/loader/detail.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"

namespace {

using tokenizer_model = emel::model::data::tokenizer_model;
using tokenizer_pre = emel::model::data::tokenizer_pre;

struct model_case {
  std::string_view name = {};
  tokenizer_model expected = tokenizer_model::UNKNOWN;
};

struct pre_case {
  std::string_view name = {};
  tokenizer_pre expected = tokenizer_pre::UNKNOWN;
  bool ignore_merges = false;
  bool add_bos = false;
};

void append_kv_entry(std::vector<uint8_t> & arena,
                     std::vector<emel::gguf::loader::kv_entry> & entries,
                     const std::string_view key,
                     const uint32_t value_type,
                     const std::span<const uint8_t> value_bytes) {
  const uint32_t key_offset = static_cast<uint32_t>(arena.size());
  arena.insert(arena.end(), key.begin(), key.end());

  const uint32_t value_offset = static_cast<uint32_t>(arena.size());
  arena.insert(arena.end(), value_bytes.begin(), value_bytes.end());

  emel::gguf::loader::kv_entry entry = {};
  entry.key_offset = key_offset;
  entry.key_length = static_cast<uint32_t>(key.size());
  entry.value_type = value_type;
  entry.value_offset = value_offset;
  entry.value_length = static_cast<uint32_t>(value_bytes.size());
  entries.push_back(entry);
}

template <class value_type>
void append_scalar(std::vector<uint8_t> & bytes, const value_type value) {
  using unsigned_type = std::make_unsigned_t<value_type>;
  const unsigned_type raw = static_cast<unsigned_type>(value);
  for (size_t i = 0u; i < sizeof(value_type); ++i) {
    bytes.push_back(static_cast<uint8_t>((raw >> (i * 8u)) & 0xffu));
  }
}

void append_string_bytes(std::vector<uint8_t> & bytes, const std::string_view value) {
  append_scalar<uint64_t>(bytes, static_cast<uint64_t>(value.size()));
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void append_kv_string(std::vector<uint8_t> & arena,
                      std::vector<emel::gguf::loader::kv_entry> & entries,
                      const std::string_view key,
                      const std::string_view value) {
  std::vector<uint8_t> encoded = {};
  append_string_bytes(encoded, value);
  append_kv_entry(arena,
                  entries,
                  key,
                  emel::gguf::loader::detail::constants::gguf_type_string,
                  std::span<const uint8_t>{encoded});
}

void append_kv_i32(std::vector<uint8_t> & arena,
                   std::vector<emel::gguf::loader::kv_entry> & entries,
                   const std::string_view key,
                   const int32_t value) {
  std::array<uint8_t, sizeof(int32_t)> bytes = {};
  std::memcpy(bytes.data(), &value, sizeof(value));
  append_kv_entry(arena,
                  entries,
                  key,
                  emel::gguf::loader::detail::constants::gguf_type_int32,
                  bytes);
}

void append_kv_u32(std::vector<uint8_t> & arena,
                   std::vector<emel::gguf::loader::kv_entry> & entries,
                   const std::string_view key,
                   const uint32_t value) {
  std::array<uint8_t, sizeof(uint32_t)> bytes = {};
  std::memcpy(bytes.data(), &value, sizeof(value));
  append_kv_entry(arena,
                  entries,
                  key,
                  emel::gguf::loader::detail::constants::gguf_type_uint32,
                  bytes);
}

void append_kv_bool(std::vector<uint8_t> & arena,
                    std::vector<emel::gguf::loader::kv_entry> & entries,
                    const std::string_view key,
                    const bool value) {
  const uint8_t byte = value ? 1u : 0u;
  append_kv_entry(arena,
                  entries,
                  key,
                  emel::gguf::loader::detail::constants::gguf_type_bool,
                  std::span<const uint8_t>(&byte, 1u));
}

void append_kv_string_array(std::vector<uint8_t> & arena,
                            std::vector<emel::gguf::loader::kv_entry> & entries,
                            const std::string_view key,
                            const std::span<const std::string_view> values) {
  std::vector<uint8_t> bytes = {};
  append_scalar<uint32_t>(bytes, emel::gguf::loader::detail::constants::gguf_type_string);
  append_scalar<uint64_t>(bytes, static_cast<uint64_t>(values.size()));
  for (const std::string_view value : values) {
    append_string_bytes(bytes, value);
  }
  append_kv_entry(arena,
                  entries,
                  key,
                  emel::gguf::loader::detail::constants::gguf_type_array,
                  bytes);
}

}  // namespace

TEST_CASE("model_detail_maps_tokenizer_model_names") {
  constexpr auto cases = std::to_array<model_case>({
      {"no_vocab", tokenizer_model::NONE},
      {"llama", tokenizer_model::SPM},
      {"gemma4", tokenizer_model::SPM},
      {"gpt2", tokenizer_model::BPE},
      {"bert", tokenizer_model::WPM},
      {"t5", tokenizer_model::UGM},
      {"rwkv", tokenizer_model::RWKV},
      {"plamo2", tokenizer_model::PLAMO2},
  });

  for (const auto & test_case : cases) {
    INFO(test_case.name);
    CHECK(emel::model::detail::tokenizer_model_from_name(test_case.name) == test_case.expected);
  }

  CHECK(emel::model::detail::tokenizer_model_from_name("unknown-model") ==
        tokenizer_model::UNKNOWN);
}

TEST_CASE("model_detail_applies_tokenizer_model_defaults") {
  {
    auto vocab = std::make_unique<emel::model::data::vocab>();
    emel::model::detail::apply_tokenizer_model_defaults("llama", *vocab);
    CHECK(vocab->bos_id == 1);
    CHECK(vocab->eos_id == 2);
    CHECK(vocab->unk_id == 0);
    CHECK(vocab->add_bos);
    CHECK(vocab->add_space_prefix);
    CHECK(vocab->escape_whitespaces);
  }

  {
    auto vocab = std::make_unique<emel::model::data::vocab>();
    emel::model::detail::apply_tokenizer_model_defaults("bert", *vocab);
    CHECK(vocab->bos_id == 101);
    CHECK(vocab->unk_id == 100);
    CHECK(vocab->sep_id == 102);
    CHECK(vocab->pad_id == 0);
    CHECK(vocab->mask_id == 103);
    CHECK(vocab->add_bos);
    CHECK(vocab->add_sep);
  }

  {
    auto vocab = std::make_unique<emel::model::data::vocab>();
    emel::model::detail::apply_tokenizer_model_defaults("gpt2", *vocab);
    CHECK(vocab->bos_id == 11);
    CHECK(vocab->eos_id == 11);
  }

  {
    auto vocab = std::make_unique<emel::model::data::vocab>();
    emel::model::detail::apply_tokenizer_model_defaults("t5", *vocab);
    CHECK(vocab->eos_id == 1);
    CHECK(vocab->unk_id == 2);
    CHECK(vocab->pad_id == 0);
  }

  {
    auto vocab = std::make_unique<emel::model::data::vocab>();
    emel::model::detail::apply_tokenizer_model_defaults("plamo2", *vocab);
    CHECK(vocab->bos_id == 1);
    CHECK(vocab->eos_id == 2);
    CHECK(vocab->unk_id == 0);
    CHECK(vocab->pad_id == 3);
  }
}

TEST_CASE("model_detail_maps_tokenizer_pre_profiles") {
  constexpr auto cases = std::to_array<pre_case>({
      {"jina-v5-nano", tokenizer_pre::LLAMA3, true, true},
      {"jais2", tokenizer_pre::JAIS2, false, false},
      {"dbrx", tokenizer_pre::DBRX, false, false},
      {"smaug", tokenizer_pre::SMAUG, false, false},
      {"deepseek-llm", tokenizer_pre::DEEPSEEK_LLM, false, false},
      {"deepseek-coder", tokenizer_pre::DEEPSEEK_CODER, false, false},
      {"deepseek-v3", tokenizer_pre::DEEPSEEK3_LLM, false, false},
      {"youtu", tokenizer_pre::YOUTU, true, false},
      {"falcon", tokenizer_pre::FALCON, false, false},
      {"mpt", tokenizer_pre::MPT, false, false},
      {"starcoder", tokenizer_pre::STARCODER, false, false},
      {"gpt2", tokenizer_pre::GPT2, false, false},
      {"gpt-2", tokenizer_pre::GPT2, false, false},
      {"jais", tokenizer_pre::JAIS, false, false},
      {"refact", tokenizer_pre::REFACT, false, false},
      {"command-r", tokenizer_pre::COMMAND_R, false, false},
      {"qwen2", tokenizer_pre::QWEN2, false, false},
      {"qwen35", tokenizer_pre::QWEN35, false, false},
      {"stablelm2", tokenizer_pre::STABLELM2, false, false},
      {"olmo", tokenizer_pre::OLMO, false, false},
      {"poro", tokenizer_pre::PORO, false, false},
      {"chatglm4", tokenizer_pre::CHATGLM4, false, false},
      {"viking", tokenizer_pre::VIKING, false, false},
      {"tekken", tokenizer_pre::TEKKEN, false, false},
      {"smollm", tokenizer_pre::SMOLLM, false, false},
      {"codeshell", tokenizer_pre::CODESHELL, false, false},
      {"bloom", tokenizer_pre::BLOOM, false, false},
      {"gpt3-finnish", tokenizer_pre::GPT3_FINNISH, false, false},
      {"exaone", tokenizer_pre::EXAONE, false, false},
      {"exaone-moe", tokenizer_pre::EXAONE_MOE, false, false},
      {"chameleon", tokenizer_pre::CHAMELEON, false, false},
      {"minerva", tokenizer_pre::MINERVA, false, false},
      {"megrez", tokenizer_pre::MEGREZ, false, false},
      {"gpt-4o", tokenizer_pre::GPT4O, false, false},
      {"tiny-aya", tokenizer_pre::TINY_AYA, false, false},
      {"superbpe", tokenizer_pre::SUPERBPE, false, false},
      {"trillion", tokenizer_pre::TRILLION, false, false},
      {"granite-docling", tokenizer_pre::GRANITE_DOCLING, false, false},
      {"bailingmoe", tokenizer_pre::BAILINGMOE, false, false},
      {"seed-coder", tokenizer_pre::SEED_CODER, false, false},
      {"hunyuan-dense", tokenizer_pre::HUNYUAN_DENSE, false, false},
      {"joyai-llm", tokenizer_pre::JOYAI_LLM, false, false},
      {"kimi-k2", tokenizer_pre::KIMI_K2, false, false},
      {"grok-2", tokenizer_pre::GROK_2, false, false},
      {"afmoe", tokenizer_pre::AFMOE, false, false},
      {"minimax-m2", tokenizer_pre::MINIMAX_M2, false, false},
      {"solar-open", tokenizer_pre::SOLAR_OPEN, false, false},
  });

  for (const auto & test_case : cases) {
    INFO(test_case.name);
    CHECK(emel::model::detail::tokenizer_pre_profile_from_name(test_case.name) ==
          test_case.expected);
    auto vocab = std::make_unique<emel::model::data::vocab>();
    emel::model::detail::apply_tokenizer_pre_defaults(test_case.name, *vocab);
    CHECK(vocab->ignore_merges == test_case.ignore_merges);
    CHECK(vocab->add_bos == test_case.add_bos);
  }

  CHECK(emel::model::detail::tokenizer_pre_profile_from_name("") == tokenizer_pre::DEFAULT);
  CHECK(emel::model::detail::tokenizer_pre_profile_from_name("unknown-pre") ==
        tokenizer_pre::UNKNOWN);
}

TEST_CASE("model_detail_load_vocab_preserves_signed_optional_token_ids") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  constexpr std::array<std::string_view, 2> tokens = {"<bos>", "<eos>"};

  append_kv_string(arena, entries, "tokenizer.model", "gpt2");
  append_kv_string(arena, entries, "tokenizer.pre", "gpt-2");
  append_kv_string_array(arena, entries, "tokenizer.tokens", tokens);
  append_kv_u32(arena, entries, "tokenizer.bos_token_id", 0u);
  append_kv_u32(arena, entries, "tokenizer.eos_token_id", 1u);
  append_kv_i32(arena, entries, "tokenizer.prefix_token_id", -1);
  append_kv_i32(arena, entries, "tokenizer.suffix_token_id", -1);
  append_kv_i32(arena, entries, "tokenizer.middle_token_id", -1);
  append_kv_i32(arena, entries, "tokenizer.fim_pre_token_id", -1);
  append_kv_bool(arena, entries, "tokenizer.add_bos_token", false);
  append_kv_bool(arena, entries, "tokenizer.add_eos_token", false);

  const emel::model::detail::kv_binding binding{
    .arena = std::span<const uint8_t>{arena},
    .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  auto vocab = std::make_unique<emel::model::data::vocab>();
  CHECK(emel::model::detail::load_vocab_from_gguf(binding, *vocab));
  CHECK(vocab->prefix_id == -1);
  CHECK(vocab->suffix_id == -1);
  CHECK(vocab->middle_id == -1);
  CHECK(vocab->fim_pre_id == -1);
}
