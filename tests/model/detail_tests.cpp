#include "doctest/doctest.h"

#include <array>
#include <memory>
#include <string_view>

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

}  // namespace

TEST_CASE("model_detail_maps_tokenizer_model_names") {
  constexpr auto cases = std::to_array<model_case>({
      {"no_vocab", tokenizer_model::NONE},
      {"llama", tokenizer_model::SPM},
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
