#include "doctest/doctest.h"

#include <array>
#include <memory>
#include <string_view>

#include "emel/model/data.hpp"
#include "emel/text/tokenizer/preprocessor/detail.hpp"

namespace {

using tokenizer_pre = emel::model::data::tokenizer_pre;

struct pre_case {
  std::string_view name = {};
  tokenizer_pre expected = tokenizer_pre::UNKNOWN;
  bool ignore_merges = false;
  bool add_bos = false;
};

}  // namespace

TEST_CASE("preprocessor_detail_maps_tokenizer_pre_profiles") {
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
    CHECK(emel::text::tokenizer::preprocessor::detail::tokenizer_pre_profile_from_name(
              test_case.name) == test_case.expected);
    auto vocab = std::make_unique<emel::model::data::vocab>();
    emel::text::tokenizer::preprocessor::detail::apply_tokenizer_pre_defaults(
        test_case.name, *vocab);
    CHECK(vocab->ignore_merges == test_case.ignore_merges);
    CHECK(vocab->add_bos == test_case.add_bos);
  }

  CHECK(emel::text::tokenizer::preprocessor::detail::tokenizer_pre_profile_from_name("") ==
        tokenizer_pre::DEFAULT);
  CHECK(emel::text::tokenizer::preprocessor::detail::tokenizer_pre_profile_from_name(
            "unknown-pre") == tokenizer_pre::UNKNOWN);
}
