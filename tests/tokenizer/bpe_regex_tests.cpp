#include <array>

#include <doctest/doctest.h>

#include "emel/model/data.hpp"
#include "emel/tokenizer/bpe/regex.hpp"

TEST_CASE("tokenizer_bpe_regex_for_vocab") {
  emel::model::data::vocab vocab = {};
  vocab.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;

  const auto from_vocab = emel::tokenizer::bpe::detail::regex_for(vocab);
  const auto from_pre = emel::tokenizer::bpe::detail::regex_for(
    emel::model::data::tokenizer_pre::GPT2);

  CHECK(from_vocab.count == from_pre.count);
  for (size_t idx = 0; idx < from_vocab.count; ++idx) {
    CHECK(from_vocab.exprs[idx] == from_pre.exprs[idx]);
  }
}

TEST_CASE("tokenizer_bpe_regex_for_presets") {
  using tokenizer_pre = emel::model::data::tokenizer_pre;
  const std::array<tokenizer_pre, 47> presets = {{
    tokenizer_pre::DEFAULT,
    tokenizer_pre::LLAMA3,
    tokenizer_pre::JAIS2,
    tokenizer_pre::DBRX,
    tokenizer_pre::SMAUG,
    tokenizer_pre::DEEPSEEK_LLM,
    tokenizer_pre::DEEPSEEK3_LLM,
    tokenizer_pre::HUNYUAN_DENSE,
    tokenizer_pre::JOYAI_LLM,
    tokenizer_pre::YOUTU,
    tokenizer_pre::DEEPSEEK_CODER,
    tokenizer_pre::FALCON,
    tokenizer_pre::STARCODER,
    tokenizer_pre::REFACT,
    tokenizer_pre::COMMAND_R,
    tokenizer_pre::SMOLLM,
    tokenizer_pre::CODESHELL,
    tokenizer_pre::EXAONE,
    tokenizer_pre::MINERVA,
    tokenizer_pre::GPT2,
    tokenizer_pre::MPT,
    tokenizer_pre::OLMO,
    tokenizer_pre::JAIS,
    tokenizer_pre::TRILLION,
    tokenizer_pre::GRANITE_DOCLING,
    tokenizer_pre::QWEN35,
    tokenizer_pre::STABLELM2,
    tokenizer_pre::QWEN2,
    tokenizer_pre::HUNYUAN,
    tokenizer_pre::SOLAR_OPEN,
    tokenizer_pre::PORO,
    tokenizer_pre::BLOOM,
    tokenizer_pre::GPT3_FINNISH,
    tokenizer_pre::CHATGLM4,
    tokenizer_pre::VIKING,
    tokenizer_pre::TEKKEN,
    tokenizer_pre::CHAMELEON,
    tokenizer_pre::GPT4O,
    tokenizer_pre::MINIMAX_M2,
    tokenizer_pre::TINY_AYA,
    tokenizer_pre::KIMI_K2,
    tokenizer_pre::SUPERBPE,
    tokenizer_pre::BAILINGMOE,
    tokenizer_pre::SEED_CODER,
    tokenizer_pre::GROK_2,
    tokenizer_pre::AFMOE,
    tokenizer_pre::EXAONE_MOE,
  }};

  for (const auto pre : presets) {
    const auto list = emel::tokenizer::bpe::detail::regex_for(pre);
    CHECK(list.count > 0);
    CHECK(!list.exprs[0].empty());
  }
}
