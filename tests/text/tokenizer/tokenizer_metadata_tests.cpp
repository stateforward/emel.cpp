#include "doctest/doctest.h"

#include <array>
#include <memory>
#include <string_view>

#include "emel/model/data.hpp"
#include "emel/text/tokenizer/detail.hpp"

namespace {

using tokenizer_model = emel::model::data::tokenizer_model;

struct model_case {
  std::string_view name = {};
  tokenizer_model expected = tokenizer_model::UNKNOWN;
};

}  // namespace

TEST_CASE("tokenizer_detail_maps_tokenizer_model_names") {
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
    CHECK(emel::text::tokenizer::detail::tokenizer_model_from_name(test_case.name) ==
          test_case.expected);
  }

  CHECK(emel::text::tokenizer::detail::tokenizer_model_from_name("unknown-model") ==
        tokenizer_model::UNKNOWN);
}

TEST_CASE("tokenizer_detail_applies_tokenizer_model_defaults") {
  {
    auto vocab = std::make_unique<emel::model::data::vocab>();
    emel::text::tokenizer::detail::apply_tokenizer_model_defaults("llama", *vocab);
    CHECK(vocab->bos_id == 1);
    CHECK(vocab->eos_id == 2);
    CHECK(vocab->unk_id == 0);
    CHECK(vocab->add_bos);
    CHECK(vocab->add_space_prefix);
    CHECK(vocab->escape_whitespaces);
  }

  {
    auto vocab = std::make_unique<emel::model::data::vocab>();
    emel::text::tokenizer::detail::apply_tokenizer_model_defaults("bert", *vocab);
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
    emel::text::tokenizer::detail::apply_tokenizer_model_defaults("gpt2", *vocab);
    CHECK(vocab->bos_id == 11);
    CHECK(vocab->eos_id == 11);
  }

  {
    auto vocab = std::make_unique<emel::model::data::vocab>();
    emel::text::tokenizer::detail::apply_tokenizer_model_defaults("t5", *vocab);
    CHECK(vocab->eos_id == 1);
    CHECK(vocab->unk_id == 2);
    CHECK(vocab->pad_id == 0);
  }

  {
    auto vocab = std::make_unique<emel::model::data::vocab>();
    emel::text::tokenizer::detail::apply_tokenizer_model_defaults("plamo2", *vocab);
    CHECK(vocab->bos_id == 1);
    CHECK(vocab->eos_id == 2);
    CHECK(vocab->unk_id == 0);
    CHECK(vocab->pad_id == 3);
  }
}
