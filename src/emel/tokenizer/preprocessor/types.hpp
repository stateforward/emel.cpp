#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/model/data.hpp"

namespace emel::tokenizer::preprocessor {

constexpr size_t k_max_fragments = 1024;
constexpr size_t k_max_special_tokens = 256;

enum class fragment_kind : uint8_t {
  raw_text = 0,
  token = 1,
};

struct fragment {
  fragment_kind kind = fragment_kind::raw_text;
  std::string_view text = {};
  int32_t token = -1;
};

struct special_token {
  std::string_view text = {};
  int32_t token = -1;
  int32_t type = 0;
  bool lstrip = false;
  bool rstrip = false;
};

struct special_token_cache {
  std::array<special_token, k_max_special_tokens> tokens = {};
  size_t count = 0;
  const emel::model::data::vocab * vocab = nullptr;
};

}  // namespace emel::tokenizer::preprocessor
