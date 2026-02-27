#pragma once

#include <cstdint>

namespace emel::gbnf::sampler::token_parser::events {

enum class token_kind : uint8_t {
  unknown = 0,
  text_token = 1,
  empty_token = 2,
};

}  // namespace emel::gbnf::sampler::token_parser::events
