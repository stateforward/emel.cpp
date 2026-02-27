#pragma once

#include <cstdint>

namespace emel::gbnf::sampler::matcher_parser::events {

enum class match_result : uint8_t {
  unknown = 0,
  accepted = 1,
  rejected = 2,
};

}  // namespace emel::gbnf::sampler::matcher_parser::events
