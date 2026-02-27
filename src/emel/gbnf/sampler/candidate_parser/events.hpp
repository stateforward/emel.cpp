#pragma once

#include <cstdint>

namespace emel::gbnf::sampler::candidate_parser::events {

enum class candidate_kind : uint8_t {
  unknown = 0,
  text = 1,
  empty = 2,
};

}  // namespace emel::gbnf::sampler::candidate_parser::events
