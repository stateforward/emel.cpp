#pragma once

#include <cstdint>

namespace emel::gbnf::sampler::accept_parser::events {

enum class accept_result : uint8_t {
  unknown = 0,
  accepted = 1,
};

}  // namespace emel::gbnf::sampler::accept_parser::events
