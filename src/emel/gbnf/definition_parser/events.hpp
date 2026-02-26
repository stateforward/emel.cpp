#pragma once

#include <cstdint>

namespace emel::gbnf::definition_parser::events {

enum class parse_result : uint8_t {
  unknown = 0,
  definition_operator = 1,
};

}  // namespace emel::gbnf::definition_parser::events
