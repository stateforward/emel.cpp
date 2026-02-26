#pragma once

#include <cstdint>

namespace emel::gbnf::nonterm_parser::events {

enum class parse_mode : uint8_t {
  none = 0,
  definition = 1,
  reference = 2,
};

}  // namespace emel::gbnf::nonterm_parser::events
