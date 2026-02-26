#pragma once

#include <cstdint>

namespace emel::gbnf::expression_parser::events {

enum class parse_kind : uint8_t {
  unknown = 0,
  identifier = 1,
  non_identifier = 2,
};

}  // namespace emel::gbnf::expression_parser::events
