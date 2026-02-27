#pragma once

#include "emel/error/error.hpp"

namespace emel::gbnf::rule_parser::definition_parser {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  parse_failed = (1u << 1),
  internal_error = (1u << 2),
  untracked = (1u << 3),
};

}  // namespace emel::gbnf::rule_parser::definition_parser
