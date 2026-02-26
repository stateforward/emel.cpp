#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "emel/gbnf/detail.hpp"
#include "emel/gbnf/lexer/sm.hpp"
#include "emel/gbnf/rule_parser/detail.hpp"

namespace emel::gbnf::rule_parser::action {

struct context {
  struct group_frame {
    uint32_t sequence_start = 0;
    uint32_t generated_rule_id = 0;
  };

  emel::gbnf::lexer::sm lexer = {};
  uint32_t current_rule_id = 0;
  detail::rule_builder current_rule = {};
  uint32_t last_sym_start = 0;
  std::array<group_frame, detail::k_max_group_nesting_depth>
      group_stack = {};
  uint32_t group_depth = 0;

  detail::symbol_table symbols = {};
  std::array<bool, emel::gbnf::k_max_gbnf_rules> rule_defined = {};
  uint32_t next_symbol_id = 0;

  // Scratch storage reused across parse steps to avoid large per-action stack clears.
  std::unique_ptr<emel::gbnf::element[]> group_scratch = {};
  std::unique_ptr<emel::gbnf::element[]> prev_scratch = {};
  std::unique_ptr<emel::gbnf::element[]> rec_scratch = {};

  context() {
    group_scratch =
        std::make_unique_for_overwrite<emel::gbnf::element[]>(emel::gbnf::k_max_gbnf_rule_elements);
    prev_scratch =
        std::make_unique_for_overwrite<emel::gbnf::element[]>(emel::gbnf::k_max_gbnf_rule_elements);
    rec_scratch =
        std::make_unique_for_overwrite<emel::gbnf::element[]>(emel::gbnf::k_max_gbnf_rule_elements);
  }
};

} // namespace emel::gbnf::rule_parser::action
