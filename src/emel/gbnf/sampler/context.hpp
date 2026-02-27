#pragma once

#include <array>
#include <cstdint>
#include <functional>

#include "emel/gbnf/detail.hpp"

namespace emel::gbnf::sampler::action {

inline const emel::gbnf::grammar & empty_grammar() noexcept {
  static const emel::gbnf::grammar grammar{};
  return grammar;
}

struct context {
  std::reference_wrapper<const emel::gbnf::grammar> grammar = std::cref(empty_grammar());
  uint32_t start_rule_id = 0;
  std::array<uint32_t, emel::gbnf::k_max_gbnf_rule_elements> frontier = {};
  std::array<uint32_t, emel::gbnf::k_max_gbnf_rule_elements> scratch = {};
  uint32_t frontier_size = 0;
  uint32_t active_rule_id = 0;
};

}  // namespace emel::gbnf::sampler::action
