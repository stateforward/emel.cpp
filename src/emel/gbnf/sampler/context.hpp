#pragma once

#include <array>
#include <cstdint>

#include "emel/gbnf/detail.hpp"

namespace emel::gbnf::sampler::action {

struct context {
  std::array<uint32_t, emel::gbnf::k_max_gbnf_rule_elements> frontier = {};
  std::array<uint32_t, emel::gbnf::k_max_gbnf_rule_elements> scratch = {};
  uint32_t frontier_size = 0;
  uint32_t active_rule_id = 0;
};

}  // namespace emel::gbnf::sampler::action
