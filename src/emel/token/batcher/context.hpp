#pragma once

#include <array>
#include <cstdint>

#include "emel/batch/planner/context.hpp"

namespace emel::token::batcher::action {

inline constexpr int32_t MAX_TOKENS = emel::batch::planner::action::MAX_PLAN_STEPS;
inline constexpr int32_t MAX_SEQ = emel::batch::planner::action::MAX_SEQ;
inline constexpr int32_t SEQ_WORDS = emel::batch::planner::action::SEQ_WORDS;

enum class position_probe_status : uint8_t {
  none = 0u,
  ok = 1u,
  backend_error = 2u,
  invalid = 3u,
};

struct context {
  position_probe_status seeded_probe_status = position_probe_status::none;
  bool unseeded_probe_valid = false;
  std::array<int32_t, MAX_SEQ> seeded_next_pos = {};
};

}  // namespace emel::token::batcher::action
