#pragma once

#include <array>
#include <cstdint>

#include "emel/batch/planner/events.hpp"
#include "emel/batch/planner/errors.hpp"

namespace emel::batch::planner::action {

inline constexpr int32_t MAX_PLAN_STEPS = 4096;
inline constexpr int32_t MAX_SEQ = 256;
inline constexpr int32_t SEQ_WORDS = (MAX_SEQ + 63) / 64;

struct context {
  int32_t effective_step_size = 0;
  std::array<int32_t, MAX_PLAN_STEPS> step_sizes = {};
  int32_t step_count = 0;
  int32_t total_outputs = 0;
  std::array<int32_t, MAX_PLAN_STEPS> step_token_indices = {};
  std::array<int32_t, MAX_PLAN_STEPS + 1> step_token_offsets = {};
  int32_t token_indices_count = 0;
};

}  // namespace emel::batch::planner::action
