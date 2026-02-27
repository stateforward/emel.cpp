#pragma once

#include <cstdint>

namespace emel::batch::planner::action {

inline constexpr int32_t MAX_PLAN_STEPS = 4096;
inline constexpr int32_t MAX_SEQ = 256;
inline constexpr int32_t SEQ_WORDS = (MAX_SEQ + 63) / 64;

struct context {};

}  // namespace emel::batch::planner::action
