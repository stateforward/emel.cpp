#pragma once

#include <cstdint>

#include "emel/batch/planner/context.hpp"

namespace emel::token::batcher::action {

inline constexpr int32_t MAX_TOKENS = emel::batch::planner::action::MAX_PLAN_STEPS;
inline constexpr int32_t MAX_SEQ = emel::batch::planner::action::MAX_SEQ;
inline constexpr int32_t SEQ_WORDS = emel::batch::planner::action::SEQ_WORDS;

struct context {};

}  // namespace emel::token::batcher::action
