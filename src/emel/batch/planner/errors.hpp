#pragma once

#include "emel/error/error.hpp"

namespace emel::batch::planner {
enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  invalid_token_data = (1u << 1),
  invalid_step_size = (1u << 2),
  invalid_sequence_metadata = (1u << 3),
  invalid_sequence_id = (1u << 4),
  invalid_sequence_mask = (1u << 5),
  multiple_bits_in_mask = (1u << 6),
  missing_mode = (1u << 7),
  invalid_mode = (1u << 8),
  output_plan_full = (1u << 9),
  output_indices_full = (1u << 10),
  output_steps_full = (1u << 11),
  planning_progress_stalled = (1u << 12),
  algorithm_failed = (1u << 13),
  unsupported_layout = (1u << 14),
  internal_error = (1u << 15),
  untracked = (1u << 16)
};

}  // namespace emel::batch::planner
