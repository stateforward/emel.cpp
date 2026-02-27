#pragma once

#include <array>
#include <cstdint>

#include "emel/batch/planner/context.hpp"
#include "emel/batch/planner/errors.hpp"
#include "emel/callback.hpp"

namespace emel::batch::planner::events {
struct plan_done;
struct plan_error;
}  // namespace emel::batch::planner::events

namespace emel::batch::planner::event {

enum class plan_mode : int32_t {
  simple = 0,
  equal = 1,
  sequential = 2,
  seq = sequential,
};

struct request {
  const int32_t * token_ids = nullptr;
  int32_t n_tokens = 0;
  int32_t n_steps = 0;
  plan_mode mode = plan_mode::simple;
  const uint64_t * seq_masks = nullptr;
  int32_t seq_masks_count = 0;
  const int32_t * seq_primary_ids = nullptr;
  int32_t seq_primary_ids_count = 0;
  bool equal_sequential = true;
  int32_t seq_mask_words = 1;
  const int8_t * output_mask = nullptr;
  int32_t output_mask_count = 0;
  bool output_all = false;

  // callbacks are required for all plan requests.
  const emel::callback<void(const events::plan_done &)> & on_done;
  const emel::callback<void(const events::plan_error &)> & on_error;
};

struct request_ctx {
  emel::error::type err = emel::error::cast(error::none);
  int32_t effective_step_size = 0;
  std::array<int32_t, action::MAX_PLAN_STEPS> step_sizes = {};
  int32_t step_count = 0;
  int32_t total_outputs = 0;
  std::array<int32_t, action::MAX_PLAN_STEPS> step_token_indices = {};
  std::array<int32_t, action::MAX_PLAN_STEPS + 1> step_token_offsets = {};
  int32_t token_indices_count = 0;
};

struct request_runtime {
  const request & request;
  request_ctx & ctx;
};

}  // namespace emel::batch::planner::event

namespace emel::batch::planner::events {

struct plan_done {
  const event::request * request = nullptr;
  const int32_t * step_sizes = nullptr;
  int32_t step_count = 0;
  int32_t total_outputs = 0;
  const int32_t * step_token_indices = nullptr;
  int32_t step_token_indices_count = 0;
  const int32_t * step_token_offsets = nullptr;
  int32_t step_token_offsets_count = 0;
};

struct plan_error {
  emel::error::type err = emel::error::type{};
  const event::request * request = nullptr;
};

}  // namespace emel::batch::planner::events
