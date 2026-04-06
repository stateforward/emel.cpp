#pragma once

#include <algorithm>

#include "emel/batch/planner/modes/simple/context.hpp"
#include "emel/batch/planner/modes/simple/detail.hpp"
#include "emel/batch/planner/modes/simple/events.hpp"

namespace emel::batch::planner::modes::simple::action {

using context = emel::batch::planner::action::context;

inline constexpr auto effect_plan_simple_batches =
    [](const event::plan_runtime & ev, context &) noexcept {
  const int32_t step_size = ev.ctx.effective_step_size;
  const int32_t token_count = ev.request.n_tokens;
  const int32_t full_chunks = token_count / step_size;
  const int32_t has_remainder = static_cast<int32_t>((token_count % step_size) != 0);
  const int32_t chunk_count = full_chunks + has_remainder;

  for (int32_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
    const int32_t chunk_start = chunk_idx * step_size;
    const int32_t remaining = token_count - chunk_start;
    const int32_t chunk_size = std::min(step_size, remaining);
    (void)detail::begin_step(ev.ctx);

    for (int32_t i = 0; i < chunk_size; ++i) {
      (void)detail::append_token_index(ev.ctx, chunk_start + i);
    }

    (void)detail::push_step_size(ev.ctx, chunk_size);
  }

  detail::finalize_token_offsets(ev.ctx);
};

inline constexpr auto effect_begin_planning = [](const event::plan_runtime & ev,
                                                 context &) noexcept {
  detail::prepare_plan(ev);
};

inline constexpr auto effect_reject_invalid_step_size =
    [](const event::plan_runtime & ev, context &) noexcept {
  detail::fail_plan(ev, error::invalid_step_size);
};

inline constexpr auto effect_reject_output_steps_full =
    [](const event::plan_runtime & ev, context &) noexcept {
  detail::fail_plan(ev, error::output_steps_full);
};

inline constexpr auto effect_reject_output_indices_full =
    [](const event::plan_runtime & ev, context &) noexcept {
  detail::fail_plan(ev, error::output_indices_full);
};

inline constexpr auto effect_reject_planning_progress_stalled =
    [](const event::plan_runtime & ev, context &) noexcept {
  detail::fail_plan(ev, error::planning_progress_stalled);
};

}  // namespace emel::batch::planner::modes::simple::action
