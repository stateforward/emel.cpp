#pragma once

#include <algorithm>

#include "emel/batch/planner/modes/detail.hpp"

namespace emel::batch::planner::modes::simple::action {

using context = emel::batch::planner::action::context;

inline void create_plan_impl(const event::request_runtime & ev) noexcept {
  if (ev.ctx.effective_step_size <= 0) {
    detail::fail_plan(ev, emel::batch::planner::error::invalid_step_size);
    return;
  }

  int32_t next_token = 0;
  while (next_token < ev.request.n_tokens) {
    if (!detail::begin_step(ev.ctx)) {
      detail::fail_plan(ev, emel::batch::planner::error::output_steps_full);
      return;
    }
    const int32_t chunk =
        std::min<int32_t>(ev.ctx.effective_step_size, ev.request.n_tokens - next_token);
    for (int32_t i = 0; i < chunk; ++i) {
      if (!detail::append_token_index(ev.ctx, next_token + i)) {
        detail::fail_plan(ev, emel::batch::planner::error::output_indices_full);
        return;
      }
    }
    next_token += chunk;
    if (!detail::push_step_size(ev.ctx, chunk)) {
      detail::fail_plan(ev, emel::batch::planner::error::output_steps_full);
      return;
    }
  }
  detail::finalize_token_offsets(ev.ctx);
}

inline constexpr auto prepare_steps = [](const event::request_runtime & ev, context &) noexcept {
  detail::prepare_plan(ev);
};

inline constexpr auto create_plan = [](const event::request_runtime & ev, context &) noexcept {
  create_plan_impl(ev);
};

}  // namespace emel::batch::planner::modes::simple::action
