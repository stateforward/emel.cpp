#pragma once

#include <algorithm>

#include "emel/batch/planner/modes/detail.hpp"
#include "emel/batch/planner/context.hpp"

namespace emel::batch::planner::modes::simple::action {

using context = emel::batch::planner::action::context;

inline void create_plan_impl(const event::request & ev, context & ctx) noexcept {
  int32_t next_token = 0;
  while (next_token < ev.n_tokens) {
    if (!detail::begin_step(ctx)) {
      detail::fail_plan(ctx, emel::batch::planner::error::output_steps_full);
      return;
    }
    const int32_t chunk =
        std::min<int32_t>(ctx.effective_step_size, ev.n_tokens - next_token);
    for (int32_t i = 0; i < chunk; ++i) {
      if (!detail::append_token_index(ctx, next_token + i)) {
        detail::fail_plan(ctx,
                          emel::batch::planner::error::output_indices_full);
        return;
      }
    }
    next_token += chunk;
    if (!detail::push_step_size(ctx, chunk)) {
      detail::fail_plan(ctx, emel::batch::planner::error::output_steps_full);
      return;
    }
  }
  detail::finalize_token_offsets(ctx);
}

inline constexpr auto prepare_steps = [](const event::request & ev, context & ctx) noexcept {
  detail::prepare_plan(ev, ctx);
};

inline constexpr auto create_plan = [](const event::request & ev, context & ctx) noexcept {
  create_plan_impl(ev, ctx);
};

}  // namespace emel::batch::planner::modes::simple::action
