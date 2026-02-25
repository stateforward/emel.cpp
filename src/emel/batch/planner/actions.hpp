#pragma once

#include <algorithm>

#include "emel/batch/planner/modes/detail.hpp"
#include "emel/batch/planner/context.hpp"

namespace emel::batch::planner::action {

// initializes context for a new plan request.
inline constexpr auto begin_plan = [](const event::request &, context & ctx) noexcept {
  ctx.effective_step_size = 0;
  ctx.step_count = 0;
  ctx.total_outputs = 0;
  ctx.step_sizes.fill(0);
  ctx.step_token_indices.fill(0);
  ctx.step_token_offsets.fill(0);
  ctx.token_indices_count = 0;
};

// normalizes the requested step size.
inline constexpr auto normalize_batch = [](const event::request & ev, context & ctx) noexcept {
  const int32_t default_step = ev.n_tokens;
  const int32_t requested = ev.n_steps > 0 ? ev.n_steps : default_step;
  ctx.effective_step_size = std::max<int32_t>(1, std::min<int32_t>(requested, ev.n_tokens));
};

// publishes plan outputs (output write-back happens in caller via callbacks).
inline constexpr auto publish = [](context &) noexcept {};

inline constexpr auto dispatch_done = [](const event::request & ev, const context & ctx) noexcept {
  ev.on_done(events::plan_done{
    .request = &ev,
    .step_sizes = ctx.step_sizes.data(),
    .step_count = ctx.step_count,
    .total_outputs = ctx.total_outputs,
    .step_token_indices = ctx.step_token_indices.data(),
    .step_token_indices_count = ctx.token_indices_count,
    .step_token_offsets = ctx.step_token_offsets.data(),
    .step_token_offsets_count = ctx.step_count + 1,
  });
};

inline constexpr auto dispatch_invalid_request = [](const event::request & ev,
                                                  const emel::error::type err) noexcept {
  ev.on_error(events::plan_error{
    .err = err,
    .request = &ev,
  });
};

inline constexpr auto dispatch_plan_failed = [](const event::request & ev,
                                              const emel::error::type err) noexcept {
  ev.on_error(events::plan_error{
    .err = err,
    .request = &ev,
  });
};

inline constexpr auto dispatch_invalid_request_default = [](const event::request & ev,
                                                            const context &) noexcept {
  dispatch_invalid_request(ev, emel::error::cast(error::invalid_request));
};

inline constexpr auto dispatch_plan_failed_default = [](const event::request & ev,
                                                        const context &) noexcept {
  dispatch_plan_failed(ev, emel::error::cast(error::internal_error));
};

inline constexpr auto dispatch_unexpected = [](const auto & ev) noexcept {
  if constexpr (requires { ev.on_error; }) {
    ev.on_error(events::plan_error{
      .err = emel::error::cast(error::invalid_request),
      .request = nullptr,
    });
  }
};

inline constexpr auto dispatch_invalid_mode = [](const event::request & ev,
                                                 context & ctx) noexcept {
      dispatch_invalid_request_default(ev, ctx);
    };

}  // namespace emel::batch::planner::action
