#pragma once

#include <algorithm>

#include "emel/batch/planner/context.hpp"
#include "emel/batch/planner/modes/detail.hpp"

namespace emel::batch::planner::action {

inline void dispatch_invalid_request(const event::request & request,
                                     const emel::error::type err) noexcept {
  request.on_error(events::plan_error{
    .err = err,
    .request = &request,
  });
}

inline void dispatch_plan_failed(const event::request & request,
                                 const emel::error::type err) noexcept {
  request.on_error(events::plan_error{
    .err = err,
    .request = &request,
  });
}

inline constexpr auto begin_plan = [](const event::request_runtime & ev, context &) noexcept {
  ev.ctx.err = emel::error::cast(error::none);
  ev.ctx.effective_step_size = 0;
  ev.ctx.step_count = 0;
  ev.ctx.total_outputs = 0;
  ev.ctx.step_sizes.fill(0);
  ev.ctx.step_token_indices.fill(0);
  ev.ctx.step_token_offsets.fill(0);
  ev.ctx.token_indices_count = 0;
};

inline constexpr auto normalize_batch = [](const event::request_runtime & ev, context &) noexcept {
  const int32_t default_step = ev.request.n_tokens;
  const int32_t requested = ev.request.n_steps > 0 ? ev.request.n_steps : default_step;
  ev.ctx.effective_step_size =
      std::max<int32_t>(1, std::min<int32_t>(requested, ev.request.n_tokens));
};

inline constexpr auto publish = [](const event::request_runtime &, context &) noexcept {};

inline constexpr auto dispatch_done = [](const event::request_runtime & ev,
                                         const context &) noexcept {
  ev.request.on_done(events::plan_done{
    .request = &ev.request,
    .step_sizes = ev.ctx.step_sizes.data(),
    .step_count = ev.ctx.step_count,
    .total_outputs = ev.ctx.total_outputs,
    .step_token_indices = ev.ctx.step_token_indices.data(),
    .step_token_indices_count = ev.ctx.token_indices_count,
    .step_token_offsets = ev.ctx.step_token_offsets.data(),
    .step_token_offsets_count = ev.ctx.step_count + 1,
  });
};

inline constexpr auto mark_invalid_request = [](const event::request_runtime & ev,
                                                context &) noexcept {
  const emel::error::type input_mask = modes::detail::collect_input_errors(ev.request);
  ev.ctx.err = emel::error::set(input_mask, error::invalid_request);
  dispatch_invalid_request(ev.request, ev.ctx.err);
};

inline constexpr auto mark_invalid_mode = [](const event::request_runtime & ev,
                                             context &) noexcept {
  ev.ctx.err = emel::error::set(ev.ctx.err, error::invalid_mode);
  ev.ctx.err = emel::error::set(ev.ctx.err, error::invalid_request);
  dispatch_invalid_request(ev.request, ev.ctx.err);
};

inline constexpr auto dispatch_plan_failed_with_ctx_error = [](const event::request_runtime & ev,
                                                               const context &) noexcept {
  dispatch_plan_failed(ev.request, ev.ctx.err);
};

inline constexpr auto dispatch_plan_failed_internal = [](const event::request_runtime & ev,
                                                         const context &) noexcept {
  dispatch_plan_failed(ev.request, emel::error::cast(error::internal_error));
};

inline constexpr auto on_unexpected = [](const auto & ev) noexcept {
  if constexpr (requires { ev.request; ev.ctx; }) {
    ev.ctx.err = emel::error::set(ev.ctx.err, error::untracked);
    ev.request.on_error(events::plan_error{
      .err = ev.ctx.err,
      .request = &ev.request,
    });
  }
};

}  // namespace emel::batch::planner::action
