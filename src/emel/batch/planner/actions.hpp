#pragma once

#include <algorithm>
#include <array>

#include "emel/batch/planner/context.hpp"
#include "emel/batch/planner/detail.hpp"
#include "emel/batch/planner/modes/equal/sm.hpp"
#include "emel/batch/planner/modes/sequential/sm.hpp"
#include "emel/batch/planner/modes/simple/sm.hpp"

namespace emel::batch::planner::action {

template <class done_event, class error_event>
struct mode_outcome_capture {
  emel::error::type err = emel::error::type{};

  void on_done(const done_event &) noexcept {}

  void on_error(const error_event & ev) noexcept {
    err = ev.err;
  }
};

template <class mode_sm, class mode_request, class mode_done, class mode_error>
inline void run_mode_request(const event::plan_runtime & ev) noexcept {
  mode_outcome_capture<mode_done, mode_error> capture{};
  const auto on_done =
      emel::callback<void(const mode_done &)>::template from<
          mode_outcome_capture<mode_done, mode_error>,
          &mode_outcome_capture<mode_done, mode_error>::on_done>(&capture);
  const auto on_error =
      emel::callback<void(const mode_error &)>::template from<
          mode_outcome_capture<mode_done, mode_error>,
          &mode_outcome_capture<mode_done, mode_error>::on_error>(&capture);

  mode_sm mode{};
  (void)mode.process_event(mode_request{
    .request = ev.request,
    .ctx = ev.ctx,
    .on_done = on_done,
    .on_error = on_error,
  });
  ev.ctx.err = capture.err;
}

inline void effect_emit_plan_error(const event::plan_request & request,
                                   const emel::error::type err) noexcept {
  request.on_error(events::plan_error{
    .err = err,
    .request = &request,
  });
}

inline constexpr auto effect_begin_planning = [](const event::plan_runtime & ev,
                                                 context &) noexcept {
  ev.ctx.err = emel::error::cast(error::none);
  ev.ctx.effective_step_size = 0;
  ev.ctx.step_count = 0;
  ev.ctx.total_outputs = 0;
  ev.ctx.step_sizes.fill(0);
  ev.ctx.step_token_indices.fill(0);
  ev.ctx.step_token_offsets.fill(0);
  ev.ctx.token_indices_count = 0;
};

inline constexpr auto effect_normalize_step_size = [](const event::plan_runtime & ev,
                                                      context &) noexcept {
  const int32_t default_step = ev.request.n_tokens;
  const std::array<int32_t, 2> requested_candidates = {
      default_step,
      ev.request.n_steps,
  };
  const int32_t requested =
      requested_candidates[static_cast<size_t>(ev.request.n_steps > 0)];
  ev.ctx.effective_step_size =
      std::max<int32_t>(1, std::min<int32_t>(requested, ev.request.n_tokens));
};

inline constexpr auto effect_publish_result = [](const event::plan_runtime &,
                                                 context &) noexcept {};

inline constexpr auto effect_emit_plan_done = [](const event::plan_runtime & ev,
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

inline constexpr auto effect_reject_invalid_request = [](const event::plan_runtime & ev,
                                                         context &) noexcept {
  const emel::error::type input_mask = detail::collect_input_errors(ev.request);
  ev.ctx.err = emel::error::set(input_mask, error::invalid_request);
  effect_emit_plan_error(ev.request, ev.ctx.err);
};

inline constexpr auto effect_reject_invalid_mode = [](const event::plan_runtime & ev,
                                                      context &) noexcept {
  ev.ctx.err = emel::error::set(ev.ctx.err, error::invalid_mode);
  ev.ctx.err = emel::error::set(ev.ctx.err, error::invalid_request);
  effect_emit_plan_error(ev.request, ev.ctx.err);
};

inline constexpr auto effect_emit_planning_error = [](const event::plan_runtime & ev,
                                                      const context &) noexcept {
  effect_emit_plan_error(ev.request, ev.ctx.err);
};

inline constexpr auto effect_emit_internal_planning_error =
    [](const event::plan_runtime & ev, const context &) noexcept {
  effect_emit_plan_error(ev.request, emel::error::cast(error::internal_error));
};

inline constexpr auto effect_plan_simple_mode = [](const event::plan_runtime & ev,
                                                   context &) noexcept {
  run_mode_request<modes::simple::sm,
                   modes::simple::event::plan_request,
                   modes::simple::events::plan_done,
                   modes::simple::events::plan_error>(ev);
};

inline constexpr auto effect_plan_equal_mode = [](const event::plan_runtime & ev,
                                                  context &) noexcept {
  run_mode_request<modes::equal::sm,
                   modes::equal::event::plan_request,
                   modes::equal::events::plan_done,
                   modes::equal::events::plan_error>(ev);
};

inline constexpr auto effect_plan_sequential_mode = [](const event::plan_runtime & ev,
                                                       context &) noexcept {
  run_mode_request<modes::sequential::sm,
                   modes::sequential::event::plan_request,
                   modes::sequential::events::plan_done,
                   modes::sequential::events::plan_error>(ev);
};

inline constexpr auto effect_reject_unexpected_event = [](const auto & ev) noexcept {
  if constexpr (requires { ev.request; ev.ctx; }) {
    ev.ctx.err = emel::error::set(ev.ctx.err, error::untracked);
    ev.request.on_error(events::plan_error{
      .err = ev.ctx.err,
      .request = &ev.request,
    });
  }
};

}  // namespace emel::batch::planner::action
