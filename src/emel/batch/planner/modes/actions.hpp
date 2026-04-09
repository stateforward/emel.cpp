#pragma once

#include "emel/batch/planner/detail.hpp"

namespace emel::batch::planner::modes::action {

template <class done_event, class mode_request>
inline void emit_plan_done(const mode_request & request) noexcept {
  request.on_done(done_event{
    .request = request,
  });
}

template <class runtime_event>
inline emel::error::type resolve_plan_error(const runtime_event & ev) noexcept {
  return detail::select_error(ev.ctx.err != emel::error::cast(error::none),
                              ev.ctx.err,
                              emel::error::cast(error::internal_error));
}

template <class error_event, class mode_request>
inline void emit_plan_error(const mode_request & request, const emel::error::type err) noexcept {
  request.on_error(error_event{
    .request = request,
    .err = err,
  });
}

}  // namespace emel::batch::planner::modes::action
