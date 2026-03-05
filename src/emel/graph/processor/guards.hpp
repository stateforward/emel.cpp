#pragma once

#include "emel/graph/processor/context.hpp"
#include "emel/graph/processor/errors.hpp"
#include "emel/graph/processor/events.hpp"

namespace emel::graph::processor::guard {

inline emel::error::type runtime_error(const event::execute_step & ev) noexcept {
  return ev.ctx.err;
}

inline bool error_is(const emel::error::type runtime_err,
                     const error expected) noexcept {
  return runtime_err == emel::error::cast(expected);
}

inline bool error_is_unknown(const emel::error::type runtime_err) noexcept {
  return !error_is(runtime_err, error::none) &&
         !error_is(runtime_err, error::invalid_request) &&
         !error_is(runtime_err, error::kernel_failed) &&
         !error_is(runtime_err, error::internal_error) &&
         !error_is(runtime_err, error::untracked);
}

struct valid_execute {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.request.step_plan != nullptr &&
           ev.request.output_out != nullptr &&
           ev.request.step_index >= 0 &&
           ev.request.step_size > 0 &&
           ev.request.kv_tokens >= 0 &&
           ev.request.expected_outputs >= 0 &&
           ev.request.positions_count >= 0 &&
           ev.request.seq_mask_words > 0 &&
           ev.request.seq_masks_count >= 0 &&
           ev.request.seq_primary_ids_count >= 0 &&
           static_cast<bool>(ev.request.dispatch_done) &&
           static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_execute {
  bool operator()(const event::execute_step & ev, const action::context & ctx) const noexcept {
    return !valid_execute{}(ev, ctx);
  }
};

struct invalid_execute_with_dispatchable_output {
  bool operator()(const event::execute_step & ev, const action::context & ctx) const noexcept {
    return invalid_execute{}(ev, ctx) &&
           ev.request.output_out != nullptr &&
           static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_execute_with_output_only {
  bool operator()(const event::execute_step & ev, const action::context & ctx) const noexcept {
    return invalid_execute{}(ev, ctx) &&
           ev.request.output_out != nullptr &&
           !static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_execute_without_output {
  bool operator()(const event::execute_step & ev, const action::context & ctx) const noexcept {
    return invalid_execute{}(ev, ctx) && ev.request.output_out == nullptr;
  }
};

struct execution_error_none {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::none);
  }
};

struct execution_error_invalid_request {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::invalid_request);
  }
};

struct execution_error_kernel_failed {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::kernel_failed);
  }
};

struct execution_error_internal_error {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::internal_error);
  }
};

struct execution_error_untracked {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::untracked);
  }
};

struct execution_error_unknown {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return error_is_unknown(runtime_error(ev));
  }
};

struct validate_done {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.validate_outcome == validate_step::events::phase_outcome::done;
  }
};

struct validate_failed {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.validate_outcome == validate_step::events::phase_outcome::failed;
  }
};

struct prepare_done_reused {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.prepare_outcome == prepare_step::events::phase_outcome::done &&
           ev.ctx.graph_reused == 1u;
  }
};

struct prepare_done_needs_allocation {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.prepare_outcome == prepare_step::events::phase_outcome::done &&
           ev.ctx.graph_reused == 0u;
  }
};

struct prepare_failed {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.prepare_outcome == prepare_step::events::phase_outcome::failed;
  }
};

struct alloc_done {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.alloc_outcome == alloc_step::events::phase_outcome::done;
  }
};

struct alloc_failed {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.alloc_outcome == alloc_step::events::phase_outcome::failed;
  }
};

struct bind_done {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.bind_outcome == bind_step::events::phase_outcome::done;
  }
};

struct bind_failed {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.bind_outcome == bind_step::events::phase_outcome::failed;
  }
};

struct kernel_done {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.kernel_outcome == kernel_step::events::phase_outcome::done;
  }
};

struct kernel_failed {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.kernel_outcome == kernel_step::events::phase_outcome::failed;
  }
};

struct extract_done {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.extract_outcome == extract_step::events::phase_outcome::done;
  }
};

struct extract_failed {
  bool operator()(const event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.extract_outcome == extract_step::events::phase_outcome::failed;
  }
};

}  // namespace emel::graph::processor::guard
