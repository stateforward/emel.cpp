#pragma once

#include "emel/graph/context.hpp"
#include "emel/graph/errors.hpp"
#include "emel/graph/events.hpp"

namespace emel::graph::guard {

inline emel::error::type runtime_error(const event::compute_graph & ev) noexcept {
  return ev.ctx.err;
}

inline emel::error::type runtime_error(const event::compute_reserved_graph & ev) noexcept {
  return ev.ctx.err;
}

inline bool error_is(const emel::error::type runtime_err,
                     const error expected) noexcept {
  return runtime_err == emel::error::cast(expected);
}

inline bool error_is_unknown(const emel::error::type runtime_err) noexcept {
  return !error_is(runtime_err, error::none) &&
         !error_is(runtime_err, error::invalid_request) &&
         !error_is(runtime_err, error::assembler_failed) &&
         !error_is(runtime_err, error::processor_failed) &&
         !error_is(runtime_err, error::busy) &&
         !error_is(runtime_err, error::internal_error) &&
         !error_is(runtime_err, error::untracked);
}

template <class compute_request>
inline bool guard_valid_compute_execution_request(const compute_request & request) noexcept {
  return request.step_plan != nullptr &&
         request.output_out != nullptr &&
         request.lifecycle != nullptr &&
         request.lifecycle->tensors != nullptr &&
         request.lifecycle->tensor_count > 0 &&
         request.step_index >= 0 &&
         request.step_size > 0 &&
         request.kv_tokens >= 0 &&
         request.expected_outputs >= 0 &&
         request.positions_count >= 0 &&
         request.seq_mask_words > 0 &&
         request.seq_masks_count >= 0 &&
         request.seq_primary_ids_count >= 0 &&
         request.prepare_graph != nullptr &&
         request.bind_inputs != nullptr &&
         request.run_kernel != nullptr &&
         request.extract_outputs != nullptr &&
         static_cast<bool>(request.dispatch_done) &&
         static_cast<bool>(request.dispatch_error);
}

struct valid_reserve {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.request.model_topology != nullptr &&
           ev.request.output_out != nullptr &&
           ev.request.lifecycle != nullptr &&
           ev.request.lifecycle->tensors != nullptr &&
           ev.request.lifecycle->tensor_count > 0 &&
           ev.request.max_node_count != 0u &&
           ev.request.max_tensor_count != 0u &&
           ev.request.bytes_per_tensor != 0u &&
           ev.request.workspace_capacity_bytes != 0u &&
           static_cast<bool>(ev.request.dispatch_done) &&
           static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_reserve {
  bool operator()(const event::reserve_graph & ev, const action::context & ctx) const noexcept {
    return !valid_reserve{}(ev, ctx);
  }
};

struct invalid_reserve_with_dispatchable_output {
  bool operator()(const event::reserve_graph & ev, const action::context & ctx) const noexcept {
    return invalid_reserve{}(ev, ctx) &&
           ev.request.output_out != nullptr &&
           static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_reserve_with_output_only {
  bool operator()(const event::reserve_graph & ev, const action::context & ctx) const noexcept {
    return invalid_reserve{}(ev, ctx) &&
           ev.request.output_out != nullptr &&
           !static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_reserve_without_output {
  bool operator()(const event::reserve_graph & ev, const action::context & ctx) const noexcept {
    return invalid_reserve{}(ev, ctx) && ev.request.output_out == nullptr;
  }
};

struct valid_compute {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return guard_valid_compute_execution_request(ev.request) &&
           ev.request.bytes_per_tensor != 0u &&
           ev.request.workspace_capacity_bytes != 0u;
  }
};

struct guard_valid_compute_reserved {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context & ctx) const noexcept {
    return guard_valid_compute_execution_request(ev.request) &&
           ev.request.lifecycle->phase != nullptr &&
           ctx.reservation.graph_topology != nullptr &&
           ctx.reservation.node_count != 0u &&
           ctx.reservation.tensor_count != 0u &&
           ctx.reservation.required_buffer_bytes != 0u &&
           ctx.reservation.lifecycle != nullptr &&
           ctx.reservation.lifecycle->tensors == ev.request.lifecycle->tensors &&
           ctx.reservation.lifecycle->tensor_count ==
               ev.request.lifecycle->tensor_count;
  }
};

struct invalid_compute {
  bool operator()(const event::compute_graph & ev, const action::context & ctx) const noexcept {
    return !valid_compute{}(ev, ctx);
  }
};

struct guard_invalid_compute_reserved {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context & ctx) const noexcept {
    return !guard_valid_compute_reserved{}(ev, ctx);
  }
};

struct invalid_compute_with_dispatchable_output {
  bool operator()(const event::compute_graph & ev, const action::context & ctx) const noexcept {
    return invalid_compute{}(ev, ctx) &&
           ev.request.output_out != nullptr &&
           static_cast<bool>(ev.request.dispatch_error);
  }
};

struct guard_invalid_compute_reserved_with_dispatchable_output {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context & ctx) const noexcept {
    return guard_invalid_compute_reserved{}(ev, ctx) &&
           ev.request.output_out != nullptr &&
           static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_compute_with_output_only {
  bool operator()(const event::compute_graph & ev, const action::context & ctx) const noexcept {
    return invalid_compute{}(ev, ctx) &&
           ev.request.output_out != nullptr &&
           !static_cast<bool>(ev.request.dispatch_error);
  }
};

struct guard_invalid_compute_reserved_with_output_only {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context & ctx) const noexcept {
    return guard_invalid_compute_reserved{}(ev, ctx) &&
           ev.request.output_out != nullptr &&
           !static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_compute_without_output {
  bool operator()(const event::compute_graph & ev, const action::context & ctx) const noexcept {
    return invalid_compute{}(ev, ctx) && ev.request.output_out == nullptr;
  }
};

struct guard_invalid_compute_reserved_without_output {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context & ctx) const noexcept {
    return guard_invalid_compute_reserved{}(ev, ctx) && ev.request.output_out == nullptr;
  }
};

struct reserve_done {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.reserve_outcome == event::phase_outcome::done;
  }
};

struct tensor_reserve_done {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.tensor_reserve_outcome == event::phase_outcome::done;
  }
};

struct tensor_reserve_failed {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.tensor_reserve_outcome == event::phase_outcome::failed;
  }
};

struct reserve_failed {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.reserve_outcome == event::phase_outcome::failed;
  }
};

struct assemble_done {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.assemble_outcome == event::phase_outcome::done;
  }
};

struct assemble_failed {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.assemble_outcome == event::phase_outcome::failed;
  }
};

struct execute_done {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.execute_outcome == event::phase_outcome::done;
  }
};

struct guard_reserved_execute_done {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.execute_outcome == event::phase_outcome::done;
  }
};

struct execute_failed {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.execute_outcome == event::phase_outcome::failed;
  }
};

struct guard_reserved_execute_failed {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.execute_outcome == event::phase_outcome::failed;
  }
};

struct compute_error_none {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::none);
  }
};

struct guard_reserved_compute_error_none {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::none);
  }
};

struct compute_error_invalid_request {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::invalid_request);
  }
};

struct guard_reserved_compute_error_invalid_request {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::invalid_request);
  }
};

struct compute_error_assembler_failed {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::assembler_failed);
  }
};

struct guard_reserved_compute_error_assembler_failed {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::assembler_failed);
  }
};

struct compute_error_processor_failed {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::processor_failed);
  }
};

struct guard_reserved_compute_error_processor_failed {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::processor_failed);
  }
};

struct compute_error_busy {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::busy);
  }
};

struct guard_reserved_compute_error_busy {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::busy);
  }
};

struct compute_error_internal_error {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::internal_error);
  }
};

struct guard_reserved_compute_error_internal_error {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::internal_error);
  }
};

struct compute_error_untracked {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::untracked);
  }
};

struct guard_reserved_compute_error_untracked {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::untracked);
  }
};

struct compute_error_unknown {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return error_is_unknown(runtime_error(ev));
  }
};

struct guard_reserved_compute_error_unknown {
  bool operator()(const event::compute_reserved_graph & ev,
                  const action::context &) const noexcept {
    return error_is_unknown(runtime_error(ev));
  }
};

}  // namespace emel::graph::guard
