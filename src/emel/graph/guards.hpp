#pragma once

#include "emel/graph/context.hpp"
#include "emel/graph/errors.hpp"
#include "emel/graph/events.hpp"

namespace emel::graph::guard {

struct valid_reserve {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.request.model_topology != nullptr &&
           ev.request.output_out != nullptr &&
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
    return ev.request.step_plan != nullptr &&
           ev.request.output_out != nullptr &&
           ev.request.bytes_per_tensor != 0u &&
           ev.request.workspace_capacity_bytes != 0u &&
           ev.request.step_index >= 0 &&
           ev.request.step_size > 0 &&
           ev.request.kv_tokens >= 0 &&
           ev.request.expected_outputs >= 0 &&
           ev.request.positions_count >= 0 &&
           ev.request.seq_mask_words > 0 &&
           ev.request.seq_masks_count >= 0 &&
           ev.request.seq_primary_ids_count >= 0 &&
           ev.request.prepare_graph != nullptr &&
           ev.request.bind_inputs != nullptr &&
           ev.request.run_kernel != nullptr &&
           ev.request.extract_outputs != nullptr &&
           static_cast<bool>(ev.request.dispatch_done) &&
           static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_compute {
  bool operator()(const event::compute_graph & ev, const action::context & ctx) const noexcept {
    return !valid_compute{}(ev, ctx);
  }
};

struct invalid_compute_with_dispatchable_output {
  bool operator()(const event::compute_graph & ev, const action::context & ctx) const noexcept {
    return invalid_compute{}(ev, ctx) &&
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

struct invalid_compute_without_output {
  bool operator()(const event::compute_graph & ev, const action::context & ctx) const noexcept {
    return invalid_compute{}(ev, ctx) && ev.request.output_out == nullptr;
  }
};

struct reserve_done {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.reserve_outcome == event::phase_outcome::done;
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

struct execute_failed {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.execute_outcome == event::phase_outcome::failed;
  }
};

struct compute_phase_ok {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct compute_phase_failed {
  bool operator()(const event::compute_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

}  // namespace emel::graph::guard
