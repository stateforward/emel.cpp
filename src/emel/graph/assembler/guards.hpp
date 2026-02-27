#pragma once

#include "emel/graph/assembler/context.hpp"
#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"

namespace emel::graph::assembler::guard {

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

struct valid_assemble {
  bool operator()(const event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.request.step_plan != nullptr &&
           ev.request.output_out != nullptr &&
           ev.request.bytes_per_tensor != 0u &&
           ev.request.workspace_capacity_bytes != 0u &&
           static_cast<bool>(ev.request.dispatch_done) &&
           static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_assemble {
  bool operator()(const event::assemble_graph & ev, const action::context & ctx) const noexcept {
    return !valid_assemble{}(ev, ctx);
  }
};

struct invalid_assemble_with_dispatchable_output {
  bool operator()(const event::assemble_graph & ev, const action::context & ctx) const noexcept {
    return invalid_assemble{}(ev, ctx) &&
           ev.request.output_out != nullptr &&
           static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_assemble_with_output_only {
  bool operator()(const event::assemble_graph & ev, const action::context & ctx) const noexcept {
    return invalid_assemble{}(ev, ctx) &&
           ev.request.output_out != nullptr &&
           !static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_assemble_without_output {
  bool operator()(const event::assemble_graph & ev, const action::context & ctx) const noexcept {
    return invalid_assemble{}(ev, ctx) && ev.request.output_out == nullptr;
  }
};

struct reserve_validate_done {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.validate_outcome == reserve_validate_pass::events::phase_outcome::done;
  }
};

struct reserve_validate_failed {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.validate_outcome == reserve_validate_pass::events::phase_outcome::failed;
  }
};

struct reserve_build_done {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.build_outcome == reserve_build_pass::events::phase_outcome::done;
  }
};

struct reserve_build_failed {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.build_outcome == reserve_build_pass::events::phase_outcome::failed;
  }
};

struct reserve_alloc_done {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.alloc_outcome == reserve_alloc_pass::events::phase_outcome::done;
  }
};

struct reserve_alloc_failed {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.alloc_outcome == reserve_alloc_pass::events::phase_outcome::failed;
  }
};

struct reserve_phase_ok {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct reserve_phase_failed {
  bool operator()(const event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

struct assemble_validate_done {
  bool operator()(const event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.validate_outcome == assemble_validate_pass::events::phase_outcome::done;
  }
};

struct assemble_validate_failed {
  bool operator()(const event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.validate_outcome == assemble_validate_pass::events::phase_outcome::failed;
  }
};

struct reuse_decision_reused {
  bool operator()(const event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.reuse_outcome == reuse_decision_pass::events::phase_outcome::reused;
  }
};

struct reuse_decision_rebuild {
  bool operator()(const event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.reuse_outcome == reuse_decision_pass::events::phase_outcome::rebuild;
  }
};

struct reuse_decision_failed {
  bool operator()(const event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.reuse_outcome == reuse_decision_pass::events::phase_outcome::failed;
  }
};

struct assemble_build_done {
  bool operator()(const event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.build_outcome == assemble_build_pass::events::phase_outcome::done;
  }
};

struct assemble_build_failed {
  bool operator()(const event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.build_outcome == assemble_build_pass::events::phase_outcome::failed;
  }
};

struct assemble_alloc_done {
  bool operator()(const event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.alloc_outcome == assemble_alloc_pass::events::phase_outcome::done;
  }
};

struct assemble_alloc_failed {
  bool operator()(const event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.alloc_outcome == assemble_alloc_pass::events::phase_outcome::failed;
  }
};

struct assemble_phase_ok {
  bool operator()(const event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct assemble_phase_failed {
  bool operator()(const event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

}  // namespace emel::graph::assembler::guard
