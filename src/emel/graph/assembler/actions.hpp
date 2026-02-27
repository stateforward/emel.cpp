#pragma once

#include <cstdint>

#include "emel/graph/assembler/context.hpp"
#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"

namespace emel::graph::assembler::action {

inline void reset_reserve_output(event::reserve_output & output) noexcept {
  output.graph_topology = nullptr;
  output.node_count = 0;
  output.tensor_count = 0;
  output.required_buffer_bytes = 0;
  output.version = 0;
}

inline void reset_assemble_output(event::assemble_output & output) noexcept {
  output.graph_topology = nullptr;
  output.node_count = 0;
  output.tensor_count = 0;
  output.required_buffer_bytes = 0;
  output.version = 0;
  output.reused_topology = 0;
}

struct reject_invalid_reserve_with_dispatch {
  void operator()(const event::reserve_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_reserve_output(*ev.request.output_out);
    ev.request.dispatch_error(events::reserve_error{
      *ev.request.output_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct reject_invalid_reserve_with_output_only {
  void operator()(const event::reserve_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_reserve_output(*ev.request.output_out);
  }
};

struct reject_invalid_reserve_without_output {
  void operator()(const event::reserve_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct reject_invalid_assemble_with_dispatch {
  void operator()(const event::assemble_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_assemble_output(*ev.request.output_out);
    ev.request.dispatch_error(events::assemble_error{
      *ev.request.output_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct reject_invalid_assemble_with_output_only {
  void operator()(const event::assemble_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_assemble_output(*ev.request.output_out);
  }
};

struct reject_invalid_assemble_without_output {
  void operator()(const event::assemble_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct begin_reserve {
  void operator()(const event::reserve_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.validate_outcome = reserve_validate_pass::events::phase_outcome::unknown;
    ev.ctx.build_outcome = reserve_build_pass::events::phase_outcome::unknown;
    ev.ctx.alloc_outcome = reserve_alloc_pass::events::phase_outcome::unknown;
    ev.ctx.assembled_node_count = 0;
    ev.ctx.assembled_tensor_count = 0;
    ev.ctx.alloc_plan = {};
    reset_reserve_output(*ev.request.output_out);
  }
};

struct begin_assemble {
  void operator()(const event::assemble_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.validate_outcome = assemble_validate_pass::events::phase_outcome::unknown;
    ev.ctx.reuse_outcome = reuse_decision_pass::events::phase_outcome::unknown;
    ev.ctx.build_outcome = assemble_build_pass::events::phase_outcome::unknown;
    ev.ctx.alloc_outcome = assemble_alloc_pass::events::phase_outcome::unknown;
    ev.ctx.assembled_node_count = 0;
    ev.ctx.assembled_tensor_count = 0;
    ev.ctx.alloc_plan = {};
    ev.ctx.reused_topology = 0u;
    reset_assemble_output(*ev.request.output_out);
  }
};

struct commit_reserve_result {
  void operator()(const event::reserve_graph & ev, context & ctx) const noexcept {
    ctx.reserved_topology = ev.request.model_topology;
    ctx.reserved_node_count = ev.ctx.assembled_node_count;
    ctx.reserved_tensor_count = ev.ctx.assembled_tensor_count;
    ctx.reserved_required_buffer_bytes = ev.ctx.alloc_plan.required_buffer_bytes;
    ++ctx.topology_version;
    ctx.has_reserved_topology = 1u;

    ev.request.output_out->graph_topology = ctx.reserved_topology;
    ev.request.output_out->node_count = ctx.reserved_node_count;
    ev.request.output_out->tensor_count = ctx.reserved_tensor_count;
    ev.request.output_out->required_buffer_bytes = ctx.reserved_required_buffer_bytes;
    ev.request.output_out->version = ctx.topology_version;
  }
};

struct commit_assemble_reuse_result {
  void operator()(const event::assemble_graph & ev, context & ctx) const noexcept {
    ev.request.output_out->graph_topology = ctx.reserved_topology;
    ev.request.output_out->node_count = ctx.reserved_node_count;
    ev.request.output_out->tensor_count = ctx.reserved_tensor_count;
    ev.request.output_out->required_buffer_bytes = ctx.reserved_required_buffer_bytes;
    ev.request.output_out->version = ctx.topology_version;
    ev.request.output_out->reused_topology = 1u;
  }
};

struct commit_assemble_rebuild_result {
  void operator()(const event::assemble_graph & ev, context & ctx) const noexcept {
    ctx.reserved_topology = ev.request.step_plan;
    ctx.reserved_node_count = ev.ctx.assembled_node_count;
    ctx.reserved_tensor_count = ev.ctx.assembled_tensor_count;
    ctx.reserved_required_buffer_bytes = ev.ctx.alloc_plan.required_buffer_bytes;
    ++ctx.topology_version;
    ctx.has_reserved_topology = 1u;

    ev.request.output_out->graph_topology = ctx.reserved_topology;
    ev.request.output_out->node_count = ctx.reserved_node_count;
    ev.request.output_out->tensor_count = ctx.reserved_tensor_count;
    ev.request.output_out->required_buffer_bytes = ctx.reserved_required_buffer_bytes;
    ev.request.output_out->version = ctx.topology_version;
    ev.request.output_out->reused_topology = 0u;
  }
};

struct dispatch_reserve_done {
  void operator()(const event::reserve_graph & ev, const context &) const noexcept {
    ev.request.dispatch_done(events::reserve_done{
      *ev.request.output_out,
    });
  }
};

struct dispatch_reserve_error {
  void operator()(const event::reserve_graph & ev, const context &) const noexcept {
    ev.request.dispatch_error(events::reserve_error{
      *ev.request.output_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct dispatch_assemble_done {
  void operator()(const event::assemble_graph & ev, const context &) const noexcept {
    ev.request.dispatch_done(events::assemble_done{
      *ev.request.output_out,
    });
  }
};

struct dispatch_assemble_error {
  void operator()(const event::assemble_graph & ev, const context &) const noexcept {
    ev.request.dispatch_error(events::assemble_error{
      *ev.request.output_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
    }
  }
};

inline constexpr reject_invalid_reserve_with_dispatch reject_invalid_reserve_with_dispatch{};
inline constexpr reject_invalid_reserve_with_output_only reject_invalid_reserve_with_output_only{};
inline constexpr reject_invalid_reserve_without_output reject_invalid_reserve_without_output{};
inline constexpr reject_invalid_assemble_with_dispatch reject_invalid_assemble_with_dispatch{};
inline constexpr reject_invalid_assemble_with_output_only reject_invalid_assemble_with_output_only{};
inline constexpr reject_invalid_assemble_without_output reject_invalid_assemble_without_output{};
inline constexpr begin_reserve begin_reserve{};
inline constexpr begin_assemble begin_assemble{};
inline constexpr commit_reserve_result commit_reserve_result{};
inline constexpr commit_assemble_reuse_result commit_assemble_reuse_result{};
inline constexpr commit_assemble_rebuild_result commit_assemble_rebuild_result{};
inline constexpr dispatch_reserve_done dispatch_reserve_done{};
inline constexpr dispatch_reserve_error dispatch_reserve_error{};
inline constexpr dispatch_assemble_done dispatch_assemble_done{};
inline constexpr dispatch_assemble_error dispatch_assemble_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::graph::assembler::action
