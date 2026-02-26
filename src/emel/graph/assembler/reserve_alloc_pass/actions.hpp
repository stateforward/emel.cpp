#pragma once

#include "emel/callback.hpp"
#include "emel/graph/assembler/context.hpp"
#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"
#include "emel/graph/assembler/reserve_alloc_pass/context.hpp"
#include "emel/graph/assembler/reserve_alloc_pass/events.hpp"

namespace emel::graph::assembler::reserve_alloc_pass::action {

struct allocator_reply {
  assembler::event::reserve_ctx * reserve_ctx = nullptr;
};

inline bool on_allocator_done(void * owner, const allocator::events::allocation_done & ev) noexcept {
  auto * reply = static_cast<allocator_reply *>(owner);
  reply->reserve_ctx->alloc_plan = ev.plan;
  reply->reserve_ctx->alloc_outcome = events::phase_outcome::done;
  reply->reserve_ctx->err = emel::error::cast(assembler::error::none);
  return true;
}

inline bool on_allocator_error(void * owner, const allocator::events::allocation_error & ev) noexcept {
  auto * reply = static_cast<allocator_reply *>(owner);
  reply->reserve_ctx->alloc_outcome = events::phase_outcome::failed;
  reply->reserve_ctx->err = static_cast<emel::error::type>(ev.err);
  return true;
}

struct request_allocator_plan {
  void operator()(const assembler::event::reserve_graph & ev, context & ctx) const noexcept {
    ev.ctx.alloc_outcome = events::phase_outcome::failed;
    ev.ctx.alloc_plan = {};
    ev.ctx.err = emel::error::cast(assembler::error::internal_error);

    allocator_reply reply{
      &ev.ctx,
    };
    const callback<bool(const allocator::events::allocation_done &)> done_cb{
      &reply,
      on_allocator_done,
    };
    const callback<bool(const allocator::events::allocation_error &)> error_cb{
      &reply,
      on_allocator_error,
    };

    const allocator::event::allocate_graph allocate_ev{
      .graph_topology = ev.request.model_topology,
      .plan_out = &ev.ctx.alloc_plan,
      .node_count = ev.ctx.assembled_node_count,
      .tensor_count = ev.ctx.assembled_tensor_count,
      .tensor_capacity = ev.ctx.assembled_tensor_count,
      .interval_capacity = ev.ctx.assembled_tensor_count,
      .bytes_per_tensor = ev.request.bytes_per_tensor,
      .workspace_capacity_bytes = ev.request.workspace_capacity_bytes,
      .dispatch_done = done_cb,
      .dispatch_error = error_cb,
    };
    (void)ctx.allocator_actor.process_event(allocate_ev);
  }
};

struct mark_failed_prereq {
  void operator()(const assembler::event::reserve_graph & ev, context &) const noexcept {
    ev.ctx.alloc_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(assembler::error::internal_error);
  }
};

struct mark_failed_invalid_request {
  void operator()(const assembler::event::reserve_graph & ev, context &) const noexcept {
    ev.ctx.alloc_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(assembler::error::invalid_request);
  }
};

struct on_unexpected {
  void operator()(const assembler::event::reserve_graph & ev, const context &) const noexcept {
    ev.ctx.alloc_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(assembler::error::internal_error);
  }

  template <class event_type>
  void operator()(const event_type &, const context &) const noexcept {
  }
};

inline constexpr request_allocator_plan request_allocator_plan{};
inline constexpr mark_failed_prereq mark_failed_prereq{};
inline constexpr mark_failed_invalid_request mark_failed_invalid_request{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::graph::assembler::reserve_alloc_pass::action
