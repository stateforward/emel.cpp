#pragma once

#include "emel/graph/allocator/placement_pass/context.hpp"
#include "emel/graph/allocator/placement_pass/events.hpp"
#include "emel/graph/allocator/errors.hpp"
#include "emel/graph/allocator/events.hpp"

namespace emel::graph::allocator::placement_pass::action {

struct mark_done {
  void operator()(const allocator::event::allocate_graph_plan & ev,
                  context &) const noexcept {
    ev.ctx.placement_outcome = events::phase_outcome::done;
    ev.ctx.err = emel::error::cast(allocator::error::none);
  }
};

struct mark_failed_prereq {
  void operator()(const allocator::event::allocate_graph_plan & ev,
                  context &) const noexcept {
    ev.ctx.placement_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(allocator::error::internal_error);
  }
};

struct mark_failed_capacity {
  void operator()(const allocator::event::allocate_graph_plan & ev,
                  context &) const noexcept {
    ev.ctx.placement_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(allocator::error::capacity);
  }
};

struct mark_failed_invalid_request {
  void operator()(const allocator::event::allocate_graph_plan & ev,
                  context &) const noexcept {
    ev.ctx.placement_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(allocator::error::invalid_request);
  }
};

struct mark_failed_internal {
  void operator()(const allocator::event::allocate_graph_plan & ev,
                  context &) const noexcept {
    ev.ctx.placement_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(allocator::error::internal_error);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, const context &) const noexcept {
    if constexpr (requires { ev.ctx.placement_outcome; ev.ctx.err; }) {
      ev.ctx.placement_outcome = events::phase_outcome::failed;
      ev.ctx.err = emel::error::cast(allocator::error::internal_error);
    }
  }
};

inline constexpr mark_done mark_done{};
inline constexpr mark_failed_prereq mark_failed_prereq{};
inline constexpr mark_failed_capacity mark_failed_capacity{};
inline constexpr mark_failed_invalid_request mark_failed_invalid_request{};
inline constexpr mark_failed_internal mark_failed_internal{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::graph::allocator::placement_pass::action
