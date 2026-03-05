#include <doctest/doctest.h>

#include <cstdint>
#include <limits>

#include "emel/error/error.hpp"
#include "emel/graph/allocator/actions.hpp"
#include "emel/graph/allocator/events.hpp"
#include "emel/graph/allocator/guards.hpp"
#include "emel/graph/allocator/liveness_pass/actions.hpp"
#include "emel/graph/allocator/liveness_pass/guards.hpp"
#include "emel/graph/allocator/ordering_pass/actions.hpp"
#include "emel/graph/allocator/ordering_pass/guards.hpp"
#include "emel/graph/allocator/placement_pass/actions.hpp"
#include "emel/graph/allocator/placement_pass/guards.hpp"

namespace {

struct allocator_dispatch_state {
  bool done_called = false;
  bool error_called = false;
  emel::graph::allocator::event::allocation_plan done_plan = {};
  emel::graph::allocator::event::allocation_plan error_plan = {};
  int32_t error_code = 0;

  static bool on_done(void * owner,
                      const emel::graph::allocator::events::allocation_done & ev) noexcept {
    auto * self = static_cast<allocator_dispatch_state *>(owner);
    self->done_called = true;
    self->done_plan = ev.plan;
    return true;
  }

  static bool on_error(void * owner,
                       const emel::graph::allocator::events::allocation_error & ev) noexcept {
    auto * self = static_cast<allocator_dispatch_state *>(owner);
    self->error_called = true;
    self->error_plan = ev.plan;
    self->error_code = ev.err;
    return true;
  }
};

emel::graph::allocator::event::allocate_graph make_valid_request(
    emel::graph::allocator::event::allocation_plan * plan,
    allocator_dispatch_state * state) {
  return emel::graph::allocator::event::allocate_graph{
    .graph_topology = reinterpret_cast<const void *>(0xCAFE),
    .plan_out = plan,
    .node_count = 4u,
    .tensor_count = 8u,
    .tensor_capacity = 8u,
    .interval_capacity = 8u,
    .bytes_per_tensor = 16u,
    .workspace_capacity_bytes = 1024u,
    .dispatch_done = {state, allocator_dispatch_state::on_done},
    .dispatch_error = {state, allocator_dispatch_state::on_error},
  };
}

}  // namespace

TEST_CASE("graph_allocator_action_and_guard_branches") {
  namespace action = emel::graph::allocator::action;
  namespace event = emel::graph::allocator::event;
  namespace guard = emel::graph::allocator::guard;

  action::context machine_ctx{};
  allocator_dispatch_state state{};
  event::allocation_plan output{};
  event::allocate_graph request = make_valid_request(&output, &state);
  event::allocate_graph_ctx phase_ctx{};
  event::allocate_graph_plan ev{request, phase_ctx};

  CHECK(guard::valid_allocate{}(ev, machine_ctx));
  CHECK_FALSE(guard::invalid_allocate{}(ev, machine_ctx));

  request.node_count = 0u;
  CHECK(guard::invalid_allocate{}(ev, machine_ctx));
  CHECK(guard::invalid_allocate_with_dispatchable_output{}(ev, machine_ctx));
  request.dispatch_error = {};
  CHECK(guard::invalid_allocate_with_output_only{}(ev, machine_ctx));
  request.plan_out = nullptr;
  CHECK(guard::invalid_allocate_without_output{}(ev, machine_ctx));

  request = make_valid_request(&output, &state);
  ev.ctx.err = emel::error::cast(emel::graph::allocator::error::none);
  action::begin_allocate(ev, machine_ctx);
  CHECK(machine_ctx.dispatch_generation == 1u);
  CHECK(ev.ctx.err == emel::error::cast(emel::graph::allocator::error::none));

  ev.ctx.sorted_tensor_count = 8u;
  ev.ctx.required_intervals = 8u;
  ev.ctx.required_buffer_bytes = 128u;
  action::commit_plan(ev, machine_ctx);
  CHECK(output.tensor_count == 8u);
  CHECK(output.interval_count == 8u);
  CHECK(output.required_buffer_bytes == 128u);

  state = {};
  action::dispatch_done(ev, machine_ctx);
  CHECK(state.done_called);
  CHECK_FALSE(state.error_called);

  state = {};
  ev.ctx.err = emel::error::cast(emel::graph::allocator::error::capacity);
  action::dispatch_error(ev, machine_ctx);
  CHECK_FALSE(state.done_called);
  CHECK(state.error_called);
  CHECK(state.error_code ==
        static_cast<int32_t>(emel::error::cast(emel::graph::allocator::error::capacity)));

  state = {};
  action::reject_invalid_allocate_with_dispatch(ev, machine_ctx);
  CHECK(state.error_called);
  CHECK(output.tensor_count == 0u);
  CHECK(output.interval_count == 0u);
  CHECK(output.required_buffer_bytes == 0u);

  output = {.tensor_count = 1u, .interval_count = 1u, .required_buffer_bytes = 1u};
  action::reject_invalid_allocate_with_output_only(ev, machine_ctx);
  CHECK(output.tensor_count == 0u);
  CHECK(output.interval_count == 0u);
  CHECK(output.required_buffer_bytes == 0u);

  request.plan_out = nullptr;
  action::reject_invalid_allocate_without_output(ev, machine_ctx);
  CHECK(ev.ctx.err == emel::error::cast(emel::graph::allocator::error::invalid_request));

  ev.ctx.err = emel::error::cast(emel::graph::allocator::error::none);
  CHECK(guard::allocation_error_none{}(ev, machine_ctx));

  ev.ctx.err = emel::error::cast(emel::graph::allocator::error::invalid_request);
  CHECK(guard::allocation_error_invalid_request{}(ev, machine_ctx));

  ev.ctx.err = emel::error::cast(emel::graph::allocator::error::capacity);
  CHECK(guard::allocation_error_capacity{}(ev, machine_ctx));

  ev.ctx.err = emel::error::cast(emel::graph::allocator::error::internal_error);
  CHECK(guard::allocation_error_internal_error{}(ev, machine_ctx));

  ev.ctx.err = emel::error::cast(emel::graph::allocator::error::untracked);
  CHECK(guard::allocation_error_untracked{}(ev, machine_ctx));

  ev.ctx.err = static_cast<emel::error::type>(0x7fff);
  CHECK(guard::allocation_error_unknown{}(ev, machine_ctx));

  ev.ctx.err = emel::error::cast(emel::graph::allocator::error::none);
  ev.ctx.liveness_outcome = emel::graph::allocator::liveness_pass::events::phase_outcome::done;
  CHECK(guard::liveness_done{}(ev, machine_ctx));
  CHECK_FALSE(guard::liveness_failed{}(ev, machine_ctx));
  ev.ctx.liveness_outcome = emel::graph::allocator::liveness_pass::events::phase_outcome::failed;
  CHECK_FALSE(guard::liveness_done{}(ev, machine_ctx));
  CHECK(guard::liveness_failed{}(ev, machine_ctx));

  ev.ctx.ordering_outcome = emel::graph::allocator::ordering_pass::events::phase_outcome::done;
  CHECK(guard::ordering_done{}(ev, machine_ctx));
  ev.ctx.ordering_outcome = emel::graph::allocator::ordering_pass::events::phase_outcome::failed;
  CHECK(guard::ordering_failed{}(ev, machine_ctx));

  ev.ctx.placement_outcome = emel::graph::allocator::placement_pass::events::phase_outcome::done;
  CHECK(guard::placement_done{}(ev, machine_ctx));
  ev.ctx.placement_outcome = emel::graph::allocator::placement_pass::events::phase_outcome::failed;
  CHECK(guard::placement_failed{}(ev, machine_ctx));

  ev.ctx.err = emel::error::cast(emel::graph::allocator::error::none);
  action::on_unexpected(ev, machine_ctx);
  CHECK(ev.ctx.err == emel::error::cast(emel::graph::allocator::error::internal_error));
}

TEST_CASE("graph_allocator_pass_action_and_guard_branches") {
  namespace event = emel::graph::allocator::event;
  using allocator_error = emel::graph::allocator::error;
  event::allocation_plan plan{};
  allocator_dispatch_state state{};
  event::allocate_graph request = make_valid_request(&plan, &state);
  event::allocate_graph_ctx phase_ctx{};
  event::allocate_graph_plan ev{request, phase_ctx};
  emel::graph::allocator::action::context machine_ctx{};

  // liveness
  ev.ctx.err = emel::error::cast(allocator_error::none);
  CHECK(emel::graph::allocator::liveness_pass::guard::phase_done{}(ev, machine_ctx));
  emel::graph::allocator::liveness_pass::action::mark_done(ev, machine_ctx);
  CHECK(ev.ctx.required_intervals == request.tensor_count);
  CHECK(ev.ctx.liveness_outcome ==
        emel::graph::allocator::liveness_pass::events::phase_outcome::done);

  request.graph_topology = nullptr;
  CHECK(emel::graph::allocator::liveness_pass::guard::phase_invalid_request{}(ev, machine_ctx));
  request.graph_topology = reinterpret_cast<const void *>(0xCAFE);
  request.tensor_capacity = 1u;
  CHECK(emel::graph::allocator::liveness_pass::guard::phase_capacity_exceeded{}(ev, machine_ctx));
  request.tensor_capacity = request.tensor_count;
  request.node_count = 0u;
  ev.ctx.err = emel::error::cast(allocator_error::internal_error);
  CHECK(emel::graph::allocator::liveness_pass::guard::phase_prefailed{}(ev, machine_ctx));
  emel::graph::allocator::liveness_pass::action::mark_failed_prefailed(ev, machine_ctx);
  ev.ctx.err = emel::error::cast(allocator_error::none);
  request.node_count = 4u;
  emel::graph::allocator::liveness_pass::action::mark_failed_invalid_request(ev, machine_ctx);
  emel::graph::allocator::liveness_pass::action::mark_failed_capacity(ev, machine_ctx);
  emel::graph::allocator::liveness_pass::action::mark_failed_internal(ev, machine_ctx);
  emel::graph::allocator::liveness_pass::action::on_unexpected(ev, machine_ctx);

  // ordering
  ev.ctx.err = emel::error::cast(allocator_error::none);
  ev.ctx.liveness_outcome = emel::graph::allocator::liveness_pass::events::phase_outcome::done;
  ev.ctx.required_intervals = request.interval_capacity;
  request.bytes_per_tensor = 16u;
  CHECK(emel::graph::allocator::ordering_pass::guard::phase_done{}(ev, machine_ctx));
  emel::graph::allocator::ordering_pass::action::mark_done(ev, machine_ctx);
  CHECK(ev.ctx.required_buffer_bytes == ev.ctx.required_intervals * request.bytes_per_tensor);

  ev.ctx.liveness_outcome = emel::graph::allocator::liveness_pass::events::phase_outcome::failed;
  CHECK(emel::graph::allocator::ordering_pass::guard::phase_prereq_failed{}(ev, machine_ctx));
  ev.ctx.liveness_outcome = emel::graph::allocator::liveness_pass::events::phase_outcome::done;
  ev.ctx.required_intervals = request.interval_capacity + 1u;
  CHECK(emel::graph::allocator::ordering_pass::guard::phase_capacity_exceeded{}(ev, machine_ctx));
  ev.ctx.required_intervals = 4u;
  request.bytes_per_tensor = std::numeric_limits<uint64_t>::max();
  CHECK(emel::graph::allocator::ordering_pass::guard::phase_overflow{}(ev, machine_ctx));
  request.bytes_per_tensor = 0u;
  CHECK(emel::graph::allocator::ordering_pass::guard::phase_invalid_request{}(ev, machine_ctx));
  request.bytes_per_tensor = 16u;
  ev.ctx.required_intervals = 0u;
  ev.ctx.err = emel::error::cast(allocator_error::internal_error);
  CHECK(emel::graph::allocator::ordering_pass::guard::phase_prefailed{}(ev, machine_ctx));
  emel::graph::allocator::ordering_pass::action::mark_failed_prefailed(ev, machine_ctx);
  ev.ctx.err = emel::error::cast(allocator_error::none);

  emel::graph::allocator::ordering_pass::action::mark_failed_prereq(ev, machine_ctx);
  emel::graph::allocator::ordering_pass::action::mark_failed_capacity(ev, machine_ctx);
  emel::graph::allocator::ordering_pass::action::mark_failed_overflow(ev, machine_ctx);
  emel::graph::allocator::ordering_pass::action::mark_failed_invalid_request(ev, machine_ctx);
  emel::graph::allocator::ordering_pass::action::mark_failed_internal(ev, machine_ctx);
  emel::graph::allocator::ordering_pass::action::on_unexpected(ev, machine_ctx);

  // placement
  ev.ctx.err = emel::error::cast(allocator_error::none);
  ev.ctx.ordering_outcome = emel::graph::allocator::ordering_pass::events::phase_outcome::done;
  ev.ctx.sorted_tensor_count = 2u;
  ev.ctx.required_buffer_bytes = 16u;
  request.workspace_capacity_bytes = 16u;
  request.plan_out = &plan;
  CHECK(emel::graph::allocator::placement_pass::guard::phase_done{}(ev, machine_ctx));
  emel::graph::allocator::placement_pass::action::mark_done(ev, machine_ctx);

  ev.ctx.ordering_outcome = emel::graph::allocator::ordering_pass::events::phase_outcome::failed;
  CHECK(emel::graph::allocator::placement_pass::guard::phase_prereq_failed{}(ev, machine_ctx));
  ev.ctx.ordering_outcome = emel::graph::allocator::ordering_pass::events::phase_outcome::done;
  ev.ctx.required_buffer_bytes = 32u;
  request.workspace_capacity_bytes = 16u;
  CHECK(emel::graph::allocator::placement_pass::guard::phase_capacity_exceeded{}(ev, machine_ctx));
  ev.ctx.required_buffer_bytes = 16u;
  request.plan_out = nullptr;
  CHECK(emel::graph::allocator::placement_pass::guard::phase_invalid_request{}(ev, machine_ctx));
  request.plan_out = &plan;
  ev.ctx.sorted_tensor_count = 0u;
  CHECK_FALSE(
      emel::graph::allocator::placement_pass::guard::phase_unclassified_failure{}(ev, machine_ctx));

  emel::graph::allocator::placement_pass::action::mark_failed_prereq(ev, machine_ctx);
  emel::graph::allocator::placement_pass::action::mark_failed_capacity(ev, machine_ctx);
  emel::graph::allocator::placement_pass::action::mark_failed_invalid_request(ev, machine_ctx);
  emel::graph::allocator::placement_pass::action::mark_failed_internal(ev, machine_ctx);
  emel::graph::allocator::placement_pass::action::on_unexpected(ev, machine_ctx);
}
