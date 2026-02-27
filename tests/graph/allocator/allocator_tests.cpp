#include <doctest/doctest.h>

#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/graph/allocator/errors.hpp"
#include "emel/graph/allocator/events.hpp"
#include "emel/graph/allocator/sm.hpp"

namespace {

struct allocation_callbacks {
  bool done_called = false;
  bool error_called = false;
  emel::graph::allocator::event::allocation_plan done_plan = {};
  emel::graph::allocator::event::allocation_plan error_plan = {};
  int32_t error_code = 0;

  static bool on_done(void * owner,
                      const emel::graph::allocator::events::allocation_done & ev) noexcept {
    auto * self = static_cast<allocation_callbacks *>(owner);
    self->done_called = true;
    self->done_plan = ev.plan;
    return true;
  }

  static bool on_error(void * owner,
                       const emel::graph::allocator::events::allocation_error & ev) noexcept {
    auto * self = static_cast<allocation_callbacks *>(owner);
    self->error_called = true;
    self->error_plan = ev.plan;
    self->error_code = ev.err;
    return true;
  }
};

}  // namespace

TEST_CASE("graph_allocator_processes_valid_request") {
  emel::graph::allocator::sm machine{};
  emel::graph::allocator::event::allocation_plan plan{};
  allocation_callbacks callbacks{};

  const emel::graph::allocator::event::allocate_graph request{
    .graph_topology = reinterpret_cast<const void *>(0x1),
    .plan_out = &plan,
    .node_count = 2u,
    .tensor_count = 4u,
    .tensor_capacity = 4u,
    .interval_capacity = 4u,
    .bytes_per_tensor = 8u,
    .workspace_capacity_bytes = 32u,
    .dispatch_done = {&callbacks, allocation_callbacks::on_done},
    .dispatch_error = {&callbacks, allocation_callbacks::on_error},
  };

  CHECK(machine.process_event(request));
  CHECK(callbacks.done_called);
  CHECK_FALSE(callbacks.error_called);
  CHECK(plan.tensor_count == 4u);
  CHECK(plan.interval_count == 4u);
  CHECK(plan.required_buffer_bytes == 32u);
}

TEST_CASE("graph_allocator_rejects_invalid_request_with_dispatchable_error") {
  emel::graph::allocator::sm machine{};
  emel::graph::allocator::event::allocation_plan plan{
    .tensor_count = 99u,
    .interval_count = 99u,
    .required_buffer_bytes = 99u,
  };
  allocation_callbacks callbacks{};

  const emel::graph::allocator::event::allocate_graph request{
    .graph_topology = reinterpret_cast<const void *>(0x1),
    .plan_out = &plan,
    .node_count = 0u,
    .tensor_count = 4u,
    .tensor_capacity = 4u,
    .interval_capacity = 4u,
    .bytes_per_tensor = 8u,
    .workspace_capacity_bytes = 32u,
    .dispatch_done = {&callbacks, allocation_callbacks::on_done},
    .dispatch_error = {&callbacks, allocation_callbacks::on_error},
  };

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(callbacks.done_called);
  CHECK(callbacks.error_called);
  CHECK(callbacks.error_code ==
        static_cast<int32_t>(emel::error::cast(emel::graph::allocator::error::invalid_request)));
  CHECK(plan.tensor_count == 0u);
  CHECK(plan.interval_count == 0u);
  CHECK(plan.required_buffer_bytes == 0u);
}
