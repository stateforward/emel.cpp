#pragma once

#include <cstdint>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/graph/allocator/errors.hpp"
#include "emel/graph/allocator/liveness_pass/events.hpp"
#include "emel/graph/allocator/ordering_pass/events.hpp"
#include "emel/graph/allocator/placement_pass/events.hpp"

namespace emel::graph::allocator::events {

struct allocation_done;
struct allocation_error;

}  // namespace emel::graph::allocator::events

namespace emel::graph::allocator::event {

struct allocation_plan {
  uint32_t tensor_count = 0;
  uint32_t interval_count = 0;
  uint64_t required_buffer_bytes = 0;
};

struct allocate_graph {
  const void * graph_topology = nullptr;
  allocation_plan * plan_out = nullptr;
  uint32_t node_count = 0;
  uint32_t tensor_count = 0;
  uint32_t tensor_capacity = 0;
  uint32_t interval_capacity = 0;
  uint64_t bytes_per_tensor = 0;
  uint64_t workspace_capacity_bytes = 0;
  ::emel::callback<bool(const ::emel::graph::allocator::events::allocation_done &)> dispatch_done =
      {};
  ::emel::callback<bool(const ::emel::graph::allocator::events::allocation_error &)>
      dispatch_error = {};
};

// Internal context object carried via completion<allocate_graph_plan>.
struct allocate_graph_ctx {
  liveness_pass::events::phase_outcome liveness_outcome =
      liveness_pass::events::phase_outcome::unknown;
  ordering_pass::events::phase_outcome ordering_outcome =
      ordering_pass::events::phase_outcome::unknown;
  placement_pass::events::phase_outcome placement_outcome =
      placement_pass::events::phase_outcome::unknown;
  uint32_t required_intervals = 0;
  uint32_t sorted_tensor_count = 0;
  uint64_t required_buffer_bytes = 0;
  emel::error::type err = emel::error::cast(error::none);
};

// Internal event used by allocator::sm wrapper; not part of public API.
struct allocate_graph_plan {
  const allocate_graph & request;
  allocate_graph_ctx & ctx;
};

}  // namespace emel::graph::allocator::event

namespace emel::graph::allocator::events {

struct allocation_done {
  event::allocation_plan & plan;
};

struct allocation_error {
  event::allocation_plan & plan;
  int32_t err = 0;
};

}  // namespace emel::graph::allocator::events
