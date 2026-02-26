#pragma once

#include <cstdint>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/graph/allocator/events.hpp"
#include "emel/graph/assembler/assemble_alloc_pass/events.hpp"
#include "emel/graph/assembler/assemble_build_pass/events.hpp"
#include "emel/graph/assembler/assemble_validate_pass/events.hpp"
#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/reserve_alloc_pass/events.hpp"
#include "emel/graph/assembler/reserve_build_pass/events.hpp"
#include "emel/graph/assembler/reserve_validate_pass/events.hpp"
#include "emel/graph/assembler/reuse_decision_pass/events.hpp"

namespace emel::graph::assembler::events {

struct reserve_done;
struct reserve_error;
struct assemble_done;
struct assemble_error;

}  // namespace emel::graph::assembler::events

namespace emel::graph::assembler::event {

struct reserve_output {
  const void * graph_topology = nullptr;
  uint32_t node_count = 0;
  uint32_t tensor_count = 0;
  uint64_t required_buffer_bytes = 0;
  uint32_t version = 0;
};

struct assemble_output {
  const void * graph_topology = nullptr;
  uint32_t node_count = 0;
  uint32_t tensor_count = 0;
  uint64_t required_buffer_bytes = 0;
  uint32_t version = 0;
  uint8_t reused_topology = 0;
};

struct reserve {
  const void * model_topology = nullptr;
  reserve_output * output_out = nullptr;
  uint32_t max_node_count = 0;
  uint32_t max_tensor_count = 0;
  uint64_t bytes_per_tensor = 0;
  uint64_t workspace_capacity_bytes = 0;
  ::emel::callback<bool(const ::emel::graph::assembler::events::reserve_done &)> dispatch_done = {};
  ::emel::callback<bool(const ::emel::graph::assembler::events::reserve_error &)> dispatch_error = {};
};

struct assemble {
  const void * step_plan = nullptr;
  assemble_output * output_out = nullptr;
  uint32_t node_count_hint = 0;
  uint32_t tensor_count_hint = 0;
  uint64_t bytes_per_tensor = 0;
  uint64_t workspace_capacity_bytes = 0;
  ::emel::callback<bool(const ::emel::graph::assembler::events::assemble_done &)> dispatch_done =
      {};
  ::emel::callback<bool(const ::emel::graph::assembler::events::assemble_error &)> dispatch_error =
      {};
};

// Internal context object carried via completion<reserve_graph>.
struct reserve_ctx {
  reserve_validate_pass::events::phase_outcome validate_outcome =
      reserve_validate_pass::events::phase_outcome::unknown;
  reserve_build_pass::events::phase_outcome build_outcome =
      reserve_build_pass::events::phase_outcome::unknown;
  reserve_alloc_pass::events::phase_outcome alloc_outcome =
      reserve_alloc_pass::events::phase_outcome::unknown;
  uint32_t assembled_node_count = 0;
  uint32_t assembled_tensor_count = 0;
  allocator::event::allocation_plan alloc_plan = {};
  emel::error::type err = emel::error::cast(error::none);
};

// Internal context object carried via completion<assemble_graph>.
struct assemble_ctx {
  assemble_validate_pass::events::phase_outcome validate_outcome =
      assemble_validate_pass::events::phase_outcome::unknown;
  reuse_decision_pass::events::phase_outcome reuse_outcome =
      reuse_decision_pass::events::phase_outcome::unknown;
  assemble_build_pass::events::phase_outcome build_outcome =
      assemble_build_pass::events::phase_outcome::unknown;
  assemble_alloc_pass::events::phase_outcome alloc_outcome =
      assemble_alloc_pass::events::phase_outcome::unknown;
  uint32_t assembled_node_count = 0;
  uint32_t assembled_tensor_count = 0;
  allocator::event::allocation_plan alloc_plan = {};
  uint8_t reused_topology = 0;
  emel::error::type err = emel::error::cast(error::none);
};

// Internal event used by assembler::sm wrapper; not part of public API.
struct reserve_graph {
  const reserve & request;
  reserve_ctx & ctx;
};

// Internal event used by assembler::sm wrapper; not part of public API.
struct assemble_graph {
  const assemble & request;
  assemble_ctx & ctx;
};

}  // namespace emel::graph::assembler::event

namespace emel::graph::assembler::events {

struct reserve_done {
  event::reserve_output & output;
};

struct reserve_error {
  event::reserve_output & output;
  int32_t err = 0;
};

struct assemble_done {
  event::assemble_output & output;
};

struct assemble_error {
  event::assemble_output & output;
  int32_t err = 0;
};

}  // namespace emel::graph::assembler::events
