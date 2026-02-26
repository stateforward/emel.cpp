#pragma once

#include <cstdint>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/graph/assembler/events.hpp"
#include "emel/graph/errors.hpp"
#include "emel/graph/processor/events.hpp"
#include "emel/memory/view.hpp"

namespace emel::graph::events {

struct reserve_done;
struct reserve_error;
struct compute_done;
struct compute_error;

}  // namespace emel::graph::events

namespace emel::graph::event {

using validate_fn = processor::event::validate_fn;
using prepare_graph_fn = processor::event::prepare_graph_fn;
using alloc_graph_fn = processor::event::alloc_graph_fn;
using bind_inputs_fn = processor::event::bind_inputs_fn;
using run_kernel_fn = processor::event::run_kernel_fn;
using extract_outputs_fn = processor::event::extract_outputs_fn;

struct reserve_output {
  const void * graph_topology = nullptr;
  uint32_t node_count = 0;
  uint32_t tensor_count = 0;
  uint64_t required_buffer_bytes = 0;
  uint32_t version = 0;
};

struct compute_output {
  const void * graph_topology = nullptr;
  uint32_t node_count = 0;
  uint32_t tensor_count = 0;
  uint64_t required_buffer_bytes = 0;
  uint32_t version = 0;
  uint8_t reused_topology = 0;
  int32_t outputs_produced = 0;
  uint8_t graph_reused = 0;
};

struct reserve {
  const void * model_topology = nullptr;
  reserve_output * output_out = nullptr;
  uint32_t max_node_count = 0;
  uint32_t max_tensor_count = 0;
  uint64_t bytes_per_tensor = 0;
  uint64_t workspace_capacity_bytes = 0;
  ::emel::callback<bool(const ::emel::graph::events::reserve_done &)> dispatch_done = {};
  ::emel::callback<bool(const ::emel::graph::events::reserve_error &)> dispatch_error = {};
};

struct compute {
  const void * step_plan = nullptr;
  compute_output * output_out = nullptr;
  uint32_t node_count_hint = 0;
  uint32_t tensor_count_hint = 0;
  uint64_t bytes_per_tensor = 0;
  uint64_t workspace_capacity_bytes = 0;
  int32_t step_index = 0;
  int32_t step_size = 0;
  int32_t kv_tokens = 0;
  void * memory_sm = nullptr;
  emel::memory::view::any memory_view = {};
  int32_t expected_outputs = 0;
  void * compute_ctx = nullptr;
  const int32_t * positions = nullptr;
  int32_t positions_count = 0;
  const uint64_t * seq_masks = nullptr;
  int32_t seq_mask_words = 1;
  int32_t seq_masks_count = 0;
  const int32_t * seq_primary_ids = nullptr;
  int32_t seq_primary_ids_count = 0;
  validate_fn validate = nullptr;
  prepare_graph_fn prepare_graph = nullptr;
  alloc_graph_fn alloc_graph = nullptr;
  bind_inputs_fn bind_inputs = nullptr;
  run_kernel_fn run_kernel = nullptr;
  extract_outputs_fn extract_outputs = nullptr;
  ::emel::callback<bool(const ::emel::graph::events::compute_done &)> dispatch_done = {};
  ::emel::callback<bool(const ::emel::graph::events::compute_error &)> dispatch_error = {};
};

// Internal context object carried via completion<reserve_graph>.
enum class phase_outcome : uint8_t {
  unknown = 0,
  done = 1,
  failed = 2,
};

struct reserve_ctx {
  phase_outcome reserve_outcome = phase_outcome::unknown;
  assembler::event::reserve_output reserve_output = {};
  emel::error::type err = emel::error::cast(error::none);
};

// Internal context object carried via completion<compute_graph>.
struct compute_ctx {
  phase_outcome assemble_outcome = phase_outcome::unknown;
  phase_outcome execute_outcome = phase_outcome::unknown;
  assembler::event::assemble_output assemble_output = {};
  processor::event::execution_output execute_output = {};
  emel::error::type err = emel::error::cast(error::none);
};

// Internal event used by graph::sm wrapper; not part of public API.
struct reserve_graph {
  const reserve & request;
  reserve_ctx & ctx;
};

// Internal event used by graph::sm wrapper; not part of public API.
struct compute_graph {
  const compute & request;
  compute_ctx & ctx;
};

}  // namespace emel::graph::event

namespace emel::graph::events {

struct reserve_done {
  event::reserve_output & output;
};

struct reserve_error {
  event::reserve_output & output;
  int32_t err = 0;
};

struct compute_done {
  event::compute_output & output;
};

struct compute_error {
  event::compute_output & output;
  int32_t err = 0;
};

}  // namespace emel::graph::events
