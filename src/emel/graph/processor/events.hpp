#pragma once

#include <cstdint>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/graph/processor/alloc_step/events.hpp"
#include "emel/graph/processor/kernel_step/events.hpp"
#include "emel/graph/processor/bind_step/events.hpp"
#include "emel/graph/processor/errors.hpp"
#include "emel/graph/processor/extract_step/events.hpp"
#include "emel/graph/processor/prepare_step/events.hpp"
#include "emel/graph/processor/validate_step/events.hpp"
#include "emel/memory/view.hpp"

namespace emel::graph::processor::events {

struct execution_done;
struct execution_error;

}  // namespace emel::graph::processor::events

namespace emel::graph::processor::event {

struct execute;

using validate_fn = bool (*)(const execute & request, int32_t * err_out);
using prepare_graph_fn =
    bool (*)(const execute & request, bool * reused_out, int32_t * err_out);
using alloc_graph_fn = bool (*)(const execute & request, int32_t * err_out);
using bind_inputs_fn = bool (*)(const execute & request, int32_t * err_out);
using run_kernel_fn = bool (*)(const execute & request, int32_t * err_out);
using extract_outputs_fn =
    bool (*)(const execute & request, int32_t * outputs_out, int32_t * err_out);

struct lifecycle_tensor_binding {
  int32_t tensor_id = 0;
  void * buffer = nullptr;
  uint64_t buffer_bytes = 0u;
  int32_t consumer_refs = 0;
  bool is_leaf = false;
};

struct lifecycle_phase {
  const int32_t * required_filled_ids = nullptr;
  int32_t required_filled_count = 0;
  const int32_t * publish_ids = nullptr;
  int32_t publish_count = 0;
  const int32_t * release_ids = nullptr;
  int32_t release_count = 0;
};

struct lifecycle_manifest {
  const lifecycle_tensor_binding * tensors = nullptr;
  int32_t tensor_count = 0;
  const lifecycle_phase * phase = nullptr;
};

struct execution_output {
  int32_t outputs_produced = 0;
  uint8_t graph_reused = 0;
  const lifecycle_manifest * lifecycle = nullptr;
};

struct execute {
  const void * step_plan = nullptr;
  execution_output * output_out = nullptr;
  const lifecycle_manifest * lifecycle = nullptr;
  int32_t step_index = 0;
  int32_t step_size = 0;
  int32_t kv_tokens = 0;
  void * memory_sm = nullptr;
  const emel::memory::view::snapshot * memory_view = nullptr;
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
  ::emel::callback<bool(const ::emel::graph::processor::events::execution_done &)> dispatch_done =
      {};
  ::emel::callback<bool(const ::emel::graph::processor::events::execution_error &)> dispatch_error =
      {};
};

// Internal context object carried via completion<execute_step>.
struct execute_ctx {
  validate_step::events::phase_outcome validate_outcome =
      validate_step::events::phase_outcome::unknown;
  prepare_step::events::phase_outcome prepare_outcome =
      prepare_step::events::phase_outcome::unknown;
  alloc_step::events::phase_outcome alloc_outcome =
      alloc_step::events::phase_outcome::unknown;
  bind_step::events::phase_outcome bind_outcome =
      bind_step::events::phase_outcome::unknown;
  kernel_step::events::phase_outcome kernel_outcome =
      kernel_step::events::phase_outcome::unknown;
  extract_step::events::phase_outcome extract_outcome =
      extract_step::events::phase_outcome::unknown;
  uint8_t graph_reused = 0;
  int32_t outputs_produced = 0;
  bool phase_callback_ok = false;
  int32_t phase_callback_err = 0;
  emel::error::type err = emel::error::cast(error::none);
};

// Internal event used by processor::sm wrapper; not part of public API.
struct execute_step {
  const execute & request;
  execute_ctx & ctx;
};

}  // namespace emel::graph::processor::event

namespace emel::graph::processor::events {

struct execution_done {
  event::execution_output & output;
};

struct execution_error {
  event::execution_output & output;
  int32_t err = 0;
};

}  // namespace emel::graph::processor::events
