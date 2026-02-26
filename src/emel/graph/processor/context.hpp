#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/graph/processor/events.hpp"

namespace emel::graph::processor::action {

struct context {
  int32_t outputs_produced = 0;
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
  event::validate_fn validate = nullptr;
  event::prepare_graph_fn prepare_graph = nullptr;
  event::alloc_graph_fn alloc_graph = nullptr;
  event::bind_inputs_fn bind_inputs = nullptr;
  event::run_backend_fn run_backend = nullptr;
  event::extract_outputs_fn extract_outputs = nullptr;
  int32_t * kv_tokens_out = nullptr;
  bool * rollback_attempted_out = nullptr;
  bool graph_reused = false;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::graph::processor::action
