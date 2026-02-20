#pragma once

#include <cstdint>

#include "emel/decoder/compute_executor/events.hpp"
#include "emel/emel.h"

namespace emel::decoder::compute_executor::action {

struct context {
  int32_t outputs_produced = 0;
  int32_t ubatch_index = 0;
  int32_t ubatch_size = 0;
  int32_t kv_tokens = 0;
  void * compute_ctx = nullptr;
  event::validate_fn validate = nullptr;
  event::prepare_graph_fn prepare_graph = nullptr;
  event::alloc_graph_fn alloc_graph = nullptr;
  event::bind_inputs_fn bind_inputs = nullptr;
  event::run_backend_fn run_backend = nullptr;
  event::extract_outputs_fn extract_outputs = nullptr;
  bool graph_reused = false;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::decoder::compute_executor::action
