#pragma once

#include <cstdint>

#include "emel/decoder/compute_executor/sm.hpp"
#include "emel/decoder/ubatch_executor/events.hpp"
#include "emel/emel.h"
#include "emel/kv/cache/sm.hpp"
#include "emel/memory/coordinator/any.hpp"

namespace emel::decoder::ubatch_executor::action {

struct context {
  int32_t ubatch_index = 0;
  int32_t ubatch_size = 0;
  int32_t expected_outputs = 0;

  int32_t outputs_produced = 0;
  int32_t kv_tokens = 0;
  emel::decoder::compute_executor::sm compute_executor = {};
  emel::memory::coordinator::any * memory_coordinator_sm = nullptr;
  emel::kv::cache::sm * kv_cache_sm = nullptr;
  void * compute_ctx = nullptr;
  event::compute_validate_fn compute_validate = nullptr;
  event::compute_prepare_graph_fn compute_prepare_graph = nullptr;
  event::compute_alloc_graph_fn compute_alloc_graph = nullptr;
  event::compute_bind_inputs_fn compute_bind_inputs = nullptr;
  event::compute_run_backend_fn compute_run_backend = nullptr;
  event::compute_extract_outputs_fn compute_extract_outputs = nullptr;
  const int32_t * positions = nullptr;
  int32_t positions_count = 0;
  const uint64_t * seq_masks = nullptr;
  int32_t seq_mask_words = 0;
  int32_t seq_masks_count = 0;
  const int32_t * seq_primary_ids = nullptr;
  int32_t seq_primary_ids_count = 0;
  int32_t phase_error = EMEL_OK;
  int32_t execution_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
  bool rollback_attempted = false;
};

}  // namespace emel::decoder::ubatch_executor::action
