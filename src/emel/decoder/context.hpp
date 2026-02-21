#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "emel/batch/sanitizer/sm.hpp"
#include "emel/batch/splitter/sm.hpp"
#include "emel/decoder/events.hpp"
#include "emel/decoder/ubatch_executor/sm.hpp"
#include "emel/emel.h"
#include "emel/kv/cache/sm.hpp"
#include "emel/memory/coordinator/any.hpp"

namespace emel::decoder::action {

enum class prepare_failure_kind : uint8_t {
  none = 0,
  retryable,
  permanent,
};

struct context {
  const int32_t * token_ids = nullptr;
  bool output_all = false;
  const int8_t * output_mask = nullptr;
  const uint64_t * seq_masks = nullptr;
  const int32_t * seq_primary_ids = nullptr;
  const int32_t * positions = nullptr;
  int32_t n_tokens = 0;
  int32_t n_ubatch = 0;
  int32_t output_mask_count = 0;
  int32_t seq_mask_words = 1;
  int32_t seq_masks_count = 0;
  int32_t seq_primary_ids_count = 0;
  int32_t positions_count = 0;
  int32_t outputs_capacity = 0;

  int32_t outputs_total = 0;
  int32_t outputs_processed = 0;

  std::array<int32_t, emel::batch::splitter::action::MAX_UBATCHES> ubatch_sizes = {};
  std::array<int32_t, emel::kv::cache::action::MAX_UBATCHES> slot_offsets = {};
  std::array<int32_t, emel::kv::cache::action::MAX_UBATCHES> ubatch_seq_ids = {};
  std::array<int32_t, emel::batch::splitter::action::MAX_UBATCHES> ubatch_token_indices = {};
  std::array<int32_t, emel::batch::splitter::action::MAX_UBATCHES + 1> ubatch_token_offsets =
      {};
  std::array<int32_t, emel::batch::splitter::action::MAX_UBATCHES> ubatch_outputs = {};
  std::array<int32_t, emel::batch::splitter::action::MAX_UBATCHES * 3> ubatch_positions = {};
  std::array<uint64_t,
             emel::batch::splitter::action::MAX_UBATCHES *
                 emel::batch::splitter::action::SEQ_WORDS>
      ubatch_seq_masks = {};
  std::array<int32_t, emel::batch::splitter::action::MAX_UBATCHES> ubatch_seq_primary_ids = {};
  std::array<int32_t, emel::batch::splitter::action::MAX_UBATCHES> sanitized_seq_primary_ids = {};
  std::array<uint64_t,
             emel::batch::splitter::action::MAX_UBATCHES *
                 emel::batch::splitter::action::SEQ_WORDS>
      sanitized_seq_masks = {};
  std::array<int32_t, emel::batch::splitter::action::MAX_UBATCHES * 3>
      sanitized_positions = {};
  std::array<int8_t, emel::batch::splitter::action::MAX_UBATCHES> sanitized_output_mask = {};
  int32_t sanitized_outputs_total = 0;
  int32_t sanitized_positions_count = 0;
  int32_t sanitized_seq_mask_words = 1;
  int32_t token_indices_count = 0;
  int32_t ubatches_total = 0;
  int32_t ubatches_processed = 0;

  std::unique_ptr<emel::batch::sanitizer::sm> batch_sanitizer;
  std::unique_ptr<emel::batch::splitter::sm> batch_splitter;
  std::unique_ptr<emel::memory::coordinator::any> memory_coordinator;
  std::unique_ptr<emel::kv::cache::sm> kv_cache;
  std::unique_ptr<emel::decoder::ubatch_executor::sm> ubatch_executor;

  void * compute_ctx = nullptr;
  event::compute_validate_fn compute_validate = nullptr;
  event::compute_prepare_graph_fn compute_prepare_graph = nullptr;
  event::compute_alloc_graph_fn compute_alloc_graph = nullptr;
  event::compute_bind_inputs_fn compute_bind_inputs = nullptr;
  event::compute_run_backend_fn compute_run_backend = nullptr;
  event::compute_extract_outputs_fn compute_extract_outputs = nullptr;

  int32_t phase_error = EMEL_OK;
  int32_t ubatch_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
  bool phase_retryable = false;
  bool rollback_needed = false;

  context();
};

inline context::context()
    : batch_sanitizer(std::make_unique<emel::batch::sanitizer::sm>()),
      batch_splitter(std::make_unique<emel::batch::splitter::sm>()),
      memory_coordinator(std::make_unique<emel::memory::coordinator::any>()),
      kv_cache(std::make_unique<emel::kv::cache::sm>()),
      ubatch_executor(std::make_unique<emel::decoder::ubatch_executor::sm>()) {
  // one-time heap allocation keeps decoder context small on the stack.
}

}  // namespace emel::decoder::action
