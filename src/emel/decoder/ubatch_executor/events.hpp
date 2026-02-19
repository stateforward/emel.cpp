#pragma once

#include <cstdint>

#include "emel/decoder/compute_executor/events.hpp"

namespace emel::kv::cache {
struct sm;
}  // namespace emel::kv::cache

namespace emel::memory::coordinator {
struct sm;
}  // namespace emel::memory::coordinator

namespace emel::decoder::ubatch_executor::event {

using compute_validate_fn = emel::decoder::compute_executor::event::validate_fn;
using compute_prepare_graph_fn = emel::decoder::compute_executor::event::prepare_graph_fn;
using compute_alloc_graph_fn = emel::decoder::compute_executor::event::alloc_graph_fn;
using compute_bind_inputs_fn = emel::decoder::compute_executor::event::bind_inputs_fn;
using compute_run_backend_fn = emel::decoder::compute_executor::event::run_backend_fn;
using compute_extract_outputs_fn = emel::decoder::compute_executor::event::extract_outputs_fn;

struct execute {
  int32_t ubatch_index = 0;
  int32_t ubatch_size = 0;
  emel::memory::coordinator::sm * memory_coordinator_sm = nullptr;
  emel::kv::cache::sm * kv_cache_sm = nullptr;
  int32_t expected_outputs = 0;
  void * compute_ctx = nullptr;
  compute_validate_fn compute_validate = nullptr;
  compute_prepare_graph_fn compute_prepare_graph = nullptr;
  compute_alloc_graph_fn compute_alloc_graph = nullptr;
  compute_bind_inputs_fn compute_bind_inputs = nullptr;
  compute_run_backend_fn compute_run_backend = nullptr;
  compute_extract_outputs_fn compute_extract_outputs = nullptr;
  int32_t * outputs_produced_out = nullptr;
  int32_t * kv_tokens_out = nullptr;
  bool * rollback_attempted_out = nullptr;
  int32_t * error_out = nullptr;
  const int32_t * positions = nullptr;
  int32_t positions_count = 0;
  const uint64_t * seq_masks = nullptr;
  int32_t seq_mask_words = 0;
  int32_t seq_masks_count = 0;
  const int32_t * seq_primary_ids = nullptr;
  int32_t seq_primary_ids_count = 0;
};

struct validate {
  const execute * request = nullptr;
  int32_t * error_out = nullptr;
};

struct prepare_memory {
  emel::memory::coordinator::sm * memory_coordinator_sm = nullptr;
  int32_t * error_out = nullptr;
};

struct prepare_kv {
  emel::kv::cache::sm * kv_cache_sm = nullptr;
  int32_t * error_out = nullptr;
};

struct run_compute {
  emel::kv::cache::sm * kv_cache_sm = nullptr;
  const execute * request = nullptr;
  int32_t * error_out = nullptr;
};

struct extract_outputs {
  int32_t * error_out = nullptr;
};

struct rollback {
  emel::kv::cache::sm * kv_cache_sm = nullptr;
  int32_t * error_out = nullptr;
};

}  // namespace emel::decoder::ubatch_executor::event

namespace emel::decoder::ubatch_executor::events {

struct validate_done {
  const event::execute * request = nullptr;
};
struct validate_error {
  int32_t err = 0;
  const event::execute * request = nullptr;
};

struct prepare_memory_done {
  const event::execute * request = nullptr;
};
struct prepare_memory_error {
  int32_t err = 0;
  const event::execute * request = nullptr;
};

struct prepare_kv_done {
  const event::execute * request = nullptr;
};
struct prepare_kv_error {
  int32_t err = 0;
  const event::execute * request = nullptr;
};

struct run_compute_done {
  const event::execute * request = nullptr;
};
struct run_compute_error {
  int32_t err = 0;
  const event::execute * request = nullptr;
};

struct extract_outputs_done {
  const event::execute * request = nullptr;
};
struct extract_outputs_error {
  int32_t err = 0;
  const event::execute * request = nullptr;
};

struct rollback_done {
  const event::execute * request = nullptr;
};
struct rollback_error {
  int32_t err = 0;
  const event::execute * request = nullptr;
};

struct ubatch_execution_done {
  int32_t outputs_produced = 0;
  int32_t kv_tokens = 0;
  int32_t * error_out = nullptr;
  const event::execute * request = nullptr;
};

struct ubatch_execution_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
  const event::execute * request = nullptr;
};

}  // namespace emel::decoder::ubatch_executor::events
