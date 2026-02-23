#pragma once

#include <cstdint>

namespace emel::graph::processor::event {

struct execute;

using validate_fn = bool (*)(const execute &, int32_t * err_out);
using prepare_graph_fn = bool (*)(const execute &, bool * reused_out, int32_t * err_out);
using alloc_graph_fn = bool (*)(const execute &, int32_t * err_out);
using bind_inputs_fn = bool (*)(const execute &, int32_t * err_out);
using run_backend_fn = bool (*)(const execute &, int32_t * err_out);
using extract_outputs_fn = bool (*)(const execute &, int32_t * outputs_out, int32_t * err_out);

struct execute {
  int32_t ubatch_index = 0;
  int32_t ubatch_size = 0;
  int32_t kv_tokens = 0;
  void * memory_coordinator_sm = nullptr;
  void * kv_cache_sm = nullptr;
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
  run_backend_fn run_backend = nullptr;
  extract_outputs_fn extract_outputs = nullptr;
  int32_t * outputs_produced_out = nullptr;
  int32_t * kv_tokens_out = nullptr;
  bool * rollback_attempted_out = nullptr;
  int32_t * error_out = nullptr;
};

struct validate {
  const execute * request = nullptr;
  int32_t * error_out = nullptr;
};

struct prepare_graph {
  const execute * request = nullptr;
  bool * reused_out = nullptr;
  int32_t * error_out = nullptr;
};

struct alloc_graph {
  const execute * request = nullptr;
  int32_t * error_out = nullptr;
};

struct bind_inputs {
  const execute * request = nullptr;
  int32_t * error_out = nullptr;
};

struct run_backend {
  const execute * request = nullptr;
  int32_t * error_out = nullptr;
};

struct extract_outputs {
  const execute * request = nullptr;
  int32_t * error_out = nullptr;
};

}  // namespace emel::graph::processor::event

namespace emel::graph::processor::events {

struct validate_done {
  const event::execute * request = nullptr;
};
struct validate_error {
  int32_t err = 0;
  const event::execute * request = nullptr;
};

struct prepare_graph_done {
  const event::execute * request = nullptr;
  bool reused = false;
};
struct prepare_graph_error {
  int32_t err = 0;
  const event::execute * request = nullptr;
};

struct alloc_graph_done {
  const event::execute * request = nullptr;
};
struct alloc_graph_error {
  int32_t err = 0;
  const event::execute * request = nullptr;
};

struct bind_inputs_done {
  const event::execute * request = nullptr;
};
struct bind_inputs_error {
  int32_t err = 0;
  const event::execute * request = nullptr;
};

struct run_backend_done {
  const event::execute * request = nullptr;
};
struct run_backend_error {
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

struct compute_done {
  int32_t outputs_produced = 0;
  int32_t * error_out = nullptr;
  const event::execute * request = nullptr;
};

struct compute_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
  const event::execute * request = nullptr;
};

}  // namespace emel::graph::processor::events
