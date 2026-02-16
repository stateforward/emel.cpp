#pragma once

#include <cstdint>

#include "emel/buffer/allocator/events.hpp"

namespace emel::buffer::planner {
struct strategy;
}  // namespace emel::buffer::planner

namespace emel::buffer::planner::events {
struct plan_done;
struct plan_error;
}  // namespace emel::buffer::planner::events

namespace emel::buffer::planner::event {

using graph_view = emel::buffer::allocator::event::graph_view;

struct plan {
  graph_view graph = {};
  const int32_t * node_buffer_ids = nullptr;
  const int32_t * leaf_buffer_ids = nullptr;
  int32_t buffer_count = 1;
  const int32_t * buffer_alignments = nullptr;
  const int32_t * buffer_max_sizes = nullptr;
  bool size_only = false;
  int32_t * sizes_out = nullptr;
  int32_t sizes_out_count = 0;
  int32_t * chunk_sizes_out = nullptr;
  int32_t chunk_sizes_out_count = 0;
  int32_t * chunk_counts_out = nullptr;
  int32_t chunk_counts_out_count = 0;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const events::plan_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const events::plan_error &) = nullptr;
  const emel::buffer::planner::strategy * strategy = nullptr;
};

struct reset_done {
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};
struct reset_error {
  int32_t err = 0;
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};

struct seed_leafs_done {
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};
struct seed_leafs_error {
  int32_t err = 0;
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};

struct count_references_done {
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};
struct count_references_error {
  int32_t err = 0;
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};

struct alloc_explicit_inputs_done {
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};
struct alloc_explicit_inputs_error {
  int32_t err = 0;
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};

struct plan_nodes_done {
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};
struct plan_nodes_error {
  int32_t err = 0;
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};

struct release_expired_done {
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};
struct release_expired_error {
  int32_t err = 0;
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};

struct finalize_done {
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};
struct finalize_error {
  int32_t err = 0;
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};

struct split_required_done {
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};
struct split_required_error {
  int32_t err = 0;
  const plan * request = nullptr;
  int32_t * error_out = nullptr;
};

}  // namespace emel::buffer::planner::event

namespace emel::buffer::planner::events {

struct plan_done {
  int32_t total_bytes = 0;
  int32_t * error_out = nullptr;
};

struct plan_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
};

using bootstrap_event = event::plan;

}  // namespace emel::buffer::planner::events
