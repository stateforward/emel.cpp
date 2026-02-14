#pragma once

#include <cstdint>

#include "emel/buffer/allocator/events.hpp"

namespace emel::buffer::planner {
struct strategy;
}  // namespace emel::buffer::planner

namespace emel::buffer::planner::event {

using graph_view = emel::buffer::allocator::event::graph_view;

struct plan {
  graph_view graph = {};
  const int32_t * node_buffer_ids = nullptr;
  const int32_t * leaf_buffer_ids = nullptr;
  int32_t buffer_count = 1;
  bool size_only = false;
  int32_t * sizes_out = nullptr;
  int32_t sizes_out_count = 0;
  int32_t * error_out = nullptr;
  const emel::buffer::planner::strategy * strategy = nullptr;
};

struct reset_done {};
struct reset_error {
  int32_t err = 0;
};

struct seed_leafs_done {};
struct seed_leafs_error {
  int32_t err = 0;
};

struct count_references_done {};
struct count_references_error {
  int32_t err = 0;
};

struct alloc_explicit_inputs_done {};
struct alloc_explicit_inputs_error {
  int32_t err = 0;
};

struct plan_nodes_done {};
struct plan_nodes_error {
  int32_t err = 0;
};

struct release_expired_done {};
struct release_expired_error {
  int32_t err = 0;
};

struct finalize_done {};
struct finalize_error {
  int32_t err = 0;
};

}  // namespace emel::buffer::planner::event

namespace emel::buffer::planner::events {

struct plan_done {
  int32_t total_bytes = 0;
};

struct plan_error {
  int32_t err = 0;
};

using bootstrap_event = event::plan;

}  // namespace emel::buffer::planner::events
