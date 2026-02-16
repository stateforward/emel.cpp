#pragma once

#include <array>
#include <cstdint>

#include "emel/buffer/allocator/events.hpp"

namespace emel::buffer::realloc_analyzer::event {

using graph_view = emel::buffer::allocator::event::graph_view;
inline constexpr int32_t k_max_sources = emel::buffer::allocator::event::k_max_sources;

struct tensor_alloc {
  int32_t tensor_id = -1;
  int32_t buffer_id = -1;
  int32_t size_max = 0;
  int32_t alignment = 0;
};

struct node_alloc {
  tensor_alloc dst = {};
  std::array<tensor_alloc, k_max_sources> src = {};
};

struct leaf_alloc {
  tensor_alloc leaf = {};
};

struct analyze {
  graph_view graph = {};
  const node_alloc * node_allocs = nullptr;
  int32_t node_alloc_count = 0;
  const leaf_alloc * leaf_allocs = nullptr;
  int32_t leaf_alloc_count = 0;
  int32_t * needs_realloc_out = nullptr;
  int32_t * error_out = nullptr;
};

struct validate {
  graph_view graph = {};
  const node_alloc * node_allocs = nullptr;
  int32_t node_alloc_count = 0;
  const leaf_alloc * leaf_allocs = nullptr;
  int32_t leaf_alloc_count = 0;
  int32_t * error_out = nullptr;
};
struct evaluate {
  graph_view graph = {};
  const node_alloc * node_allocs = nullptr;
  int32_t node_alloc_count = 0;
  const leaf_alloc * leaf_allocs = nullptr;
  int32_t leaf_alloc_count = 0;
  int32_t * error_out = nullptr;
};
struct publish {
  int32_t * needs_realloc_out = nullptr;
  int32_t * error_out = nullptr;
};
struct reset {
  int32_t * error_out = nullptr;
};

}  // namespace emel::buffer::realloc_analyzer::event

namespace emel::buffer::realloc_analyzer::events {

struct validate_done {
  const event::analyze * request = nullptr;
};
struct validate_error {
  int32_t err = 0;
  const event::analyze * request = nullptr;
};

struct evaluate_done {
  const event::analyze * request = nullptr;
};
struct evaluate_error {
  int32_t err = 0;
  const event::analyze * request = nullptr;
};

struct publish_done {
  const event::analyze * request = nullptr;
};
struct publish_error {
  int32_t err = 0;
  const event::analyze * request = nullptr;
};

struct analyze_done {
  int32_t needs_realloc = 0;
  int32_t * needs_realloc_out = nullptr;
  int32_t * error_out = nullptr;
  const event::analyze * request = nullptr;
};
struct analyze_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
  const event::analyze * request = nullptr;
};

struct reset_done {
  int32_t * error_out = nullptr;
  const event::reset * request = nullptr;
};
struct reset_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
  const event::reset * request = nullptr;
};

using bootstrap_event = event::analyze;

}  // namespace emel::buffer::realloc_analyzer::events
