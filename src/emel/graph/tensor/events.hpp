#pragma once

#include <cstdint>

namespace emel::graph::tensor::event {

enum class lifecycle : uint8_t {
  unallocated = 0u,
  empty = 1u,
  filled = 2u,
  leaf_filled = 3u,
  internal_error = 4u,
};

struct tensor_state {
  lifecycle lifecycle_state = lifecycle::unallocated;
  uint8_t is_leaf = 0u;
  uint32_t seed_refs = 0u;
  uint32_t live_refs = 0u;
  void * buffer = nullptr;
  uint64_t buffer_bytes = 0u;
};

struct reserve_tensor {
  int32_t tensor_id = 0;
  void * buffer = nullptr;
  uint64_t buffer_bytes = 0u;
  int32_t consumer_refs = 0;
  bool is_leaf = false;
  int32_t * error_out = nullptr;
};

struct publish_filled_tensor {
  int32_t tensor_id = 0;
  int32_t * error_out = nullptr;
};

struct release_tensor_ref {
  int32_t tensor_id = 0;
  int32_t * error_out = nullptr;
};

struct reset_tensor_epoch {
  int32_t tensor_id = 0;
  int32_t * error_out = nullptr;
};

struct capture_tensor_state {
  int32_t tensor_id = 0;
  tensor_state * state_out = nullptr;
  int32_t * error_out = nullptr;
};

}  // namespace emel::graph::tensor::event

namespace emel::graph::tensor::events {

struct reserve_tensor_done {
  const event::reserve_tensor * request = nullptr;
};
struct reserve_tensor_error {
  int32_t err = 0;
  const event::reserve_tensor * request = nullptr;
};

struct publish_filled_tensor_done {
  const event::publish_filled_tensor * request = nullptr;
};
struct publish_filled_tensor_error {
  int32_t err = 0;
  const event::publish_filled_tensor * request = nullptr;
};

struct release_tensor_ref_done {
  const event::release_tensor_ref * request = nullptr;
};
struct release_tensor_ref_error {
  int32_t err = 0;
  const event::release_tensor_ref * request = nullptr;
};

struct reset_tensor_epoch_done {
  const event::reset_tensor_epoch * request = nullptr;
};
struct reset_tensor_epoch_error {
  int32_t err = 0;
  const event::reset_tensor_epoch * request = nullptr;
};

struct capture_tensor_state_done {
  const event::capture_tensor_state * request = nullptr;
};
struct capture_tensor_state_error {
  int32_t err = 0;
  const event::capture_tensor_state * request = nullptr;
};

}  // namespace emel::graph::tensor::events
