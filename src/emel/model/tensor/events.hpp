#pragma once

#include <cstdint>

#include "emel/model/data.hpp"

namespace emel::model::tensor::event {

enum class lifecycle : uint8_t {
  unbound = 0u,
  resident = 1u,
  evicted = 2u,
  internal_error = 3u,
};

struct tensor_state {
  lifecycle lifecycle_state = lifecycle::unbound;
  const void * buffer = nullptr;
  uint64_t buffer_bytes = 0u;
  uint64_t file_offset = 0u;
  uint64_t data_size = 0u;
  uint16_t file_index = 0u;
  int32_t tensor_type = 0;
};

struct bind_tensor {
  int32_t tensor_id = 0;
  const emel::model::data::tensor_record & tensor_record;
  const void * buffer = nullptr;
  uint64_t buffer_bytes = 0u;
  int32_t * error_out = nullptr;

  bind_tensor(const int32_t tensor_id_in,
              const emel::model::data::tensor_record & tensor_record_in,
              const void * buffer_in,
              const uint64_t buffer_bytes_in) noexcept
      : tensor_id(tensor_id_in),
        tensor_record(tensor_record_in),
        buffer(buffer_in),
        buffer_bytes(buffer_bytes_in) {}
};

struct evict_tensor {
  int32_t tensor_id = 0;
  int32_t * error_out = nullptr;
};

struct capture_tensor_state {
  int32_t tensor_id = 0;
  tensor_state * state_out = nullptr;
  int32_t * error_out = nullptr;
};

}  // namespace emel::model::tensor::event

namespace emel::model::tensor::events {

struct bind_tensor_done {
  const event::bind_tensor * request = nullptr;
};
struct bind_tensor_error {
  int32_t err = 0;
  const event::bind_tensor * request = nullptr;
};

struct evict_tensor_done {
  const event::evict_tensor * request = nullptr;
};
struct evict_tensor_error {
  int32_t err = 0;
  const event::evict_tensor * request = nullptr;
};

struct capture_tensor_state_done {
  const event::capture_tensor_state * request = nullptr;
};
struct capture_tensor_state_error {
  int32_t err = 0;
  const event::capture_tensor_state * request = nullptr;
};

}  // namespace emel::model::tensor::events
