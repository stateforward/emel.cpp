#pragma once

#include <cstdint>

namespace emel::decoder::compute_executor::event {

struct execute {
  int32_t ubatch_index = 0;
  int32_t ubatch_size = 0;
  int32_t kv_tokens = 0;
  int32_t * outputs_produced_out = nullptr;
  int32_t * error_out = nullptr;
};

struct validate {
  int32_t * error_out = nullptr;
};

struct bind_inputs {
  int32_t * error_out = nullptr;
};

struct run_backend {
  int32_t * error_out = nullptr;
};

struct extract_outputs {
  int32_t * error_out = nullptr;
};

}  // namespace emel::decoder::compute_executor::event

namespace emel::decoder::compute_executor::events {

struct validate_done {
  const event::execute * request = nullptr;
};
struct validate_error {
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

}  // namespace emel::decoder::compute_executor::events
