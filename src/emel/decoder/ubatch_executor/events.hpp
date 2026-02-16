#pragma once

#include <cstdint>

namespace emel::kv::cache {
struct sm;
}  // namespace emel::kv::cache

namespace emel::memory::coordinator {
struct sm;
}  // namespace emel::memory::coordinator

namespace emel::decoder::ubatch_executor::event {

struct execute {
  int32_t ubatch_index = 0;
  int32_t ubatch_size = 0;
  emel::memory::coordinator::sm * memory_coordinator_sm = nullptr;
  emel::kv::cache::sm * kv_cache_sm = nullptr;
  int32_t * outputs_produced_out = nullptr;
  int32_t * kv_tokens_out = nullptr;
  bool * rollback_attempted_out = nullptr;
  int32_t * error_out = nullptr;
};

struct validate {
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
