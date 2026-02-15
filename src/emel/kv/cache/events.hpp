#pragma once

#include <cstdint>

namespace emel::kv::cache::event {

struct prepare {
  const int32_t * ubatch_sizes = nullptr;
  int32_t ubatch_count = 0;
  int32_t requested_capacity = 0;

  int32_t * slot_offsets_out = nullptr;
  int32_t slot_offsets_capacity = 0;
  int32_t * ubatch_count_out = nullptr;
};

struct apply_ubatch {
  int32_t ubatch_index = 0;
  int32_t * kv_tokens_out = nullptr;
};

struct rollback {
  int32_t from_ubatch_index = 0;
};

struct validate {
  int32_t * error_out = nullptr;
};

struct prepare_slots {
  int32_t * error_out = nullptr;
};

struct apply_step {
  int32_t * error_out = nullptr;
};

struct rollback_step {
  int32_t * error_out = nullptr;
};

struct publish {
  int32_t * error_out = nullptr;
};

}  // namespace emel::kv::cache::event

namespace emel::kv::cache::events {

struct validate_done {};
struct validate_error {
  int32_t err = 0;
};

struct prepare_slots_done {};
struct prepare_slots_error {
  int32_t err = 0;
};

struct apply_done {};
struct apply_error {
  int32_t err = 0;
};

struct rollback_done {};
struct rollback_error {
  int32_t err = 0;
};

struct publish_done {};
struct publish_error {
  int32_t err = 0;
};

struct kv_done {};
struct kv_error {
  int32_t err = 0;
};

}  // namespace emel::kv::cache::events
