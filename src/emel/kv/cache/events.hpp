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
  int32_t * error_out = nullptr;
};

struct apply_ubatch {
  int32_t ubatch_index = 0;
  int32_t * kv_tokens_out = nullptr;
  int32_t * error_out = nullptr;
};

struct rollback {
  int32_t from_ubatch_index = 0;
  int32_t * error_out = nullptr;
};

struct validate_prepare {
  const prepare * request = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_apply {
  const apply_ubatch * request = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_rollback {
  const rollback * request = nullptr;
  int32_t * error_out = nullptr;
};

struct prepare_slots {
  int32_t * error_out = nullptr;
};

struct apply_step {
  const apply_ubatch * request = nullptr;
  int32_t * error_out = nullptr;
};

struct rollback_step {
  const rollback * request = nullptr;
  int32_t * error_out = nullptr;
};

struct publish {
  int32_t * error_out = nullptr;
};

}  // namespace emel::kv::cache::event

namespace emel::kv::cache::events {

struct request_ref {
  const event::prepare * prepare = nullptr;
  const event::apply_ubatch * apply = nullptr;
  const event::rollback * rollback = nullptr;
};

struct validate_done {
  request_ref request = {};
};
struct validate_error {
  int32_t err = 0;
  request_ref request = {};
};

struct prepare_slots_done {
  request_ref request = {};
};
struct prepare_slots_error {
  int32_t err = 0;
  request_ref request = {};
};

struct apply_done {
  request_ref request = {};
};
struct apply_error {
  int32_t err = 0;
  request_ref request = {};
};

struct rollback_done {
  request_ref request = {};
};
struct rollback_error {
  int32_t err = 0;
  request_ref request = {};
};

struct publish_done {
  request_ref request = {};
};
struct publish_error {
  int32_t err = 0;
  request_ref request = {};
};

struct kv_done {};
struct kv_error {
  int32_t err = 0;
};

}  // namespace emel::kv::cache::events
