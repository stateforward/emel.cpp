#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/model/data.hpp"

namespace emel::parser::gguf {

struct requirements {
  uint32_t tensor_count = 0;
  uint32_t kv_count = 0;
  uint32_t max_key_bytes = 0;
  uint32_t max_value_bytes = 0;
};

namespace events {
struct probe_done;
struct probe_error;
struct bind_done;
struct bind_error;
struct parse_done;
struct parse_error;
}  // namespace events

namespace event {

struct probe {
  const void * file_image = nullptr;
  uint64_t size = 0;
  requirements * requirements_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const events::probe_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const events::probe_error &) = nullptr;
};

struct bind_storage {
  void * kv_arena = nullptr;
  uint64_t kv_arena_size = 0;
  void * kv_entries = nullptr;
  uint32_t kv_entry_capacity = 0;
  emel::model::data::tensor_record * tensors = nullptr;
  uint32_t tensor_capacity = 0;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const events::bind_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const events::bind_error &) = nullptr;
};

struct parse {
  const void * file_image = nullptr;
  uint64_t size = 0;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const events::parse_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const events::parse_error &) = nullptr;
};

}  // namespace event

namespace events {

struct probe_done {
  const event::probe * request = nullptr;
  requirements requirements_out = {};
};

struct probe_error {
  const event::probe * request = nullptr;
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
};

struct bind_done {
  const event::bind_storage * request = nullptr;
};

struct bind_error {
  const event::bind_storage * request = nullptr;
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
};

struct parse_done {
  const event::parse * request = nullptr;
};

struct parse_error {
  const event::parse * request = nullptr;
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
};

}  // namespace events

}  // namespace emel::parser::gguf
