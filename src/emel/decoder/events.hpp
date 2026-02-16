#pragma once

#include <cstdint>

namespace emel::decoder::events {

struct decoding_done;
struct decoding_error;
struct owner_event;

}  // namespace emel::decoder::events

namespace emel::decoder::event {

struct decode {
  const int32_t * token_ids = nullptr;
  int32_t n_tokens = 0;
  int32_t n_ubatch = 0;
  void * owner_sm = nullptr;
  bool (*dispatch_event)(void * owner_sm, const events::owner_event &) = nullptr;
  int32_t * error_out = nullptr;
};

struct validate {
  int32_t * error_out = nullptr;
};

struct initialize_batch {
  int32_t * error_out = nullptr;
};

struct update_memory {
  int32_t * error_out = nullptr;
};

struct prepare_memory_batch {
  int32_t * error_out = nullptr;
  bool * retryable_out = nullptr;
};

struct optimize_memory {
  int32_t * error_out = nullptr;
};

struct reserve_output {
  int32_t * error_out = nullptr;
};

struct process_ubatch {
  int32_t * error_out = nullptr;
  bool * rollback_needed_out = nullptr;
};

struct rollback_ubatch {
  int32_t * error_out = nullptr;
  bool rollback_needed = false;
};

struct finalize_outputs {
  int32_t * error_out = nullptr;
};

}  // namespace emel::decoder::event

namespace emel::decoder::events {

struct validate_done {
  const event::decode * request = nullptr;
};
struct validate_error {
  int32_t err = 0;
  const event::decode * request = nullptr;
};

struct initialize_batch_done {
  const event::decode * request = nullptr;
};
struct initialize_batch_error {
  int32_t err = 0;
  const event::decode * request = nullptr;
};

struct update_memory_done {
  const event::decode * request = nullptr;
};
struct update_memory_error {
  int32_t err = 0;
  const event::decode * request = nullptr;
};

struct prepare_memory_batch_done {
  const event::decode * request = nullptr;
};
struct prepare_memory_batch_retryable_error {
  int32_t err = 0;
  const event::decode * request = nullptr;
};
struct prepare_memory_batch_permanent_error {
  int32_t err = 0;
  const event::decode * request = nullptr;
};

struct optimize_memory_done {
  const event::decode * request = nullptr;
};
struct optimize_memory_error {
  int32_t err = 0;
  const event::decode * request = nullptr;
};

struct reserve_output_done {
  const event::decode * request = nullptr;
};
struct reserve_output_error {
  int32_t err = 0;
  const event::decode * request = nullptr;
};

struct ubatch_done {
  const event::decode * request = nullptr;
};
struct ubatch_error {
  int32_t err = 0;
  bool rollback_needed = false;
  const event::decode * request = nullptr;
};

struct rollback_done {
  int32_t err = 0;
  const event::decode * request = nullptr;
};
struct rollback_error {
  int32_t err = 0;
  const event::decode * request = nullptr;
};

struct finalize_outputs_done {
  const event::decode * request = nullptr;
};
struct finalize_outputs_error {
  int32_t err = 0;
  const event::decode * request = nullptr;
};

struct decoding_done {
  int32_t outputs = 0;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_event)(void * owner_sm, const events::owner_event &) = nullptr;
  const event::decode * request = nullptr;
};

struct decoding_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_event)(void * owner_sm, const events::owner_event &) = nullptr;
  const event::decode * request = nullptr;
};

struct owner_event {
  enum class kind : uint8_t {
    done = 0,
    error,
  };

  kind type = kind::error;
  decoding_done done = {};
  decoding_error error = {};
};

}  // namespace emel::decoder::events
