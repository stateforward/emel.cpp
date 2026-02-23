#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/model/data.hpp"

namespace emel::text::detokenizer::events {

struct binding_done;
struct binding_error;
struct detokenize_done;
struct detokenize_error;

}  // namespace emel::text::detokenizer::events

namespace emel::text::detokenizer::event {

struct bind {
  const emel::model::data::vocab * vocab = nullptr;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm,
                        const events::binding_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm,
                         const events::binding_error &) = nullptr;
};

struct detokenize {
  int32_t token_id = -1;
  bool emit_special = false;
  uint8_t * pending_bytes = nullptr;
  size_t pending_length = 0;
  size_t pending_capacity = 0;
  char * output = nullptr;
  size_t output_capacity = 0;
  size_t * output_length_out = nullptr;
  size_t * pending_length_out = nullptr;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm,
                        const events::detokenize_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm,
                         const events::detokenize_error &) = nullptr;
};

}  // namespace emel::text::detokenizer::event

namespace emel::text::detokenizer::events {

struct binding_done {
  const event::bind * request = nullptr;
};

struct binding_error {
  const event::bind * request = nullptr;
  int32_t err = 0;
};

struct detokenize_done {
  const event::detokenize * request = nullptr;
  size_t output_length = 0;
  size_t pending_length = 0;
};

struct detokenize_error {
  const event::detokenize * request = nullptr;
  int32_t err = 0;
};

}  // namespace emel::text::detokenizer::events
