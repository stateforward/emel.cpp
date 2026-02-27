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
  const emel::model::data::vocab & vocab;
  int32_t & error_out;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm,
                        const events::binding_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm,
                         const events::binding_error &) = nullptr;

  bind(const emel::model::data::vocab & vocab_in,
       int32_t & error_out_in,
       void * owner_sm_in = nullptr,
       bool (*dispatch_done_in)(void *, const events::binding_done &) = nullptr,
       bool (*dispatch_error_in)(void *, const events::binding_error &) = nullptr) noexcept
      : vocab(vocab_in),
        error_out(error_out_in),
        owner_sm(owner_sm_in),
        dispatch_done(dispatch_done_in),
        dispatch_error(dispatch_error_in) {}
};

struct detokenize {
  int32_t token_id;
  bool emit_special;
  uint8_t * pending_bytes;
  size_t pending_length;
  size_t pending_capacity;
  char * output;
  size_t output_capacity;
  size_t & output_length_out;
  size_t & pending_length_out;
  int32_t & error_out;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm,
                        const events::detokenize_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm,
                         const events::detokenize_error &) = nullptr;

  detokenize(int32_t token_id_in,
             bool emit_special_in,
             uint8_t * pending_bytes_in,
             size_t pending_length_in,
             size_t pending_capacity_in,
             char * output_in,
             size_t output_capacity_in,
             size_t & output_length_out_in,
             size_t & pending_length_out_in,
             int32_t & error_out_in,
             void * owner_sm_in = nullptr,
             bool (*dispatch_done_in)(void *, const events::detokenize_done &) = nullptr,
             bool (*dispatch_error_in)(void *, const events::detokenize_error &) = nullptr) noexcept
      : token_id(token_id_in),
        emit_special(emit_special_in),
        pending_bytes(pending_bytes_in),
        pending_length(pending_length_in),
        pending_capacity(pending_capacity_in),
        output(output_in),
        output_capacity(output_capacity_in),
        output_length_out(output_length_out_in),
        pending_length_out(pending_length_out_in),
        error_out(error_out_in),
        owner_sm(owner_sm_in),
        dispatch_done(dispatch_done_in),
        dispatch_error(dispatch_error_in) {}
};

}  // namespace emel::text::detokenizer::event

namespace emel::text::detokenizer::events {

struct binding_done {
  const event::bind & request;
};

struct binding_error {
  const event::bind & request;
  int32_t err;
};

struct detokenize_done {
  const event::detokenize & request;
  size_t output_length;
  size_t pending_length;
};

struct detokenize_error {
  const event::detokenize & request;
  int32_t err;
};

}  // namespace emel::text::detokenizer::events
