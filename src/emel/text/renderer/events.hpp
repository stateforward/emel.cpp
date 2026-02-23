#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/model/data.hpp"
#include "emel/text/detokenizer/events.hpp"

namespace emel::text::renderer {

enum class sequence_status : uint8_t {
  running = 0,
  stop_sequence_matched = 1,
};

}  // namespace emel::text::renderer

namespace emel::text::renderer::events {

struct binding_done;
struct binding_error;
struct rendering_done;
struct rendering_error;
struct flush_done;
struct flush_error;

}  // namespace emel::text::renderer::events

namespace emel::text::renderer::event {

struct bind {
  const emel::model::data::vocab * vocab = nullptr;
  void * detokenizer_sm = nullptr;
  bool (*dispatch_detokenizer_bind)(
      void * detokenizer_sm,
      const emel::text::detokenizer::event::bind &) = nullptr;
  bool (*dispatch_detokenizer_detokenize)(
      void * detokenizer_sm,
      const emel::text::detokenizer::event::detokenize &) = nullptr;
  bool strip_leading_space = false;
  const std::string_view * stop_sequences = nullptr;
  size_t stop_sequence_count = 0;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm,
                        const events::binding_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm,
                         const events::binding_error &) = nullptr;
};

struct render {
  int32_t token_id = -1;
  int32_t sequence_id = 0;
  bool emit_special = false;
  char * output = nullptr;
  size_t output_capacity = 0;
  size_t * output_length_out = nullptr;
  sequence_status * status_out = nullptr;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm,
                        const events::rendering_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm,
                         const events::rendering_error &) = nullptr;
};

struct flush {
  int32_t sequence_id = 0;
  char * output = nullptr;
  size_t output_capacity = 0;
  size_t * output_length_out = nullptr;
  sequence_status * status_out = nullptr;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm,
                        const events::flush_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm,
                         const events::flush_error &) = nullptr;
};

}  // namespace emel::text::renderer::event

namespace emel::text::renderer::events {

struct binding_done {
  const event::bind * request = nullptr;
};

struct binding_error {
  const event::bind * request = nullptr;
  int32_t err = 0;
};

struct rendering_done {
  const event::render * request = nullptr;
  size_t output_length = 0;
  sequence_status status = sequence_status::running;
};

struct rendering_error {
  const event::render * request = nullptr;
  int32_t err = 0;
};

struct flush_done {
  const event::flush * request = nullptr;
  size_t output_length = 0;
  sequence_status status = sequence_status::running;
};

struct flush_error {
  const event::flush * request = nullptr;
  int32_t err = 0;
};

}  // namespace emel::text::renderer::events
