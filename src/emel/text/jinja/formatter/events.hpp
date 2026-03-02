#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/text/jinja/ast.hpp"
#include "emel/text/jinja/formatter/errors.hpp"
#include "emel/text/jinja/value.hpp"

namespace emel::text::jinja::events {

struct rendering_done;
struct rendering_error;

}  // namespace emel::text::jinja::events

namespace emel::text::jinja::event {

struct render {
  using done_callback = ::emel::callback<bool(const ::emel::text::jinja::events::rendering_done &)>;
  using error_callback = ::emel::callback<bool(const ::emel::text::jinja::events::rendering_error &)>;

  render(const emel::text::jinja::program & program_ref,
         std::string_view source_text_ref,
         char & output_ref,
         const size_t output_capacity_ref,
         const done_callback dispatch_done_ref,
         const error_callback dispatch_error_ref,
         const emel::text::jinja::object_value * globals_ref = nullptr,
         size_t * output_length_ref = nullptr,
         bool * output_truncated_ref = nullptr,
         int32_t * error_out_ref = nullptr,
         size_t * error_pos_out_ref = nullptr) noexcept
      : program(program_ref),
        source_text(source_text_ref),
        output(output_ref),
        output_capacity(output_capacity_ref),
        dispatch_done(dispatch_done_ref),
        dispatch_error(dispatch_error_ref),
        globals(globals_ref),
        output_length(output_length_ref),
        output_truncated(output_truncated_ref),
        error_out(error_out_ref),
        error_pos_out(error_pos_out_ref) {}

  const emel::text::jinja::program & program;
  const std::string_view source_text;
  char & output;
  const size_t output_capacity;
  const done_callback dispatch_done;
  const error_callback dispatch_error;
  const emel::text::jinja::object_value * const globals;
  size_t * const output_length;
  bool * const output_truncated;
  int32_t * const error_out;
  size_t * const error_pos_out;
};

struct render_ctx {
  render_ctx(size_t & output_length_ref,
             bool & output_truncated_ref,
             int32_t & error_out_ref,
             size_t & error_pos_out_ref) noexcept
      : output_length(output_length_ref),
        output_truncated(output_truncated_ref),
        error_out(error_out_ref),
        error_pos_out(error_pos_out_ref) {}

  formatter::error err = formatter::error::none;
  size_t & output_length;
  bool & output_truncated;
  int32_t & error_out;
  size_t & error_pos_out;
};

struct render_runtime {
  const render & request;
  render_ctx & ctx;
};

}  // namespace emel::text::jinja::event

namespace emel::text::jinja::events {

struct rendering_done {
  const event::render & request;
  size_t output_length;
  bool output_truncated;
};

struct rendering_error {
  const event::render & request;
  int32_t err;
  size_t error_pos;
};

}  // namespace emel::text::jinja::events
