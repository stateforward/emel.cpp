#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/text/jinja/formatter/errors.hpp"
#include "emel/text/jinja/formatter/events.hpp"

namespace emel::text::jinja::formatter::detail {

template <class value_type>
inline value_type & bind_optional(value_type * destination,
                                  value_type & sink) noexcept {
  value_type * destinations[2] = {&sink, destination};
  return *destinations[static_cast<size_t>(destination != nullptr)];
}

inline constexpr int32_t to_error_code(const error err) noexcept {
  return static_cast<int32_t>(err);
}

inline void reset_result(event::render_ctx & runtime_ctx) noexcept {
  runtime_ctx.err = error::none;
  runtime_ctx.output_length = 0;
  runtime_ctx.output_truncated = false;
  runtime_ctx.error_out = to_error_code(error::none);
  runtime_ctx.error_pos_out = 0;
}

inline void mark_done(event::render_ctx & runtime_ctx,
                      const size_t output_length,
                      const bool output_truncated) noexcept {
  runtime_ctx.err = error::none;
  runtime_ctx.output_length = output_length;
  runtime_ctx.output_truncated = output_truncated;
  runtime_ctx.error_out = to_error_code(error::none);
  runtime_ctx.error_pos_out = 0;
}

inline void mark_error(event::render_ctx & runtime_ctx,
                       const error err,
                       const bool output_truncated,
                       const size_t error_pos) noexcept {
  runtime_ctx.err = err;
  runtime_ctx.output_length = 0;
  runtime_ctx.output_truncated = output_truncated;
  runtime_ctx.error_out = to_error_code(err);
  runtime_ctx.error_pos_out = error_pos;
}

inline void emit_done(const event::render & request,
                      const event::render_ctx & runtime_ctx) noexcept {
  const events::rendering_done done_ev{
      request,
      runtime_ctx.output_length,
      runtime_ctx.output_truncated,
  };
  (void)request.dispatch_done(done_ev);
}

inline void emit_error(const event::render & request,
                       const event::render_ctx & runtime_ctx) noexcept {
  const events::rendering_error error_ev{
      request,
      to_error_code(runtime_ctx.err),
      runtime_ctx.error_pos_out,
  };
  (void)request.dispatch_error(error_ev);
}

}  // namespace emel::text::jinja::formatter::detail
