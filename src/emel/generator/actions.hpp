#pragma once

#include "emel/generator/context.hpp"
#include "emel/generator/errors.hpp"
#include "emel/generator/events.hpp"

namespace emel::generator::action {

struct begin_generate {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.tokens_generated = 0;
    ev.ctx.target_tokens = ev.request.max_tokens;
  }
};

struct reject_invalid_generate {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    ev.ctx.tokens_generated = 0;
    ev.ctx.target_tokens = 0;
  }
};

struct request_conditioning {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
  }
};

struct request_planning {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.target_tokens = ev.request.max_tokens;
  }
};

struct request_prefill {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
  }
};

struct request_decode_compute {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
  }
};

struct request_decode_sample {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
  }
};

struct request_decode_render {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.tokens_generated += 1;
  }
};

struct dispatch_done_with_error_out {
  void operator()(const event::generate_run & ev, const context &) const noexcept {
    *ev.request.error_out = emel::error::cast(error::none);
    ev.request.on_done(
      events::generation_done{
        .request = &ev.request,
        .tokens_generated = ev.ctx.tokens_generated,
      });
  }
};

struct dispatch_done_without_error_out {
  void operator()(const event::generate_run & ev, const context &) const noexcept {
    ev.request.on_done(
      events::generation_done{
        .request = &ev.request,
        .tokens_generated = ev.ctx.tokens_generated,
      });
  }
};

struct dispatch_error_with_dispatch_and_error_out {
  void operator()(const event::generate_run & ev, const context &) const noexcept {
    *ev.request.error_out = ev.ctx.err;
    ev.request.on_error(
      events::generation_error{
        .request = &ev.request,
        .err = ev.ctx.err,
        .tokens_generated = ev.ctx.tokens_generated,
      });
  }
};

struct dispatch_error_with_dispatch_only {
  void operator()(const event::generate_run & ev, const context &) const noexcept {
    ev.request.on_error(
      events::generation_error{
        .request = &ev.request,
        .err = ev.ctx.err,
        .tokens_generated = ev.ctx.tokens_generated,
      });
  }
};

struct dispatch_error_with_error_out_only {
  void operator()(const event::generate_run & ev, const context &) const noexcept {
    *ev.request.error_out = ev.ctx.err;
  }
};

struct dispatch_error_without_error_channels {
  void operator()(const event::generate_run &, const context &) const noexcept {}
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::backend);
    }
  }
};

inline constexpr begin_generate begin_generate{};
inline constexpr reject_invalid_generate reject_invalid_generate{};
inline constexpr request_conditioning request_conditioning{};
inline constexpr request_planning request_planning{};
inline constexpr request_prefill request_prefill{};
inline constexpr request_decode_compute request_decode_compute{};
inline constexpr request_decode_sample request_decode_sample{};
inline constexpr request_decode_render request_decode_render{};
inline constexpr dispatch_done_with_error_out dispatch_done_with_error_out{};
inline constexpr dispatch_done_without_error_out dispatch_done_without_error_out{};
inline constexpr dispatch_error_with_dispatch_and_error_out
    dispatch_error_with_dispatch_and_error_out{};
inline constexpr dispatch_error_with_dispatch_only dispatch_error_with_dispatch_only{};
inline constexpr dispatch_error_with_error_out_only dispatch_error_with_error_out_only{};
inline constexpr dispatch_error_without_error_channels dispatch_error_without_error_channels{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::generator::action
