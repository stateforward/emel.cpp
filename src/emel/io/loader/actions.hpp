#pragma once

#include "emel/io/loader/context.hpp"
#include "emel/io/loader/detail.hpp"
#include "emel/io/loader/errors.hpp"
#include "emel/io/loader/events.hpp"

namespace emel::io::loader::action {

struct effect_begin_load_tensor {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.ok = false;
  }
};

struct effect_mark_invalid_request {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    ev.ctx.ok = false;
  }
};

struct effect_mark_unsupported_strategy {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::unsupported_strategy);
    ev.ctx.ok = false;
  }
};

struct effect_publish_load_tensor_error {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::load_tensor_error{
        .request = ev.request,
        .err = ev.ctx.err,
    });
  }
};

struct effect_record_load_tensor_error {
  void operator()(const detail::load_tensor_runtime &,
                  context &) const noexcept {}
};

struct effect_publish_load_tensor_done {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &) const noexcept {
    ev.request.on_done(events::load_tensor_done{
        .request = ev.request,
        .strategy = ev.request.policy.strategy,
        .buffer = ev.request.tensor.target,
        .buffer_bytes = ev.request.tensor.byte_size,
    });
  }
};

struct effect_record_load_tensor_done {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.ok = true;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
      ev.ctx.ok = false;
    }
  }
};

inline constexpr effect_begin_load_tensor effect_begin_load_tensor{};
inline constexpr effect_mark_invalid_request effect_mark_invalid_request{};
inline constexpr effect_mark_unsupported_strategy
    effect_mark_unsupported_strategy{};
inline constexpr effect_publish_load_tensor_error
    effect_publish_load_tensor_error{};
inline constexpr effect_record_load_tensor_error
    effect_record_load_tensor_error{};
inline constexpr effect_publish_load_tensor_done
    effect_publish_load_tensor_done{};
inline constexpr effect_record_load_tensor_done
    effect_record_load_tensor_done{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::io::loader::action
