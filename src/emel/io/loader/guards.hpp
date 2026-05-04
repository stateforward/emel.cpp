#pragma once

#include "emel/io/loader/context.hpp"
#include "emel/io/loader/detail.hpp"
#include "emel/io/loader/events.hpp"

namespace emel::io::loader::guard {

struct tensor_span_valid {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.tensor.byte_size > 0u;
  }
};

struct tensor_span_invalid {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !tensor_span_valid{}(ev, ctx);
  }
};

struct strategy_none {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.request.policy.strategy == event::strategy_kind::none;
  }
};

struct strategy_mapped_file {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.request.policy.strategy == event::strategy_kind::mapped_file;
  }
};

struct strategy_staged_read {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.request.policy.strategy == event::strategy_kind::staged_read;
  }
};

struct strategy_external_buffer {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.request.policy.strategy == event::strategy_kind::external_buffer;
  }
};

struct done_callback_present {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct done_callback_absent {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return !done_callback_present{}(ev);
  }
};

struct error_callback_present {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct error_callback_absent {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return !error_callback_present{}(ev);
  }
};

} // namespace emel::io::loader::guard
