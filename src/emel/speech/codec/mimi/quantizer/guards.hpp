#pragma once

#include "emel/speech/codec/mimi/detail.hpp"
#include "emel/speech/codec/mimi/quantizer/context.hpp"
#include "emel/speech/codec/mimi/quantizer/events.hpp"

namespace emel::speech::codec::mimi::quantizer::guard {

namespace detail {

inline bool runtime_bound(const mimi::detail::codec_runtime &runtime) noexcept {
  return runtime.model != nullptr && runtime.n_q > 0 && runtime.dim > 0 &&
         runtime.quantizer.codebook_dim > 0 &&
         runtime.quantizer.codebook_entries > 0;
}

inline uint64_t
rvq_workspace_floats(const mimi::detail::codec_runtime &runtime) noexcept {
  return 3u * static_cast<uint64_t>(runtime.quantizer.codebook_dim) +
         static_cast<uint64_t>(runtime.dim) + 8u;
}

} // namespace detail

template <class runtime_event_type> struct guard_runtime_bound {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return detail::runtime_bound(runtime_ev.request.runtime);
  }
};

template <class runtime_event_type> struct guard_runtime_unbound {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_runtime_bound<runtime_event_type>{}(runtime_ev, ctx);
  }
};

struct guard_encode_shape_valid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &request = runtime_ev.request;
    return request.latent.size() >= static_cast<size_t>(request.runtime.dim) &&
           request.codes_out.size() >=
               static_cast<size_t>(request.runtime.n_q) &&
           request.workspace.size() >=
               detail::rvq_workspace_floats(request.runtime);
  }
};

struct guard_encode_shape_invalid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_encode_shape_valid{}(runtime_ev, ctx);
  }
};

struct guard_decode_shape_valid {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &request = runtime_ev.request;
    return request.codes.size() >= static_cast<size_t>(request.runtime.n_q) &&
           request.latent_out.size() >=
               static_cast<size_t>(request.runtime.dim) &&
           request.workspace.size() >=
               detail::rvq_workspace_floats(request.runtime);
  }
};

struct guard_decode_shape_invalid {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_decode_shape_valid{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type> struct guard_conv_f16 {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.runtime.conv_f16;
  }
};

template <class runtime_event_type> struct guard_conv_f32 {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return !runtime_ev.request.runtime.conv_f16;
  }
};

template <class runtime_event_type> struct guard_stage_ok {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.stage_ok;
  }
};

template <class runtime_event_type> struct guard_stage_failed {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return !runtime_ev.ctx.stage_ok;
  }
};

template <class runtime_event_type> struct guard_has_error_out {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

template <class runtime_event_type> struct guard_no_error_out {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.error_out == nullptr;
  }
};

template <class runtime_event_type> struct guard_has_done_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

template <class runtime_event_type> struct guard_no_done_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_done_callback<runtime_event_type>{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type> struct guard_has_error_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

template <class runtime_event_type> struct guard_no_error_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_error_callback<runtime_event_type>{}(runtime_ev, ctx);
  }
};

} // namespace emel::speech::codec::mimi::quantizer::guard
