#pragma once

#include "emel/speech/codec/mimi/context.hpp"
#include "emel/speech/codec/mimi/detail.hpp"
#include "emel/speech/codec/mimi/events.hpp"

namespace emel::speech::codec::mimi::guard {

// Bind validation: the contract guard runs the pure dry-run instantiation of
// the bind walk; the capacity guard is pure size arithmetic against the
// required_* sizing contract plus the facade latent staging cap. Both are
// one-time initialization predicates, never per-frame.
struct guard_bind_contract_valid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return detail::validate_codec_contract(runtime_ev.request.model);
  }
};

struct guard_bind_contract_invalid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_bind_contract_valid{}(runtime_ev, ctx);
  }
};

struct guard_arena_capacity_valid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &request = runtime_ev.request;
    const auto &model = request.model;
    return model.mimi.dim <= action::k_max_latent_floats &&
           request.prepared.size() >= detail::required_prepared_floats(model) &&
           request.state_arena.size() >= detail::required_state_floats(model) &&
           request.workspace.size() >= detail::required_workspace_floats(model) &&
           request.frame.size() >= detail::required_frame_floats(model);
  }
};

struct guard_arena_capacity_invalid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_arena_capacity_valid{}(runtime_ev, ctx);
  }
};

// Per-frame request validation against the bound runtime (O(1) reads plus a
// bounded n_q code scan); the compute actions selected behind these guards
// are non-failing by contract.
struct guard_encode_request_valid {
  bool operator()(const event::encode_frame_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &request = runtime_ev.request;
    return request.pcm.size() ==
               static_cast<size_t>(ctx.runtime.frame_samples) &&
           request.codes_out.size() >= static_cast<size_t>(ctx.runtime.n_q);
  }
};

struct guard_encode_request_invalid {
  bool operator()(const event::encode_frame_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_encode_request_valid{}(runtime_ev, ctx);
  }
};

struct guard_decode_request_valid {
  bool operator()(const event::decode_frame_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &request = runtime_ev.request;
    return request.codes.size() >= static_cast<size_t>(ctx.runtime.n_q) &&
           request.pcm_out.size() >=
               static_cast<size_t>(ctx.runtime.frame_samples);
  }
};

struct guard_decode_request_invalid {
  bool operator()(const event::decode_frame_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_decode_request_valid{}(runtime_ev, ctx);
  }
};

struct guard_decode_codes_valid {
  bool operator()(const event::decode_frame_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return detail::validate_codes_in_range(ctx.runtime,
                                           runtime_ev.request.codes);
  }
};

struct guard_decode_codes_invalid {
  bool operator()(const event::decode_frame_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_decode_codes_valid{}(runtime_ev, ctx);
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

} // namespace emel::speech::codec::mimi::guard
