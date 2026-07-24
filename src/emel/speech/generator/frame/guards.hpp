#pragma once

#include <cstddef>

#include "emel/speech/generator/frame/context.hpp"
#include "emel/speech/generator/frame/events.hpp"

namespace emel::speech::generator::frame::guard {

template <class dependencies_type> struct guard_request_valid {
  bool
  operator()(const detail::run_frame &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    const auto count = static_cast<size_t>(ctx.collaborators.codebook_count);
    return ctx.collaborators.codebook_count > 0 &&
           runtime_ev.request.encoded_tokens.size() == count &&
           runtime_ev.request.generated_tokens_out.size() == count &&
           ctx.collaborators.model_codes.size() >= count &&
           ctx.collaborators.predicted_codes.size() >= count &&
           ctx.collaborators.frame_plan_steps > 0 &&
           ctx.collaborators.frame_plan_token_count > 0 &&
           static_cast<size_t>(ctx.collaborators.frame_plan_token_count) <=
               ctx.collaborators.model_codes.size();
  }
};

template <class dependencies_type> struct guard_request_invalid {
  bool
  operator()(const detail::run_frame &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_request_valid<dependencies_type>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type> struct guard_child_succeeded {
  bool operator()(const detail::run_frame &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0;
  }
};

template <class dependencies_type> struct guard_child_failed {
  bool
  operator()(const detail::run_frame &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_child_succeeded<dependencies_type>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type> struct guard_prediction_succeeded {
  bool operator()(const detail::run_frame &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0 &&
           runtime_ev.ctx.graph_err == 0;
  }
};

template <class dependencies_type> struct guard_prediction_failed {
  bool
  operator()(const detail::run_frame &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_prediction_succeeded<dependencies_type>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type> struct guard_frame_produced {
  bool operator()(const detail::run_frame &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0 &&
           runtime_ev.ctx.produced;
  }
};

template <class dependencies_type> struct guard_frame_pending {
  bool operator()(const detail::run_frame &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0 &&
           !runtime_ev.ctx.produced;
  }
};

template <class dependencies_type> struct guard_frame_failed {
  bool operator()(const detail::run_frame &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return !runtime_ev.ctx.child_accepted || runtime_ev.ctx.child_err != 0;
  }
};

template <class dependencies_type> struct guard_done_callback_present {
  bool operator()(const detail::run_frame &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

template <class dependencies_type> struct guard_done_callback_absent {
  bool
  operator()(const detail::run_frame &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_done_callback_present<dependencies_type>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type> struct guard_error_callback_present {
  bool operator()(const detail::run_frame &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

template <class dependencies_type> struct guard_error_callback_absent {
  bool
  operator()(const detail::run_frame &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_error_callback_present<dependencies_type>{}(runtime_ev, ctx);
  }
};

} // namespace emel::speech::generator::frame::guard
