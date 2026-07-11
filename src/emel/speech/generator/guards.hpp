#pragma once

#include <cstddef>

#include "emel/speech/generator/context.hpp"
#include "emel/speech/generator/events.hpp"

namespace emel::speech::generator::guard {

template <class dependencies_type> struct guard_initialize_request_valid {
  bool
  operator()(const event::initialize_run &,
             const action::context<dependencies_type> &ctx) const noexcept {
    const auto count = static_cast<size_t>(ctx.collaborators.codebook_count);
    const auto samples = static_cast<size_t>(ctx.collaborators.frame_samples);
    return ctx.collaborators.codebook_count > 0 &&
           ctx.collaborators.frame_samples > 0 &&
           ctx.collaborators.input_codes.size() == count &&
           ctx.collaborators.output_codes.size() == count &&
           ctx.collaborators.silence_pcm.size() == samples &&
           ctx.collaborators.frame_plan_token_count > 0 &&
           ctx.collaborators.frame_plan_steps > 0 &&
           static_cast<size_t>(ctx.collaborators.frame_plan_token_count) <=
               ctx.collaborators.model_codes.size();
  }
};

template <class dependencies_type> struct guard_initialize_request_invalid {
  bool
  operator()(const event::initialize_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_initialize_request_valid<dependencies_type>{}(runtime_ev,
                                                                ctx);
  }
};

template <class dependencies_type, class runtime_event_type>
struct guard_child_succeeded {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0;
  }
};

template <class dependencies_type, class runtime_event_type>
struct guard_child_failed {
  bool
  operator()(const runtime_event_type &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_child_succeeded<dependencies_type, runtime_event_type>{}(
        runtime_ev, ctx);
  }
};

template <class dependencies_type> struct guard_condition_succeeded_pending {
  bool operator()(const event::condition_run &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0 &&
           runtime_ev.ctx.graph_err == 0 && !runtime_ev.ctx.complete;
  }
};

template <class dependencies_type> struct guard_condition_succeeded_complete {
  bool operator()(const event::condition_run &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0 &&
           runtime_ev.ctx.graph_err == 0 && runtime_ev.ctx.complete;
  }
};

template <class dependencies_type> struct guard_condition_failed {
  bool operator()(const event::condition_run &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return !runtime_ev.ctx.child_accepted || runtime_ev.ctx.child_err != 0 ||
           runtime_ev.ctx.graph_err != 0;
  }
};

template <class dependencies_type> struct guard_generate_request_valid {
  bool operator()(const event::generate_run &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return !runtime_ev.request.text.empty() &&
           !runtime_ev.request.pcm_out.empty();
  }
};

template <class dependencies_type> struct guard_generate_request_invalid {
  bool
  operator()(const event::generate_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_generate_request_valid<dependencies_type>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type> struct guard_stream_request_valid {
  bool
  operator()(const event::stream_frame_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    const auto count = static_cast<size_t>(ctx.collaborators.codebook_count);
    const auto samples = static_cast<size_t>(ctx.collaborators.frame_samples);
    return runtime_ev.request.pcm_in.size() == samples &&
           runtime_ev.request.pcm_out.size() >= samples &&
           runtime_ev.request.encoded_tokens_out.size() >= count &&
           runtime_ev.request.generated_tokens_out.size() >= count;
  }
};

template <class dependencies_type> struct guard_stream_request_invalid {
  bool
  operator()(const event::stream_frame_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_stream_request_valid<dependencies_type>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type> struct guard_flush_request_valid {
  bool
  operator()(const event::flush_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    const auto count = static_cast<size_t>(ctx.collaborators.codebook_count);
    const auto samples = static_cast<size_t>(ctx.collaborators.frame_samples);
    return runtime_ev.request.pcm_out.size() >= samples &&
           runtime_ev.request.encoded_tokens_out.size() >= count &&
           runtime_ev.request.generated_tokens_out.size() >= count;
  }
};

template <class dependencies_type> struct guard_flush_request_invalid {
  bool
  operator()(const event::flush_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_flush_request_valid<dependencies_type>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type, class runtime_event_type>
struct guard_frame_produced {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0 &&
           runtime_ev.ctx.graph_err == 0 && runtime_ev.ctx.produced;
  }
};

template <class dependencies_type, class runtime_event_type>
struct guard_prediction_succeeded {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0 &&
           runtime_ev.ctx.graph_err == 0;
  }
};

template <class dependencies_type, class runtime_event_type>
struct guard_prediction_failed {
  bool
  operator()(const runtime_event_type &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_prediction_succeeded<dependencies_type, runtime_event_type>{}(
        runtime_ev, ctx);
  }
};

template <class dependencies_type, class runtime_event_type>
struct guard_frame_pending {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0 &&
           runtime_ev.ctx.graph_err == 0 && !runtime_ev.ctx.produced;
  }
};

template <class dependencies_type, class runtime_event_type>
struct guard_frame_failed {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return !runtime_ev.ctx.child_accepted || runtime_ev.ctx.child_err != 0 ||
           runtime_ev.ctx.graph_err != 0;
  }
};

template <class dependencies_type, class runtime_event_type>
struct guard_done_callback_present {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

template <class dependencies_type, class runtime_event_type>
struct guard_done_callback_absent {
  bool
  operator()(const runtime_event_type &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_done_callback_present<dependencies_type,
                                        runtime_event_type>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type, class runtime_event_type>
struct guard_error_callback_present {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

template <class dependencies_type, class runtime_event_type>
struct guard_error_callback_absent {
  bool
  operator()(const runtime_event_type &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_error_callback_present<dependencies_type,
                                         runtime_event_type>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type>
struct guard_condition_pending_done_callback_present {
  bool
  operator()(const event::condition_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return guard_condition_succeeded_pending<dependencies_type>{}(runtime_ev,
                                                                  ctx) &&
           guard_done_callback_present<dependencies_type,
                                       event::condition_run>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type>
struct guard_condition_pending_done_callback_absent {
  bool
  operator()(const event::condition_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return guard_condition_succeeded_pending<dependencies_type>{}(runtime_ev,
                                                                  ctx) &&
           guard_done_callback_absent<dependencies_type,
                                      event::condition_run>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type>
struct guard_condition_complete_done_callback_present {
  bool
  operator()(const event::condition_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return guard_condition_succeeded_complete<dependencies_type>{}(runtime_ev,
                                                                   ctx) &&
           guard_done_callback_present<dependencies_type,
                                       event::condition_run>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type>
struct guard_condition_complete_done_callback_absent {
  bool
  operator()(const event::condition_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return guard_condition_succeeded_complete<dependencies_type>{}(runtime_ev,
                                                                   ctx) &&
           guard_done_callback_absent<dependencies_type,
                                      event::condition_run>{}(runtime_ev, ctx);
  }
};

} // namespace emel::speech::generator::guard
