#pragma once

#include <cstddef>

#include "emel/speech/generator/context.hpp"
#include "emel/speech/generator/events.hpp"
#include "emel/speech/generator/frame/errors.hpp"

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

template <class dependencies_type>
struct guard_wavefront_stage_configuration_valid {
  template <class runtime_event_type>
  bool
  operator()(const runtime_event_type &,
             const action::context<dependencies_type> &ctx) const noexcept {
    return ctx.collaborators.stage_mode !=
               action::wavefront_stage_mode::parallel ||
           static_cast<action::wavefront_stage_pool *>(
               ctx.collaborators.stage_pool) != nullptr;
  }
};

template <class dependencies_type> struct guard_wavefront_frame_valid {
  bool
  operator()(const detail::wavefront_frame_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    const auto samples = static_cast<size_t>(ctx.collaborators.frame_samples);
    const auto codebooks =
        static_cast<size_t>(ctx.collaborators.codebook_count);
    return ctx.collaborators.frame_samples > 0 &&
           ctx.collaborators.codebook_count > 0 &&
           guard_wavefront_stage_configuration_valid<dependencies_type>{}(
               runtime_ev, ctx) &&
           samples <= dependencies_type::wavefront_frame_capacity &&
           codebooks <= dependencies_type::wavefront_codebook_capacity &&
           runtime_ev.request.pcm_in.size() == samples &&
           runtime_ev.request.pcm_out.size() >= samples &&
           runtime_ev.request.encoded_tokens_out.size() >= codebooks &&
           runtime_ev.request.generated_tokens_out.size() >= codebooks &&
           runtime_ev.request.input_attribution.sequence !=
               event::k_invalid_wavefront_sequence &&
           runtime_ev.request.input_attribution.sequence ==
               ctx.expected_input.sequence &&
           (ctx.expected_input.sequence == 0u ||
            runtime_ev.request.input_attribution.source ==
                ctx.expected_input.source);
  }
};

template <class dependencies_type> struct guard_wavefront_frame_invalid {
  bool
  operator()(const detail::wavefront_frame_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_wavefront_frame_valid<dependencies_type>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type> struct guard_wavefront_flush_valid {
  bool
  operator()(const detail::wavefront_flush_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    const auto samples = static_cast<size_t>(ctx.collaborators.frame_samples);
    const auto codebooks =
        static_cast<size_t>(ctx.collaborators.codebook_count);
    return ctx.collaborators.frame_samples > 0 &&
           ctx.collaborators.codebook_count > 0 &&
           guard_wavefront_stage_configuration_valid<dependencies_type>{}(
               runtime_ev, ctx) &&
           samples <= dependencies_type::wavefront_frame_capacity &&
           codebooks <= dependencies_type::wavefront_codebook_capacity &&
           runtime_ev.request.pcm_out.size() >= samples &&
           runtime_ev.request.generated_tokens_out.size() >= codebooks;
  }
};

template <class dependencies_type> struct guard_wavefront_flush_invalid {
  bool
  operator()(const detail::wavefront_flush_run &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return !guard_wavefront_flush_valid<dependencies_type>{}(runtime_ev, ctx);
  }
};

template <class dependencies_type, action::wavefront_stage_mode stage_mode>
struct guard_wavefront_stage_mode {
  template <class runtime_event_type>
  bool
  operator()(const runtime_event_type &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return ctx.collaborators.stage_mode == stage_mode &&
           guard_wavefront_stage_configuration_valid<dependencies_type>{}(
               runtime_ev, ctx);
  }
};

template <class dependencies_type, class request_guard_type,
          action::wavefront_stage_mode stage_mode>
struct guard_wavefront_request_stage_mode {
  template <class runtime_event_type>
  bool
  operator()(const runtime_event_type &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return request_guard_type{}(runtime_ev, ctx) &&
           guard_wavefront_stage_mode<dependencies_type, stage_mode>{}(
               runtime_ev, ctx);
  }
};

template <class lane_type, class dependencies_type>
const event::wavefront_attribution &guard_encoded_attribution(
    const action::context<dependencies_type> &ctx) noexcept {
  if constexpr (std::same_as<lane_type, action::lane_zero>) {
    return ctx.encoded_lane0_attribution;
  } else {
    return ctx.encoded_lane1_attribution;
  }
}

template <class lane_type, class dependencies_type>
const event::wavefront_attribution &guard_generated_attribution(
    const action::context<dependencies_type> &ctx) noexcept {
  if constexpr (std::same_as<lane_type, action::lane_zero>) {
    return ctx.generated_lane0_attribution;
  } else {
    return ctx.generated_lane1_attribution;
  }
}

inline bool
guard_attribution_equal(const event::wavefront_attribution lhs,
                        const event::wavefront_attribution rhs) noexcept {
  return lhs.sequence == rhs.sequence && lhs.source == rhs.source;
}

template <class dependencies_type, class encode_lane_type,
          class middle_lane_type, class decode_lane_type, bool encode_active,
          bool middle_active, bool decode_active, bool two_frame_latency>
struct guard_wavefront_phase_succeeded {
  static constexpr bool has_encode = encode_active;
  static constexpr bool has_middle = middle_active;
  static constexpr bool has_decode = decode_active;

  template <class runtime_event_type>
  bool
  operator()(const runtime_event_type &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    bool succeeded = runtime_ev.ctx.all_submitted && runtime_ev.ctx.joined;
    if constexpr (encode_active) {
      succeeded = succeeded && runtime_ev.ctx.encode_accepted &&
                  runtime_ev.ctx.encode_err == 0 &&
                  guard_attribution_equal(
                      guard_encoded_attribution<encode_lane_type>(ctx),
                      runtime_ev.request.input_attribution);
    }
    if constexpr (middle_active) {
      succeeded = succeeded && runtime_ev.ctx.middle_accepted &&
                  runtime_ev.ctx.middle_err == 0 &&
                  guard_attribution_equal(
                      guard_generated_attribution<middle_lane_type>(ctx),
                      guard_encoded_attribution<middle_lane_type>(ctx));
    }
    if constexpr (decode_active) {
      succeeded = succeeded && runtime_ev.ctx.decode_accepted &&
                  runtime_ev.ctx.decode_err == 0 &&
                  guard_attribution_equal(
                      runtime_ev.ctx.decoded_attribution,
                      guard_generated_attribution<decode_lane_type>(ctx)) &&
                  runtime_ev.ctx.decoded_attribution.sequence !=
                      event::k_invalid_wavefront_sequence;
    }
    if constexpr (two_frame_latency) {
      succeeded = succeeded &&
                  runtime_ev.request.input_attribution.sequence >= 2u &&
                  runtime_ev.ctx.decoded_attribution.sequence ==
                      runtime_ev.request.input_attribution.sequence - 2u;
    }
    return succeeded;
  }
};

template <class runtime_event_type, class dependencies_type,
          class phase_guard_type>
struct guard_wavefront_encode_rejected {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    if constexpr (!phase_guard_type::has_encode) {
      return false;
    }
    return runtime_ev.ctx.all_submitted && runtime_ev.ctx.joined &&
           !runtime_ev.ctx.encode_accepted && runtime_ev.ctx.encode_err == 0u;
  }
};

template <class runtime_event_type, class dependencies_type,
          class phase_guard_type>
struct guard_wavefront_encode_failed {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    if constexpr (!phase_guard_type::has_encode) {
      return false;
    }
    return runtime_ev.ctx.all_submitted && runtime_ev.ctx.joined &&
           runtime_ev.ctx.encode_err != 0u;
  }
};

template <class runtime_event_type, class dependencies_type,
          class phase_guard_type>
struct guard_wavefront_middle_non_production {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    if constexpr (!phase_guard_type::has_middle) {
      return false;
    }
    return runtime_ev.ctx.all_submitted && runtime_ev.ctx.joined &&
           !runtime_ev.ctx.middle_accepted &&
           runtime_ev.ctx.middle_err ==
               static_cast<emel::error::type>(frame::error::frame_pending);
  }
};

template <class runtime_event_type, class dependencies_type,
          class phase_guard_type>
struct guard_wavefront_middle_failed {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    if constexpr (!phase_guard_type::has_middle) {
      return false;
    }
    return runtime_ev.ctx.all_submitted && runtime_ev.ctx.joined &&
           runtime_ev.ctx.middle_err != 0u &&
           runtime_ev.ctx.middle_err !=
               static_cast<emel::error::type>(frame::error::frame_pending);
  }
};

template <class runtime_event_type, class dependencies_type,
          class phase_guard_type>
struct guard_wavefront_middle_rejected {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    if constexpr (!phase_guard_type::has_middle) {
      return false;
    }
    return runtime_ev.ctx.all_submitted && runtime_ev.ctx.joined &&
           !runtime_ev.ctx.middle_accepted && runtime_ev.ctx.middle_err == 0u;
  }
};

template <class runtime_event_type, class dependencies_type,
          class phase_guard_type>
struct guard_wavefront_decode_rejected {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    if constexpr (!phase_guard_type::has_decode) {
      return false;
    }
    return runtime_ev.ctx.all_submitted && runtime_ev.ctx.joined &&
           !runtime_ev.ctx.decode_accepted && runtime_ev.ctx.decode_err == 0u;
  }
};

template <class runtime_event_type, class dependencies_type,
          class phase_guard_type>
struct guard_wavefront_decode_failed {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    if constexpr (!phase_guard_type::has_decode) {
      return false;
    }
    return runtime_ev.ctx.all_submitted && runtime_ev.ctx.joined &&
           runtime_ev.ctx.decode_err != 0u;
  }
};

template <class runtime_event_type, class dependencies_type,
          class phase_guard_type>
struct guard_wavefront_attribution_failed {
  bool
  operator()(const runtime_event_type &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    const bool encode_ok =
        !phase_guard_type::has_encode ||
        (runtime_ev.ctx.encode_accepted && runtime_ev.ctx.encode_err == 0u);
    const bool middle_ok =
        !phase_guard_type::has_middle ||
        (runtime_ev.ctx.middle_accepted && runtime_ev.ctx.middle_err == 0u);
    const bool decode_ok =
        !phase_guard_type::has_decode ||
        (runtime_ev.ctx.decode_accepted && runtime_ev.ctx.decode_err == 0u);
    return runtime_ev.ctx.all_submitted && runtime_ev.ctx.joined && encode_ok &&
           middle_ok && decode_ok && !phase_guard_type{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type, class dependencies_type,
          class phase_guard_type>
struct guard_wavefront_phase_success_done_present {
  bool
  operator()(const runtime_event_type &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return phase_guard_type{}(runtime_ev, ctx) &&
           static_cast<bool>(runtime_ev.request.on_done);
  }
};

template <class runtime_event_type, class dependencies_type,
          class phase_guard_type>
struct guard_wavefront_phase_success_done_absent {
  bool
  operator()(const runtime_event_type &runtime_ev,
             const action::context<dependencies_type> &ctx) const noexcept {
    return phase_guard_type{}(runtime_ev, ctx) &&
           !static_cast<bool>(runtime_ev.request.on_done);
  }
};

template <class runtime_event_type, class dependencies_type,
          class phase_guard_type>
struct guard_wavefront_submission_failed {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return !runtime_ev.ctx.all_submitted || !runtime_ev.ctx.joined;
  }
};

template <class dependencies_type>
struct guard_wavefront_reset_submission_failed {
  bool operator()(const detail::event_wavefront_reset_run &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return !runtime_ev.ctx.all_submitted || !runtime_ev.ctx.joined;
  }
};

template <class dependencies_type>
struct guard_wavefront_reset_children_failed {
  bool operator()(const detail::event_wavefront_reset_run &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return runtime_ev.ctx.all_submitted && runtime_ev.ctx.joined &&
           (!runtime_ev.ctx.encode_accepted ||
            runtime_ev.ctx.encode_err != 0u ||
            !runtime_ev.ctx.middle_accepted ||
            runtime_ev.ctx.middle_err != 0u ||
            !runtime_ev.ctx.decode_accepted || runtime_ev.ctx.decode_err != 0u);
  }
};

template <class dependencies_type> struct guard_wavefront_reset_succeeded {
  bool operator()(const detail::event_wavefront_reset_run &runtime_ev,
                  const action::context<dependencies_type> &) const noexcept {
    return runtime_ev.ctx.all_submitted && runtime_ev.ctx.joined &&
           runtime_ev.ctx.encode_accepted && runtime_ev.ctx.encode_err == 0u &&
           runtime_ev.ctx.middle_accepted && runtime_ev.ctx.middle_err == 0u &&
           runtime_ev.ctx.decode_accepted && runtime_ev.ctx.decode_err == 0u;
  }
};

} // namespace emel::speech::generator::guard
