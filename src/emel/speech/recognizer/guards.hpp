#pragma once

#include "emel/speech/recognizer/context.hpp"
#include "emel/speech/recognizer/detail.hpp"
#include "emel/speech/recognizer/events.hpp"

namespace emel::speech::recognizer::guard {

struct guard_valid_initialize {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.tokenizer.model_json.data() != nullptr &&
           !runtime_ev.request.tokenizer.model_json.empty();
  }
};

struct guard_invalid_initialize {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_valid_initialize{}(runtime_ev, ctx);
  }
};

template <class route_policy> struct guard_initialize_tokenizer_supported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return typename route_policy::guard_tokenizer_supported{}(
        runtime_ev.request.tokenizer);
  }
};

template <class route_policy> struct guard_initialize_tokenizer_unsupported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_initialize_tokenizer_supported<route_policy>{}(runtime_ev,
                                                                 ctx);
  }
};

template <class route_policy> struct guard_initialize_model_supported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return typename route_policy::guard_model_supported{}(
        runtime_ev.request.model);
  }
};

template <class route_policy>
struct guard_initialize_model_supported_and_route_storage_ready {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_initialize_model_supported<route_policy>{}(runtime_ev, ctx);
  }
};

template <class route_policy> struct guard_initialize_unsupported_model {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_initialize_model_supported<route_policy>{}(runtime_ev, ctx);
  }
};

struct guard_has_initialize_done_callback {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

struct guard_no_initialize_done_callback {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_initialize_done_callback{}(runtime_ev, ctx);
  }
};

struct guard_has_initialize_error_callback {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

struct guard_no_initialize_error_callback {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_initialize_error_callback{}(runtime_ev, ctx);
  }
};

struct guard_has_initialize_error_out {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

struct guard_no_initialize_error_out {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_initialize_error_out{}(runtime_ev, ctx);
  }
};

struct guard_valid_recognize {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.pcm.data() != nullptr &&
           !runtime_ev.request.pcm.empty() &&
           runtime_ev.request.transcript.data() != nullptr &&
           runtime_ev.request.transcript.size() > 0u;
  }
};

struct guard_invalid_recognize {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_valid_recognize{}(runtime_ev, ctx);
  }
};

template <class route_policy> struct guard_recognizer_route_ready {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &) const noexcept {
    return typename route_policy::guard_recognition_ready{}(runtime_ev.request);
  }
};

template <class route_policy> struct guard_recognizer_route_unsupported {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_recognizer_route_ready<route_policy>{}(runtime_ev, ctx);
  }
};

struct guard_encoder_success {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.encoder_accepted &&
           runtime_ev.ctx.err == detail::to_error(error::none);
  }
};

struct guard_encoder_failure {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_encoder_success{}(runtime_ev, ctx);
  }
};

struct guard_decoder_success {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.decoder_accepted &&
           runtime_ev.ctx.err == detail::to_error(error::none);
  }
};

struct guard_decoder_failure {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_decoder_success{}(runtime_ev, ctx);
  }
};

struct guard_has_recognize_done_callback {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

struct guard_no_recognize_done_callback {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_recognize_done_callback{}(runtime_ev, ctx);
  }
};

struct guard_has_recognize_error_callback {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

struct guard_no_recognize_error_callback {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_recognize_error_callback{}(runtime_ev, ctx);
  }
};

struct guard_has_recognize_error_out {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

struct guard_no_recognize_error_out {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_recognize_error_out{}(runtime_ev, ctx);
  }
};

} // namespace emel::speech::recognizer::guard

namespace emel::speech::recognizer::route {

struct guard_unsupported_tokenizer {
  bool operator()(const event::tokenizer_assets &assets) const noexcept {
    return assets.model_json.data() != nullptr && !assets.model_json.empty() &&
           assets.sha256.data() != nullptr && !assets.sha256.empty();
  }
};

struct guard_unsupported_model {
  bool operator()(const emel::model::data &) const noexcept { return false; }
};

struct guard_unsupported_recognition {
  bool operator()(const event::recognize &) const noexcept { return false; }
};

} // namespace emel::speech::recognizer::route
