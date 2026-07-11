#pragma once

#include <cstddef>

#include "emel/speech/transcriber/context.hpp"
#include "emel/speech/transcriber/detail.hpp"
#include "emel/speech/transcriber/events.hpp"

namespace emel::speech::transcriber::guard {

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

// Injected-dependency support predicates: the engine accepts a request only
// when the caller injected supported component kinds, the component contracts
// were bound against the same model the event carries, and the tokenizer assets
// match the pinned checksum in the dependencies. Content-level validation
// (model tensors, workspace capacity, tokenizer JSON control tokens) is owned
// by the component machines and re-checked on every component dispatch.
//
// Component kinds are matched against the facade's supported-variant set, not
// merely against the `unsupported` sentinel: sm_any clamps any out-of-range
// enum value to the first variant, so an owner that accepted every
// non-`unsupported` value would silently run the default variant for a
// deserialized/cast kind it never meant to support. any::is_supported keeps
// that variant knowledge in the component facade, so the engine stays
// model-family-neutral.
struct guard_tokenizer_assets_supported {
  bool operator()(const event::tokenizer_assets &assets,
                  const dependencies &deps) const noexcept {
    return speech::tokenizer::any::is_supported(deps.tokenizer_kind) &&
           assets.model_json.data() != nullptr && !assets.model_json.empty() &&
           !deps.tokenizer_sha256.empty() &&
           assets.sha256 == deps.tokenizer_sha256;
  }
};

// The encoder output feeds the decoder cross-attention directly, and the
// decode phase slices the shared encoder-state buffer as frame_count x
// decoder embedding_length, so injected contracts whose embedding lengths
// disagree would make the pipeline mis-slice the buffer; they are rejected
// here as an unsupported dependency combination.
struct guard_model_contracts_supported {
  bool operator()(const emel::model::data &model,
                  const dependencies &deps) const noexcept {
    return speech::encoder::any::is_supported(deps.encoder_kind) &&
           speech::decoder::any::is_supported(deps.decoder_kind) &&
           deps.encoder_contract.model == &model &&
           deps.encoder_contract.embedding_length > 0 &&
           deps.decoder_contract.model == &model &&
           deps.decoder_contract.vocab_size > 0 &&
           deps.decoder_contract.embedding_length > 0 &&
           deps.encoder_contract.embedding_length ==
               deps.decoder_contract.embedding_length;
  }
};

struct guard_initialize_tokenizer_supported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_tokenizer_assets_supported{}(runtime_ev.request.tokenizer,
                                              ctx.deps);
  }
};

struct guard_initialize_tokenizer_unsupported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_initialize_tokenizer_supported{}(runtime_ev, ctx);
  }
};

// Outcome of the component-driven asset validation dispatched by
// effect_validate_tokenizer_assets: the injected tokenizer variant validated
// the tokenizer JSON and the bound decode policy, so initialization fails fast
// with tokenizer_invalid instead of deferring the failure to the first
// recognize after encode/decode work already ran.
struct guard_tokenizer_validation_accepted {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.tokenizer_validation_accepted;
  }
};

struct guard_tokenizer_validation_rejected {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_tokenizer_validation_accepted{}(runtime_ev, ctx);
  }
};

struct guard_initialize_model_supported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_model_contracts_supported{}(runtime_ev.request.model,
                                             ctx.deps);
  }
};

struct guard_initialize_unsupported_model {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_initialize_model_supported{}(runtime_ev, ctx);
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

struct guard_transcriber_ready {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_model_contracts_supported{}(runtime_ev.request.model,
                                             ctx.deps) &&
           guard_tokenizer_assets_supported{}(runtime_ev.request.tokenizer,
                                              ctx.deps);
  }
};

struct guard_transcriber_unsupported {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_transcriber_ready{}(runtime_ev, ctx);
  }
};

// The decode phase reads frame_count x decoder embedding_length floats from
// the caller-owned encoder-state buffer, so the encode outcome only counts as
// a success when that slice provably fits the buffer the caller provided;
// otherwise the pipeline takes the backend-error path instead of letting the
// decoder read past the span.
struct guard_encoder_success {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return runtime_ev.ctx.encoder_accepted &&
           runtime_ev.ctx.err == detail::to_error(error::none) &&
           runtime_ev.ctx.encoder_frame_count >= 0 &&
           static_cast<size_t>(runtime_ev.ctx.encoder_frame_count) *
                   static_cast<size_t>(
                       ctx.deps.decoder_contract.embedding_length) <=
               runtime_ev.request.storage.encoder_state.size();
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

struct guard_detokenize_success {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.detokenize_accepted &&
           runtime_ev.ctx.err == detail::to_error(error::none);
  }
};

struct guard_detokenize_failure {
  bool operator()(const event::recognize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_detokenize_success{}(runtime_ev, ctx);
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

} // namespace emel::speech::transcriber::guard
