#pragma once

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
struct guard_tokenizer_assets_supported {
  bool operator()(const event::tokenizer_assets &assets,
                  const dependencies &deps) const noexcept {
    return deps.tokenizer_kind !=
               speech::tokenizer::tokenizer_kind::unsupported &&
           assets.model_json.data() != nullptr && !assets.model_json.empty() &&
           !deps.tokenizer_sha256.empty() &&
           assets.sha256 == deps.tokenizer_sha256;
  }
};

struct guard_model_contracts_supported {
  bool operator()(const emel::model::data &model,
                  const dependencies &deps) const noexcept {
    return deps.encoder_kind != speech::encoder::encoder_kind::unsupported &&
           deps.decoder_kind != speech::decoder::decoder_kind::unsupported &&
           deps.encoder_contract.model == &model &&
           deps.encoder_contract.embedding_length > 0 &&
           deps.decoder_contract.model == &model &&
           deps.decoder_contract.vocab_size > 0 &&
           deps.decoder_contract.embedding_length > 0;
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
