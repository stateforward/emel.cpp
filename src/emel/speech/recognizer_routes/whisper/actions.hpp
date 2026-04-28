#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "emel/error/error.hpp"
#include "emel/speech/decoder/whisper/any.hpp"
#include "emel/speech/decoder/whisper/sm.hpp"
#include "emel/speech/encoder/whisper/any.hpp"
#include "emel/speech/encoder/whisper/sm.hpp"
#include "emel/speech/recognizer/actions.hpp"
#include "emel/speech/recognizer/errors.hpp"
#include "emel/speech/recognizer/events.hpp"
#include "emel/speech/tokenizer/whisper/any.hpp"

namespace emel::speech::recognizer_routes::whisper::action {

namespace recognizer = emel::speech::recognizer;
namespace whisper_decoder = emel::speech::decoder::whisper;
namespace whisper_encoder = emel::speech::encoder::whisper;
namespace whisper_tokenizer = emel::speech::tokenizer::whisper;

struct effect_encode {
  void operator()(const recognizer::event::recognize_run &runtime_ev,
                  recognizer::action::context &) const noexcept {
    auto contract =
        whisper_encoder::bind_execution_contract(runtime_ev.request.model);
    whisper_encoder::sm encoder{};
    whisper_encoder::event::encode encode_ev{
        contract,
        runtime_ev.request.pcm,
        runtime_ev.request.sample_rate,
        runtime_ev.request.channel_count,
        runtime_ev.request.storage.encoder_workspace,
        runtime_ev.request.storage.encoder_state,
        runtime_ev.ctx.encoder_frame_count,
        runtime_ev.ctx.encoder_width,
        runtime_ev.ctx.encoder_digest};
    runtime_ev.ctx.encoder_accepted = encoder.process_event(encode_ev);
    runtime_ev.ctx.err = emel::error::cast(recognizer::error::none);
  }
};

struct effect_decode {
  void operator()(const recognizer::event::recognize_run &runtime_ev,
                  recognizer::action::context &) const noexcept {
    auto contract =
        whisper_decoder::bind_execution_contract(runtime_ev.request.model);
    const auto encoder_state_size =
        static_cast<size_t>(runtime_ev.ctx.encoder_frame_count) *
        static_cast<size_t>(contract.embedding_length);
    whisper_decoder::sm decoder{};
    whisper_decoder::event::decode decode_ev{
        contract,
        std::span<const float>{runtime_ev.request.storage.encoder_state.data(),
                               encoder_state_size},
        runtime_ev.ctx.encoder_frame_count,
        whisper_tokenizer::tiny_asr_decode_policy(),
        runtime_ev.request.storage.generated_tokens,
        runtime_ev.ctx.generated_token_count,
        runtime_ev.request.storage.decoder_workspace,
        runtime_ev.request.storage.logits,
        runtime_ev.ctx.selected_token,
        runtime_ev.ctx.confidence,
        runtime_ev.ctx.decoder_digest};
    runtime_ev.ctx.decoder_accepted = decoder.process_event(decode_ev);
    runtime_ev.ctx.err = emel::error::cast(recognizer::error::none);
  }
};

struct effect_detokenize {
  void operator()(const recognizer::event::recognize_run &runtime_ev,
                  recognizer::action::context &) const noexcept {
    runtime_ev.ctx.transcript_size =
        static_cast<int32_t>(whisper_tokenizer::decode_token_ids(
            runtime_ev.request.tokenizer.model_json,
            std::span<const int32_t>{
                runtime_ev.request.storage.generated_tokens.data(),
                static_cast<size_t>(runtime_ev.ctx.generated_token_count),
            },
            runtime_ev.request.transcript.data(),
            static_cast<uint64_t>(runtime_ev.request.transcript.size())));
    runtime_ev.ctx.err = emel::error::cast(recognizer::error::none);
  }
};

} // namespace emel::speech::recognizer_routes::whisper::action
