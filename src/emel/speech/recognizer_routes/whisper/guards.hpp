#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/whisper/any.hpp"
#include "emel/speech/decoder/whisper/any.hpp"
#include "emel/speech/encoder/whisper/any.hpp"
#include "emel/speech/recognizer/events.hpp"
#include "emel/speech/tokenizer/whisper/any.hpp"

namespace emel::speech::recognizer_routes::whisper::guard {

namespace recognizer = emel::speech::recognizer;
namespace whisper_decoder = emel::speech::decoder::whisper;
namespace whisper_encoder = emel::speech::encoder::whisper;
namespace whisper_tokenizer = emel::speech::tokenizer::whisper;

struct guard_tokenizer_supported {
  bool
  operator()(const recognizer::event::tokenizer_assets &assets) const noexcept {
    return assets.model_json.data() != nullptr && !assets.model_json.empty() &&
           assets.sha256 == whisper_tokenizer::tiny_tokenizer_sha256() &&
           whisper_tokenizer::validate_tiny_control_tokens(assets.model_json) &&
           whisper_tokenizer::is_tiny_asr_decode_policy_supported(
               whisper_tokenizer::tiny_asr_decode_policy());
  }
};

struct guard_model_supported {
  bool operator()(const emel::model::data &model) const noexcept {
    emel::model::whisper::execution_contract contract = {};
    return emel::model::whisper::build_execution_contract(model, contract) ==
           emel::error::cast(emel::model::loader::error::none);
  }
};

struct guard_recognition_ready {
  bool operator()(const recognizer::event::recognize &request) const noexcept {
    const auto encoder_contract =
        whisper_encoder::bind_execution_contract(request.model);
    const uint64_t encoder_state_size =
        whisper_encoder::required_encoder_output_floats(request.pcm.size());
    const uint64_t encoder_frames =
        encoder_state_size /
        static_cast<uint64_t>(encoder_contract.embedding_length);
    return guard_model_supported{}(request.model) &&
           guard_tokenizer_supported{}(request.tokenizer) &&
           request.storage.encoder_workspace.data() != nullptr &&
           request.storage.encoder_workspace.size() >=
               static_cast<size_t>(whisper_encoder::required_workspace_floats(
                   request.pcm.size())) &&
           request.storage.encoder_state.data() != nullptr &&
           request.storage.encoder_state.size() >=
               static_cast<size_t>(encoder_state_size) &&
           request.storage.decoder_workspace.data() != nullptr &&
           request.storage.decoder_workspace.size() >=
               static_cast<size_t>(whisper_decoder::required_workspace_floats(
                   encoder_frames)) &&
           request.storage.logits.data() != nullptr &&
           request.storage.logits.size() >=
               static_cast<size_t>(whisper_decoder::vocab_size()) &&
           request.storage.generated_tokens.data() != nullptr &&
           !request.storage.generated_tokens.empty();
  }
};

} // namespace emel::speech::recognizer_routes::whisper::guard
