#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/speech/decoder/whisper/errors.hpp"
#include "emel/speech/tokenizer/whisper/any.hpp"

namespace emel::speech::decoder::whisper::detail {

struct execution_contract;

} // namespace emel::speech::decoder::whisper::detail

namespace emel::speech::decoder::whisper::events {

struct decode_done;
struct decode_error;

} // namespace emel::speech::decoder::whisper::events

namespace emel::speech::decoder::whisper::event {

struct decode {
  decode(const detail::execution_contract &contract_ref,
         std::span<const float> encoder_state_ref,
         const int32_t encoder_frame_count_ref,
         const emel::speech::tokenizer::whisper::asr_decode_policy
             &policy_ref,
         std::span<float> workspace_ref, std::span<float> logits_ref,
         int32_t &token_out_ref, float &confidence_out_ref,
         uint64_t &digest_out_ref) noexcept
      : contract(contract_ref), encoder_state(encoder_state_ref),
        encoder_frame_count(encoder_frame_count_ref),
        policy(policy_ref),
        generated_token_storage{}, generated_tokens(generated_token_storage),
        generated_token_count_storage(0),
        generated_token_count_out(generated_token_count_storage),
        workspace(workspace_ref), logits(logits_ref),
        token_out(token_out_ref), confidence_out(confidence_out_ref),
        digest_out(digest_out_ref) {}

  decode(const detail::execution_contract &contract_ref,
         std::span<const float> encoder_state_ref,
         const int32_t encoder_frame_count_ref,
         const emel::speech::tokenizer::whisper::asr_decode_policy
             &policy_ref,
         std::span<int32_t> generated_tokens_ref,
         int32_t &generated_token_count_out_ref, std::span<float> workspace_ref,
         std::span<float> logits_ref, int32_t &token_out_ref,
         float &confidence_out_ref, uint64_t &digest_out_ref) noexcept
      : contract(contract_ref), encoder_state(encoder_state_ref),
        encoder_frame_count(encoder_frame_count_ref),
        policy(policy_ref), generated_token_storage{},
        generated_tokens(generated_tokens_ref),
        generated_token_count_storage(0),
        generated_token_count_out(generated_token_count_out_ref),
        workspace(workspace_ref), logits(logits_ref),
        token_out(token_out_ref), confidence_out(confidence_out_ref),
        digest_out(digest_out_ref) {}

  const detail::execution_contract &contract;
  std::span<const float> encoder_state = {};
  int32_t encoder_frame_count = 0;
  const emel::speech::tokenizer::whisper::asr_decode_policy &policy;
  std::array<int32_t, 1> generated_token_storage = {};
  std::span<int32_t> generated_tokens = {};
  int32_t generated_token_count_storage = 0;
  int32_t &generated_token_count_out;
  std::span<float> workspace = {};
  std::span<float> logits = {};
  int32_t &token_out;
  float &confidence_out;
  uint64_t &digest_out;
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::decode_done &)> on_done = {};
  emel::callback<void(const events::decode_error &)> on_error = {};
};

struct decode_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct decode_run {
  const decode &request;
  decode_ctx &ctx;
};

} // namespace emel::speech::decoder::whisper::event

namespace emel::speech::decoder::whisper::events {

struct decode_done {
  const event::decode *request = nullptr;
  int32_t token = 0;
  float confidence = 0.0f;
  uint64_t digest = 0;
};

struct decode_error {
  const event::decode *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::speech::decoder::whisper::events
