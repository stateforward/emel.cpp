#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/speech/tokenizer/events.hpp"

namespace emel::speech::decoder {

// Component execution contract: variant-neutral decoder dimensions bound from a
// loaded model by a variant binder (e.g.
// decoder::whisper::bind_execution_contract) at the variant boundary. The
// component machines validate the bound values before running; the contract
// itself carries no behavior.
struct execution_contract {
  const emel::model::data *model = nullptr;
  int32_t vocab_size = 0;
  int32_t embedding_length = 0;
  int32_t decoder_block_count = 0;
};

} // namespace emel::speech::decoder

namespace emel::speech::decoder::events {

struct decode_done;
struct decode_error;

} // namespace emel::speech::decoder::events

namespace emel::speech::decoder::event {

struct decode {
  decode(const execution_contract &contract_ref,
         std::span<const float> encoder_state_ref,
         const int32_t encoder_frame_count_ref,
         const emel::speech::tokenizer::asr_decode_policy &policy_ref,
         std::span<float> workspace_ref, std::span<float> logits_ref,
         int32_t &token_out_ref, float &confidence_out_ref,
         uint64_t &digest_out_ref) noexcept
      : contract(contract_ref), encoder_state(encoder_state_ref),
        encoder_frame_count(encoder_frame_count_ref), policy(policy_ref),
        generated_token_storage{}, generated_tokens(generated_token_storage),
        generated_token_count_storage(0),
        generated_token_count_out(generated_token_count_storage),
        workspace(workspace_ref), logits(logits_ref), token_out(token_out_ref),
        confidence_out(confidence_out_ref), digest_out(digest_out_ref) {}

  decode(const execution_contract &contract_ref,
         std::span<const float> encoder_state_ref,
         const int32_t encoder_frame_count_ref,
         const emel::speech::tokenizer::asr_decode_policy &policy_ref,
         std::span<int32_t> generated_tokens_ref,
         int32_t &generated_token_count_out_ref, std::span<float> workspace_ref,
         std::span<float> logits_ref, int32_t &token_out_ref,
         float &confidence_out_ref, uint64_t &digest_out_ref) noexcept
      : contract(contract_ref), encoder_state(encoder_state_ref),
        encoder_frame_count(encoder_frame_count_ref), policy(policy_ref),
        generated_token_storage{}, generated_tokens(generated_tokens_ref),
        generated_token_count_storage(0),
        generated_token_count_out(generated_token_count_out_ref),
        workspace(workspace_ref), logits(logits_ref), token_out(token_out_ref),
        confidence_out(confidence_out_ref), digest_out(digest_out_ref) {}

  const execution_contract &contract;
  std::span<const float> encoder_state = {};
  int32_t encoder_frame_count = 0;
  const emel::speech::tokenizer::asr_decode_policy &policy;
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

} // namespace emel::speech::decoder::event

namespace emel::speech::decoder::events {

struct decode_done {
  const event::decode *request = nullptr;
  int32_t token = 0;
  float confidence = 0.0f;
  uint64_t digest = 0;
};

struct decode_error {
  const event::decode *request = nullptr;
  emel::error::type err = {};
};

} // namespace emel::speech::decoder::events
