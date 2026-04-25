#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/diarization/sortformer/executor/errors.hpp"
#include "emel/error/error.hpp"
#include "emel/model/sortformer/detail.hpp"

namespace emel::diarization::sortformer::executor::events {

struct execute_done;
struct execute_error;

}  // namespace emel::diarization::sortformer::executor::events

namespace emel::diarization::sortformer::executor::event {

struct execute {
  execute(const emel::model::sortformer::detail::execution_contract & contract_ref,
          std::span<const float> encoder_frames_ref,
          std::span<float> hidden_out_ref,
          int32_t & frame_count_out_ref,
          int32_t & hidden_dim_out_ref) noexcept
      : contract(contract_ref),
        encoder_frames(encoder_frames_ref),
        hidden_out(hidden_out_ref),
        frame_count_out(frame_count_out_ref),
        hidden_dim_out(hidden_dim_out_ref) {}

  const emel::model::sortformer::detail::execution_contract & contract;
  std::span<const float> encoder_frames = {};
  std::span<float> hidden_out = {};
  int32_t & frame_count_out;
  int32_t & hidden_dim_out;
  emel::error::type * error_out = nullptr;
  emel::callback<void(const events::execute_done &)> on_done = {};
  emel::callback<void(const events::execute_error &)> on_error = {};
};

struct execute_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct execute_run {
  const execute & request;
  execute_ctx & ctx;
};

}  // namespace emel::diarization::sortformer::executor::event

namespace emel::diarization::sortformer::executor::events {

struct execute_done {
  const event::execute * request = nullptr;
  int32_t frame_count = 0;
  int32_t hidden_dim = 0;
};

struct execute_error {
  const event::execute * request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

}  // namespace emel::diarization::sortformer::executor::events
