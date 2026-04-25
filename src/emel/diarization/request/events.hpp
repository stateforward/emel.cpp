#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/diarization/request/errors.hpp"
#include "emel/error/error.hpp"
#include "emel/model/sortformer/detail.hpp"

namespace emel::diarization::request::events {

struct prepare_done;
struct prepare_error;

}  // namespace emel::diarization::request::events

namespace emel::diarization::request::event {

struct prepare {
  prepare(const emel::model::sortformer::detail::execution_contract & contract_ref,
          std::span<const float> pcm_ref,
          const int32_t sample_rate_ref,
          const int32_t channel_count_ref,
          std::span<float> features_ref,
          int32_t & frame_count_out_ref,
          int32_t & feature_bin_count_out_ref) noexcept
      : contract(contract_ref),
        pcm(pcm_ref),
        sample_rate(sample_rate_ref),
        channel_count(channel_count_ref),
        features(features_ref),
        frame_count_out(frame_count_out_ref),
        feature_bin_count_out(feature_bin_count_out_ref) {}

  const emel::model::sortformer::detail::execution_contract & contract;
  std::span<const float> pcm = {};
  int32_t sample_rate = 0;
  int32_t channel_count = 0;
  std::span<float> features = {};
  int32_t & frame_count_out;
  int32_t & feature_bin_count_out;
  emel::error::type * error_out = nullptr;
  emel::callback<void(const events::prepare_done &)> on_done = {};
  emel::callback<void(const events::prepare_error &)> on_error = {};
};

struct prepare_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct prepare_run {
  const prepare & request;
  prepare_ctx & ctx;
};

}  // namespace emel::diarization::request::event

namespace emel::diarization::request::events {

struct prepare_done {
  const event::prepare * request = nullptr;
  int32_t frame_count = 0;
  int32_t feature_bin_count = 0;
};

struct prepare_error {
  const event::prepare * request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

}  // namespace emel::diarization::request::events
