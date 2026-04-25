#pragma once

#include <cstdint>
#include <span>

#include "emel/diarization/sortformer/output/detail.hpp"
#include "emel/diarization/sortformer/pipeline/errors.hpp"
#include "emel/error/error.hpp"
#include "emel/model/sortformer/detail.hpp"

namespace emel::diarization::sortformer::pipeline::event {

struct run {
  run(const emel::model::sortformer::detail::execution_contract & contract_ref,
      std::span<const float> pcm_ref,
      const int32_t sample_rate_ref,
      const int32_t channel_count_ref,
      std::span<float> probabilities_ref,
      std::span<emel::diarization::sortformer::output::detail::segment_record> segments_ref,
      int32_t & frame_count_out_ref,
      int32_t & probability_count_out_ref,
      int32_t & segment_count_out_ref,
      emel::error::type & error_out_ref) noexcept
      : contract(contract_ref),
        pcm(pcm_ref),
        sample_rate(sample_rate_ref),
        channel_count(channel_count_ref),
        probabilities(probabilities_ref),
        segments(segments_ref),
        frame_count_out(frame_count_out_ref),
        probability_count_out(probability_count_out_ref),
        segment_count_out(segment_count_out_ref),
        error_out(error_out_ref) {}

  const emel::model::sortformer::detail::execution_contract & contract;
  std::span<const float> pcm = {};
  int32_t sample_rate = 0;
  int32_t channel_count = 0;
  std::span<float> probabilities = {};
  std::span<emel::diarization::sortformer::output::detail::segment_record> segments = {};
  int32_t & frame_count_out;
  int32_t & probability_count_out;
  int32_t & segment_count_out;
  emel::error::type & error_out;
};

struct run_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct run_flow {
  const run & request;
  run_ctx & ctx;
};

}  // namespace emel::diarization::sortformer::pipeline::event
