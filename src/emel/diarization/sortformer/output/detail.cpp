#include "emel/diarization/sortformer/output/detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "emel/diarization/sortformer/detail.hpp"

namespace emel::diarization::sortformer::output::detail {

namespace {

bool speaker_head_bound(
    const emel::diarization::sortformer::modules::detail::contract & modules_contract) noexcept {
  return modules_contract.frame_hidden_weight.tensor != nullptr &&
      modules_contract.frame_hidden_bias.tensor != nullptr &&
      modules_contract.speaker_hidden_to_speaker_weight.tensor != nullptr &&
      modules_contract.speaker_hidden_to_speaker_bias.tensor != nullptr;
}

float sigmoid(const float value) noexcept {
  const float clamped = std::clamp(value, -80.0f, 80.0f);
  return 1.0f / (1.0f + std::exp(-clamped));
}

float relu(const float value) noexcept { return std::max(value, 0.0f); }

bool append_segment(std::span<segment_record> segments_out,
                    int32_t & segment_count_out,
                    const int32_t speaker,
                    const int32_t start_frame,
                    const int32_t end_frame,
                    const float max_probability) noexcept {
  if (segment_count_out >= static_cast<int32_t>(segments_out.size())) {
    return false;
  }

  auto & segment = segments_out[static_cast<size_t>(segment_count_out)];
  segment.speaker = speaker;
  segment.start_frame = start_frame;
  segment.end_frame = end_frame;
  segment.start_seconds = static_cast<float>(start_frame) * k_frame_shift_seconds;
  segment.end_seconds = static_cast<float>(end_frame) * k_frame_shift_seconds;
  segment.max_probability = max_probability;
  ++segment_count_out;
  return true;
}

}  // namespace

bool compute_speaker_probabilities(
    std::span<const float> hidden_frames,
    const emel::diarization::sortformer::modules::detail::contract & modules_contract,
    std::span<float> probabilities_out) noexcept {
  if (hidden_frames.data() == nullptr ||
      hidden_frames.size() != static_cast<size_t>(k_required_hidden_value_count) ||
      probabilities_out.data() == nullptr ||
      probabilities_out.size() != static_cast<size_t>(k_required_probability_value_count) ||
      !speaker_head_bound(modules_contract)) {
    return false;
  }

  const auto weights = tensor_data<k_speaker_count * k_hidden_dim>(
      *modules_contract.speaker_hidden_to_speaker_weight.tensor);
  const auto bias =
      tensor_data<k_speaker_count>(*modules_contract.speaker_hidden_to_speaker_bias.tensor);
  const auto frame_hidden_weights = tensor_data<k_hidden_dim * k_hidden_dim>(
      *modules_contract.frame_hidden_weight.tensor);
  const auto frame_hidden_bias =
      tensor_data<k_hidden_dim>(*modules_contract.frame_hidden_bias.tensor);
  std::array<float, k_hidden_dim> intermediate = {};
  std::array<float, k_hidden_dim> frame_hidden = {};
  std::array<float, k_speaker_count> logits = {};

  for (int32_t frame = 0; frame < k_frame_count; ++frame) {
    const size_t hidden_offset = static_cast<size_t>(frame) * static_cast<size_t>(k_hidden_dim);
    const auto hidden = std::span<const float, k_hidden_dim>{
        hidden_frames.data() + hidden_offset, k_hidden_dim};

    for (size_t index = 0u; index < intermediate.size(); ++index) {
      intermediate[index] = relu(hidden[index]);
    }

    if (!emel::diarization::sortformer::detail::compute_dense(
            intermediate, frame_hidden_weights, frame_hidden_bias, frame_hidden)) {
      return false;
    }

    for (size_t index = 0u; index < frame_hidden.size(); ++index) {
      intermediate[index] = relu(frame_hidden[index]);
    }

    if (!emel::diarization::sortformer::detail::compute_dense(intermediate, weights, bias, logits)) {
      return false;
    }

    const size_t probability_offset =
        static_cast<size_t>(frame) * static_cast<size_t>(k_speaker_count);
    for (int32_t speaker = 0; speaker < k_speaker_count; ++speaker) {
      probabilities_out[probability_offset + static_cast<size_t>(speaker)] =
          sigmoid(logits[static_cast<size_t>(speaker)]);
    }
  }

  return true;
}

bool decode_segments(std::span<const float> probabilities,
                     const float threshold,
                     std::span<segment_record> segments_out,
                     int32_t & segment_count_out) noexcept {
  segment_count_out = 0;
  if (probabilities.data() == nullptr ||
      probabilities.size() != static_cast<size_t>(k_required_probability_value_count) ||
      !std::isfinite(threshold) ||
      threshold < 0.0f ||
      threshold > 1.0f) {
    return false;
  }

  for (int32_t speaker = 0; speaker < k_speaker_count; ++speaker) {
    int32_t start_frame = -1;
    float max_probability = 0.0f;

    for (int32_t frame = 0; frame < k_frame_count; ++frame) {
      const size_t probability_offset =
          (static_cast<size_t>(frame) * static_cast<size_t>(k_speaker_count)) +
          static_cast<size_t>(speaker);
      const float probability = probabilities[probability_offset];
      const bool active = probability >= threshold;

      if (active && start_frame < 0) {
        start_frame = frame;
        max_probability = probability;
      } else if (active) {
        max_probability = std::max(max_probability, probability);
      } else if (start_frame >= 0) {
        if (!append_segment(segments_out,
                            segment_count_out,
                            speaker,
                            start_frame,
                            frame,
                            max_probability)) {
          return false;
        }
        start_frame = -1;
        max_probability = 0.0f;
      }
    }

    if (start_frame >= 0 &&
        !append_segment(segments_out,
                        segment_count_out,
                        speaker,
                        start_frame,
                        k_frame_count,
                        max_probability)) {
      return false;
    }
  }

  return true;
}

std::string_view speaker_label(const int32_t speaker) noexcept {
  switch (speaker) {
    case 0:
      return "speaker_0";
    case 1:
      return "speaker_1";
    case 2:
      return "speaker_2";
    case 3:
      return "speaker_3";
    default:
      return {};
  }
}

}  // namespace emel::diarization::sortformer::output::detail
