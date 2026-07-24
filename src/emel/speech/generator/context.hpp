#pragma once

#include <array>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <span>

#include "emel/batch/planner/events.hpp"
#include "emel/sm.hpp"
#include "emel/speech/generator/events.hpp"

namespace emel::speech::generator::action {

namespace mode {

struct duplex {};
struct synthesis {};
struct wavefront {};

} // namespace mode

template <class dependencies_type>
concept duplex_dependencies = requires(dependencies_type deps) {
  requires std::same_as<typename dependencies_type::generator_mode,
                        mode::duplex>;
  typename dependencies_type::voice_condition_event;
  typename dependencies_type::prompt_begin_event;
  typename dependencies_type::prompt_condition_event;
  typename dependencies_type::encode_event;
  typename dependencies_type::tokenizer_initialize_event;
  typename dependencies_type::tokenize_event;
  typename dependencies_type::predict_event;
  typename dependencies_type::graph_event;
  typename dependencies_type::sample_event;
  typename dependencies_type::detokenize_event;
  typename dependencies_type::capture_tokenizer_state_event;
  typename dependencies_type::restore_tokenizer_state_event;
  typename dependencies_type::decode_event;
  deps.temporal_positions;
  deps.secondary_positions;
  deps.planner;
  deps.encoder;
  deps.tokenizer;
  deps.decoder;
  deps.predictor;
  deps.graph;
  deps.sampler;
  deps.prediction_workspace;
  deps.encoder_initialize;
  deps.decoder_initialize;
  deps.predictor_initialize;
  deps.conditioning_initialize;
  deps.prompt_begin;
  deps.silence_pcm;
  deps.input_codes;
  deps.tokenize_input_codes;
  deps.model_codes;
  deps.predicted_codes;
  deps.output_codes;
  deps.tokenizer_cache_snapshot;
  deps.frame_samples;
  deps.codebook_count;
  deps.frame_plan_mode;
  deps.frame_plan_steps;
  deps.frame_plan_token_count;
  deps.frame_plan_output_all;
};

template <class dependencies_type>
concept synthesis_dependencies = requires(dependencies_type deps) {
  requires std::same_as<typename dependencies_type::generator_mode,
                        mode::synthesis>;
  typename dependencies_type::condition_event;
  typename dependencies_type::prefill_event;
  typename dependencies_type::predict_event;
  typename dependencies_type::sample_event;
  typename dependencies_type::decode_event;
  typename dependencies_type::postprocess_event;
  deps.conditioner;
  deps.prefiller;
  deps.predictor;
  deps.sampler;
  deps.decoder;
  deps.postprocessor;
  deps.conditioner_initialize;
  deps.prefiller_initialize;
  deps.predictor_initialize;
  deps.sampler_initialize;
  deps.decoder_initialize;
  deps.postprocessor_initialize;
};

using wavefront_stage_pool =
    emel::policy::fork_join_lane_pool<2u, 256u, 1048576u>;

enum class wavefront_stage_mode : uint8_t { serial, parallel };

struct wavefront_diagnostics {
  std::atomic<uint64_t> submissions = 0u;
  std::atomic<uint64_t> joins = 0u;
  std::atomic<uint64_t> worker_entries = 0u;
  std::atomic<uint64_t> worker_exits = 0u;
};

template <class dependencies_type>
concept wavefront_dependencies = requires(dependencies_type deps) {
  requires std::same_as<typename dependencies_type::generator_mode,
                        mode::wavefront>;
  requires dependencies_type::wavefront_frame_capacity > 0u;
  requires dependencies_type::wavefront_codebook_capacity > 0u;
  typename dependencies_type::wavefront_encode_event;
  typename dependencies_type::wavefront_encode_reset_event;
  typename dependencies_type::wavefront_middle_reset_event;
  typename dependencies_type::wavefront_decode_event;
  typename dependencies_type::wavefront_decode_reset_event;
  deps.wavefront_encoder;
  deps.wavefront_middle;
  deps.wavefront_decoder;
  { deps.stage_pool } -> std::convertible_to<wavefront_stage_pool *>;
  { deps.stage_mode } -> std::convertible_to<wavefront_stage_mode>;
  deps.frame_samples;
  deps.codebook_count;
};

template <class dependencies_type>
concept generator_dependencies = duplex_dependencies<dependencies_type> ||
                                 synthesis_dependencies<dependencies_type> ||
                                 wavefront_dependencies<dependencies_type>;

template <generator_dependencies dependencies_type,
          class mode_type = typename dependencies_type::generator_mode>
struct context;

template <duplex_dependencies dependencies_type>
struct context<dependencies_type, mode::duplex> {
  explicit context(const dependencies_type &deps) noexcept
      : collaborators(deps) {}

  const dependencies_type collaborators;
};

template <synthesis_dependencies dependencies_type>
struct context<dependencies_type, mode::synthesis> {
  explicit context(const dependencies_type &deps) noexcept
      : collaborators(deps) {}

  const dependencies_type collaborators;
};

template <wavefront_dependencies dependencies_type>
struct context<dependencies_type, mode::wavefront> {
  explicit context(const dependencies_type &deps) noexcept
      : collaborators(deps) {}

  std::span<int32_t> encoded_lane0() noexcept {
    return {encoded_lane0_storage.data(),
            static_cast<size_t>(collaborators.codebook_count)};
  }

  std::span<int32_t> encoded_lane1() noexcept {
    return {encoded_lane1_storage.data(),
            static_cast<size_t>(collaborators.codebook_count)};
  }

  std::span<int32_t> generated_lane0() noexcept {
    return {generated_lane0_storage.data(),
            static_cast<size_t>(collaborators.codebook_count)};
  }

  std::span<int32_t> generated_lane1() noexcept {
    return {generated_lane1_storage.data(),
            static_cast<size_t>(collaborators.codebook_count)};
  }

  std::span<float> decoded_pcm() noexcept {
    return {decoded_pcm_storage.data(),
            static_cast<size_t>(collaborators.frame_samples)};
  }

  const dependencies_type collaborators;
  std::array<int32_t, dependencies_type::wavefront_codebook_capacity>
      encoded_lane0_storage{};
  std::array<int32_t, dependencies_type::wavefront_codebook_capacity>
      encoded_lane1_storage{};
  std::array<int32_t, dependencies_type::wavefront_codebook_capacity>
      generated_lane0_storage{};
  std::array<int32_t, dependencies_type::wavefront_codebook_capacity>
      generated_lane1_storage{};
  std::array<float, dependencies_type::wavefront_frame_capacity>
      decoded_pcm_storage{};
  event::wavefront_attribution encoded_lane0_attribution = {};
  event::wavefront_attribution encoded_lane1_attribution = {};
  event::wavefront_attribution generated_lane0_attribution = {};
  event::wavefront_attribution generated_lane1_attribution = {};
  event::wavefront_attribution expected_input{.sequence = 0u, .source = 0u};
  int32_t generated_lane0_text_token = -1;
  int32_t generated_lane1_text_token = -1;
};

} // namespace emel::speech::generator::action
