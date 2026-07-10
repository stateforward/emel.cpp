#pragma once

#include <concepts>

namespace emel::speech::generator::action {

namespace mode {

struct duplex {};
struct synthesis {};

} // namespace mode

template <class dependencies_type>
concept duplex_dependencies = requires(dependencies_type deps) {
  requires std::same_as<typename dependencies_type::generator_mode,
                        mode::duplex>;
  typename dependencies_type::voice_condition_event;
  typename dependencies_type::prompt_begin_event;
  typename dependencies_type::prompt_condition_event;
  typename dependencies_type::encode_event;
  typename dependencies_type::predict_event;
  typename dependencies_type::decode_event;
  deps.temporal_positions;
  deps.secondary_positions;
  deps.encoder;
  deps.decoder;
  deps.runtime;
  deps.predictor;
  deps.encoder_initialize;
  deps.decoder_initialize;
  deps.runtime_initialize;
  deps.predictor_initialize;
  deps.conditioning_initialize;
  deps.silence_pcm;
  deps.input_codes;
  deps.output_codes;
  deps.frame_samples;
  deps.codebook_count;
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

template <class dependencies_type>
concept generator_dependencies = duplex_dependencies<dependencies_type> ||
                                 synthesis_dependencies<dependencies_type>;

template <generator_dependencies dependencies_type> struct context {
  explicit context(const dependencies_type &deps) noexcept
      : collaborators(deps) {}

  const dependencies_type collaborators;
};

} // namespace emel::speech::generator::action
