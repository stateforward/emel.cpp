#pragma once

namespace emel::speech::generator::action {

template <class dependencies_type>
concept generator_dependencies = requires(dependencies_type deps) {
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

template <generator_dependencies dependencies_type> struct context {
  explicit context(const dependencies_type &deps) noexcept
      : collaborators(deps) {}

  const dependencies_type collaborators;
};

} // namespace emel::speech::generator::action
