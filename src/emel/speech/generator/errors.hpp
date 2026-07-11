#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::generator {

enum class error : emel::error::type {
  none = 0u,
  uninitialized = (1u << 0),
  already_initialized = (1u << 1),
  invalid_request = (1u << 2),
  internal_error = (1u << 4),
  memory_initialize_failed = (1u << 5),
  encoder_initialize_failed = (1u << 6),
  decoder_initialize_failed = (1u << 7),
  predictor_initialize_failed = (1u << 9),
  conditioning_failed = (1u << 10),
  encode_failed = (1u << 11),
  predict_failed = (1u << 12),
  decode_failed = (1u << 13),
  conditioner_initialize_failed = (1u << 14),
  prefiller_initialize_failed = (1u << 15),
  sampler_initialize_failed = (1u << 16),
  postprocessor_initialize_failed = (1u << 17),
  prefill_failed = (1u << 18),
  sample_failed = (1u << 19),
  postprocess_failed = (1u << 20),
  unsupported_request = (1u << 21),
  tokenizer_initialize_failed = (1u << 22),
  tokenize_failed = (1u << 23),
  detokenize_failed = (1u << 24),
  planning_failed = (1u << 25),
  graph_failed = (1u << 26),
};

} // namespace emel::speech::generator
