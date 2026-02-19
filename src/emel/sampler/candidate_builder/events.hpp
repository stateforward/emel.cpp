#pragma once

#include <cstdint>

namespace emel::sampler::candidate_builder::event {

struct build {
  const float * logits = nullptr;
  int32_t vocab_size = 0;
  int32_t * candidate_ids_out = nullptr;
  float * candidate_scores_out = nullptr;
  int32_t candidate_capacity = 0;
  int32_t * candidate_count_out = nullptr;
  int32_t * error_out = nullptr;
};

}  // namespace emel::sampler::candidate_builder::event

namespace emel::sampler::candidate_builder::events {

struct build_done {
  int32_t candidate_count = 0;
  int32_t * candidate_count_out = nullptr;
  const event::build * request = nullptr;
};

struct build_error {
  int32_t err = 0;
  int32_t * candidate_count_out = nullptr;
  const event::build * request = nullptr;
};

}  // namespace emel::sampler::candidate_builder::events
