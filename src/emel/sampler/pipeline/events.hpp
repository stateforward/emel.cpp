#pragma once

#include <cstdint>

namespace emel::sampler::pipeline::event {

using sampler_fn = bool (*)(
    int32_t * candidate_ids,
    float * candidate_scores,
    int32_t candidate_count,
    void * user_data,
    int32_t * error_out);

struct sample {
  const float * logits = nullptr;
  int32_t vocab_size = 0;
  int32_t * candidate_ids = nullptr;
  float * candidate_scores = nullptr;
  int32_t candidate_capacity = 0;
  sampler_fn * sampler_fns = nullptr;
  int32_t sampler_count = 0;
  void * sampler_user_data = nullptr;
  int32_t * selected_token_out = nullptr;
  int32_t * error_out = nullptr;
};

struct prepare_candidates {
  const float * logits = nullptr;
  int32_t vocab_size = 0;
  int32_t * candidate_ids = nullptr;
  float * candidate_scores = nullptr;
  int32_t candidate_capacity = 0;
  int32_t * error_out = nullptr;
};

struct apply_sampling {
  int32_t * candidate_ids = nullptr;
  float * candidate_scores = nullptr;
  sampler_fn * sampler_fns = nullptr;
  void * sampler_user_data = nullptr;
  int32_t * error_out = nullptr;
};

struct select_token {
  int32_t * candidate_ids = nullptr;
  float * candidate_scores = nullptr;
  int32_t * selected_token_out = nullptr;
  int32_t * error_out = nullptr;
};

}  // namespace emel::sampler::pipeline::event

namespace emel::sampler::pipeline::events {

struct prepare_candidates_done {
  const event::sample * request = nullptr;
};
struct prepare_candidates_error {
  int32_t err = 0;
  const event::sample * request = nullptr;
};

struct apply_sampling_done {
  const event::sample * request = nullptr;
};
struct apply_sampling_error {
  int32_t err = 0;
  const event::sample * request = nullptr;
};

struct select_token_done {
  const event::sample * request = nullptr;
};
struct select_token_error {
  int32_t err = 0;
  const event::sample * request = nullptr;
};

struct sampling_done {
  int32_t token_id = -1;
  const event::sample * request = nullptr;
};

struct sampling_error {
  int32_t err = 0;
  const event::sample * request = nullptr;
};

}  // namespace emel::sampler::pipeline::events
