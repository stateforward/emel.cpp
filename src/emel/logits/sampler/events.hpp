#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/logits/sampler/errors.hpp"

namespace emel::logits::sampler {

using fn = emel::callback<emel::error::type(int32_t & candidate_ids,
                                            float & candidate_scores,
                                            int32_t & candidate_count,
                                            int32_t & selected_token_out)>;

}  // namespace emel::logits::sampler

namespace emel::logits::sampler::event {

struct configure {
  fn & sampler_fns;
  int32_t sampler_count = 0;
  emel::error::type & error_out;

  configure(fn & sampler_fns_ref,
            int32_t sampler_count_value,
            emel::error::type & error_out_ref) noexcept
    : sampler_fns(sampler_fns_ref),
      sampler_count(sampler_count_value),
      error_out(error_out_ref) {}
};

struct sample_logits {
  const float & logits;
  int32_t vocab_size = 0;
  int32_t & candidate_ids;
  float & candidate_scores;
  int32_t candidate_capacity = 0;
  int32_t & selected_token_out;
  emel::error::type & error_out;

  sample_logits(const float & logits_ref,
                int32_t vocab_size_value,
                int32_t & candidate_ids_ref,
                float & candidate_scores_ref,
                int32_t candidate_capacity_value,
                int32_t & selected_token_out_ref,
                emel::error::type & error_out_ref) noexcept
    : logits(logits_ref),
      vocab_size(vocab_size_value),
      candidate_ids(candidate_ids_ref),
      candidate_scores(candidate_scores_ref),
      candidate_capacity(candidate_capacity_value),
      selected_token_out(selected_token_out_ref),
      error_out(error_out_ref) {}
};

struct sample_preselected {
  int32_t vocab_size = 0;
  int32_t & selected_token_out;
  emel::error::type & error_out;

  sample_preselected(int32_t vocab_size_value,
                     int32_t & selected_token_out_ref,
                     emel::error::type & error_out_ref) noexcept
    : vocab_size(vocab_size_value),
      selected_token_out(selected_token_out_ref),
      error_out(error_out_ref) {}
};

struct sample_temperature_top_k {
  sample_temperature_top_k(std::span<float> logits_ref, const int32_t card_ref,
                           const float temperature_ref, const int32_t top_k_ref,
                           std::span<int32_t> sorted_indices_ref,
                           std::span<float> top_probabilities_ref,
                           std::span<int32_t> top_indices_ref,
                           uint32_t &random_state_ref,
                           int32_t &selected_token_out_ref,
                           float &selected_score_out_ref,
                           emel::error::type &error_out_ref) noexcept
      : logits(logits_ref), card(card_ref), temperature(temperature_ref),
        top_k(top_k_ref), sorted_indices(sorted_indices_ref),
        top_probabilities(top_probabilities_ref), top_indices(top_indices_ref),
        random_state(random_state_ref),
        selected_token_out(selected_token_out_ref),
        selected_score_out(selected_score_out_ref), error_out(error_out_ref) {}

  std::span<float> logits;
  int32_t card;
  float temperature;
  int32_t top_k;
  std::span<int32_t> sorted_indices;
  std::span<float> top_probabilities;
  std::span<int32_t> top_indices;
  uint32_t &random_state;
  int32_t &selected_token_out;
  float &selected_score_out;
  emel::error::type &error_out;
};

struct configure_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct sample_logits_ctx {
  emel::error::type err = emel::error::cast(error::none);
  int32_t candidate_count = 0;
  int32_t sampler_index = 0;
  emel::error::type sampler_call_error = emel::error::cast(error::none);
};

struct sample_preselected_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct sample_temperature_top_k_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct configure_runtime {
  const configure & request;
  configure_ctx & ctx;
};

struct sample_logits_runtime {
  const sample_logits & request;
  sample_logits_ctx & ctx;
};

struct sample_preselected_runtime {
  const sample_preselected & request;
  sample_preselected_ctx & ctx;
};

struct sample_temperature_top_k_runtime {
  const sample_temperature_top_k &request;
  sample_temperature_top_k_ctx &ctx;
};

}  // namespace emel::logits::sampler::event

namespace emel::logits::sampler::events {

struct configure_done {};

struct configure_error {
  emel::error::type err = emel::error::cast(error::none);
};

struct sample_done {
  int32_t token_id = -1;
};

struct sample_error {
  emel::error::type err = emel::error::cast(error::none);
};

}  // namespace emel::logits::sampler::events
