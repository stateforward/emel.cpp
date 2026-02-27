#pragma once

#include <cstdint>

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

struct sample_logits_ctx {
  emel::error::type err = emel::error::cast(error::none);
  int32_t candidate_count = 0;
  int32_t sampler_index = 0;
  emel::error::type sampler_call_error = emel::error::cast(error::none);
};

struct sample_preselected_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct sample_logits_runtime {
  const sample_logits & request;
  sample_logits_ctx & ctx;
};

struct sample_preselected_runtime {
  const sample_preselected & request;
  sample_preselected_ctx & ctx;
};

}  // namespace emel::logits::sampler::event

namespace emel::logits::sampler::events {

struct sample_done {
  int32_t token_id = -1;
};

struct sample_error {
  emel::error::type err = emel::error::cast(error::none);
};

}  // namespace emel::logits::sampler::events
