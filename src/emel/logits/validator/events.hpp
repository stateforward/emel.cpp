#pragma once

#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/logits/validator/errors.hpp"

namespace emel::logits::validator::event {

struct build {
  const float & logits;
  int32_t vocab_size = 0;
  int32_t & candidate_ids;
  float & candidate_scores;
  int32_t candidate_capacity = 0;
  int32_t & candidate_count_out;
  emel::error::type & error_out;

  build(const float & logits_ref,
        int32_t vocab_size_value,
        int32_t & candidate_ids_ref,
        float & candidate_scores_ref,
        int32_t candidate_capacity_value,
        int32_t & candidate_count_out_ref,
        emel::error::type & error_out_ref) noexcept
    : logits(logits_ref),
      vocab_size(vocab_size_value),
      candidate_ids(candidate_ids_ref),
      candidate_scores(candidate_scores_ref),
      candidate_capacity(candidate_capacity_value),
      candidate_count_out(candidate_count_out_ref),
      error_out(error_out_ref) {}
};

struct build_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct build_runtime {
  const build & request;
  build_ctx & ctx;
};

}  // namespace emel::logits::validator::event

namespace emel::logits::validator::events {

struct build_done {
  int32_t candidate_count = 0;
};

struct build_error {
  emel::error::type err = emel::error::cast(error::none);
};

}  // namespace emel::logits::validator::events
