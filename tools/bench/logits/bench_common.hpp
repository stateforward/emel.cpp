#pragma once
#include "bench_cases.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "emel/error/error.hpp"
#include "emel/logits/sampler/errors.hpp"
#include "emel/logits/sampler/events.hpp"
#include "emel/logits/sampler/sm.hpp"
#include "emel/logits/validator/errors.hpp"
#include "emel/logits/validator/events.hpp"
#include "emel/logits/validator/sm.hpp"

namespace {

constexpr std::array<int32_t, 3> k_vocab_sizes = {32000, 128000, 256000};
constexpr int32_t k_top_k = 64;

struct logits_case_data {
  explicit logits_case_data(const int32_t vocab_size)
      : logits(static_cast<std::size_t>(vocab_size)),
        candidate_ids(static_cast<std::size_t>(vocab_size)),
        candidate_scores(static_cast<std::size_t>(vocab_size)) {
    for (std::size_t i = 0; i < logits.size(); ++i) {
      const float x = static_cast<float>(i);
      const float wave = (std::sin(x * 0.013F) * 6.5F) + (std::cos(x * 0.0017F) * 2.0F);
      const float bucket = static_cast<float>(static_cast<int32_t>(i % 29U) - 14) * 0.03125F;
      logits[i] = wave + bucket;
    }
  }

  std::vector<float> logits;
  std::vector<int32_t> candidate_ids;
  std::vector<float> candidate_scores;
};

std::string make_case_name(const char * family, const char * mode, const int32_t vocab_size) {
  return std::string{"logits/"} + family + "_" + mode + "/vocab_" + std::to_string(vocab_size);
}

emel::error::type top_k_sampler(int32_t &,
                                float &,
                                int32_t & candidate_count,
                                int32_t &) noexcept {
  if (candidate_count <= 0) {
    return emel::error::cast(emel::logits::sampler::error::invalid_request);
  }
  candidate_count = std::min(candidate_count, k_top_k);
  return emel::error::cast(emel::logits::sampler::error::none);
}

emel::error::type argmax_sampler(int32_t & candidate_ids,
                                 float & candidate_scores,
                                 int32_t & candidate_count,
                                 int32_t & selected_token_out) noexcept {
  if (candidate_count <= 0) {
    return emel::error::cast(emel::logits::sampler::error::invalid_request);
  }

  const int32_t * ids = &candidate_ids;
  const float * scores = &candidate_scores;

  int32_t best_index = 0;
  float best_score = scores[0];
  for (int32_t i = 1; i < candidate_count; ++i) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      best_index = i;
    }
  }

  selected_token_out = ids[best_index];
  return emel::error::cast(emel::logits::sampler::error::none);
}

emel::error::type run_validator_raw(const emel::logits::validator::event::build & ev) noexcept {
  if (ev.vocab_size <= 0 || ev.candidate_capacity < ev.vocab_size) {
    ev.candidate_count_out = 0;
    ev.error_out = emel::error::cast(emel::logits::validator::error::invalid_request);
    return ev.error_out;
  }

  const int32_t vocab_size = ev.vocab_size;
  const float * logits = &ev.logits;
  int32_t * candidate_ids = &ev.candidate_ids;
  float * candidate_scores = &ev.candidate_scores;

  for (int32_t i = 0; i < vocab_size; ++i) {
    candidate_ids[i] = i;
    candidate_scores[i] = logits[i];
  }

  float max_score = candidate_scores[0];
  for (int32_t i = 1; i < vocab_size; ++i) {
    if (candidate_scores[i] > max_score) {
      max_score = candidate_scores[i];
    }
  }

  for (int32_t i = 0; i < vocab_size; ++i) {
    candidate_scores[i] -= max_score;
  }

  ev.candidate_count_out = vocab_size;
  ev.error_out = emel::error::cast(emel::logits::validator::error::none);
  return ev.error_out;
}

emel::error::type run_sampler_raw(const emel::logits::sampler::event::sample_logits & ev,
                                  emel::logits::sampler::fn * sampler_fns,
                                  const int32_t sampler_count) noexcept {
  if (ev.vocab_size <= 0 || ev.candidate_capacity < ev.vocab_size ||
      sampler_fns == nullptr || sampler_count <= 0) {
    ev.selected_token_out = -1;
    ev.error_out = emel::error::cast(emel::logits::sampler::error::invalid_request);
    return ev.error_out;
  }

  const int32_t vocab_size = ev.vocab_size;
  const float * logits = &ev.logits;
  int32_t * candidate_ids = &ev.candidate_ids;
  float * candidate_scores = &ev.candidate_scores;
  int32_t candidate_count = vocab_size;

  ev.selected_token_out = -1;
  ev.error_out = emel::error::cast(emel::logits::sampler::error::none);

  for (int32_t i = 0; i < vocab_size; ++i) {
    candidate_ids[i] = i;
    candidate_scores[i] = logits[i];
  }

  for (int32_t i = 0; i < sampler_count; ++i) {
    if (!sampler_fns[i]) {
      ev.error_out = emel::error::cast(emel::logits::sampler::error::invalid_request);
      return ev.error_out;
    }

    const emel::error::type err = sampler_fns[i](
      ev.candidate_ids,
      ev.candidate_scores,
      candidate_count,
      ev.selected_token_out);

    if (err != emel::error::cast(emel::logits::sampler::error::none)) {
      ev.error_out = err;
      return ev.error_out;
    }

    if (candidate_count <= 0 || candidate_count > vocab_size) {
      ev.error_out = emel::error::cast(emel::logits::sampler::error::invalid_request);
      return ev.error_out;
    }
  }

  if (ev.selected_token_out < 0 || ev.selected_token_out >= vocab_size) {
    ev.error_out = emel::error::cast(emel::logits::sampler::error::invalid_request);
    return ev.error_out;
  }

  ev.error_out = emel::error::cast(emel::logits::sampler::error::none);
  return ev.error_out;
}


}  // namespace
