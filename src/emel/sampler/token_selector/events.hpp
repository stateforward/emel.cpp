#pragma once

#include <cstdint>

namespace emel::sampler::token_selector::event {

enum class selection_policy : uint8_t {
  argmax = 0,
  categorical,
};

struct select_token {
  const int32_t * candidate_ids = nullptr;
  const float * candidate_scores = nullptr;
  int32_t candidate_count = 0;
  selection_policy policy = selection_policy::argmax;
  float random_01 = 0.0f;
  int32_t * selected_token_out = nullptr;
  int32_t * error_out = nullptr;
};

struct validate {
  const int32_t * candidate_ids = nullptr;
  const float * candidate_scores = nullptr;
  int32_t candidate_count = 0;
  selection_policy policy = selection_policy::argmax;
  float random_01 = 0.0f;
  int32_t * error_out = nullptr;
};

struct select {
  const int32_t * candidate_ids = nullptr;
  const float * candidate_scores = nullptr;
  int32_t candidate_count = 0;
  selection_policy policy = selection_policy::argmax;
  float random_01 = 0.0f;
  int32_t * selected_token_out = nullptr;
  int32_t * error_out = nullptr;
};

}  // namespace emel::sampler::token_selector::event

namespace emel::sampler::token_selector::events {

struct validate_done {
  const event::select_token * request = nullptr;
};
struct validate_error {
  int32_t err = 0;
  const event::select_token * request = nullptr;
};

struct select_done {
  const event::select_token * request = nullptr;
};
struct select_error {
  int32_t err = 0;
  const event::select_token * request = nullptr;
};

struct token_selection_done {
  int32_t token_id = -1;
  const event::select_token * request = nullptr;
};

struct token_selection_error {
  int32_t err = 0;
  const event::select_token * request = nullptr;
};

}  // namespace emel::sampler::token_selector::events
