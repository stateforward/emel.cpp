#pragma once

#include <concepts>
#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/speech/generator/frame/events.hpp"

namespace emel::speech::generator::frame::action {

template <class dependencies_type>
concept frame_dependencies = requires(dependencies_type deps) {
  typename dependencies_type::tokenize_event;
  typename dependencies_type::predict_event;
  typename dependencies_type::graph_event;
  typename dependencies_type::sample_event;
  typename dependencies_type::detokenize_event;
  deps.planner;
  deps.tokenizer;
  deps.predictor;
  deps.graph;
  deps.sampler;
  deps.prediction_workspace;
  deps.model_codes;
  deps.predicted_codes;
  deps.codebook_count;
  deps.frame_plan_mode;
  deps.frame_plan_steps;
  deps.frame_plan_token_count;
  deps.frame_plan_output_all;
};

template <frame_dependencies dependencies_type> struct context {
  explicit context(const dependencies_type &deps) noexcept
      : collaborators(deps) {}

  const dependencies_type collaborators;
};

} // namespace emel::speech::generator::frame::action

namespace emel::speech::generator::frame::detail {

struct run_ctx {
  emel::error::type err = {};
  emel::error::type child_err = {};
  emel::error::type graph_err = {};
  bool child_accepted = false;
  bool produced = false;
  int32_t tokenizer_err = 0;
  int32_t predicted_text_token = -1;
  int32_t text_token = -1;
  int32_t plan_step_size = 0;
  int32_t plan_output_count = 0;
};

struct run_frame {
  const event::run &request;
  run_ctx &ctx;
};

} // namespace emel::speech::generator::frame::detail
