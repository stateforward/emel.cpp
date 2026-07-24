#pragma once

#include <algorithm>
#include <span>

#include <stateforward/sml.hpp>

#include "emel/batch/planner/events.hpp"
#include "emel/error/error.hpp"
#include "emel/speech/generator/frame/context.hpp"
#include "emel/speech/generator/frame/errors.hpp"
#include "emel/speech/generator/frame/events.hpp"

namespace emel::speech::generator::frame::action {

inline emel::error::type error_code(const error value) noexcept {
  return emel::error::cast(value);
}

struct frame_plan_capture {
  detail::run_ctx &ctx;

  void on_done(const emel::batch::planner::events::plan_done &ev) noexcept {
    ctx.child_err = emel::error::cast(emel::batch::planner::error::none);
    ctx.plan_step_size = ev.step_sizes[0];
    ctx.plan_output_count = ev.total_outputs;
  }

  void on_error(const emel::batch::planner::events::plan_error &ev) noexcept {
    ctx.child_err = ev.err;
  }
};

template <class dependencies_type> struct effect_tokenize {
  void operator()(const detail::run_frame &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.tokenizer_err = 0;
    typename dependencies_type::tokenize_event request{
        runtime_ev.request.encoded_tokens, ctx.collaborators.model_codes,
        runtime_ev.ctx.tokenizer_err};
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.tokenizer.process_event(request);
    runtime_ev.ctx.child_err =
        static_cast<emel::error::type>(runtime_ev.ctx.tokenizer_err);
  }
};

template <class dependencies_type> struct effect_plan {
  void operator()(const detail::run_frame &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err =
        emel::error::cast(emel::batch::planner::error::none);
    runtime_ev.ctx.plan_step_size = 0;
    runtime_ev.ctx.plan_output_count = 0;
    frame_plan_capture capture{runtime_ev.ctx};
    const auto on_done =
        emel::callback<void(const emel::batch::planner::events::plan_done &)>::
            template from<frame_plan_capture, &frame_plan_capture::on_done>(
                &capture);
    const auto on_error =
        emel::callback<void(const emel::batch::planner::events::plan_error &)>::
            template from<frame_plan_capture, &frame_plan_capture::on_error>(
                &capture);
    const emel::batch::planner::event::plan_request request{
        .token_ids = ctx.collaborators.model_codes.data(),
        .n_tokens = ctx.collaborators.frame_plan_token_count,
        .n_steps = ctx.collaborators.frame_plan_steps,
        .mode = ctx.collaborators.frame_plan_mode,
        .output_all = ctx.collaborators.frame_plan_output_all,
        .on_done = on_done,
        .on_error = on_error,
    };
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.planner.process_event(request);
  }
};

template <class dependencies_type> struct effect_predict {
  void operator()(const detail::run_frame &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    typename dependencies_type::predict_event request{
        std::span<const int32_t>{ctx.collaborators.model_codes},
        ctx.collaborators.prediction_workspace, runtime_ev.ctx.plan_step_size,
        runtime_ev.ctx.plan_output_count};
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.predictor.process_event(request);
  }
};

template <class dependencies_type> struct effect_execute_graph {
  void operator()(const detail::run_frame &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    runtime_ev.ctx.graph_err = 0;
    typename dependencies_type::graph_event request{
        ctx.collaborators.prediction_workspace,
        std::span<const int32_t>{ctx.collaborators.model_codes}};
    request.error_out = &runtime_ev.ctx.child_err;
    request.graph_error_out = &runtime_ev.ctx.graph_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.graph.process_event(request);
  }
};

template <class dependencies_type> struct effect_sample {
  void operator()(const detail::run_frame &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    runtime_ev.ctx.graph_err = 0;
    runtime_ev.ctx.predicted_text_token = -1;
    std::fill(ctx.collaborators.predicted_codes.begin(),
              ctx.collaborators.predicted_codes.end(), -1);
    typename dependencies_type::sample_event request{
        ctx.collaborators.prediction_workspace,
        std::span<const int32_t>{ctx.collaborators.model_codes},
        ctx.collaborators.predicted_codes, runtime_ev.ctx.predicted_text_token};
    request.error_out = &runtime_ev.ctx.child_err;
    request.graph_error_out = &runtime_ev.ctx.graph_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.sampler.process_event(request);
  }
};

template <class dependencies_type> struct effect_detokenize {
  void operator()(const detail::run_frame &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.tokenizer_err = 0;
    runtime_ev.ctx.produced = false;
    runtime_ev.ctx.text_token = -1;
    std::fill(runtime_ev.request.generated_tokens_out.begin(),
              runtime_ev.request.generated_tokens_out.end(), -1);
    typename dependencies_type::detokenize_event request{
        runtime_ev.ctx.predicted_text_token,
        std::span<const int32_t>{ctx.collaborators.predicted_codes},
        runtime_ev.ctx.text_token,
        runtime_ev.request.generated_tokens_out,
        runtime_ev.ctx.produced,
        runtime_ev.ctx.tokenizer_err};
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.tokenizer.process_event(request);
    runtime_ev.ctx.child_err =
        static_cast<emel::error::type>(runtime_ev.ctx.tokenizer_err);
  }
};

template <class dependencies_type> struct effect_publish_done {
  void operator()(const detail::run_frame &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.text_token_out = runtime_ev.ctx.text_token;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <class dependencies_type, error error_value> struct effect_fail {
  void operator()(const detail::run_frame &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.text_token_out = -1;
    runtime_ev.ctx.err = error_code(error_value);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <class dependencies_type> struct effect_emit_done {
  void operator()(const detail::run_frame &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.on_done(
        events::run_done{runtime_ev.request, runtime_ev.ctx.text_token});
  }
};

template <class dependencies_type> struct effect_emit_error {
  void operator()(const detail::run_frame &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.on_error(
        events::run_error{runtime_ev.request, runtime_ev.ctx.err});
  }
};

template <class dependencies_type> struct effect_reset {
  void operator()(const event::reset &ev,
                  const context<dependencies_type> &) const noexcept {
    ev.error_out = error_code(error::none);
  }
};

template <class dependencies_type> struct effect_unexpected {
  template <class unexpected_type>
  void operator()(const unexpected_type &unexpected,
                  const context<dependencies_type> &) const noexcept {
    effect_reject_origin(stateforward::sml::back::get_origin_event(unexpected));
  }

private:
  template <class event_type>
  static void effect_reject_origin(const event_type &ev) noexcept {
    if constexpr (requires { ev.error_out; }) {
      ev.error_out = error_code(error::unexpected_event);
    }
  }
};

} // namespace emel::speech::generator::frame::action
