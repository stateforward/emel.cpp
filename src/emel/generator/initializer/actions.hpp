#pragma once

#include "emel/generator/actions.hpp"
#include "emel/generator/errors.hpp"
#include "emel/generator/initializer/context.hpp"
#include "emel/generator/initializer/detail.hpp"

namespace emel::generator::initializer::action {

struct begin_initialize {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    auto & generator = ctx.generator;
    ev.ctx.err = emel::error::cast(emel::generator::error::none);
    ev.ctx.phase_accepted = false;
    ev.ctx.phase_code = 0;
    ev.ctx.buffers_ready = false;

    generator.conditioning.actor = ev.request.tokenizer_sm;
    generator.conditioning.dispatch_bind = ev.request.dispatch_tokenizer_bind;
    generator.conditioning.dispatch_tokenize = ev.request.dispatch_tokenizer_tokenize;
    generator.conditioning.preprocessor = ev.request.preprocessor_variant;
    generator.conditioning.encoder = ev.request.encoder_variant;
    generator.conditioning.add_special = ev.request.add_special;
    generator.conditioning.parse_special = ev.request.parse_special;

    generator.limits.prompt_capacity = ev.request.max_prompt_tokens;
    generator.limits.decode_capacity = ev.request.max_generated_tokens;
    generator.limits.block_capacity = ev.request.max_blocks;
    generator.limits.block_tokens = ev.request.block_tokens;
    generator.state.selection_mode = ev.request.selection_mode;

    generator.buffers.seq_masks[0] = 1u;
    generator.buffers.seq_primary_ids[0] = emel::generator::action::k_sequence_id;

    emel::generator::action::capture_renderer_session(ev.request, generator);
    emel::generator::action::reset_generation_tensor_epochs(generator);
    generator.state.sequence_live = false;
    generator.state.memory_snapshot = {};
  }
};

struct request_backend_prepare {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    auto & generator = ctx.generator;
    generator.compute.backend_ready = false;
    ev.ctx.phase_code = static_cast<int32_t>(
        emel::generator::detail::prepare(generator.compute.backend, *generator.model));
    ev.ctx.phase_accepted =
        ev.ctx.phase_code ==
        static_cast<int32_t>(emel::error::cast(emel::model::loader::error::none));
  }
};

struct accept_prepared_backend {
  void operator()(const event::run &, context & ctx) const noexcept {
    auto & generator = ctx.generator;
    generator.compute.model_topology = generator.compute.backend.topology;
    generator.compute.prefill_plan = generator.compute.backend.prefill_plan;
    generator.compute.decode_plan = generator.compute.backend.decode_plan;
    generator.compute.backend_ready = true;
  }
};

struct request_conditioner_bind {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    auto & generator = ctx.generator;
    ev.ctx.phase_code = 0;
    emel::text::conditioner::event::bind bind_ev{generator.model->vocab_data};
    bind_ev.preprocessor_variant = generator.conditioning.preprocessor;
    bind_ev.encoder_variant = generator.conditioning.encoder;
    bind_ev.tokenizer_sm = generator.conditioning.actor;
    bind_ev.dispatch_tokenizer_bind = generator.conditioning.dispatch_bind;
    bind_ev.dispatch_tokenizer_tokenize = generator.conditioning.dispatch_tokenize;
    bind_ev.formatter_ctx = generator.formatter_ctx;
    bind_ev.format_prompt = generator.format_prompt;
    bind_ev.add_special = generator.conditioning.add_special;
    bind_ev.parse_special = generator.conditioning.parse_special;
    bind_ev.error_out = &ev.ctx.phase_code;
    ev.ctx.phase_accepted = generator.conditioner->process_event(bind_ev);
  }
};

struct request_renderer_initialize {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = 0;
    ev.ctx.phase_accepted =
        emel::generator::action::dispatch_renderer_initialize(ctx.generator, ev.ctx.phase_code);
  }
};

struct request_memory_reserve {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    emel::memory::event::reserve reserve_ev{
      .max_sequences = 1,
      .max_blocks = ctx.generator.limits.block_capacity,
      .block_tokens = ctx.generator.limits.block_tokens,
      .error_out = &ev.ctx.phase_code,
    };
    ev.ctx.phase_accepted = ctx.generator.memory.process_event(reserve_ev);
  }
};

struct request_graph_reserve {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    auto & generator = ctx.generator;
    ev.ctx.phase_code = static_cast<int32_t>(emel::error::cast(emel::graph::error::none));
    const auto * lifecycle = emel::generator::detail::reserve_lifecycle(
        generator.compute.backend,
        generator.buffers.prompt_tokens.data(),
        generator.limits.prompt_capacity,
        generator.buffers.positions.data(),
        generator.limits.prompt_capacity,
        generator.buffers.logits.get(),
        generator.buffers.vocab_size);
    const auto on_done =
        emel::callback<bool(const emel::graph::events::reserve_done &)>::from<
            emel::generator::action::capture_graph_reserve_done>();
    const auto on_error =
        emel::callback<bool(const emel::graph::events::reserve_error &)>::from<
            emel::generator::event::initialize_ctx,
            emel::generator::action::capture_graph_reserve_error>(&ev.ctx);
    emel::graph::event::reserve reserve_ev{
      .model_topology = &generator.compute.model_topology,
      .output_out = &generator.state.graph_reservation,
      .lifecycle = lifecycle,
      .max_node_count = generator.compute.model_topology.node_count,
      .max_tensor_count = generator.compute.model_topology.tensor_count,
      .bytes_per_tensor = generator.compute.model_topology.bytes_per_tensor,
      .workspace_capacity_bytes = generator.compute.model_topology.workspace_capacity_bytes,
      .dispatch_done = on_done,
      .dispatch_error = on_error,
    };
    ev.ctx.phase_accepted = generator.graph.process_event(reserve_ev);
  }
};

struct configure_sampler {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    emel::error::type sampler_error = emel::error::cast(emel::logits::sampler::error::none);
    emel::logits::sampler::event::configure configure_ev{
      ev.request.sampler_fns.front(),
      static_cast<int32_t>(ev.request.sampler_fns.size()),
      sampler_error,
    };
    const bool sampler_ready = ctx.generator.sampler.process_event(configure_ev);
    ev.ctx.buffers_ready = ctx.generator.buffers.vocab_size > 0 &&
                           ctx.generator.buffers.logits != nullptr &&
                           ctx.generator.buffers.candidate_ids != nullptr &&
                           ctx.generator.buffers.candidate_scores != nullptr &&
                           sampler_ready;
    ev.ctx.phase_accepted = ev.ctx.buffers_ready;
    ev.ctx.phase_code = static_cast<int32_t>(sampler_error);
  }
};

struct configure_preselected_argmax {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    ev.ctx.buffers_ready =
        ctx.generator.buffers.vocab_size > 0 && ctx.generator.buffers.logits != nullptr;
    ev.ctx.phase_accepted = ev.ctx.buffers_ready;
    ev.ctx.phase_code =
        static_cast<int32_t>(emel::error::cast(emel::generator::error::none));
  }
};

struct mark_invalid_request {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(emel::generator::error::invalid_request);
  }
};

struct mark_backend_error {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(emel::generator::error::backend);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(emel::generator::error::backend);
    }
  }
};

inline constexpr begin_initialize begin_initialize{};
inline constexpr request_backend_prepare request_backend_prepare{};
inline constexpr accept_prepared_backend accept_prepared_backend{};
inline constexpr request_conditioner_bind request_conditioner_bind{};
inline constexpr request_renderer_initialize request_renderer_initialize{};
inline constexpr request_memory_reserve request_memory_reserve{};
inline constexpr request_graph_reserve request_graph_reserve{};
inline constexpr configure_sampler configure_sampler{};
inline constexpr configure_preselected_argmax configure_preselected_argmax{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_backend_error mark_backend_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::generator::initializer::action
