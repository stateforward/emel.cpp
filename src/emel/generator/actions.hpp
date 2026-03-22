#pragma once

#include "emel/batch/planner/events.hpp"
#include "emel/generator/context.hpp"
#include "emel/generator/errors.hpp"
#include "emel/generator/events.hpp"
#include "emel/graph/events.hpp"
#include "emel/logits/sampler/errors.hpp"
#include "emel/memory/events.hpp"
#include "emel/text/conditioner/events.hpp"
#include "emel/text/renderer/events.hpp"

namespace emel::generator::action {

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return (ev.event_);
  } else {
    return (ev);
  }
}

inline void capture_plan_done(event::generate_ctx & ctx,
                              const emel::batch::planner::events::plan_done & ev) noexcept {
  ctx.prefill_step_size = ev.step_sizes[0];
  ctx.plan_step_count = ev.step_count;
  ctx.plan_outputs = ev.total_outputs;
}

inline void capture_plan_error(event::generate_ctx & ctx,
                               const emel::batch::planner::events::plan_error & ev) noexcept {
  ctx.phase_code = static_cast<int32_t>(ev.err);
}

inline bool capture_graph_reserve_done(const emel::graph::events::reserve_done &) noexcept {
  return true;
}

inline bool capture_graph_reserve_error(event::initialize_ctx & ctx,
                                        const emel::graph::events::reserve_error & ev) noexcept {
  ctx.phase_code = ev.err;
  return true;
}

inline bool capture_graph_compute_done(const emel::graph::events::compute_done &) noexcept {
  return true;
}

inline bool capture_graph_compute_error(event::generate_ctx & ctx,
                                        const emel::graph::events::compute_error & ev) noexcept {
  ctx.phase_code = ev.err;
  return true;
}

struct begin_initialize {
  void operator()(const event::initialize_run & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.phase_accepted = false;
    ev.ctx.phase_code = 0;
    ev.ctx.buffers_ready = false;

    ctx.conditioning.actor = ev.request.tokenizer_sm;
    ctx.conditioning.dispatch_bind = ev.request.dispatch_tokenizer_bind;
    ctx.conditioning.dispatch_tokenize = ev.request.dispatch_tokenizer_tokenize;
    ctx.conditioning.preprocessor = ev.request.preprocessor_variant;
    ctx.conditioning.encoder = ev.request.encoder_variant;
    ctx.conditioning.add_special = ev.request.add_special;
    ctx.conditioning.parse_special = ev.request.parse_special;

    ctx.limits.prompt_capacity = ev.request.max_prompt_tokens;
    ctx.limits.decode_capacity = ev.request.max_generated_tokens;
    ctx.limits.block_capacity = ev.request.max_blocks;
    ctx.limits.block_tokens = ev.request.block_tokens;

    ctx.buffers.seq_masks[0] = 1u;
    ctx.buffers.seq_primary_ids[0] = k_sequence_id;

    ctx.state.sequence_live = false;
    ctx.state.memory_snapshot = {};

  }
};

struct reject_initialize {
  void operator()(const event::initialize_run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    ev.ctx.phase_accepted = false;
    ev.ctx.phase_code = 0;
    ev.ctx.buffers_ready = false;
  }
};

struct request_conditioner_bind {
  void operator()(const event::initialize_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = 0;
    emel::text::conditioner::event::bind bind_ev{ctx.model->vocab_data};
    bind_ev.preprocessor_variant = ctx.conditioning.preprocessor;
    bind_ev.encoder_variant = ctx.conditioning.encoder;
    bind_ev.tokenizer_sm = ctx.conditioning.actor;
    bind_ev.dispatch_tokenizer_bind = ctx.conditioning.dispatch_bind;
    bind_ev.dispatch_tokenizer_tokenize = ctx.conditioning.dispatch_tokenize;
    bind_ev.formatter_ctx = ctx.formatter_ctx;
    bind_ev.format_prompt = ctx.format_prompt;
    bind_ev.add_special = ctx.conditioning.add_special;
    bind_ev.parse_special = ctx.conditioning.parse_special;
    bind_ev.error_out = &ev.ctx.phase_code;
    ev.ctx.phase_accepted = ctx.conditioner->process_event(bind_ev);
  }
};

struct request_renderer_initialize {
  void operator()(const event::initialize_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = 0;
    emel::text::renderer::event::initialize initialize_ev{ctx.model->vocab_data};
    initialize_ev.strip_leading_space = ev.request.strip_leading_space;
    initialize_ev.stop_sequences = ev.request.stop_sequences.data();
    initialize_ev.stop_sequence_count = ev.request.stop_sequences.size();
    initialize_ev.error_out = &ev.ctx.phase_code;
    ev.ctx.phase_accepted = ctx.renderer.process_event(initialize_ev);
  }
};

struct request_memory_reserve {
  void operator()(const event::initialize_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    emel::memory::event::reserve reserve_ev{
      .max_sequences = 1,
      .max_blocks = ctx.limits.block_capacity,
      .block_tokens = ctx.limits.block_tokens,
      .error_out = &ev.ctx.phase_code,
    };
    ev.ctx.phase_accepted = ctx.memory.process_event(reserve_ev);
  }
};

struct request_graph_reserve {
  void operator()(const event::initialize_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(emel::error::cast(emel::graph::error::none));
    const auto on_done =
        emel::callback<bool(const emel::graph::events::reserve_done &)>::from<capture_graph_reserve_done>();
    const auto on_error =
        emel::callback<bool(const emel::graph::events::reserve_error &)>::from<
            event::initialize_ctx,
            capture_graph_reserve_error>(&ev.ctx);
    emel::graph::event::reserve reserve_ev{
      .model_topology = &ctx.compute.model_topology,
      .output_out = &ctx.state.graph_reservation,
      .max_node_count = ctx.compute.model_topology.node_count,
      .max_tensor_count = ctx.compute.model_topology.tensor_count,
      .bytes_per_tensor = ctx.compute.model_topology.bytes_per_tensor,
      .workspace_capacity_bytes = ctx.compute.model_topology.workspace_capacity_bytes,
      .dispatch_done = on_done,
      .dispatch_error = on_error,
    };
    ev.ctx.phase_accepted = ctx.graph.process_event(reserve_ev);
  }
};

struct configure_sampler {
  void operator()(const event::initialize_run & ev, context & ctx) const noexcept {
    emel::error::type sampler_error = emel::error::cast(emel::logits::sampler::error::none);
    emel::logits::sampler::event::configure configure_ev{
      ev.request.sampler_fns.front(),
      static_cast<int32_t>(ev.request.sampler_fns.size()),
      sampler_error,
    };
    const bool sampler_ready = ctx.sampler.process_event(configure_ev);
    ev.ctx.buffers_ready = ctx.buffers.vocab_size > 0 &&
                           ctx.buffers.logits != nullptr &&
                           ctx.buffers.candidate_ids != nullptr &&
                           ctx.buffers.candidate_scores != nullptr &&
                           sampler_ready;
    ev.ctx.phase_accepted = ev.ctx.buffers_ready;
    ev.ctx.phase_code = static_cast<int32_t>(sampler_error);
  }
};

struct begin_generate {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.phase_accepted = false;
    ev.ctx.phase_code = 0;
    ev.ctx.tokens_generated = 0;
    ev.ctx.target_tokens = ev.request.max_tokens;
    ev.ctx.prompt_token_count = 0;
    ev.ctx.prefill_step_size = 0;
    ev.ctx.plan_step_count = 0;
    ev.ctx.plan_outputs = 0;
    ev.ctx.kv_tokens = 0;
    ev.ctx.selected_token = -1;
    ev.ctx.output_length = 0;
    ev.ctx.phase_output_length = 0;
    ev.ctx.render_status = emel::text::renderer::sequence_status::running;
    ev.ctx.graph_output = {};
    ev.ctx.io = {};
    ev.request.output_length_out = 0;
  }
};

struct reject_invalid_generate {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    ev.ctx.phase_accepted = false;
    ev.ctx.phase_code = 0;
    ev.ctx.tokens_generated = 0;
    ev.ctx.output_length = 0;
  }
};

struct reject_uninitialized_generate {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    ev.ctx.phase_accepted = false;
    ev.ctx.phase_code = 0;
    ev.ctx.tokens_generated = 0;
    ev.ctx.output_length = 0;
  }
};

struct request_reset_sequence {
  void operator()(const event::generate_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    emel::memory::event::free_sequence free_ev{
      .seq_id = k_sequence_id,
      .error_out = &ev.ctx.phase_code,
    };
    ev.ctx.phase_accepted = ctx.memory.process_event(free_ev);
  }
};

struct request_conditioning {
  void operator()(const event::generate_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = 0;
    emel::text::conditioner::event::prepare prepare_ev{
      ev.ctx.prompt_token_count,
      ev.ctx.phase_code,
    };
    prepare_ev.input = ev.request.prompt;
    prepare_ev.add_special = ctx.conditioning.add_special;
    prepare_ev.parse_special = ctx.conditioning.parse_special;
    prepare_ev.use_bind_defaults = true;
    prepare_ev.token_ids_out = ctx.buffers.prompt_tokens.data();
    prepare_ev.token_capacity = ctx.limits.prompt_capacity;
    ev.ctx.phase_accepted = ctx.conditioner->process_event(prepare_ev);
  }
};

struct request_planning {
  void operator()(const event::generate_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(emel::error::cast(emel::batch::planner::error::none));
    ev.ctx.prefill_step_size = 0;
    ev.ctx.plan_step_count = 0;
    ev.ctx.plan_outputs = 0;
    const auto on_done = emel::callback<void(const emel::batch::planner::events::plan_done &)>::from<
        event::generate_ctx,
        capture_plan_done>(&ev.ctx);
    const auto on_error =
        emel::callback<void(const emel::batch::planner::events::plan_error &)>::from<
            event::generate_ctx,
            capture_plan_error>(&ev.ctx);
    emel::batch::planner::event::request request{
      .token_ids = ctx.buffers.prompt_tokens.data(),
      .n_tokens = ev.ctx.prompt_token_count,
      .n_steps = 1,
      .mode = emel::batch::planner::event::plan_mode::simple,
      .seq_masks = ctx.buffers.seq_masks.data(),
      .seq_masks_count = 1,
      .seq_primary_ids = ctx.buffers.seq_primary_ids.data(),
      .seq_primary_ids_count = 1,
      .equal_sequential = true,
      .seq_mask_words = k_sequence_mask_words,
      .output_mask = nullptr,
      .output_mask_count = 0,
      .output_all = false,
      .on_done = on_done,
      .on_error = on_error,
    };
    ev.ctx.phase_accepted = ctx.planner.process_event(request);
  }
};

struct request_allocate_sequence {
  void operator()(const event::generate_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    emel::memory::event::allocate_sequence allocate_ev{
      .seq_id = k_sequence_id,
      .error_out = &ev.ctx.phase_code,
    };
    ev.ctx.phase_accepted = ctx.memory.process_event(allocate_ev);
  }
};

struct request_prefill_slots {
  void operator()(const event::generate_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    emel::memory::event::allocate_slots allocate_ev{
      .seq_id = k_sequence_id,
      .token_count = ev.ctx.prompt_token_count,
      .block_count_out = nullptr,
      .error_out = &ev.ctx.phase_code,
    };
    ev.ctx.phase_accepted = ctx.memory.process_event(allocate_ev);
  }
};

struct request_memory_snapshot {
  void operator()(const event::generate_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    emel::memory::event::capture_view capture_ev{
      .snapshot_out = &ctx.state.memory_snapshot,
      .error_out = &ev.ctx.phase_code,
    };
    ev.ctx.phase_accepted = ctx.memory.process_event(capture_ev);
  }
};

struct request_prefill_compute {
  void operator()(const event::generate_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(emel::error::cast(emel::graph::error::none));
    for (int32_t idx = 0; idx < ev.ctx.prompt_token_count; ++idx) {
      ctx.buffers.positions[static_cast<size_t>(idx)] = idx;
    }
    ev.ctx.graph_output = {};
    ev.ctx.io.backend_ctx = &ctx.compute.backend;
    ev.ctx.io.token_ids = ctx.buffers.prompt_tokens.data();
    ev.ctx.io.token_count = ev.ctx.prompt_token_count;
    ev.ctx.io.logits = ctx.buffers.logits.get();
    ev.ctx.io.logits_capacity = ctx.buffers.vocab_size;
    const auto on_done =
        emel::callback<bool(const emel::graph::events::compute_done &)>::from<capture_graph_compute_done>();
    const auto on_error =
        emel::callback<bool(const emel::graph::events::compute_error &)>::from<
            event::generate_ctx,
            capture_graph_compute_error>(&ev.ctx);
    emel::graph::event::compute compute_ev{
      .step_plan = &ctx.compute.prefill_plan,
      .output_out = &ev.ctx.graph_output,
      .node_count_hint = ctx.state.graph_reservation.node_count,
      .tensor_count_hint = ctx.state.graph_reservation.tensor_count,
      .bytes_per_tensor = ctx.compute.model_topology.bytes_per_tensor,
      .workspace_capacity_bytes = ctx.compute.model_topology.workspace_capacity_bytes,
      .step_index = 0,
      .step_size = ev.ctx.prefill_step_size,
      .kv_tokens = 0,
      .memory_sm = &ctx.memory,
      .memory_view = &ctx.state.memory_snapshot,
      .expected_outputs = ev.ctx.plan_outputs,
      .compute_ctx = &ev.ctx.io,
      .positions = ctx.buffers.positions.data(),
      .positions_count = ev.ctx.prompt_token_count,
      .seq_masks = ctx.buffers.seq_masks.data(),
      .seq_mask_words = k_sequence_mask_words,
      .seq_masks_count = 1,
      .seq_primary_ids = ctx.buffers.seq_primary_ids.data(),
      .seq_primary_ids_count = 1,
      .validate = emel::generator::detail::validate,
      .prepare_graph = emel::generator::detail::prepare_graph,
      .alloc_graph = emel::generator::detail::alloc_graph,
      .bind_inputs = emel::generator::detail::bind_inputs,
      .run_kernel = emel::generator::detail::run_kernel,
      .extract_outputs = emel::generator::detail::extract_outputs,
      .dispatch_done = on_done,
      .dispatch_error = on_error,
    };
    ev.ctx.phase_accepted = ctx.graph.process_event(compute_ev);
  }
};

struct request_decode_slots {
  void operator()(const event::generate_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    emel::memory::event::allocate_slots allocate_ev{
      .seq_id = k_sequence_id,
      .token_count = 1,
      .block_count_out = nullptr,
      .error_out = &ev.ctx.phase_code,
    };
    ev.ctx.phase_accepted = ctx.memory.process_event(allocate_ev);
  }
};

struct request_decode_compute {
  void operator()(const event::generate_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(emel::error::cast(emel::graph::error::none));
    ctx.buffers.positions[0] = ev.ctx.kv_tokens;
    ev.ctx.graph_output = {};
    ev.ctx.io.backend_ctx = &ctx.compute.backend;
    ev.ctx.io.token_ids = &ev.ctx.selected_token;
    ev.ctx.io.token_count = 1;
    ev.ctx.io.logits = ctx.buffers.logits.get();
    ev.ctx.io.logits_capacity = ctx.buffers.vocab_size;
    const auto on_done =
        emel::callback<bool(const emel::graph::events::compute_done &)>::from<capture_graph_compute_done>();
    const auto on_error =
        emel::callback<bool(const emel::graph::events::compute_error &)>::from<
            event::generate_ctx,
            capture_graph_compute_error>(&ev.ctx);
    emel::graph::event::compute compute_ev{
      .step_plan = &ctx.compute.decode_plan,
      .output_out = &ev.ctx.graph_output,
      .node_count_hint = ctx.state.graph_reservation.node_count,
      .tensor_count_hint = ctx.state.graph_reservation.tensor_count,
      .bytes_per_tensor = ctx.compute.model_topology.bytes_per_tensor,
      .workspace_capacity_bytes = ctx.compute.model_topology.workspace_capacity_bytes,
      .step_index = 0,
      .step_size = 1,
      .kv_tokens = ev.ctx.kv_tokens,
      .memory_sm = &ctx.memory,
      .memory_view = &ctx.state.memory_snapshot,
      .expected_outputs = 1,
      .compute_ctx = &ev.ctx.io,
      .positions = ctx.buffers.positions.data(),
      .positions_count = 1,
      .seq_masks = ctx.buffers.seq_masks.data(),
      .seq_mask_words = k_sequence_mask_words,
      .seq_masks_count = 1,
      .seq_primary_ids = ctx.buffers.seq_primary_ids.data(),
      .seq_primary_ids_count = 1,
      .validate = emel::generator::detail::validate,
      .prepare_graph = emel::generator::detail::prepare_graph,
      .alloc_graph = emel::generator::detail::alloc_graph,
      .bind_inputs = emel::generator::detail::bind_inputs,
      .run_kernel = emel::generator::detail::run_kernel,
      .extract_outputs = emel::generator::detail::extract_outputs,
      .dispatch_done = on_done,
      .dispatch_error = on_error,
    };
    ev.ctx.phase_accepted = ctx.graph.process_event(compute_ev);
  }
};

struct request_decode_sample {
  void operator()(const event::generate_run & ev, context & ctx) const noexcept {
    emel::error::type sample_error = emel::error::cast(emel::logits::sampler::error::none);
    emel::logits::sampler::event::sample_logits sample_ev{
      ctx.buffers.logits[0],
      ctx.buffers.vocab_size,
      ctx.buffers.candidate_ids[0],
      ctx.buffers.candidate_scores[0],
      ctx.buffers.candidate_capacity,
      ev.ctx.selected_token,
      sample_error,
    };
    ev.ctx.phase_accepted = ctx.sampler.process_event(sample_ev);
    ev.ctx.phase_code = static_cast<int32_t>(sample_error);
  }
};

struct request_decode_render {
  void operator()(const event::generate_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = 0;
    ev.ctx.phase_output_length = 0;
    emel::text::renderer::event::render render_ev = {};
    render_ev.token_id = ev.ctx.selected_token;
    render_ev.sequence_id = k_sequence_id;
    render_ev.emit_special = false;
    render_ev.output = ev.request.output.data() + ev.ctx.output_length;
    render_ev.output_capacity = ev.request.output.size() - ev.ctx.output_length;
    render_ev.output_length_out = &ev.ctx.phase_output_length;
    render_ev.status_out = &ev.ctx.render_status;
    render_ev.error_out = &ev.ctx.phase_code;
    ev.ctx.phase_accepted = ctx.renderer.process_event(render_ev);
  }
};

struct request_flush {
  void operator()(const event::generate_run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = 0;
    ev.ctx.phase_output_length = 0;
    emel::text::renderer::event::flush flush_ev = {};
    flush_ev.sequence_id = k_sequence_id;
    flush_ev.output = ev.request.output.data() + ev.ctx.output_length;
    flush_ev.output_capacity = ev.request.output.size() - ev.ctx.output_length;
    flush_ev.output_length_out = &ev.ctx.phase_output_length;
    flush_ev.status_out = &ev.ctx.render_status;
    flush_ev.error_out = &ev.ctx.phase_code;
    ev.ctx.phase_accepted = ctx.renderer.process_event(flush_ev);
  }
};

struct mark_invalid_request {
  template <class runtime_event>
  void operator()(const runtime_event & ev, context &) const noexcept {
    auto & runtime_ev = unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct mark_backend_error {
  template <class runtime_event>
  void operator()(const runtime_event & ev, context &) const noexcept {
    auto & runtime_ev = unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::backend);
  }
};

struct mark_sequence_clear {
  void operator()(const event::generate_run &, context & ctx) const noexcept {
    ctx.state.sequence_live = false;
  }
};

struct mark_sequence_live {
  void operator()(const event::generate_run &, context & ctx) const noexcept {
    ctx.state.sequence_live = true;
  }
};

struct mark_prefill_cached {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.kv_tokens = ev.ctx.prompt_token_count;
  }
};

struct advance_kv_cache {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.kv_tokens += 1;
  }
};

struct commit_render_output {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.output_length += ev.ctx.phase_output_length;
    ev.ctx.tokens_generated += 1;
    ev.request.output_length_out = ev.ctx.output_length;
  }
};

struct commit_flush_output {
  void operator()(const event::generate_run & ev, context &) const noexcept {
    ev.ctx.output_length += ev.ctx.phase_output_length;
    ev.request.output_length_out = ev.ctx.output_length;
  }
};

struct dispatch_initialize_done_with_callback_and_error_out {
  void operator()(const event::initialize_run & ev, const context &) const noexcept {
    *ev.request.error_out = emel::error::cast(error::none);
    ev.request.on_done(events::initialize_done{.request = &ev.request});
  }
};

struct dispatch_initialize_done_with_callback_only {
  void operator()(const event::initialize_run & ev, const context &) const noexcept {
    ev.request.on_done(events::initialize_done{.request = &ev.request});
  }
};

struct dispatch_initialize_done_with_error_out_only {
  void operator()(const event::initialize_run & ev, const context &) const noexcept {
    *ev.request.error_out = emel::error::cast(error::none);
  }
};

struct dispatch_initialize_done_without_channels {
  void operator()(const event::initialize_run &, const context &) const noexcept {}
};

struct dispatch_initialize_error_with_callback_and_error_out {
  void operator()(const event::initialize_run & ev, const context &) const noexcept {
    *ev.request.error_out = ev.ctx.err;
    ev.request.on_error(events::initialize_error{
      .request = &ev.request,
      .err = ev.ctx.err,
    });
  }
};

struct dispatch_initialize_error_with_callback_only {
  void operator()(const event::initialize_run & ev, const context &) const noexcept {
    ev.request.on_error(events::initialize_error{
      .request = &ev.request,
      .err = ev.ctx.err,
    });
  }
};

struct dispatch_initialize_error_with_error_out_only {
  void operator()(const event::initialize_run & ev, const context &) const noexcept {
    *ev.request.error_out = ev.ctx.err;
  }
};

struct dispatch_initialize_error_without_channels {
  void operator()(const event::initialize_run &, const context &) const noexcept {}
};

struct dispatch_generate_done_with_callback_and_error_out {
  void operator()(const event::generate_run & ev, const context &) const noexcept {
    *ev.request.error_out = emel::error::cast(error::none);
    ev.request.on_done(events::generation_done{
      .request = &ev.request,
      .tokens_generated = ev.ctx.tokens_generated,
      .output_length = ev.ctx.output_length,
    });
  }
};

struct dispatch_generate_done_with_callback_only {
  void operator()(const event::generate_run & ev, const context &) const noexcept {
    ev.request.on_done(events::generation_done{
      .request = &ev.request,
      .tokens_generated = ev.ctx.tokens_generated,
      .output_length = ev.ctx.output_length,
    });
  }
};

struct dispatch_generate_done_with_error_out_only {
  void operator()(const event::generate_run & ev, const context &) const noexcept {
    *ev.request.error_out = emel::error::cast(error::none);
  }
};

struct dispatch_generate_done_without_channels {
  void operator()(const event::generate_run &, const context &) const noexcept {}
};

struct dispatch_generate_error_with_callback_and_error_out {
  void operator()(const event::generate_run & ev, const context &) const noexcept {
    *ev.request.error_out = ev.ctx.err;
    ev.request.on_error(events::generation_error{
      .request = &ev.request,
      .err = ev.ctx.err,
      .tokens_generated = ev.ctx.tokens_generated,
      .output_length = ev.ctx.output_length,
    });
  }
};

struct dispatch_generate_error_with_callback_only {
  void operator()(const event::generate_run & ev, const context &) const noexcept {
    ev.request.on_error(events::generation_error{
      .request = &ev.request,
      .err = ev.ctx.err,
      .tokens_generated = ev.ctx.tokens_generated,
      .output_length = ev.ctx.output_length,
    });
  }
};

struct dispatch_generate_error_with_error_out_only {
  void operator()(const event::generate_run & ev, const context &) const noexcept {
    *ev.request.error_out = ev.ctx.err;
  }
};

struct dispatch_generate_error_without_channels {
  void operator()(const event::generate_run &, const context &) const noexcept {}
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::backend);
    }
  }
};

inline constexpr begin_initialize begin_initialize{};
inline constexpr reject_initialize reject_initialize{};
inline constexpr request_conditioner_bind request_conditioner_bind{};
inline constexpr request_renderer_initialize request_renderer_initialize{};
inline constexpr request_memory_reserve request_memory_reserve{};
inline constexpr request_graph_reserve request_graph_reserve{};
inline constexpr configure_sampler configure_sampler{};
inline constexpr begin_generate begin_generate{};
inline constexpr reject_invalid_generate reject_invalid_generate{};
inline constexpr reject_uninitialized_generate reject_uninitialized_generate{};
inline constexpr request_reset_sequence request_reset_sequence{};
inline constexpr request_conditioning request_conditioning{};
inline constexpr request_planning request_planning{};
inline constexpr request_allocate_sequence request_allocate_sequence{};
inline constexpr request_prefill_slots request_prefill_slots{};
inline constexpr request_memory_snapshot request_memory_snapshot{};
inline constexpr request_prefill_compute request_prefill_compute{};
inline constexpr request_decode_slots request_decode_slots{};
inline constexpr request_decode_compute request_decode_compute{};
inline constexpr request_decode_sample request_decode_sample{};
inline constexpr request_decode_render request_decode_render{};
inline constexpr request_flush request_flush{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_backend_error mark_backend_error{};
inline constexpr mark_sequence_clear mark_sequence_clear{};
inline constexpr mark_sequence_live mark_sequence_live{};
inline constexpr mark_prefill_cached mark_prefill_cached{};
inline constexpr advance_kv_cache advance_kv_cache{};
inline constexpr commit_render_output commit_render_output{};
inline constexpr commit_flush_output commit_flush_output{};
inline constexpr dispatch_initialize_done_with_callback_and_error_out
    dispatch_initialize_done_with_callback_and_error_out{};
inline constexpr dispatch_initialize_done_with_callback_only
    dispatch_initialize_done_with_callback_only{};
inline constexpr dispatch_initialize_done_with_error_out_only
    dispatch_initialize_done_with_error_out_only{};
inline constexpr dispatch_initialize_done_without_channels
    dispatch_initialize_done_without_channels{};
inline constexpr dispatch_initialize_error_with_callback_and_error_out
    dispatch_initialize_error_with_callback_and_error_out{};
inline constexpr dispatch_initialize_error_with_callback_only
    dispatch_initialize_error_with_callback_only{};
inline constexpr dispatch_initialize_error_with_error_out_only
    dispatch_initialize_error_with_error_out_only{};
inline constexpr dispatch_initialize_error_without_channels
    dispatch_initialize_error_without_channels{};
inline constexpr dispatch_generate_done_with_callback_and_error_out
    dispatch_generate_done_with_callback_and_error_out{};
inline constexpr dispatch_generate_done_with_callback_only
    dispatch_generate_done_with_callback_only{};
inline constexpr dispatch_generate_done_with_error_out_only
    dispatch_generate_done_with_error_out_only{};
inline constexpr dispatch_generate_done_without_channels
    dispatch_generate_done_without_channels{};
inline constexpr dispatch_generate_error_with_callback_and_error_out
    dispatch_generate_error_with_callback_and_error_out{};
inline constexpr dispatch_generate_error_with_callback_only
    dispatch_generate_error_with_callback_only{};
inline constexpr dispatch_generate_error_with_error_out_only
    dispatch_generate_error_with_error_out_only{};
inline constexpr dispatch_generate_error_without_channels
    dispatch_generate_error_without_channels{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::generator::action
