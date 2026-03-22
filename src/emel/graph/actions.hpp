#pragma once

#include <array>
#include <cstdint>

#include "emel/graph/context.hpp"
#include "emel/graph/errors.hpp"
#include "emel/graph/events.hpp"
#include "emel/tensor/errors.hpp"
#include "emel/tensor/events.hpp"

namespace emel::graph::action {

namespace detail {

struct reserve_capture {
  event::reserve_ctx * ctx = nullptr;
};

struct compute_capture {
  event::compute_ctx * ctx = nullptr;
};

inline bool on_reserve_done(void * object,
                            const assembler::events::reserve_done & done) noexcept {
  const auto * capture = static_cast<reserve_capture *>(object);
  capture->ctx->reserve_outcome = event::phase_outcome::done;
  capture->ctx->reserve_output = done.output;
  capture->ctx->err = emel::error::cast(error::none);
  return true;
}

inline bool on_reserve_error(void * object,
                             const assembler::events::reserve_error & err) noexcept {
  const auto * capture = static_cast<reserve_capture *>(object);
  capture->ctx->reserve_outcome = event::phase_outcome::failed;
  capture->ctx->err = static_cast<emel::error::type>(err.err);
  return true;
}

inline bool on_assemble_done(void * object,
                             const assembler::events::assemble_done & done) noexcept {
  const auto * capture = static_cast<compute_capture *>(object);
  capture->ctx->assemble_outcome = event::phase_outcome::done;
  capture->ctx->assemble_output = done.output;
  capture->ctx->err = emel::error::cast(error::none);
  return true;
}

inline bool on_assemble_error(void * object,
                              const assembler::events::assemble_error & err) noexcept {
  const auto * capture = static_cast<compute_capture *>(object);
  capture->ctx->assemble_outcome = event::phase_outcome::failed;
  capture->ctx->err = static_cast<emel::error::type>(err.err);
  return true;
}

inline bool on_execute_done(void * object,
                            const processor::events::execution_done & done) noexcept {
  const auto * capture = static_cast<compute_capture *>(object);
  capture->ctx->execute_outcome = event::phase_outcome::done;
  capture->ctx->execute_output = done.output;
  capture->ctx->err = emel::error::cast(error::none);
  return true;
}

inline bool on_execute_error(void * object,
                             const processor::events::execution_error & err) noexcept {
  const auto * capture = static_cast<compute_capture *>(object);
  capture->ctx->execute_outcome = event::phase_outcome::failed;
  capture->ctx->err = static_cast<emel::error::type>(err.err);
  return true;
}

}  // namespace detail

inline void reset_reserve_output(event::reserve_output & output) noexcept {
  output.graph_topology = nullptr;
  output.node_count = 0;
  output.tensor_count = 0;
  output.required_buffer_bytes = 0;
  output.version = 0;
  output.lifecycle = nullptr;
}

inline void reset_compute_output(event::compute_output & output) noexcept {
  output.graph_topology = nullptr;
  output.node_count = 0;
  output.tensor_count = 0;
  output.required_buffer_bytes = 0;
  output.version = 0;
  output.reused_topology = 0;
  output.outputs_produced = 0;
  output.graph_reused = 0;
  output.lifecycle = nullptr;
}

struct reject_invalid_reserve_with_dispatch {
  void operator()(const event::reserve_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_reserve_output(*ev.request.output_out);
    ev.request.dispatch_error(events::reserve_error{
      *ev.request.output_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct reject_invalid_reserve_with_output_only {
  void operator()(const event::reserve_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_reserve_output(*ev.request.output_out);
  }
};

struct reject_invalid_reserve_without_output {
  void operator()(const event::reserve_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct reject_invalid_compute_with_dispatch {
  void operator()(const event::compute_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_compute_output(*ev.request.output_out);
    ev.request.dispatch_error(events::compute_error{
      *ev.request.output_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct reject_invalid_compute_with_output_only {
  void operator()(const event::compute_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_compute_output(*ev.request.output_out);
  }
};

struct reject_invalid_compute_without_output {
  void operator()(const event::compute_graph & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct begin_reserve {
  void operator()(const event::reserve_graph & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.reserve_outcome = event::phase_outcome::unknown;
    ev.ctx.tensor_reserve_outcome = event::phase_outcome::unknown;
    ev.ctx.reserve_output = {};
    ++ctx.dispatch_generation;
    reset_reserve_output(*ev.request.output_out);
  }
};

struct begin_compute {
  void operator()(const event::compute_graph & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.assemble_outcome = event::phase_outcome::unknown;
    ev.ctx.execute_outcome = event::phase_outcome::unknown;
    ev.ctx.assemble_output = {};
    ev.ctx.execute_output = {};
    ++ctx.dispatch_generation;
    reset_compute_output(*ev.request.output_out);
  }
};

struct request_reserve {
  void operator()(const event::reserve_graph & ev, context & ctx) const noexcept {
    detail::reserve_capture capture{&ev.ctx};
    ev.ctx.err = emel::error::cast(error::assembler_failed);

    const assembler::event::reserve request{
      .model_topology = ev.request.model_topology,
      .output_out = &ev.ctx.reserve_output,
      .lifecycle = ev.request.lifecycle,
      .max_node_count = ev.request.max_node_count,
      .max_tensor_count = ev.request.max_tensor_count,
      .bytes_per_tensor = ev.request.bytes_per_tensor,
      .workspace_capacity_bytes = ev.request.workspace_capacity_bytes,
      .dispatch_done = {&capture, detail::on_reserve_done},
      .dispatch_error = {&capture, detail::on_reserve_error},
    };

    (void)ctx.assembler_actor.process_event(request);
  }
};

struct request_assemble {
  void operator()(const event::compute_graph & ev, context & ctx) const noexcept {
    detail::compute_capture capture{&ev.ctx};
    ev.ctx.err = emel::error::cast(error::assembler_failed);

    const assembler::event::assemble request{
      .step_plan = ev.request.step_plan,
      .output_out = &ev.ctx.assemble_output,
      .lifecycle = ev.request.lifecycle,
      .node_count_hint = ev.request.node_count_hint,
      .tensor_count_hint = ev.request.tensor_count_hint,
      .bytes_per_tensor = ev.request.bytes_per_tensor,
      .workspace_capacity_bytes = ev.request.workspace_capacity_bytes,
      .dispatch_done = {&capture, detail::on_assemble_done},
      .dispatch_error = {&capture, detail::on_assemble_error},
    };

    (void)ctx.assembler_actor.process_event(request);
  }
};

struct request_execute {
  void operator()(const event::compute_graph & ev, context & ctx) const noexcept {
    detail::compute_capture capture{&ev.ctx};
    ev.ctx.err = emel::error::cast(error::processor_failed);

    const processor::event::execute request{
      .step_plan = ev.request.step_plan,
      .output_out = &ev.ctx.execute_output,
      .lifecycle = ev.request.lifecycle,
      .step_index = ev.request.step_index,
      .step_size = ev.request.step_size,
      .kv_tokens = ev.request.kv_tokens,
      .memory_sm = ev.request.memory_sm,
      .memory_view = ev.request.memory_view,
      .expected_outputs = ev.request.expected_outputs,
      .compute_ctx = ev.request.compute_ctx,
      .positions = ev.request.positions,
      .positions_count = ev.request.positions_count,
      .seq_masks = ev.request.seq_masks,
      .seq_mask_words = ev.request.seq_mask_words,
      .seq_masks_count = ev.request.seq_masks_count,
      .seq_primary_ids = ev.request.seq_primary_ids,
      .seq_primary_ids_count = ev.request.seq_primary_ids_count,
      .validate = ev.request.validate,
      .prepare_graph = ev.request.prepare_graph,
      .alloc_graph = ev.request.alloc_graph,
      .bind_inputs = ev.request.bind_inputs,
      .run_kernel = ev.request.run_kernel,
      .extract_outputs = ev.request.extract_outputs,
      .dispatch_done = {&capture, detail::on_execute_done},
      .dispatch_error = {&capture, detail::on_execute_error},
    };

    (void)ctx.processor_actor.process_event(request);
  }
};

namespace detail {

inline bool reserve_lifecycle_manifest(const processor::event::lifecycle_manifest & lifecycle,
                                       tensor::sm & tensor_actor) noexcept {
  int32_t tensor_err = static_cast<int32_t>(emel::error::cast(tensor::error::none));
  bool all_ok = true;
  for (int32_t idx = 0; idx < lifecycle.tensor_count; ++idx) {
    const auto & binding = lifecycle.tensors[idx];
    const tensor::event::reserve_tensor reserve_ev{
      .tensor_id = binding.tensor_id,
      .buffer = binding.buffer,
      .buffer_bytes = binding.buffer_bytes,
      .consumer_refs = binding.consumer_refs,
      .is_leaf = binding.is_leaf,
      .error_out = &tensor_err,
    };
    all_ok = tensor_actor.process_event(reserve_ev) && all_ok;
  }
  return all_ok;
}

}  // namespace detail

struct request_tensor_reserve {
  void operator()(const event::reserve_graph & ev, context & ctx) const noexcept {
    const bool all_ok = detail::reserve_lifecycle_manifest(*ev.request.lifecycle, ctx.tensor_actor);
    const std::array<event::phase_outcome, 2> outcomes{
      event::phase_outcome::failed,
      event::phase_outcome::done,
    };
    const std::array<emel::error::type, 2> errors{
      emel::error::cast(error::internal_error),
      emel::error::cast(error::none),
    };
    ev.ctx.tensor_reserve_outcome = outcomes[static_cast<size_t>(all_ok)];
    ev.ctx.err = errors[static_cast<size_t>(all_ok)];
  }
};

struct dispatch_reserve_done {
  void operator()(const event::reserve_graph & ev, const context &) const noexcept {
    ev.request.output_out->graph_topology = ev.ctx.reserve_output.graph_topology;
    ev.request.output_out->node_count = ev.ctx.reserve_output.node_count;
    ev.request.output_out->tensor_count = ev.ctx.reserve_output.tensor_count;
    ev.request.output_out->required_buffer_bytes = ev.ctx.reserve_output.required_buffer_bytes;
    ev.request.output_out->version = ev.ctx.reserve_output.version;
    ev.request.output_out->lifecycle = ev.ctx.reserve_output.lifecycle;

    ev.request.dispatch_done(events::reserve_done{*ev.request.output_out});
  }
};

struct dispatch_reserve_error {
  void operator()(const event::reserve_graph & ev, const context &) const noexcept {
    ev.request.dispatch_error(events::reserve_error{
      *ev.request.output_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct dispatch_compute_done {
  void operator()(const event::compute_graph & ev, const context &) const noexcept {
    ev.request.output_out->graph_topology = ev.ctx.assemble_output.graph_topology;
    ev.request.output_out->node_count = ev.ctx.assemble_output.node_count;
    ev.request.output_out->tensor_count = ev.ctx.assemble_output.tensor_count;
    ev.request.output_out->required_buffer_bytes = ev.ctx.assemble_output.required_buffer_bytes;
    ev.request.output_out->version = ev.ctx.assemble_output.version;
    ev.request.output_out->reused_topology = ev.ctx.assemble_output.reused_topology;
    ev.request.output_out->outputs_produced = ev.ctx.execute_output.outputs_produced;
    ev.request.output_out->graph_reused = ev.ctx.execute_output.graph_reused;
    ev.request.output_out->lifecycle = ev.request.lifecycle;

    ev.request.dispatch_done(events::compute_done{*ev.request.output_out});
  }
};

struct dispatch_compute_error {
  void operator()(const event::compute_graph & ev, const context &) const noexcept {
    ev.request.dispatch_error(events::compute_error{
      *ev.request.output_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
    }
  }
};

inline constexpr reject_invalid_reserve_with_dispatch reject_invalid_reserve_with_dispatch{};
inline constexpr reject_invalid_reserve_with_output_only reject_invalid_reserve_with_output_only{};
inline constexpr reject_invalid_reserve_without_output reject_invalid_reserve_without_output{};
inline constexpr reject_invalid_compute_with_dispatch reject_invalid_compute_with_dispatch{};
inline constexpr reject_invalid_compute_with_output_only reject_invalid_compute_with_output_only{};
inline constexpr reject_invalid_compute_without_output reject_invalid_compute_without_output{};
inline constexpr begin_reserve begin_reserve{};
inline constexpr begin_compute begin_compute{};
inline constexpr request_reserve request_reserve{};
inline constexpr request_tensor_reserve request_tensor_reserve{};
inline constexpr request_assemble request_assemble{};
inline constexpr request_execute request_execute{};
inline constexpr dispatch_reserve_done dispatch_reserve_done{};
inline constexpr dispatch_reserve_error dispatch_reserve_error{};
inline constexpr dispatch_compute_done dispatch_compute_done{};
inline constexpr dispatch_compute_error dispatch_compute_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::graph::action
