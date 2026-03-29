#pragma once

#include "emel/batch/planner/errors.hpp"
#include "emel/generator/actions.hpp"
#include "emel/generator/events.hpp"
#include "emel/graph/errors.hpp"
#include "emel/logits/sampler/errors.hpp"
#include "emel/memory/hybrid/errors.hpp"
#include "emel/text/conditioner/errors.hpp"
#include "emel/text/renderer/errors.hpp"

namespace emel::generator::guard {

namespace detail {

template <class runtime_event>
bool has_phase_success(const runtime_event & ev) noexcept {
  return ev.ctx.phase_accepted && ev.ctx.phase_code == 0;
}

template <class runtime_event>
bool phase_rejected_without_code(const runtime_event & ev) noexcept {
  return !ev.ctx.phase_accepted && ev.ctx.phase_code == 0;
}

constexpr int32_t conditioner_code(const emel::text::conditioner::error err) noexcept {
  return static_cast<int32_t>(err);
}

constexpr int32_t renderer_code(const emel::text::renderer::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

constexpr int32_t memory_code(const emel::memory::hybrid::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

constexpr int32_t graph_code(const emel::graph::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

constexpr int32_t sampler_code(const emel::logits::sampler::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

inline bool conditioner_invalid_code(const int32_t code) noexcept {
  return code == conditioner_code(emel::text::conditioner::error::invalid_argument) ||
         code == conditioner_code(emel::text::conditioner::error::model_invalid) ||
         code == conditioner_code(emel::text::conditioner::error::capacity);
}

inline bool conditioner_backend_code(const int32_t code) noexcept {
  return code == conditioner_code(emel::text::conditioner::error::backend) ||
         code == conditioner_code(emel::text::conditioner::error::untracked);
}

inline bool renderer_invalid_code(const int32_t code) noexcept {
  return code == renderer_code(emel::text::renderer::error::invalid_request) ||
         code == renderer_code(emel::text::renderer::error::model_invalid);
}

inline bool renderer_backend_code(const int32_t code) noexcept {
  return code == renderer_code(emel::text::renderer::error::backend_error) ||
         code == renderer_code(emel::text::renderer::error::internal_error) ||
         code == renderer_code(emel::text::renderer::error::untracked);
}

inline bool memory_invalid_code(const int32_t code) noexcept {
  return code == memory_code(emel::memory::hybrid::error::invalid_request);
}

inline bool memory_backend_code(const int32_t code) noexcept {
  return code == memory_code(emel::memory::hybrid::error::backend_error) ||
         code == memory_code(emel::memory::hybrid::error::internal_error) ||
         code == memory_code(emel::memory::hybrid::error::out_of_memory) ||
         code == memory_code(emel::memory::hybrid::error::untracked);
}

inline bool graph_invalid_code(const int32_t code) noexcept {
  return code == graph_code(emel::graph::error::invalid_request);
}

inline bool graph_backend_code(const int32_t code) noexcept {
  return code == graph_code(emel::graph::error::assembler_failed) ||
         code == graph_code(emel::graph::error::processor_failed) ||
         code == graph_code(emel::graph::error::busy) ||
         code == graph_code(emel::graph::error::internal_error) ||
         code == graph_code(emel::graph::error::untracked);
}

inline bool planner_invalid_code(const int32_t code) noexcept {
  const auto err = static_cast<emel::error::type>(code);
  return emel::error::has(err, emel::batch::planner::error::invalid_request) ||
         emel::error::has(err, emel::batch::planner::error::invalid_token_data) ||
         emel::error::has(err, emel::batch::planner::error::invalid_step_size) ||
         emel::error::has(err, emel::batch::planner::error::invalid_sequence_metadata) ||
         emel::error::has(err, emel::batch::planner::error::invalid_sequence_id) ||
         emel::error::has(err, emel::batch::planner::error::invalid_sequence_mask) ||
         emel::error::has(err, emel::batch::planner::error::multiple_bits_in_mask) ||
         emel::error::has(err, emel::batch::planner::error::missing_mode) ||
         emel::error::has(err, emel::batch::planner::error::invalid_mode) ||
         emel::error::has(err, emel::batch::planner::error::output_plan_full) ||
         emel::error::has(err, emel::batch::planner::error::output_indices_full) ||
         emel::error::has(err, emel::batch::planner::error::output_steps_full) ||
         emel::error::has(err, emel::batch::planner::error::unsupported_layout);
}

inline bool planner_backend_code(const int32_t code) noexcept {
  const auto err = static_cast<emel::error::type>(code);
  return emel::error::has(err, emel::batch::planner::error::planning_progress_stalled) ||
         emel::error::has(err, emel::batch::planner::error::algorithm_failed) ||
         emel::error::has(err, emel::batch::planner::error::internal_error) ||
         emel::error::has(err, emel::batch::planner::error::untracked);
}

inline bool sampler_invalid_code(const int32_t code) noexcept {
  return code == sampler_code(emel::logits::sampler::error::invalid_request);
}

inline bool sampler_backend_code(const int32_t code) noexcept {
  return code == sampler_code(emel::logits::sampler::error::backend_error) ||
         code == sampler_code(emel::logits::sampler::error::internal_error) ||
         code == sampler_code(emel::logits::sampler::error::untracked);
}

template <class runtime_event>
bool has_invalid_result(const runtime_event & ev, bool (*is_invalid)(int32_t)) noexcept {
  return !has_phase_success(ev) && is_invalid(ev.ctx.phase_code);
}

template <class runtime_event>
bool has_backend_result(const runtime_event & ev, bool (*is_backend)(int32_t)) noexcept {
  return !has_phase_success(ev) &&
         (phase_rejected_without_code(ev) || is_backend(ev.ctx.phase_code) ||
          (!is_backend(ev.ctx.phase_code) && !is_invalid_result(ev, is_backend)));
}

template <class runtime_event>
bool has_done_callback(const runtime_event & ev) noexcept {
  return static_cast<bool>(ev.request.on_done);
}

template <class runtime_event>
bool has_error_callback(const runtime_event & ev) noexcept {
  return static_cast<bool>(ev.request.on_error);
}

template <class runtime_event>
bool has_error_out(const runtime_event & ev) noexcept {
  return ev.request.error_out != nullptr;
}

template <class runtime_event>
bool result_none(const runtime_event & ev) noexcept {
  return ev.ctx.err == emel::error::cast(error::none);
}

template <class runtime_event>
bool result_invalid_request(const runtime_event & ev) noexcept {
  return ev.ctx.err == emel::error::cast(error::invalid_request);
}

template <class runtime_event>
bool result_backend(const runtime_event & ev) noexcept {
  return ev.ctx.err == emel::error::cast(error::backend);
}

template <class runtime_event>
bool sampled_stop_token(const runtime_event & ev, const action::context & ctx) noexcept {
  if (ctx.model == nullptr) {
    return false;
  }
  const auto & vocab = ctx.model->vocab_data;
  return ev.ctx.selected_token == vocab.eos_id || ev.ctx.selected_token == vocab.eot_id;
}

inline bool uses_preselected_argmax_direct(const action::context & ctx) noexcept {
  return ctx.state.selection_mode == emel::generator::selection_mode::preselected_argmax &&
      emel::generator::detail::preselected_argmax_direct_supported(ctx.compute.backend);
}

inline bool uses_prefill_chunk4_q8_gemm(const event::generate_run & ev,
                                        const action::context & ctx) noexcept {
  return ev.ctx.prompt_token_count >= emel::generator::detail::k_prefill_q8_chunk_rows &&
      emel::generator::detail::prefill_chunk4_q8_gemm_supported(ctx.compute.backend);
}

}  // namespace detail

struct valid_initialize {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    const bool sample_logits =
        ev.request.selection_mode == emel::generator::selection_mode::sample_logits;
    const bool preselected_argmax =
        ev.request.selection_mode == emel::generator::selection_mode::preselected_argmax;
    const bool sampler_contract_valid =
        (sample_logits && !ev.request.sampler_fns.empty()) ||
        (preselected_argmax && ev.request.sampler_fns.empty());
    return ctx.model != nullptr &&
           ctx.conditioner != nullptr &&
           ctx.format_prompt != nullptr &&
           ev.request.tokenizer_sm != nullptr &&
           sampler_contract_valid &&
           ev.request.max_prompt_tokens > 0 &&
           ev.request.max_prompt_tokens <= action::MAX_GENERATION_STEPS &&
           ev.request.max_generated_tokens > 0 &&
           ev.request.max_generated_tokens <= action::MAX_GENERATION_STEPS &&
           ev.request.max_blocks > 0 &&
           ev.request.block_tokens > 0;
  }
};

struct invalid_initialize {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    return !valid_initialize{}(ev, ctx);
  }
};

struct initialize_uses_materialized_logits {
  bool operator()(const event::initialize_run &, const action::context & ctx) const noexcept {
    return ctx.state.selection_mode == emel::generator::selection_mode::sample_logits;
  }
};

struct initialize_uses_preselected_argmax {
  bool operator()(const event::initialize_run &, const action::context & ctx) const noexcept {
    return ctx.state.selection_mode == emel::generator::selection_mode::preselected_argmax;
  }
};

struct compute_uses_materialized_logits {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return !detail::uses_preselected_argmax_direct(ctx);
  }
};

struct compute_uses_preselected_argmax_direct {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return detail::uses_preselected_argmax_direct(ctx);
  }
};

struct conditioner_bind_ok {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct conditioner_bind_invalid_request {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::conditioner_invalid_code);
  }
};

struct conditioner_bind_backend_error {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::conditioner_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::conditioner_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct renderer_initialize_ok {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct renderer_initialize_invalid_request {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::renderer_invalid_code);
  }
};

struct renderer_initialize_backend_error {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::renderer_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::renderer_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct memory_reserve_ok {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct graph_reservation_present {
  bool operator()(const event::initialize_run &,
                  const action::context & ctx) const noexcept {
    return ctx.state.graph_reservation.node_count > 0u;
  }
};

struct graph_reservation_missing {
  bool operator()(const event::initialize_run & ev,
                  const action::context & ctx) const noexcept {
    return !graph_reservation_present{}(ev, ctx);
  }
};

struct memory_reserve_with_existing_graph {
  bool operator()(const event::initialize_run & ev,
                  const action::context & ctx) const noexcept {
    return memory_reserve_ok{}(ev, ctx) && graph_reservation_present{}(ev, ctx);
  }
};

struct memory_reserve_with_missing_graph {
  bool operator()(const event::initialize_run & ev,
                  const action::context & ctx) const noexcept {
    return memory_reserve_ok{}(ev, ctx) && graph_reservation_missing{}(ev, ctx);
  }
};

struct memory_reserve_invalid_request {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::memory_invalid_code);
  }
};

struct memory_reserve_backend_error {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::memory_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::memory_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct graph_reserve_ok {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct graph_reserve_invalid_request {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::graph_invalid_code);
  }
};

struct graph_reserve_backend_error {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::graph_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::graph_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct sampler_configured {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return ev.ctx.buffers_ready;
  }
};

struct sampler_config_failed {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return !ev.ctx.buffers_ready;
  }
};

struct valid_generate {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !ev.request.messages.empty() &&
           ev.request.max_tokens > 0 &&
           ev.request.max_tokens <= ctx.limits.decode_capacity &&
           !ev.request.output.empty();
  }
};

struct invalid_generate {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !valid_generate{}(ev, ctx);
  }
};

struct valid_generate_with_reset {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return valid_generate{}(ev, ctx) && ctx.state.sequence_live;
  }
};

struct valid_generate_without_reset {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return valid_generate{}(ev, ctx) && !ctx.state.sequence_live;
  }
};

struct sequence_needs_reset {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return ctx.state.sequence_live;
  }
};

struct sequence_is_clear {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !sequence_needs_reset{}(ev, ctx);
  }
};

struct reset_sequence_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct reset_sequence_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::memory_invalid_code);
  }
};

struct reset_sequence_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::memory_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::memory_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct conditioning_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev) && ev.ctx.prompt_token_count > 0;
  }
};

struct conditioning_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::conditioner_invalid_code);
  }
};

struct conditioning_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::conditioner_invalid_code(ev.ctx.phase_code);
    const bool empty_prompt_tokens = detail::has_phase_success(ev) && ev.ctx.prompt_token_count <= 0;
    return empty_prompt_tokens ||
           (!detail::has_phase_success(ev) &&
            (detail::phase_rejected_without_code(ev) ||
             detail::conditioner_backend_code(ev.ctx.phase_code) ||
             !invalid));
  }
};

struct planning_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev) &&
           ev.ctx.plan_step_count > 0 &&
           ev.ctx.prefill_step_size > 0 &&
           ev.ctx.plan_outputs >= 0;
  }
};

struct planning_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::planner_invalid_code);
  }
};

struct planning_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::planner_invalid_code(ev.ctx.phase_code);
    const bool malformed_plan = detail::has_phase_success(ev) &&
                                (ev.ctx.plan_step_count <= 0 || ev.ctx.prefill_step_size <= 0 ||
                                 ev.ctx.plan_outputs < 0);
    return malformed_plan ||
           (!detail::has_phase_success(ev) &&
            (detail::phase_rejected_without_code(ev) ||
             detail::planner_backend_code(ev.ctx.phase_code) ||
             !invalid));
  }
};

struct allocate_sequence_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct allocate_sequence_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::memory_invalid_code);
  }
};

struct allocate_sequence_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::memory_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::memory_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct prefill_slots_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct prefill_slots_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::memory_invalid_code);
  }
};

struct prefill_slots_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::memory_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::memory_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct snapshot_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct snapshot_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::memory_invalid_code);
  }
};

struct snapshot_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::memory_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::memory_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct prefill_flash_runtime_supported {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return ev.ctx.prompt_token_count > 0 &&
           emel::generator::detail::flash_attention_supported(
               ctx.compute.backend, ev.ctx.prompt_token_count - 1);
  }
};

struct prefill_nonflash_runtime_required {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !prefill_flash_runtime_supported{}(ev, ctx);
  }
};

struct prefill_chunk4_q8_gemm_supported {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::uses_prefill_chunk4_q8_gemm(ev, ctx);
  }
};

struct prefill_chunk4_q8_gemm_required {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !prefill_chunk4_q8_gemm_supported{}(ev, ctx);
  }
};

struct compute_uses_materialized_logits_with_prefill_chunk4_q8_gemm {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return compute_uses_materialized_logits{}(ev, ctx) &&
        prefill_chunk4_q8_gemm_supported{}(ev, ctx);
  }
};

struct compute_uses_materialized_logits_with_prefill_scalar_runtime {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return compute_uses_materialized_logits{}(ev, ctx) &&
        prefill_chunk4_q8_gemm_required{}(ev, ctx);
  }
};

struct compute_uses_preselected_argmax_direct_with_prefill_chunk4_q8_gemm {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return compute_uses_preselected_argmax_direct{}(ev, ctx) &&
        prefill_chunk4_q8_gemm_supported{}(ev, ctx);
  }
};

struct compute_uses_preselected_argmax_direct_with_prefill_scalar_runtime {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return compute_uses_preselected_argmax_direct{}(ev, ctx) &&
        prefill_chunk4_q8_gemm_required{}(ev, ctx);
  }
};

struct prefill_compute_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct prefill_compute_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::graph_invalid_code);
  }
};

struct prefill_compute_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::graph_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::graph_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct decode_slots_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct decode_slots_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::memory_invalid_code);
  }
};

struct decode_slots_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::memory_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::memory_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct decode_snapshot_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct decode_snapshot_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::memory_invalid_code);
  }
};

struct decode_snapshot_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::memory_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::memory_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct decode_flash_runtime_supported {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return ev.ctx.kv_tokens >= 0 &&
           emel::generator::detail::flash_attention_supported(
               ctx.compute.backend, ev.ctx.kv_tokens);
  }
};

struct decode_nonflash_runtime_required {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !decode_flash_runtime_supported{}(ev, ctx);
  }
};

struct decode_compute_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct decode_compute_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::graph_invalid_code);
  }
};

struct decode_compute_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::graph_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::graph_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct decode_sample_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct decode_sample_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::sampler_invalid_code);
  }
};

struct decode_sample_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::sampler_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::sampler_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct decode_uses_materialized_logits {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return ctx.state.selection_mode == emel::generator::selection_mode::sample_logits;
  }
};

struct decode_uses_preselected_argmax {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return ctx.state.selection_mode == emel::generator::selection_mode::preselected_argmax;
  }
};

struct decode_render_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct decode_render_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::renderer_invalid_code);
  }
};

struct decode_render_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::renderer_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::renderer_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct flush_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct flush_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_invalid_result(ev, detail::renderer_invalid_code);
  }
};

struct flush_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    const bool invalid = detail::renderer_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::renderer_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct decode_should_continue {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::result_none(ev) &&
           ev.ctx.render_status == emel::text::renderer::sequence_status::running &&
           ev.ctx.tokens_generated < ev.ctx.target_tokens &&
           !detail::sampled_stop_token(ev, ctx);
  }
};

struct decode_complete {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::result_none(ev) && !decode_should_continue{}(ev, ctx);
  }
};

struct initialize_result_none {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::result_none(ev);
  }
};

struct initialize_result_invalid_request {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::result_invalid_request(ev);
  }
};

struct initialize_result_backend {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::result_backend(ev);
  }
};

struct has_initialize_done_callback {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::has_done_callback(ev);
  }
};

struct no_initialize_done_callback {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    return !has_initialize_done_callback{}(ev, ctx);
  }
};

struct has_initialize_error_callback {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::has_error_callback(ev);
  }
};

struct no_initialize_error_callback {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    return !has_initialize_error_callback{}(ev, ctx);
  }
};

struct has_initialize_error_out {
  bool operator()(const event::initialize_run & ev, const action::context &) const noexcept {
    return detail::has_error_out(ev);
  }
};

struct no_initialize_error_out {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    return !has_initialize_error_out{}(ev, ctx);
  }
};

struct initialize_done_callback_with_error_out {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    return has_initialize_done_callback{}(ev, ctx) && has_initialize_error_out{}(ev, ctx);
  }
};

struct initialize_done_callback_without_error_out {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    return has_initialize_done_callback{}(ev, ctx) && no_initialize_error_out{}(ev, ctx);
  }
};

struct initialize_no_done_callback_with_error_out {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    return no_initialize_done_callback{}(ev, ctx) && has_initialize_error_out{}(ev, ctx);
  }
};

struct initialize_no_done_callback_without_error_out {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    return no_initialize_done_callback{}(ev, ctx) && no_initialize_error_out{}(ev, ctx);
  }
};

struct initialize_error_callback_with_error_out {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    return has_initialize_error_callback{}(ev, ctx) && has_initialize_error_out{}(ev, ctx);
  }
};

struct initialize_error_callback_without_error_out {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    return has_initialize_error_callback{}(ev, ctx) && no_initialize_error_out{}(ev, ctx);
  }
};

struct initialize_no_error_callback_with_error_out {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    return no_initialize_error_callback{}(ev, ctx) && has_initialize_error_out{}(ev, ctx);
  }
};

struct initialize_no_error_callback_without_error_out {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    return no_initialize_error_callback{}(ev, ctx) && no_initialize_error_out{}(ev, ctx);
  }
};

struct generate_result_none {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::result_none(ev);
  }
};

struct generate_result_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::result_invalid_request(ev);
  }
};

struct generate_result_backend {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::result_backend(ev);
  }
};

struct has_generate_done_callback {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_done_callback(ev);
  }
};

struct no_generate_done_callback {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !has_generate_done_callback{}(ev, ctx);
  }
};

struct has_generate_error_callback {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_error_callback(ev);
  }
};

struct no_generate_error_callback {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !has_generate_error_callback{}(ev, ctx);
  }
};

struct has_generate_error_out {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::has_error_out(ev);
  }
};

struct no_generate_error_out {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !has_generate_error_out{}(ev, ctx);
  }
};

struct generate_done_callback_with_error_out {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return has_generate_done_callback{}(ev, ctx) && has_generate_error_out{}(ev, ctx);
  }
};

struct generate_done_callback_without_error_out {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return has_generate_done_callback{}(ev, ctx) && no_generate_error_out{}(ev, ctx);
  }
};

struct generate_no_done_callback_with_error_out {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return no_generate_done_callback{}(ev, ctx) && has_generate_error_out{}(ev, ctx);
  }
};

struct generate_no_done_callback_without_error_out {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return no_generate_done_callback{}(ev, ctx) && no_generate_error_out{}(ev, ctx);
  }
};

struct generate_error_callback_with_error_out {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return has_generate_error_callback{}(ev, ctx) && has_generate_error_out{}(ev, ctx);
  }
};

struct generate_error_callback_without_error_out {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return has_generate_error_callback{}(ev, ctx) && no_generate_error_out{}(ev, ctx);
  }
};

struct generate_no_error_callback_with_error_out {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return no_generate_error_callback{}(ev, ctx) && has_generate_error_out{}(ev, ctx);
  }
};

struct generate_no_error_callback_without_error_out {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return no_generate_error_callback{}(ev, ctx) && no_generate_error_out{}(ev, ctx);
  }
};

}  // namespace emel::generator::guard
