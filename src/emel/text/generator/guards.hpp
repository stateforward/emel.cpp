#pragma once

#include "emel/batch/planner/errors.hpp"
#include "emel/text/generator/actions.hpp"
#include "emel/text/generator/events.hpp"
#include "emel/graph/errors.hpp"
#include "emel/logits/sampler/errors.hpp"
#include "emel/memory/hybrid/errors.hpp"
#include "emel/text/conditioner/errors.hpp"
#include "emel/text/renderer/errors.hpp"

namespace emel::text::generator::guard {

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

inline bool packed_q8_0_input_path_supported(
    const emel::text::generator::detail::native_backend & backend,
    const emel::text::generator::detail::tensor_matrix & matrix) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64 ||
      backend.packed_q8_0_input_storage.empty() ||
      matrix.tensor == nullptr) {
    return false;
  }

  return emel::kernel::detail::is_packed_q8_0_vector_dtype(
      static_cast<uint8_t>(matrix.tensor->type));
#else
  (void) backend;
  (void) matrix;
  return false;
#endif
}

inline bool q8_input_path_supported(
    const emel::text::generator::detail::native_backend & backend,
    const emel::text::generator::detail::tensor_matrix & matrix) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64 ||
      backend.q8_input_storage.empty() ||
      matrix.tensor == nullptr) {
    return false;
  }

  const uint8_t dtype = static_cast<uint8_t>(matrix.tensor->type);
#if defined(__ARM_FEATURE_MATMUL_INT8)
  if (dtype == emel::kernel::detail::dtype_q4_k_x8_bl8) {
    return true;
  }
  if (dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared) {
    return true;
  }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
  if (dtype == emel::kernel::detail::dtype_q4_k_x8_bl4) {
    return true;
  }
  if (dtype == emel::kernel::detail::dtype_q6_k_x8) {
    return true;
  }
#endif
  return false;
#else
  (void) backend;
  (void) matrix;
  return false;
#endif
}

inline bool q8_input_argmax_path_supported(
    const emel::text::generator::detail::native_backend & backend,
    const emel::text::generator::detail::tensor_matrix & matrix) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if (q8_input_path_supported(backend, matrix)) {
    return true;
  }
  return backend.kernel_kind == emel::kernel::kernel_kind::aarch64 &&
      !backend.q8_input_storage.empty() &&
      matrix.tensor != nullptr &&
#if defined(__ARM_FEATURE_MATMUL_INT8)
      static_cast<uint8_t>(matrix.tensor->type) ==
          emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared;
#else
      false;
#endif
#else
  (void) backend;
  (void) matrix;
  return false;
#endif
}

inline bool scalar_matmul_matrix_packed_q8_0_supported(
    const emel::text::generator::detail::native_backend & backend,
    const emel::text::generator::detail::tensor_matrix & matrix) noexcept {
  return packed_q8_0_input_path_supported(backend, matrix);
}

inline bool scalar_matmul_matrix_q8_k_supported(
    const emel::text::generator::detail::native_backend & backend,
    const emel::text::generator::detail::tensor_matrix & matrix) noexcept {
  return q8_input_path_supported(backend, matrix);
}

inline bool scalar_matmul_matrix_native_quantized_supported(
    const emel::text::generator::detail::native_backend & backend,
    const emel::text::generator::detail::tensor_matrix & matrix) noexcept {
  (void) backend;
  return matrix.tensor != nullptr;
}

template <auto matrix_supported_fn>
inline bool scalar_matmul_block_supported(
    const emel::text::generator::detail::native_backend & backend,
    const emel::text::generator::detail::block_weights & block) noexcept {
  const bool residual_ok = block.uses_attention
      ? matrix_supported_fn(backend, block.attention_q) &&
            matrix_supported_fn(backend, block.attention_k) &&
            matrix_supported_fn(backend, block.attention_v) &&
            matrix_supported_fn(backend, block.attention_output)
      : backend.shortconv_kernel_size > 1 &&
            matrix_supported_fn(backend, block.shortconv_in_proj) &&
            matrix_supported_fn(backend, block.shortconv_out_proj) &&
            !block.shortconv_conv.empty();
  return residual_ok &&
      matrix_supported_fn(backend, block.feed_forward_gate) &&
      matrix_supported_fn(backend, block.feed_forward_up) &&
      matrix_supported_fn(backend, block.feed_forward_down);
}

template <auto matrix_supported_fn>
inline bool scalar_matmul_route_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  if (backend.blocks.empty() || backend.n_layer <= 0) {
    return false;
  }
  for (const auto & block : backend.blocks) {
    if (!scalar_matmul_block_supported<matrix_supported_fn>(backend, block)) {
      return false;
    }
  }
  return matrix_supported_fn(backend, backend.output);
}

inline bool scalar_matmul_packed_q8_0_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return scalar_matmul_route_supported<scalar_matmul_matrix_packed_q8_0_supported>(backend);
}

inline bool scalar_matmul_q8_k_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return scalar_matmul_route_supported<scalar_matmul_matrix_q8_k_supported>(backend);
}

inline bool scalar_matmul_native_quantized_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return scalar_matmul_route_supported<scalar_matmul_matrix_native_quantized_supported>(backend);
}

// Streamed decode executes the per-layer matmuls on raw GGUF bytes copied
// into window slots, so streamed route classification must scan the raw
// stream records captured at engage - the resident records can be packed
// repacks on aarch64 whose layout the slots never carry. The logits stage
// keeps reading the resident output matrix, so the output probe stays on
// the bound record.
template <auto matrix_supported_fn>
inline bool scalar_matmul_stream_route_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  if (backend.blocks.empty() || backend.n_layer <= 0 ||
      backend.stream.raw.size() != backend.blocks.size()) {
    return false;
  }
  for (size_t layer = 0; layer < backend.blocks.size(); ++layer) {
    const auto & block = backend.blocks[layer];
    const auto & raw = backend.stream.raw[layer];
    const auto probe_ok = [&](const emel::text::generator::detail::tensor_matrix & bound,
                              const size_t role) noexcept {
      emel::text::generator::detail::tensor_matrix probe = bound;
      if (raw[role] != nullptr) {
        probe.tensor = raw[role];
      }
      return matrix_supported_fn(backend, probe);
    };
    namespace gd = emel::text::generator::detail;
    const bool residual_ok = block.uses_attention
        ? probe_ok(block.attention_q, gd::k_stream_role_attention_q) &&
              probe_ok(block.attention_k, gd::k_stream_role_attention_k) &&
              probe_ok(block.attention_v, gd::k_stream_role_attention_v) &&
              probe_ok(block.attention_output, gd::k_stream_role_attention_output)
        : backend.shortconv_kernel_size > 1 &&
              probe_ok(block.shortconv_in_proj, gd::k_stream_role_shortconv_in_proj) &&
              probe_ok(block.shortconv_out_proj, gd::k_stream_role_shortconv_out_proj) &&
              !block.shortconv_conv.empty();
    if (!residual_ok ||
        !probe_ok(block.feed_forward_gate, gd::k_stream_role_feed_forward_gate) ||
        !probe_ok(block.feed_forward_up, gd::k_stream_role_feed_forward_up) ||
        !probe_ok(block.feed_forward_down, gd::k_stream_role_feed_forward_down)) {
      return false;
    }
  }
  emel::text::generator::detail::tensor_matrix output_probe = backend.output;
  if (backend.stream.raw_output != nullptr) {
    output_probe.tensor = backend.stream.raw_output;
  }
  return matrix_supported_fn(backend, output_probe);
}

// Streamed logits/argmax output-stage probes: the streamed drivers bind the
// raw output records for the whole step, so the q8-output split for streamed
// rows classifies the raw record, not the packed/prepared resident one.
inline bool stream_materialized_output_q8_k_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  emel::text::generator::detail::tensor_matrix probe = backend.output;
  if (backend.stream.raw_output != nullptr) {
    probe.tensor = backend.stream.raw_output;
  }
  return q8_input_path_supported(backend, probe);
}

inline bool stream_preselected_argmax_output_q8_k_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  emel::text::generator::detail::tensor_matrix probe =
      backend.output_argmax.tensor != nullptr ? backend.output_argmax : backend.output;
  if (backend.stream.raw_output_argmax != nullptr) {
    probe.tensor = backend.stream.raw_output_argmax;
  }
  return q8_input_argmax_path_supported(backend, probe);
}

inline bool scalar_matmul_stream_packed_q8_0_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return scalar_matmul_stream_route_supported<scalar_matmul_matrix_packed_q8_0_supported>(
      backend);
}

inline bool scalar_matmul_stream_q8_k_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return scalar_matmul_stream_route_supported<scalar_matmul_matrix_q8_k_supported>(backend);
}

inline bool scalar_matmul_stream_native_quantized_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return scalar_matmul_stream_route_supported<scalar_matmul_matrix_native_quantized_supported>(
      backend);
}

inline bool scalar_preselected_argmax_q8_k_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  const auto & output_matrix =
      backend.output_argmax.tensor != nullptr ? backend.output_argmax : backend.output;
  return scalar_matmul_q8_k_supported(backend) &&
      q8_input_argmax_path_supported(backend, output_matrix);
}

inline bool preselected_argmax_output_q8_k_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  const auto & output_matrix =
      backend.output_argmax.tensor != nullptr ? backend.output_argmax : backend.output;
  return q8_input_argmax_path_supported(backend, output_matrix);
}

inline bool materialized_output_q8_k_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return q8_input_path_supported(backend, backend.output);
}

inline bool q8_input_chunk4_path_supported(
    const emel::text::generator::detail::native_backend & backend,
    const emel::text::generator::detail::tensor_matrix & matrix) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64 ||
      backend.q8_input_chunk4_storage.empty()) {
    return false;
  }
  return q8_input_path_supported(backend, matrix);
#else
  (void) backend;
  (void) matrix;
  return false;
#endif
}

inline bool q8_input_chunk8_path_supported(
    const emel::text::generator::detail::native_backend & backend,
    const emel::text::generator::detail::tensor_matrix & matrix) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64 ||
      backend.q8_input_chunk8_storage.empty() ||
      matrix.tensor == nullptr) {
    return false;
  }

  const uint8_t dtype = static_cast<uint8_t>(matrix.tensor->type);
  return dtype == emel::kernel::detail::dtype_q4_k_x8_bl8 ||
      dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared;
#else
  (void) backend;
  (void) matrix;
  return false;
#endif
}

inline bool packed_q8_0_chunk4_input_path_supported(
    const emel::text::generator::detail::native_backend & backend,
    const emel::text::generator::detail::tensor_matrix & matrix) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64 ||
      backend.packed_q8_0_chunk4_rows.empty() ||
      backend.packed_q8_0_chunk4_input_storage.empty() ||
      matrix.tensor == nullptr) {
    return false;
  }

  return static_cast<uint8_t>(matrix.tensor->type) ==
          emel::kernel::detail::dtype_q8_0_x4_bl8 &&
      (matrix.rows % emel::text::generator::detail::k_prefill_q8_chunk_rows) == 0;
#else
  (void) backend;
  (void) matrix;
  return false;
#endif
}

template <emel::text::generator::detail::chunk4_rhs_route route>
inline bool chunk4_matmul_backend_ready(
    const emel::text::generator::detail::native_backend & backend,
    const emel::text::generator::detail::tensor_matrix & matrix) noexcept {
  if constexpr (route == emel::text::generator::detail::chunk4_rhs_route::packed_q8_0) {
    return packed_q8_0_chunk4_input_path_supported(backend, matrix);
  } else {
    return q8_input_chunk4_path_supported(backend, matrix);
  }
}

inline int32_t max_q_dim(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return backend.max_q_dim > 0 ? backend.max_q_dim : backend.n_head * backend.head_dim;
}

inline int32_t max_kv_dim(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return backend.max_kv_dim > 0 ? backend.max_kv_dim : backend.n_head_kv * backend.head_dim_kv;
}

inline int32_t max_ffn_dim(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return backend.max_ffn_dim > 0
      ? backend.max_ffn_dim
      : (backend.blocks.empty() ? 0 : backend.blocks.front().feed_forward_gate.rows);
}

template <emel::text::generator::detail::chunk4_rhs_route route>
inline bool prefill_chunk4_backend_ready(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  const int32_t route_max_q_dim = max_q_dim(backend);
  const int32_t route_max_kv_dim = max_kv_dim(backend);
  const int32_t route_max_ffn_dim = max_ffn_dim(backend);
  if (backend.blocks.empty() || backend.n_layer <= 0) {
    return false;
  }

  for (const auto & block : backend.blocks) {
    const bool residual_ok = block.uses_attention
        ? chunk4_matmul_backend_ready<route>(backend, block.attention_q) &&
              chunk4_matmul_backend_ready<route>(backend, block.attention_k) &&
              chunk4_matmul_backend_ready<route>(backend, block.attention_v) &&
              chunk4_matmul_backend_ready<route>(backend, block.attention_output)
        : backend.shortconv_kernel_size > 1 &&
              chunk4_matmul_backend_ready<route>(backend, block.shortconv_in_proj) &&
              chunk4_matmul_backend_ready<route>(backend, block.shortconv_out_proj) &&
              !block.shortconv_conv.empty();
    if (!residual_ok ||
        !chunk4_matmul_backend_ready<route>(backend, block.feed_forward_gate) ||
        !chunk4_matmul_backend_ready<route>(backend, block.feed_forward_down) ||
        !chunk4_matmul_backend_ready<route>(backend, block.feed_forward_up)) {
      return false;
    }
  }

  const bool shortconv_ok = backend.shortconv_state_size == 0 ||
      (backend.shortconv_bcx_chunk4.size() ==
           static_cast<size_t>(emel::text::generator::detail::k_prefill_q8_chunk_rows) *
               static_cast<size_t>(3 * backend.n_embd) &&
       backend.shortconv_conv_out_chunk4.size() == backend.hidden_chunk4.size());
  return shortconv_ok &&
      backend.hidden_chunk4.size() ==
          static_cast<size_t>(emel::text::generator::detail::k_prefill_q8_chunk_rows) *
              static_cast<size_t>(backend.n_embd) &&
      backend.norm_chunk4.size() == backend.hidden_chunk4.size() &&
      backend.projected_chunk4.size() == backend.hidden_chunk4.size() &&
      backend.attn_ctx_chunk4.size() ==
          static_cast<size_t>(emel::text::generator::detail::k_prefill_q8_chunk_rows) *
              static_cast<size_t>(route_max_q_dim) &&
      backend.q_chunk4.size() == backend.attn_ctx_chunk4.size() &&
      backend.k_chunk4.size() ==
          static_cast<size_t>(emel::text::generator::detail::k_prefill_q8_chunk_rows) *
              static_cast<size_t>(route_max_kv_dim) &&
      backend.v_chunk4.size() == backend.k_chunk4.size() &&
      backend.gate_chunk4.size() ==
          static_cast<size_t>(emel::text::generator::detail::k_prefill_q8_chunk_rows) *
              static_cast<size_t>(route_max_ffn_dim) &&
      backend.up_chunk4.size() == backend.gate_chunk4.size() &&
      backend.ffn_hidden_chunk4.size() == backend.gate_chunk4.size();
}

inline bool preselected_argmax_direct_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept;

inline bool uses_preselected_argmax_direct(const action::context & ctx) noexcept {
  return ctx.state.selection_mode == emel::text::generator::selection_mode::preselected_argmax &&
      preselected_argmax_direct_supported(ctx.compute.backend);
}

inline bool preselected_argmax_direct_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64 ||
      backend.output_argmax.tensor == nullptr ||
      backend.q8_input_storage.empty()) {
    return false;
  }

  const uint8_t dtype = static_cast<uint8_t>(backend.output_argmax.tensor->type);
#if defined(__ARM_FEATURE_DOTPROD)
  if (dtype == emel::kernel::detail::dtype_q6_k_x8) {
    return true;
  }
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
  if (dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared ||
      dtype == emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared) {
    return true;
  }
#endif
  return false;
#else
  (void) backend;
  return false;
#endif
}

inline bool prefill_chunk4_q8_gemm_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return prefill_chunk4_backend_ready<
             emel::text::generator::detail::chunk4_rhs_route::packed_q8_0>(backend) ||
         prefill_chunk4_backend_ready<
             emel::text::generator::detail::chunk4_rhs_route::q8_k>(backend);
}

inline bool prefill_chunk8_q8_k_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  const int32_t route_max_q_dim = max_q_dim(backend);
  const int32_t route_max_kv_dim = max_kv_dim(backend);
  const int32_t route_max_ffn_dim = max_ffn_dim(backend);
  if (backend.blocks.empty() || backend.n_layer <= 0) {
    return false;
  }

  for (const auto & block : backend.blocks) {
    const bool residual_ok = block.uses_attention
        ? q8_input_chunk8_path_supported(backend, block.attention_q) &&
              q8_input_chunk8_path_supported(backend, block.attention_k) &&
              q8_input_chunk8_path_supported(backend, block.attention_v) &&
              q8_input_chunk8_path_supported(backend, block.attention_output)
        : backend.shortconv_kernel_size > 1 &&
              q8_input_chunk8_path_supported(backend, block.shortconv_in_proj) &&
              q8_input_chunk8_path_supported(backend, block.shortconv_out_proj) &&
              !block.shortconv_conv.empty();
    if (!residual_ok ||
        !q8_input_chunk8_path_supported(backend, block.feed_forward_gate) ||
        !q8_input_chunk8_path_supported(backend, block.feed_forward_down) ||
        !q8_input_chunk8_path_supported(backend, block.feed_forward_up)) {
      return false;
    }
  }

  const bool shortconv_ok = backend.shortconv_state_size == 0 ||
      (backend.shortconv_bcx_chunk8.size() ==
           static_cast<size_t>(emel::text::generator::detail::k_prefill_q8_chunk8_rows) *
               static_cast<size_t>(3 * backend.n_embd) &&
       backend.shortconv_conv_out_chunk8.size() == backend.hidden_chunk8.size());
  return shortconv_ok &&
      backend.hidden_chunk8.size() ==
          static_cast<size_t>(emel::text::generator::detail::k_prefill_q8_chunk8_rows) *
              static_cast<size_t>(backend.n_embd) &&
      backend.norm_chunk8.size() == backend.hidden_chunk8.size() &&
      backend.projected_chunk8.size() == backend.hidden_chunk8.size() &&
      backend.attn_ctx_chunk8.size() ==
          static_cast<size_t>(emel::text::generator::detail::k_prefill_q8_chunk8_rows) *
              static_cast<size_t>(route_max_q_dim) &&
      backend.q_chunk8.size() == backend.attn_ctx_chunk8.size() &&
      backend.k_chunk8.size() ==
          static_cast<size_t>(emel::text::generator::detail::k_prefill_q8_chunk8_rows) *
              static_cast<size_t>(route_max_kv_dim) &&
      backend.v_chunk8.size() == backend.k_chunk8.size() &&
      backend.gate_chunk8.size() ==
          static_cast<size_t>(emel::text::generator::detail::k_prefill_q8_chunk8_rows) *
              static_cast<size_t>(route_max_ffn_dim) &&
      backend.up_chunk8.size() == backend.gate_chunk8.size() &&
      backend.ffn_hidden_chunk8.size() == backend.gate_chunk8.size();
}

inline bool prefill_chunk4_packed_q8_0_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return prefill_chunk4_backend_ready<
      emel::text::generator::detail::chunk4_rhs_route::packed_q8_0>(backend);
}

inline bool prefill_chunk4_q8_k_supported(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return prefill_chunk4_backend_ready<
      emel::text::generator::detail::chunk4_rhs_route::q8_k>(backend);
}

inline const emel::text::generator::detail::block_weights * guard_first_flash_attention_block(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  for (const auto & block : backend.blocks) {
    if (block.uses_attention) {
      return &block;
    }
  }

  return nullptr;
}

inline bool guard_flash_attention_supported(
    const emel::text::generator::detail::native_backend & backend,
    const int32_t position) noexcept {
  if (backend.n_layer <= 0 || position < 0 || position >= backend.n_ctx) {
    return false;
  }

  emel::text::generator::detail::block_weights fallback_block{};
  const auto * attention_block = guard_first_flash_attention_block(backend);
  if (attention_block == nullptr) {
    fallback_block.uses_attention = true;
    fallback_block.attention_q_dim = backend.n_head * backend.head_dim;
    fallback_block.attention_kv_dim = backend.n_head_kv * backend.head_dim_kv;
    fallback_block.attention_head_dim = backend.head_dim;
    fallback_block.attention_head_dim_kv = backend.head_dim_kv;
    attention_block = &fallback_block;
  }

  return emel::kernel::detail::can_run_flash_attn_ext(
      emel::text::generator::detail::make_flash_attn_request(
          backend, *attention_block, 0, position));
}

inline bool uses_prefill_chunk4_q8_gemm(const event::generate_run & ev,
                                        const action::context & ctx) noexcept {
  return ev.ctx.prompt_token_count >= emel::text::generator::detail::k_prefill_q8_chunk_rows &&
      prefill_chunk4_q8_gemm_supported(ctx.compute.backend);
}

inline bool uses_prefill_chunk8_q8_gemm(const event::generate_run & ev,
                                        const action::context & ctx) noexcept {
  return ev.ctx.prompt_token_count >= emel::text::generator::detail::k_prefill_q8_chunk8_rows &&
      prefill_chunk8_q8_k_supported(ctx.compute.backend);
}

inline bool prefill_contract_uses_materialized_logits(
    const emel::text::generator::prefill_compute_contract contract) noexcept {
  return contract == emel::text::generator::prefill_compute_contract::flash_materialized_scalar ||
         contract == emel::text::generator::prefill_compute_contract::flash_materialized_chunk8_q8_k ||
         contract ==
             emel::text::generator::prefill_compute_contract::flash_materialized_chunk4_packed_q8_0 ||
         contract == emel::text::generator::prefill_compute_contract::flash_materialized_chunk4_q8_k ||
         contract == emel::text::generator::prefill_compute_contract::nonflash_materialized_scalar ||
         contract == emel::text::generator::prefill_compute_contract::nonflash_materialized_chunk8_q8_k ||
         contract == emel::text::generator::prefill_compute_contract::
                         nonflash_materialized_chunk4_packed_q8_0 ||
         contract == emel::text::generator::prefill_compute_contract::nonflash_materialized_chunk4_q8_k;
}

inline bool prefill_contract_uses_preselected_argmax(
    const emel::text::generator::prefill_compute_contract contract) noexcept {
  return contract == emel::text::generator::prefill_compute_contract::flash_preselected_scalar ||
         contract == emel::text::generator::prefill_compute_contract::flash_preselected_chunk8_q8_k ||
         contract ==
             emel::text::generator::prefill_compute_contract::flash_preselected_chunk4_packed_q8_0 ||
         contract == emel::text::generator::prefill_compute_contract::flash_preselected_chunk4_q8_k ||
         contract == emel::text::generator::prefill_compute_contract::nonflash_preselected_scalar ||
         contract == emel::text::generator::prefill_compute_contract::nonflash_preselected_chunk8_q8_k ||
         contract == emel::text::generator::prefill_compute_contract::
                         nonflash_preselected_chunk4_packed_q8_0 ||
         contract == emel::text::generator::prefill_compute_contract::nonflash_preselected_chunk4_q8_k;
}

inline bool guard_compute_backend_shape_ready(
    const emel::text::generator::detail::native_backend & backend) noexcept {
  return backend.model != nullptr &&
         backend.n_embd > 0 &&
         backend.n_head > 0 &&
         backend.n_head_kv > 0 &&
         backend.n_layer > 0 &&
         backend.n_vocab > 0 &&
         backend.n_ctx > 0 &&
         backend.head_dim > 0 &&
         backend.head_dim_kv > 0 &&
         backend.blocks.size() == static_cast<size_t>(backend.n_layer);
}

inline bool guard_compute_backend_ready(const action::context & ctx) noexcept {
  return ctx.compute.backend_ready && guard_compute_backend_shape_ready(ctx.compute.backend);
}

inline bool guard_graph_reservation_ready(const action::context & ctx) noexcept {
  const auto & reservation = ctx.state.graph_reservation;
  return reservation.lifecycle != nullptr &&
         reservation.lifecycle->tensors != nullptr &&
         reservation.lifecycle->tensor_count > 0 &&
         reservation.node_count > 0u &&
         reservation.tensor_count > 0u;
}

inline bool guard_compute_lifecycle_ready(
    const emel::graph::processor::event::lifecycle_manifest & lifecycle) noexcept {
  return lifecycle.tensors != nullptr &&
         lifecycle.tensor_count > 0 &&
         lifecycle.phase != nullptr;
}

inline bool guard_step_plan_ready(
    const emel::text::generator::detail::step_plan & plan,
    const emel::text::generator::detail::step_kind kind) noexcept {
  return plan.graph != nullptr &&
         plan.graph->execution != nullptr &&
         plan.kind == kind &&
         plan.expected_outputs >= 0 &&
         plan.max_step_tokens > 0;
}

inline bool guard_materialized_logits_ready(const action::context & ctx) noexcept {
  return ctx.buffers.logits != nullptr &&
         ctx.buffers.vocab_size >= ctx.compute.backend.n_vocab;
}

inline bool guard_materialized_output_ready(const action::context & ctx) noexcept {
  return guard_materialized_logits_ready(ctx) &&
         ctx.compute.backend.bound_logits.size() ==
             static_cast<size_t>(ctx.compute.backend.n_vocab);
}

inline bool guard_bound_request_capacity_ready(const action::context & ctx,
                                               const int32_t token_count) noexcept {
  return token_count > 0 &&
         static_cast<size_t>(token_count) <= ctx.compute.backend.bound_tokens.size() &&
         static_cast<size_t>(token_count) <= ctx.compute.backend.bound_positions.size() &&
         static_cast<size_t>(token_count) <= ctx.buffers.prompt_tokens.size() &&
         static_cast<size_t>(token_count) <= ctx.buffers.positions.size();
}

// Snapshot addressing truth: the per-step captured memory::view::snapshot must
// agree with the prepared backend geometry and account for exactly the tokens
// the compute phase is about to address. Incoherent snapshots fail the ready
// predicates below and route through the existing explicit invalid transitions.
inline bool guard_snapshot_geometry_coherent(const action::context & ctx) noexcept {
  return ctx.state.memory_snapshot.block_tokens == ctx.compute.backend.kv_block_tokens &&
         ctx.state.memory_snapshot.is_sequence_active(action::k_sequence_id);
}

inline bool guard_snapshot_covers_tokens(const action::context & ctx,
                                         const int32_t token_count) noexcept {
  const auto & snapshot = ctx.state.memory_snapshot;
  return token_count > 0 &&
         snapshot.sequence_length(action::k_sequence_id) == token_count &&
         snapshot.lookup_kv_block(action::k_sequence_id, token_count - 1) >= 0;
}

// Flash kernels consume contiguous strided views rooted at each layer's cache
// base, so the flash route additionally requires the snapshot block map to be
// the identity over the tokens it reads. Permuted or offset mappings take the
// scalar span-walking route through the existing non-flash transitions.
inline bool guard_flash_kv_map_identity(const action::context & ctx,
                                        const int32_t token_count) noexcept {
  const auto & snapshot = ctx.state.memory_snapshot;
  const int32_t block_count =
      emel::memory::view::blocks_for_tokens(snapshot.block_tokens, token_count);
  for (int32_t block = 0; block < block_count; ++block) {
    if (snapshot.lookup_kv_block(action::k_sequence_id, block * snapshot.block_tokens) !=
        block) {
      return false;
    }
  }
  return true;
}

inline bool guard_prefill_request_ready(const event::generate_ctx & runtime,
                                        const action::context & ctx) noexcept {
  return guard_step_plan_ready(
             ctx.compute.backend.prefill_plan,
             emel::text::generator::detail::step_kind::prefill) &&
         runtime.plan_outputs == ctx.compute.backend.prefill_plan.expected_outputs &&
         runtime.prefill_step_size > 0 &&
         runtime.prefill_step_size <= ctx.compute.backend.prefill_plan.max_step_tokens &&
         runtime.prompt_token_count > 0 &&
         runtime.prompt_token_count <= ctx.limits.prompt_capacity &&
         guard_bound_request_capacity_ready(ctx, runtime.prompt_token_count) &&
         guard_snapshot_geometry_coherent(ctx) &&
         guard_snapshot_covers_tokens(ctx, runtime.prompt_token_count);
}

inline bool guard_decode_request_ready(const event::generate_ctx & runtime,
                                       const action::context & ctx) noexcept {
  return guard_step_plan_ready(
             ctx.compute.backend.decode_plan,
             emel::text::generator::detail::step_kind::decode) &&
         ctx.compute.backend.decode_plan.expected_outputs == 1 &&
         guard_bound_request_capacity_ready(ctx, 1) &&
         runtime.selected_token >= 0 &&
         runtime.selected_token < ctx.compute.backend.token_embedding.rows &&
         runtime.kv_tokens >= 0 &&
         runtime.kv_tokens < ctx.compute.backend.n_ctx &&
         ctx.compute.backend.kv_cache_tokens == runtime.kv_tokens &&
         guard_snapshot_geometry_coherent(ctx) &&
         guard_snapshot_covers_tokens(ctx, runtime.kv_tokens + 1);
}

inline bool guard_prefill_materialized_compute_ready(const event::generate_ctx & runtime,
                                                     const action::context & ctx) noexcept {
  return guard_compute_backend_ready(ctx) &&
         guard_graph_reservation_ready(ctx) &&
         guard_compute_lifecycle_ready(ctx.compute.backend.prefill_lifecycle) &&
         guard_materialized_output_ready(ctx) &&
         guard_prefill_request_ready(runtime, ctx);
}

inline bool guard_prefill_preselected_compute_ready(const event::generate_ctx & runtime,
                                                    const action::context & ctx) noexcept {
  return guard_compute_backend_ready(ctx) &&
         guard_graph_reservation_ready(ctx) &&
         guard_compute_lifecycle_ready(ctx.compute.backend.prefill_lifecycle) &&
         guard_prefill_request_ready(runtime, ctx);
}

inline bool guard_prefill_materialized_compute_invalid(const event::generate_ctx & runtime,
                                                       const action::context & ctx) noexcept {
  return guard_compute_backend_ready(ctx) &&
         guard_graph_reservation_ready(ctx) &&
         guard_compute_lifecycle_ready(ctx.compute.backend.prefill_lifecycle) &&
         (!guard_materialized_output_ready(ctx) || !guard_prefill_request_ready(runtime, ctx));
}

inline bool guard_prefill_preselected_compute_invalid(const event::generate_ctx & runtime,
                                                      const action::context & ctx) noexcept {
  return guard_compute_backend_ready(ctx) &&
         guard_graph_reservation_ready(ctx) &&
         guard_compute_lifecycle_ready(ctx.compute.backend.prefill_lifecycle) &&
         !guard_prefill_request_ready(runtime, ctx);
}

inline bool guard_prefill_compute_backend_unavailable(const action::context & ctx) noexcept {
  return !guard_compute_backend_ready(ctx) ||
         !guard_graph_reservation_ready(ctx) ||
         !guard_compute_lifecycle_ready(ctx.compute.backend.prefill_lifecycle);
}

inline bool guard_decode_materialized_compute_ready(const event::generate_ctx & runtime,
                                                    const action::context & ctx) noexcept {
  return guard_compute_backend_ready(ctx) &&
         guard_graph_reservation_ready(ctx) &&
         guard_compute_lifecycle_ready(ctx.compute.backend.decode_lifecycle) &&
         guard_materialized_output_ready(ctx) &&
         guard_decode_request_ready(runtime, ctx);
}

inline bool guard_decode_preselected_compute_ready(const event::generate_ctx & runtime,
                                                   const action::context & ctx) noexcept {
  return guard_compute_backend_ready(ctx) &&
         guard_graph_reservation_ready(ctx) &&
         guard_compute_lifecycle_ready(ctx.compute.backend.decode_lifecycle) &&
         guard_decode_request_ready(runtime, ctx);
}

inline bool guard_decode_materialized_compute_invalid(const event::generate_ctx & runtime,
                                                      const action::context & ctx) noexcept {
  return guard_compute_backend_ready(ctx) &&
         guard_graph_reservation_ready(ctx) &&
         guard_compute_lifecycle_ready(ctx.compute.backend.decode_lifecycle) &&
         (!guard_materialized_output_ready(ctx) || !guard_decode_request_ready(runtime, ctx));
}

inline bool guard_decode_preselected_compute_invalid(const event::generate_ctx & runtime,
                                                     const action::context & ctx) noexcept {
  return guard_compute_backend_ready(ctx) &&
         guard_graph_reservation_ready(ctx) &&
         guard_compute_lifecycle_ready(ctx.compute.backend.decode_lifecycle) &&
         !guard_decode_request_ready(runtime, ctx);
}

inline bool guard_decode_compute_backend_unavailable(const action::context & ctx) noexcept {
  return !guard_compute_backend_ready(ctx) ||
         !guard_graph_reservation_ready(ctx) ||
         !guard_compute_lifecycle_ready(ctx.compute.backend.decode_lifecycle);
}

}  // namespace detail

struct valid_initialize {
  bool operator()(const event::initialize_run & ev, const action::context & ctx) const noexcept {
    const bool sample_logits =
        ev.request.selection_mode == emel::text::generator::selection_mode::sample_logits;
    const bool preselected_argmax =
        ev.request.selection_mode == emel::text::generator::selection_mode::preselected_argmax;
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

struct compute_scalar_packed_q8_0_supported {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return detail::scalar_matmul_packed_q8_0_supported(ctx.compute.backend);
  }
};

struct compute_scalar_q8_k_supported {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return !detail::scalar_matmul_packed_q8_0_supported(ctx.compute.backend) &&
           detail::scalar_matmul_q8_k_supported(ctx.compute.backend);
  }
};

struct compute_scalar_native_quantized_supported {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return !detail::scalar_matmul_packed_q8_0_supported(ctx.compute.backend) &&
           !detail::scalar_matmul_q8_k_supported(ctx.compute.backend) &&
           detail::scalar_matmul_native_quantized_supported(ctx.compute.backend);
  }
};

struct compute_scalar_kernel_required {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return !detail::scalar_matmul_packed_q8_0_supported(ctx.compute.backend) &&
           !detail::scalar_matmul_q8_k_supported(ctx.compute.backend) &&
           !detail::scalar_matmul_native_quantized_supported(ctx.compute.backend);
  }
};

struct compute_preselected_argmax_q8_k_supported {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return detail::scalar_preselected_argmax_q8_k_supported(ctx.compute.backend);
  }
};

struct compute_preselected_argmax_kernel_required {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !compute_preselected_argmax_q8_k_supported{}(ev, ctx) &&
           !compute_scalar_native_quantized_supported{}(ev, ctx);
  }
};

struct compute_materialized_scalar_packed_q8_0_supported {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return compute_uses_materialized_logits{}(ev, ctx) &&
           compute_scalar_packed_q8_0_supported{}(ev, ctx);
  }
};

struct compute_materialized_scalar_q8_k_supported {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return compute_uses_materialized_logits{}(ev, ctx) &&
           compute_scalar_q8_k_supported{}(ev, ctx);
  }
};

struct compute_materialized_scalar_kernel_required {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return compute_uses_materialized_logits{}(ev, ctx) &&
           compute_scalar_kernel_required{}(ev, ctx);
  }
};

struct compute_materialized_scalar_native_quantized_q8_k_supported {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return compute_uses_materialized_logits{}(ev, ctx) &&
           compute_scalar_native_quantized_supported{}(ev, ctx) &&
           detail::materialized_output_q8_k_supported(ctx.compute.backend);
  }
};

struct compute_materialized_scalar_native_quantized_kernel_required {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return compute_uses_materialized_logits{}(ev, ctx) &&
           compute_scalar_native_quantized_supported{}(ev, ctx) &&
           !detail::materialized_output_q8_k_supported(ctx.compute.backend);
  }
};

struct compute_preselected_argmax_native_quantized_q8_k_supported {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return compute_scalar_native_quantized_supported{}(ev, ctx) &&
           detail::preselected_argmax_output_q8_k_supported(ctx.compute.backend);
  }
};

struct compute_preselected_argmax_native_quantized_kernel_required {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return compute_scalar_native_quantized_supported{}(ev, ctx) &&
           !detail::preselected_argmax_output_q8_k_supported(ctx.compute.backend);
  }
};

struct guard_decode_compute_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return (detail::uses_preselected_argmax_direct(ctx) &&
            detail::guard_decode_preselected_compute_invalid(ev.ctx, ctx)) ||
           (!detail::uses_preselected_argmax_direct(ctx) &&
            detail::guard_decode_materialized_compute_invalid(ev.ctx, ctx));
  }
};

struct guard_decode_preselected_compute_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::guard_decode_preselected_compute_invalid(ev.ctx, ctx);
  }
};

struct guard_decode_compute_backend_unavailable {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return detail::guard_decode_compute_backend_unavailable(ctx);
  }
};

struct guard_decode_parallel_lanes_ready {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return ctx.compute.backend.lane_pool.has_value() &&
           ctx.compute.backend.n_embd >=
               emel::text::generator::detail::k_parallel_min_gemv_dim;
  }
};

struct guard_decode_materialized_scalar_packed_q8_0_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::guard_decode_materialized_compute_ready(ev.ctx, ctx) &&
           compute_materialized_scalar_packed_q8_0_supported{}(ev, ctx);
  }
};

struct guard_decode_materialized_parallel_scalar_packed_q8_0_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_parallel_lanes_ready{}(ev, ctx) &&
           guard_decode_materialized_scalar_packed_q8_0_ready{}(ev, ctx);
  }
};

struct guard_decode_materialized_scalar_q8_k_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::guard_decode_materialized_compute_ready(ev.ctx, ctx) &&
           compute_materialized_scalar_q8_k_supported{}(ev, ctx);
  }
};

struct guard_decode_materialized_parallel_scalar_q8_k_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_parallel_lanes_ready{}(ev, ctx) &&
           guard_decode_materialized_scalar_q8_k_ready{}(ev, ctx);
  }
};

struct guard_decode_materialized_scalar_native_quantized_q8_k_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::guard_decode_materialized_compute_ready(ev.ctx, ctx) &&
           compute_materialized_scalar_native_quantized_q8_k_supported{}(ev, ctx);
  }
};

struct guard_decode_materialized_parallel_scalar_native_quantized_q8_k_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_parallel_lanes_ready{}(ev, ctx) &&
           guard_decode_materialized_scalar_native_quantized_q8_k_ready{}(ev, ctx);
  }
};

struct guard_decode_materialized_scalar_native_quantized_kernel_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::guard_decode_materialized_compute_ready(ev.ctx, ctx) &&
           compute_materialized_scalar_native_quantized_kernel_required{}(ev, ctx);
  }
};

struct guard_decode_materialized_parallel_scalar_native_quantized_kernel_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_parallel_lanes_ready{}(ev, ctx) &&
           guard_decode_materialized_scalar_native_quantized_kernel_ready{}(ev, ctx);
  }
};

struct guard_decode_materialized_scalar_kernel_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::guard_decode_materialized_compute_ready(ev.ctx, ctx) &&
           compute_materialized_scalar_kernel_required{}(ev, ctx);
  }
};

struct guard_decode_materialized_parallel_scalar_kernel_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_parallel_lanes_ready{}(ev, ctx) &&
           guard_decode_materialized_scalar_kernel_ready{}(ev, ctx);
  }
};

// Streaming window engagement: the owner injected a bound tensor window actor
// that reported streaming_active at bind. Mirrors the lane-pool capability
// gate; composed with route readiness below so the streamed route row sits
// above its resident siblings only when the window is live.
struct guard_decode_stream_window_ready {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return ctx.compute.backend.stream.window != nullptr &&
           ctx.compute.backend.stream.active;
  }
};

// Streamed route classes scan the raw stream records instead of the bound
// (possibly packed) resident records: the acquired slots hold raw GGUF
// bytes, so a route selected from the resident class would misread them.
struct compute_stream_scalar_packed_q8_0_supported {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return detail::scalar_matmul_stream_packed_q8_0_supported(ctx.compute.backend);
  }
};

struct compute_stream_scalar_q8_k_supported {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !compute_stream_scalar_packed_q8_0_supported{}(ev, ctx) &&
           detail::scalar_matmul_stream_q8_k_supported(ctx.compute.backend);
  }
};

struct compute_stream_scalar_native_quantized_supported {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !compute_stream_scalar_packed_q8_0_supported{}(ev, ctx) &&
           !detail::scalar_matmul_stream_q8_k_supported(ctx.compute.backend) &&
           detail::scalar_matmul_stream_native_quantized_supported(ctx.compute.backend);
  }
};

struct compute_stream_scalar_kernel_required {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return !detail::scalar_matmul_stream_packed_q8_0_supported(ctx.compute.backend) &&
           !detail::scalar_matmul_stream_q8_k_supported(ctx.compute.backend) &&
           !detail::scalar_matmul_stream_native_quantized_supported(ctx.compute.backend);
  }
};

struct guard_decode_materialized_streamed_scalar_kernel_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_stream_window_ready{}(ev, ctx) &&
           detail::guard_decode_materialized_compute_ready(ev.ctx, ctx) &&
           compute_uses_materialized_logits{}(ev, ctx) &&
           compute_stream_scalar_kernel_required{}(ev, ctx);
  }
};

struct guard_decode_materialized_streamed_scalar_native_quantized_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_stream_window_ready{}(ev, ctx) &&
           detail::guard_decode_materialized_compute_ready(ev.ctx, ctx) &&
           compute_uses_materialized_logits{}(ev, ctx) &&
           compute_stream_scalar_native_quantized_supported{}(ev, ctx) &&
           !detail::stream_materialized_output_q8_k_supported(ctx.compute.backend);
  }
};

struct guard_decode_materialized_streamed_scalar_packed_q8_0_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_stream_window_ready{}(ev, ctx) &&
           detail::guard_decode_materialized_compute_ready(ev.ctx, ctx) &&
           compute_uses_materialized_logits{}(ev, ctx) &&
           compute_stream_scalar_packed_q8_0_supported{}(ev, ctx);
  }
};

struct guard_decode_materialized_streamed_scalar_q8_k_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_stream_window_ready{}(ev, ctx) &&
           detail::guard_decode_materialized_compute_ready(ev.ctx, ctx) &&
           compute_uses_materialized_logits{}(ev, ctx) &&
           compute_stream_scalar_q8_k_supported{}(ev, ctx);
  }
};

struct guard_decode_materialized_streamed_scalar_native_quantized_q8_k_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_stream_window_ready{}(ev, ctx) &&
           detail::guard_decode_materialized_compute_ready(ev.ctx, ctx) &&
           compute_uses_materialized_logits{}(ev, ctx) &&
           compute_stream_scalar_native_quantized_supported{}(ev, ctx) &&
           detail::stream_materialized_output_q8_k_supported(ctx.compute.backend);
  }
};

struct guard_decode_preselected_direct_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::guard_decode_preselected_compute_ready(ev.ctx, ctx) &&
           compute_uses_preselected_argmax_direct{}(ev, ctx);
  }
};

struct guard_decode_preselected_argmax_q8_k_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::guard_decode_preselected_compute_ready(ev.ctx, ctx) &&
           compute_preselected_argmax_q8_k_supported{}(ev, ctx);
  }
};

struct guard_decode_preselected_argmax_native_quantized_q8_k_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::guard_decode_preselected_compute_ready(ev.ctx, ctx) &&
           compute_preselected_argmax_native_quantized_q8_k_supported{}(ev, ctx);
  }
};

struct guard_decode_preselected_argmax_native_quantized_kernel_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::guard_decode_preselected_compute_ready(ev.ctx, ctx) &&
           compute_preselected_argmax_native_quantized_kernel_required{}(ev, ctx);
  }
};

struct guard_decode_preselected_argmax_kernel_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::guard_decode_preselected_compute_ready(ev.ctx, ctx) &&
           compute_preselected_argmax_kernel_required{}(ev, ctx);
  }
};

// Streamed preselected routes: an active tensor window overrides the
// resident/parallel siblings (the streamed decode is serial by construction,
// one layer slot at a time). Like the sample-mode streamed rows, the block
// route class scans the raw stream records; the argmax output stage stays
// resident-classified.
struct guard_decode_preselected_argmax_q8_k_streamed_ready {
  bool operator()(const event::generate_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_decode_stream_window_ready{}(ev, ctx) &&
           detail::guard_decode_preselected_compute_ready(ev.ctx, ctx) &&
           compute_stream_scalar_q8_k_supported{}(ev, ctx) &&
           detail::stream_preselected_argmax_output_q8_k_supported(
               ctx.compute.backend);
  }
};

struct guard_decode_preselected_argmax_native_quantized_q8_k_streamed_ready {
  bool operator()(const event::generate_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_decode_stream_window_ready{}(ev, ctx) &&
           detail::guard_decode_preselected_compute_ready(ev.ctx, ctx) &&
           compute_stream_scalar_native_quantized_supported{}(ev, ctx) &&
           detail::stream_preselected_argmax_output_q8_k_supported(
               ctx.compute.backend);
  }
};

struct guard_decode_preselected_argmax_native_quantized_kernel_streamed_ready {
  bool operator()(const event::generate_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_decode_stream_window_ready{}(ev, ctx) &&
           detail::guard_decode_preselected_compute_ready(ev.ctx, ctx) &&
           compute_stream_scalar_native_quantized_supported{}(ev, ctx) &&
           !detail::stream_preselected_argmax_output_q8_k_supported(
               ctx.compute.backend);
  }
};

struct guard_decode_preselected_argmax_kernel_streamed_ready {
  bool operator()(const event::generate_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_decode_stream_window_ready{}(ev, ctx) &&
           detail::guard_decode_preselected_compute_ready(ev.ctx, ctx) &&
           !(compute_stream_scalar_q8_k_supported{}(ev, ctx) &&
             detail::stream_preselected_argmax_output_q8_k_supported(
                 ctx.compute.backend)) &&
           !compute_stream_scalar_native_quantized_supported{}(ev, ctx);
  }
};

struct guard_decode_preselected_parallel_argmax_q8_k_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_parallel_lanes_ready{}(ev, ctx) &&
           guard_decode_preselected_argmax_q8_k_ready{}(ev, ctx);
  }
};

struct guard_decode_preselected_parallel_argmax_native_quantized_q8_k_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_parallel_lanes_ready{}(ev, ctx) &&
           guard_decode_preselected_argmax_native_quantized_q8_k_ready{}(ev, ctx);
  }
};

struct guard_decode_preselected_parallel_argmax_native_quantized_kernel_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_parallel_lanes_ready{}(ev, ctx) &&
           guard_decode_preselected_argmax_native_quantized_kernel_ready{}(ev, ctx);
  }
};

struct guard_decode_preselected_parallel_argmax_kernel_ready {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return guard_decode_parallel_lanes_ready{}(ev, ctx) &&
           guard_decode_preselected_argmax_kernel_ready{}(ev, ctx);
  }
};

struct valid_generate {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !ev.request.messages.empty() &&
           ev.request.messages.data() != nullptr &&
           ev.request.max_tokens > 0 &&
           ev.request.max_tokens <= ctx.limits.decode_capacity &&
           !ev.request.output.empty() &&
           ev.request.output.data() != nullptr;
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

struct planning_uses_chunk8_prefill {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return detail::uses_prefill_chunk8_q8_gemm(ev, ctx);
  }
};

struct planning_uses_chunk4_prefill {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !planning_uses_chunk8_prefill{}(ev, ctx) &&
        detail::uses_prefill_chunk4_q8_gemm(ev, ctx);
  }
};

struct planning_uses_scalar_prefill {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !planning_uses_chunk8_prefill{}(ev, ctx) &&
        !planning_uses_chunk4_prefill{}(ev, ctx);
  }
};

struct conditioning_ok_with_chunk8_prefill {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return conditioning_ok{}(ev, ctx) && planning_uses_chunk8_prefill{}(ev, ctx);
  }
};

struct conditioning_ok_with_chunk4_prefill {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return conditioning_ok{}(ev, ctx) && planning_uses_chunk4_prefill{}(ev, ctx);
  }
};

struct conditioning_ok_with_scalar_prefill {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return conditioning_ok{}(ev, ctx) && planning_uses_scalar_prefill{}(ev, ctx);
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

struct prefill_dispatch_available {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return ctx.prefill_actor != nullptr && ctx.dispatch_prefill != nullptr;
  }
};

struct prefill_dispatch_unavailable {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !prefill_dispatch_available{}(ev, ctx);
  }
};

struct prefill_result_ok_with_materialized_logits_contract {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::result_none(ev) &&
           detail::prefill_contract_uses_materialized_logits(ev.ctx.prefill_contract);
  }
};

struct prefill_result_ok_with_preselected_argmax_contract {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::result_none(ev) &&
           detail::prefill_contract_uses_preselected_argmax(ev.ctx.prefill_contract);
  }
};

struct prefill_result_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::result_invalid_request(ev);
  }
};

struct prefill_result_backend_error {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return detail::result_backend(ev) || (!ev.ctx.phase_accepted && detail::result_none(ev));
  }
};

struct decode_argmax_ready {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return ctx.buffers.vocab_size > 0 && ctx.buffers.logits != nullptr;
  }
};

struct decode_argmax_invalid_request {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !decode_argmax_ready{}(ev, ctx);
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

struct graph_lifecycle_runtime_tensor_available {
  bool operator()(const event::capture_graph_lifecycle &,
                  const action::context & ctx) const noexcept {
    return ctx.state.graph_reservation.lifecycle != nullptr &&
        ctx.state.graph_reservation.lifecycle->tensor_count > 0;
  }
};

struct graph_lifecycle_runtime_tensor_unavailable {
  bool operator()(const event::capture_graph_lifecycle & ev,
                  const action::context & ctx) const noexcept {
    return !graph_lifecycle_runtime_tensor_available{}(ev, ctx);
  }
};

struct decode_flash_runtime_supported {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return ev.ctx.kv_tokens >= 0 &&
           detail::guard_flash_attention_supported(ctx.compute.backend, ev.ctx.kv_tokens) &&
           detail::guard_flash_kv_map_identity(ctx, ev.ctx.kv_tokens + 1);
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
    return ctx.state.selection_mode == emel::text::generator::selection_mode::sample_logits;
  }
};

struct decode_uses_preselected_argmax {
  bool operator()(const event::generate_run &, const action::context & ctx) const noexcept {
    return ctx.state.selection_mode == emel::text::generator::selection_mode::preselected_argmax;
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

}  // namespace emel::text::generator::guard
