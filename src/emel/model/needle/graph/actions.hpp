#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "emel/kernel/detail.hpp"
#include "emel/model/needle/graph/context.hpp"
#include "emel/model/needle/graph/events.hpp"

namespace emel::model::needle::graph::action {

// Branch-free error fold: a rejected child dispatch sets kernel_rejected on
// the step/init context; guards route on the folded error.
inline void fold_error(emel::error::type &err, const bool ok) noexcept {
  err = emel::error::set(err, static_cast<emel::error::type>(!ok) *
                                  emel::error::cast(error::kernel_rejected));
}

inline std::span<const float> codebook_span(const context &ctx) noexcept {
  return {ctx.bound->geo.codebook.data(), emel::cact::loader::k_codebook_len};
}

inline std::span<const uint8_t> payload_span(const tensor_view &view,
                                             const uint64_t byte_offset,
                                             const uint64_t bytes) noexcept {
  return {view.data + byte_offset, bytes};
}

// All CQ tensors in the supported geometry are 4-bit (validated by
// guard_init_supported); the route is selected by explicit machine states.
template <route_kind route>
inline bool
compute_gemv(context &ctx, const tensor_view &view,
             const emel::kernel::cq::event::prepared_q4_view &prepared,
             const std::span<const float> activation,
             const std::span<float> output) noexcept {
  emel::kernel::cq::event::dispatch_result result{};
  if constexpr (route == route_kind::prepared_avx2) {
    const emel::kernel::cq::event::prepared_gemv_request request{
        prepared,
        codebook_span(ctx),
        activation,
        output,
        std::span<float>{ctx.cq_workspace},
        std::span<float>{ctx.pair_lut},
        std::span<float>{ctx.pair_scratch}};
    return ctx.cq.process_event(
        emel::kernel::cq::event::execute_prepared_pair_lut_q4{request, result});
  } else {
    const emel::kernel::cq::event::gemv_request request{
        view, codebook_span(ctx), activation, output,
        std::span<float>{ctx.cq_workspace}};
    return ctx.cq.process_event(
        emel::kernel::cq::event::execute_scalar_q4{request, result});
  }
}

template <route_kind route>
inline bool
compute_gemv_rows(context &ctx, const tensor_view &view,
                  const emel::kernel::cq::event::prepared_q4_view &prepared,
                  const std::span<const float> activation,
                  const uint32_t row_begin, const uint32_t row_count,
                  const std::span<float> output) noexcept {
  emel::kernel::cq::event::dispatch_result result{};
  if constexpr (route == route_kind::prepared_avx2) {
    const emel::kernel::cq::event::prepared_gemv_rows_request request{
        prepared,
        codebook_span(ctx),
        activation,
        row_begin,
        row_count,
        output,
        std::span<float>{ctx.cq_workspace},
        std::span<float>{ctx.pair_lut},
        std::span<float>{ctx.pair_scratch}};
    return ctx.cq.process_event(
        emel::kernel::cq::event::execute_prepared_pair_lut_rows_q4{request,
                                                                   result});
  } else {
    const emel::kernel::cq::event::gemv_rows_request request{
        view,
        codebook_span(ctx),
        activation,
        row_begin,
        row_count,
        output,
        std::span<float>{ctx.cq_workspace}};
    return ctx.cq.process_event(
        emel::kernel::cq::event::execute_scalar_rows_q4{request, result});
  }
}

template <route_kind route>
inline bool
compute_dequant_row(context &ctx, const tensor_view &view,
                    const emel::kernel::cq::event::prepared_q4_view &prepared,
                    const uint32_t row, const float scale,
                    const std::span<float> output) noexcept {
  emel::kernel::cq::event::dispatch_result result{};
  if constexpr (route == route_kind::prepared_avx2) {
    const emel::kernel::cq::event::prepared_dequant_rows_request request{
        prepared, codebook_span(ctx), row, 1u, scale, output};
    return ctx.cq.process_event(
        emel::kernel::cq::event::execute_prepared_dequant_q4{request, result});
  } else {
    const emel::kernel::cq::event::dequant_rows_request request{
        view, codebook_span(ctx), row, 1u, scale, output};
    return ctx.cq.process_event(
        emel::kernel::cq::event::execute_scalar_dequant_q4{request, result});
  }
}

inline bool
compute_gemv_batch4(context &ctx, const std::span<const float> activation,
                    const emel::kernel::cq::event::prepared_q4_view &first,
                    const std::span<float> first_output,
                    const emel::kernel::cq::event::prepared_q4_view &second,
                    const std::span<float> second_output,
                    const emel::kernel::cq::event::prepared_q4_view &third,
                    const std::span<float> third_output,
                    const emel::kernel::cq::event::prepared_q4_view &fourth,
                    const std::span<float> fourth_output) noexcept {
  const emel::kernel::cq::event::prepared_gemv_batch4_request request{
      .targets = {{{&first, first_output},
                   {&second, second_output},
                   {&third, third_output},
                   {&fourth, fourth_output}}},
      .codebook = codebook_span(ctx),
      .activation = activation,
      .workspace = std::span<float>{ctx.cq_workspace},
      .pair_lut = std::span<float>{ctx.pair_lut},
      .pair_scratch = std::span<float>{ctx.pair_scratch}};
  emel::kernel::cq::event::dispatch_result result{};
  return ctx.cq.process_event(
      emel::kernel::cq::event::execute_prepared_pair_lut_batch4_q4{request,
                                                                   result});
}

inline bool compute_zcrms_norm(context &ctx, const std::span<const float> input,
                               const std::span<const float> scale,
                               const uint32_t rows, const uint32_t dim,
                               const std::span<float> output) noexcept {
  const emel::kernel::zcrms::event::norm_rows_request request{.input = input,
                                                              .scale = scale,
                                                              .rows = rows,
                                                              .dim = dim,
                                                              .output = output};
  emel::kernel::zcrms::event::dispatch_result result{};
  return ctx.zcrms.process_event(
      emel::kernel::zcrms::event::execute_norm_rows{request, result});
}

inline bool compute_rms_unit(context &ctx, const std::span<const float> input,
                             const uint32_t dim,
                             const std::span<float> output) noexcept {
  const emel::kernel::zcrms::event::unit_rows_request request{
      .input = input, .rows = 1u, .dim = dim, .output = output};
  emel::kernel::zcrms::event::dispatch_result result{};
  return ctx.zcrms.process_event(
      emel::kernel::zcrms::event::execute_unit_rows{request, result});
}

// Init helpers prepare exact packed CQ4 bytes, decode the fp16 scale tensors,
// precompute RoPE, and clear mutable state.
inline bool prepare_view(context &ctx, const tensor_view &view,
                         emel::kernel::cq::event::prepared_q4_view &prepared,
                         size_t &packed_offset) noexcept {
  const size_t packed_row = context::compute_in_pad(view) / 2u;
  const size_t packed_count =
      static_cast<size_t>(view.shape[0] / 8u * 8u) * packed_row;
  const emel::kernel::cq::event::prepare_q4_request request{
      .weights = view,
      .packed_by_pair8 = std::span<uint8_t>{ctx.prepared_packed}.subspan(
          packed_offset, packed_count),
      .prepared = prepared};
  emel::kernel::cq::event::dispatch_result result{};
  const bool ok = ctx.cq.process_event(
      emel::kernel::cq::event::prepare_q4{request, result});
  packed_offset += packed_count;
  return ok;
}

inline bool prepare_graph_weights(context &ctx) noexcept {
  const auto &bound = *ctx.bound;
  size_t packed_offset = 0u;
  bool ok =
      prepare_view(ctx, bound.embedding, ctx.prepared_embedding, packed_offset);
  for (uint32_t i = 0u; i < bound.layer_count; ++i) {
    const auto &layer = bound.layers[i];
    auto &prepared = ctx.prepared_layers[i];
    ok = ok && prepare_view(ctx, layer.q_proj, prepared.q_proj, packed_offset);
    ok = ok && prepare_view(ctx, layer.k_proj, prepared.k_proj, packed_offset);
    ok = ok && prepare_view(ctx, layer.v_proj, prepared.v_proj, packed_offset);
    ok = ok &&
         prepare_view(ctx, layer.gate_proj, prepared.gate_proj, packed_offset);
    ok = ok &&
         prepare_view(ctx, layer.out_proj, prepared.out_proj, packed_offset);
  }
  ok = ok && prepare_view(ctx, bound.mhc.phi_pre, ctx.prepared_mhc.phi_pre,
                          packed_offset);
  ok = ok && prepare_view(ctx, bound.mhc.phi_post, ctx.prepared_mhc.phi_post,
                          packed_offset);
  ok = ok && prepare_view(ctx, bound.mhc.phi_res, ctx.prepared_mhc.phi_res,
                          packed_offset);
  for (uint32_t i = 0u; i < bound.engram_site_count; ++i) {
    const auto &site = bound.engram_sites[i];
    auto &prepared = ctx.prepared_engram_sites[i];
    ok = ok && prepare_view(ctx, site.tables, prepared.tables, packed_offset);
    ok = ok &&
         prepare_view(ctx, site.key_proj, prepared.key_proj, packed_offset);
    ok = ok &&
         prepare_view(ctx, site.value_proj, prepared.value_proj, packed_offset);
  }
  return ok && packed_offset == ctx.prepared_packed.size();
}

inline bool compute_init(context &ctx) noexcept {
  namespace quant = emel::kernel::detail::quant;
  const auto &bound = *ctx.bound;
  const auto &geo = bound.geo;
  const uint32_t d_model = geo.d_model;
  for (uint32_t i = 0u; i < bound.layer_count; ++i) {
    const auto &layer = bound.layers[i];
    const uint16_t *norm_in =
        reinterpret_cast<const uint16_t *>(layer.norm_in.data);
    const uint16_t *post_norm =
        reinterpret_cast<const uint16_t *>(layer.post_norm.data);
    const uint16_t *pre_hada =
        reinterpret_cast<const uint16_t *>(layer.pre_hada.data);
    for (uint32_t c = 0u; c < d_model; ++c) {
      ctx.norm_in_scale[static_cast<size_t>(i) * d_model + c] =
          quant::fp16_to_fp32(norm_in[c]);
      ctx.post_norm_scale[static_cast<size_t>(i) * d_model + c] =
          quant::fp16_to_fp32(post_norm[c]);
      ctx.pre_hada_scale[static_cast<size_t>(i) * d_model + c] =
          quant::fp16_to_fp32(pre_hada[c]);
    }
    const uint16_t *q_norm =
        reinterpret_cast<const uint16_t *>(layer.q_norm.data);
    const uint16_t *k_norm =
        reinterpret_cast<const uint16_t *>(layer.k_norm.data);
    for (uint32_t c = 0u; c < geo.head_dim; ++c) {
      ctx.q_norm_scale[static_cast<size_t>(i) * geo.head_dim + c] =
          quant::fp16_to_fp32(q_norm[c]);
      ctx.k_norm_scale[static_cast<size_t>(i) * geo.head_dim + c] =
          quant::fp16_to_fp32(k_norm[c]);
    }
    ctx.attn_gate_scale[i] = quant::fp16_to_fp32(
        reinterpret_cast<const uint16_t *>(layer.attn_gate.data)[0]);
  }
  const uint16_t *final_norm =
      reinterpret_cast<const uint16_t *>(bound.final_norm.data);
  for (uint32_t c = 0u; c < d_model; ++c)
    ctx.final_norm_scale[c] = quant::fp16_to_fp32(final_norm[c]);

  const emel::kernel::rope::event::precompute_request rope_request{
      .theta = geo.rope_theta,
      .head_dim = geo.head_dim,
      .positions = geo.max_seq_len,
      .cos_out = ctx.rope_cos,
      .sin_out = ctx.rope_sin};
  emel::kernel::rope::event::dispatch_result rope_result{};
  const bool rope_ok = ctx.rope.process_event(
      emel::kernel::rope::event::execute_precompute{rope_request, rope_result});

  ctx.position = 0u;
  for (auto &value : ctx.key_cache)
    value = 0.0f;
  for (auto &value : ctx.value_cache)
    value = 0.0f;
  for (auto &value : ctx.lanes)
    value = 0.0f;
  for (auto &token : ctx.history_tokens)
    token = 0;
  for (auto &valid : ctx.history_valid)
    valid = 0u;
  return rope_ok && prepare_graph_weights(ctx);
}

// Step begin: record the token in the engram history, gather its embedding
// row scaled by sqrt(d_model), and broadcast it across the mHC lanes.
template <route_kind route>
inline bool compute_step_begin(context &ctx, event::step_ctx &step) noexcept {
  const auto &geo = ctx.bound->geo;
  const uint32_t d_model = geo.d_model;
  ctx.history_tokens[ctx.position] = step.token;
  ctx.history_valid[ctx.position] = 1u;
  const float scale = std::sqrt(static_cast<float>(d_model));
  const bool embed_ok = compute_dequant_row<route>(
      ctx, ctx.bound->embedding, ctx.prepared_embedding,
      static_cast<uint32_t>(step.token), scale, std::span<float>{ctx.mean});
  for (uint32_t lane = 0u; lane < geo.mhc_lanes; ++lane)
    for (uint32_t c = 0u; c < d_model; ++c)
      ctx.lanes[static_cast<size_t>(lane) * d_model + c] = ctx.mean[c];
  step.layer_index = 0u;
  return embed_ok;
}

// Engram K/V for the current position: FNV-mix hash over the token window,
// masked table-row gathers, key/value projections, and the dilated causal
// tap convolution — one K row and one V row per site.
template <route_kind route> inline bool compute_engram(context &ctx) noexcept {
  const auto &bound = *ctx.bound;
  const auto &geo = bound.geo;
  const uint32_t d_model = geo.d_model;
  const uint32_t max_order =
      static_cast<uint32_t>(context::compute_max_order(geo));
  const uint32_t window = geo.engram_conv_taps * max_order;
  const uint32_t positions = window + 1u;
  const uint32_t tables = geo.num_engram_tables;
  const uint32_t sub_dim = geo.engram_sub_dim;
  const uint32_t e_dim = tables * sub_dim;

  // Window positions map to logical token positions
  // [position - window, position]; out-of-range sources read token 0 invalid.
  for (uint32_t p = 0u; p < positions; ++p) {
    const uint32_t in_range = static_cast<uint32_t>(p + ctx.position >= window);
    const uint32_t source = (ctx.position + p - window) * in_range;
    ctx.engram_hash_tokens[p] = in_range * ctx.history_tokens[source];
    ctx.engram_hash_valid[p] = static_cast<uint8_t>(
        in_range * static_cast<uint32_t>(ctx.history_valid[source] != 0u));
  }

  const emel::kernel::engram::event::hash_rows_request hash_request{
      .tokens = ctx.engram_hash_tokens,
      .valid = ctx.engram_hash_valid,
      .positions = positions,
      .orders = std::span<const uint32_t>{geo.engram_orders.data(),
                                          geo.num_engram_orders},
      .num_orders = geo.num_engram_orders,
      .heads = tables / geo.num_engram_orders,
      .slots = geo.engram_slots,
      .indices = ctx.engram_hash_indices,
      .ngram_ok = ctx.engram_ngram_ok};
  emel::kernel::engram::event::dispatch_result hash_result{};
  bool ok =
      ctx.engram.process_event(emel::kernel::engram::event::execute_hash_rows{
          hash_request, hash_result});

  for (uint32_t site = 0u; site < bound.engram_site_count; ++site) {
    const auto &views = bound.engram_sites[site];
    // Tap j reads window position window - j * dilation; tap 0 is the
    // current position and also feeds the key projection.
    // decode.py taps shift by j * max(orders); the header's conv dilation
    // equals max(orders) for this deployment, but decode.py is ground truth.
    for (uint32_t tap = 0u; tap < geo.engram_conv_taps; ++tap) {
      const uint32_t p = window - tap * max_order;
      ctx.engram_tap_valid[tap] = ctx.engram_hash_valid[p];
      float *e_row =
          ctx.engram_e_rows.data() + static_cast<size_t>(tap) * e_dim;
      for (uint32_t table = 0u; table < tables; ++table) {
        const size_t hash_at = static_cast<size_t>(p) * tables + table;
        const uint32_t row =
            table * geo.engram_slots + ctx.engram_hash_indices[hash_at];
        ok = ok &&
             compute_dequant_row<route>(
                 ctx, views.tables, ctx.prepared_engram_sites[site].tables, row,
                 ctx.engram_ngram_ok[hash_at],
                 std::span<float>{e_row + static_cast<size_t>(table) * sub_dim,
                                  sub_dim});
      }
      ok = ok && compute_gemv<route>(
                     ctx, views.value_proj,
                     ctx.prepared_engram_sites[site].value_proj,
                     std::span<const float>{e_row, e_dim},
                     std::span<float>{ctx.engram_v_taps.data() +
                                          static_cast<size_t>(tap) * d_model,
                                      d_model});
    }
    ok =
        ok && compute_gemv<route>(
                  ctx, views.key_proj, ctx.prepared_engram_sites[site].key_proj,
                  std::span<const float>{ctx.engram_e_rows.data(), e_dim},
                  std::span<float>{ctx.engram_keys.data() +
                                       static_cast<size_t>(site) * d_model,
                                   d_model});
    const emel::kernel::engram::event::conv_taps_request conv_request{
        .value_rows = ctx.engram_v_taps,
        .tap_valid = ctx.engram_tap_valid,
        .taps = payload_span(views.taps, 0u,
                             static_cast<uint64_t>(geo.engram_conv_taps) *
                                 d_model * 2u),
        .conv_taps = geo.engram_conv_taps,
        .dim = d_model,
        .output = std::span<float>{ctx.engram_values.data() +
                                       static_cast<size_t>(site) * d_model,
                                   d_model}};
    emel::kernel::engram::event::dispatch_result conv_result{};
    ok = ok && ctx.engram.process_event(
                   emel::kernel::engram::event::execute_conv_taps{conv_request,
                                                                  conv_result});
  }
  return ok;
}

// One transformer layer over the current single-token step. `engram_site` and
// `window_full` are guard-selected at the transition; everything inside is
// compile-time conditionals plus bounded data-plane work.
template <route_kind route, bool engram_site, bool window_full>
inline bool compute_layer(context &ctx, event::step_ctx &step) noexcept {
  const auto &bound = *ctx.bound;
  const auto &geo = bound.geo;
  const uint32_t layer_index = step.layer_index;
  const auto &layer = bound.layers[layer_index];
  const auto &prepared_layer = ctx.prepared_layers[layer_index];
  const uint32_t d_model = geo.d_model;
  const uint32_t lane_count = geo.mhc_lanes;
  const uint32_t lane = layer_index % lane_count;
  const uint32_t heads = geo.num_heads;
  const uint32_t kv_heads = geo.num_kv_heads;
  const uint32_t head_dim = geo.head_dim;
  const uint32_t attn_dim = heads * head_dim;
  const uint32_t kv_dim = kv_heads * head_dim;

  bool ok = compute_rms_unit(ctx, ctx.lanes, lane_count * d_model,
                             std::span<float>{ctx.nx});
  ok = ok && compute_gemv_rows<route>(ctx, bound.mhc.phi_pre,
                                      ctx.prepared_mhc.phi_pre, ctx.nx,
                                      layer_index * lane_count, lane_count,
                                      std::span<float>{ctx.pre_dots});

  const emel::kernel::mhc::event::pre_mix_request pre_request{
      .lanes = ctx.lanes,
      .phi_dots = ctx.pre_dots,
      .a = payload_span(bound.mhc.a_pre, layer_index * 2u, 2u),
      .b = payload_span(bound.mhc.b_pre,
                        static_cast<uint64_t>(layer_index) * lane_count * 2u,
                        static_cast<uint64_t>(lane_count) * 2u),
      .lane_index = lane,
      .lane_count = lane_count,
      .dim = d_model,
      .output = std::span<float>{ctx.u}};
  emel::kernel::mhc::event::dispatch_result pre_result{};
  ok = ok && ctx.mhc.process_event(emel::kernel::mhc::event::execute_pre_mix{
                 pre_request, pre_result});

  if constexpr (engram_site) {
    uint32_t site = 0u;
    for (uint32_t s = 0u; s < geo.num_engram_sites && s < 4u; ++s)
      site += s * static_cast<uint32_t>(geo.engram_sites[s] == layer_index);
    const emel::kernel::engram::event::alpha_gate_request gate_request{
        .u = ctx.u,
        .key = std::span<const float>{ctx.engram_keys.data() +
                                          static_cast<size_t>(site) * d_model,
                                      d_model},
        .value = std::span<const float>{ctx.engram_values.data() +
                                            static_cast<size_t>(site) * d_model,
                                        d_model},
        .dim = d_model,
        .output = std::span<float>{ctx.bx}};
    emel::kernel::engram::event::dispatch_result gate_result{};
    ok = ok && ctx.engram.process_event(
                   emel::kernel::engram::event::execute_alpha_gate{
                       gate_request, gate_result});
  } else {
    for (uint32_t c = 0u; c < d_model; ++c)
      ctx.bx[c] = ctx.u[c];
  }

  const std::span<const float> norm_in_scale{
      ctx.norm_in_scale.data() + static_cast<size_t>(layer_index) * d_model,
      d_model};
  ok = ok && compute_zcrms_norm(ctx, ctx.bx, norm_in_scale, 1u, d_model,
                                std::span<float>{ctx.h_norm});
  if constexpr (route == route_kind::prepared_avx2) {
    ok = ok && compute_gemv_batch4(
                   ctx, ctx.h_norm, prepared_layer.q_proj,
                   std::span<float>{ctx.q_rows}, prepared_layer.k_proj,
                   std::span<float>{ctx.k_rows}, prepared_layer.v_proj,
                   std::span<float>{ctx.v_rows}, prepared_layer.gate_proj,
                   std::span<float>{ctx.gate_logits});
  } else {
    ok = ok && compute_gemv<route>(ctx, layer.q_proj, prepared_layer.q_proj,
                                   ctx.h_norm, std::span<float>{ctx.q_rows});
    ok = ok && compute_gemv<route>(ctx, layer.k_proj, prepared_layer.k_proj,
                                   ctx.h_norm, std::span<float>{ctx.k_rows});
    ok = ok && compute_gemv<route>(ctx, layer.v_proj, prepared_layer.v_proj,
                                   ctx.h_norm, std::span<float>{ctx.v_rows});
    ok = ok &&
         compute_gemv<route>(ctx, layer.gate_proj, prepared_layer.gate_proj,
                             ctx.h_norm, std::span<float>{ctx.gate_logits});
  }

  const std::span<const float> q_norm_scale{
      ctx.q_norm_scale.data() + static_cast<size_t>(layer_index) * head_dim,
      head_dim};
  const std::span<const float> k_norm_scale{
      ctx.k_norm_scale.data() + static_cast<size_t>(layer_index) * head_dim,
      head_dim};
  ok = ok && compute_zcrms_norm(ctx, ctx.q_rows, q_norm_scale, heads, head_dim,
                                std::span<float>{ctx.q_rows});
  ok = ok && compute_zcrms_norm(ctx, ctx.k_rows, k_norm_scale, kv_heads,
                                head_dim, std::span<float>{ctx.k_rows});

  const emel::kernel::rope::event::apply_rows_request rope_q{
      .cos_table = ctx.rope_cos,
      .sin_table = ctx.rope_sin,
      .position = ctx.position,
      .head_count = heads,
      .head_dim = head_dim,
      .rows = ctx.q_rows};
  emel::kernel::rope::event::dispatch_result rope_q_result{};
  ok = ok &&
       ctx.rope.process_event(emel::kernel::rope::event::execute_apply_rows{
           rope_q, rope_q_result});
  const emel::kernel::rope::event::apply_rows_request rope_k{
      .cos_table = ctx.rope_cos,
      .sin_table = ctx.rope_sin,
      .position = ctx.position,
      .head_count = kv_heads,
      .head_dim = head_dim,
      .rows = ctx.k_rows};
  emel::kernel::rope::event::dispatch_result rope_k_result{};
  ok = ok &&
       ctx.rope.process_event(emel::kernel::rope::event::execute_apply_rows{
           rope_k, rope_k_result});

  const size_t cache_layer_floats = static_cast<size_t>(kv_dim) * geo.kv_window;
  const std::span<float> key_slice{ctx.key_cache.data() +
                                       static_cast<size_t>(layer_index) *
                                           cache_layer_floats,
                                   cache_layer_floats};
  const std::span<float> value_slice{ctx.value_cache.data() +
                                         static_cast<size_t>(layer_index) *
                                             cache_layer_floats,
                                     cache_layer_floats};
  const emel::kernel::swa::event::cache_write_request write_request{
      .key_rows = ctx.k_rows,
      .value_rows = ctx.v_rows,
      .position = ctx.position,
      .capacity = geo.kv_window,
      .kv_heads = kv_heads,
      .head_dim = head_dim,
      .key_cache = key_slice,
      .value_cache = value_slice};
  emel::kernel::swa::event::dispatch_result write_result{};
  ok =
      ok && ctx.swa.process_event(emel::kernel::swa::event::execute_cache_write{
                write_request, write_result});

  uint32_t window_begin = 0u;
  if constexpr (window_full)
    window_begin = ctx.position + 1u - geo.kv_window;
  const emel::kernel::swa::event::attend_request attend_request{
      .query = ctx.q_rows,
      .key_cache = key_slice,
      .value_cache = value_slice,
      .position = ctx.position,
      .window_begin = window_begin,
      .capacity = geo.kv_window,
      .heads = heads,
      .kv_heads = kv_heads,
      .head_dim = head_dim,
      .workspace = ctx.attend_workspace,
      .output = ctx.attn_out};
  emel::kernel::swa::event::dispatch_result attend_result{};
  ok = ok && ctx.swa.process_event(emel::kernel::swa::event::execute_attend{
                 attend_request, attend_result});

  const emel::kernel::swa::event::gate_mul_request gate_request{
      .values = ctx.attn_out, .gate_logits = ctx.gate_logits, .dim = attn_dim};
  emel::kernel::swa::event::dispatch_result gate_result{};
  ok = ok && ctx.swa.process_event(emel::kernel::swa::event::execute_gate_mul{
                 gate_request, gate_result});
  ok = ok && compute_gemv<route>(ctx, layer.out_proj, prepared_layer.out_proj,
                                 ctx.attn_out, std::span<float>{ctx.attn_proj});

  const std::span<const float> post_norm_scale{
      ctx.post_norm_scale.data() + static_cast<size_t>(layer_index) * d_model,
      d_model};
  ok = ok && compute_zcrms_norm(ctx, ctx.attn_proj, post_norm_scale, 1u,
                                d_model, std::span<float>{ctx.attn_norm});
  const emel::kernel::swa::event::residual_gate_request residual_request{
      .skip = ctx.bx,
      .gate = ctx.attn_gate_scale[layer_index],
      .values = ctx.attn_norm,
      .dim = d_model,
      .output = std::span<float>{ctx.xb}};
  emel::kernel::swa::event::dispatch_result residual_result{};
  ok = ok &&
       ctx.swa.process_event(emel::kernel::swa::event::execute_residual_gate{
           residual_request, residual_result});

  const std::span<const float> pre_hada_scale{
      ctx.pre_hada_scale.data() + static_cast<size_t>(layer_index) * d_model,
      d_model};
  ok = ok && compute_zcrms_norm(ctx, ctx.xb, pre_hada_scale, 1u, d_model,
                                std::span<float>{ctx.h_norm});
  const uint64_t hada_bytes = static_cast<uint64_t>(geo.hada_n) * 2u;
  const emel::kernel::hadamard::event::mlp_row_request hadamard_request{
      .input = ctx.h_norm,
      .skip = ctx.xb,
      .d1 = payload_span(layer.d1, 0u, hada_bytes),
      .d2 = payload_span(layer.d2, 0u, hada_bytes),
      .d3 = payload_span(layer.d3, 0u, hada_bytes),
      .d_model = d_model,
      .hada_n = geo.hada_n,
      .workspace = ctx.hada_workspace,
      .output = std::span<float>{ctx.block_out}};
  emel::kernel::hadamard::event::dispatch_result hadamard_result{};
  ok = ok && ctx.hadamard.process_event(
                 emel::kernel::hadamard::event::execute_mlp_row{
                     hadamard_request, hadamard_result});

  ok = ok && compute_gemv_rows<route>(ctx, bound.mhc.phi_post,
                                      ctx.prepared_mhc.phi_post, ctx.nx,
                                      layer_index * lane_count, lane_count,
                                      std::span<float>{ctx.post_dots});
  ok = ok && compute_gemv_rows<route>(
                 ctx, bound.mhc.phi_res, ctx.prepared_mhc.phi_res, ctx.nx,
                 layer_index * lane_count * lane_count, lane_count * lane_count,
                 std::span<float>{ctx.res_dots});
  const uint64_t square = static_cast<uint64_t>(lane_count) * lane_count;
  const emel::kernel::mhc::event::post_mix_request post_request{
      .lanes = ctx.lanes,
      .block_out = ctx.block_out,
      .u = ctx.u,
      .post_dots = ctx.post_dots,
      .res_dots = ctx.res_dots,
      .a_post = payload_span(bound.mhc.a_post, layer_index * 2u, 2u),
      .b_post =
          payload_span(bound.mhc.b_post,
                       static_cast<uint64_t>(layer_index) * lane_count * 2u,
                       static_cast<uint64_t>(lane_count) * 2u),
      .a_res = payload_span(bound.mhc.a_res, layer_index * 2u, 2u),
      .b_res = payload_span(bound.mhc.b_res,
                            static_cast<uint64_t>(layer_index) * square * 2u,
                            square * 2u),
      .lane_index = lane,
      .lane_count = lane_count,
      .dim = d_model,
      .output = std::span<float>{ctx.lanes_next}};
  emel::kernel::mhc::event::dispatch_result post_result{};
  ok = ok && ctx.mhc.process_event(emel::kernel::mhc::event::execute_post_mix{
                 post_request, post_result});
  std::swap(ctx.lanes, ctx.lanes_next);
  return ok;
}

// Final head: mean over lanes, final ZCRMSNorm, tied-embedding logits.
template <route_kind route>
inline bool compute_logits(context &ctx, event::step_ctx &step) noexcept {
  const auto &geo = ctx.bound->geo;
  const emel::kernel::mhc::event::mean_lanes_request mean_request{
      .lanes = ctx.lanes,
      .lane_count = geo.mhc_lanes,
      .dim = geo.d_model,
      .output = std::span<float>{ctx.mean}};
  emel::kernel::mhc::event::dispatch_result mean_result{};
  bool ok = ctx.mhc.process_event(
      emel::kernel::mhc::event::execute_mean_lanes{mean_request, mean_result});
  ok =
      ok && compute_zcrms_norm(ctx, ctx.mean, ctx.final_norm_scale, 1u,
                               geo.d_model, std::span<float>{ctx.final_normed});
  ok = ok &&
       compute_gemv<route>(ctx, ctx.bound->embedding, ctx.prepared_embedding,
                           ctx.final_normed, step.logits_out);
  return ok;
}

//------------------------------------------------------------------------------//
// Effects.
//------------------------------------------------------------------------------//

struct effect_begin_init {
  void operator()(const event::init_run &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
  }
};

struct effect_mark_init_unsupported {
  void operator()(const event::init_run &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::geometry_unsupported);
  }
};

struct effect_exec_init {
  void operator()(const event::init_run &ev, context &ctx) const noexcept {
    fold_error(ev.ctx.err, compute_init(ctx));
  }
};

struct effect_mark_step_invalid {
  void operator()(const event::step_run &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

template <route_kind route> struct effect_step_begin {
  void operator()(const event::step_run &ev, context &ctx) const noexcept {
    fold_error(ev.ctx.err, compute_step_begin<route>(ctx, ev.ctx));
  }
};

template <route_kind route> struct effect_compute_engram {
  void operator()(const event::step_run &ev, context &ctx) const noexcept {
    fold_error(ev.ctx.err, compute_engram<route>(ctx));
  }
};

template <route_kind route, bool engram_site, bool window_full>
struct effect_run_layer {
  void operator()(const event::step_run &ev, context &ctx) const noexcept {
    fold_error(ev.ctx.err,
               compute_layer<route, engram_site, window_full>(ctx, ev.ctx));
  }
};

struct effect_advance_layer {
  void operator()(const event::step_run &ev, context &) const noexcept {
    ++ev.ctx.layer_index;
  }
};

template <route_kind route> struct effect_emit_logits {
  void operator()(const event::step_run &ev, context &ctx) const noexcept {
    fold_error(ev.ctx.err, compute_logits<route>(ctx, ev.ctx));
  }
};

struct effect_finish_step {
  void operator()(const event::step_run &, context &ctx) const noexcept {
    ++ctx.position;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.event_.ctx.err; }) {
      ev.event_.ctx.err = emel::error::cast(error::internal_error);
    } else if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
    }
  }
};

} // namespace emel::model::needle::graph::action
