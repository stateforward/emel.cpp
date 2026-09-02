#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <limits>

#include <vector>

#include "emel/kernel/cq/sm.hpp"
#include "emel/kernel/engram/sm.hpp"
#include "emel/kernel/hadamard/sm.hpp"
#include "emel/kernel/mhc/sm.hpp"
#include "emel/kernel/rope/sm.hpp"
#include "emel/kernel/swa/sm.hpp"
#include "emel/kernel/zcrms/sm.hpp"
#include "emel/model/needle/graph/events.hpp"
#include "emel/model/needle/graph/errors.hpp"
#include "emel/model/needle/events.hpp"
#include "emel/model/needle/detail.hpp"
#include "emel/kernel/cq/detail.hpp"


namespace emel::model::needle::graph::action {

struct prepared_layer_views {
  emel::kernel::cq::event::prepared_q4_view q_proj = {};
  emel::kernel::cq::event::prepared_q4_view k_proj = {};
  emel::kernel::cq::event::prepared_q4_view v_proj = {};
  emel::kernel::cq::event::prepared_q4_view gate_proj = {};
  emel::kernel::cq::event::prepared_q4_view out_proj = {};
};

struct prepared_mhc_views {
  emel::kernel::cq::event::prepared_q4_view phi_pre = {};
  emel::kernel::cq::event::prepared_q4_view phi_post = {};
  emel::kernel::cq::event::prepared_q4_view phi_res = {};
};

struct prepared_engram_site_views {
  emel::kernel::cq::event::prepared_q4_view tables = {};
  emel::kernel::cq::event::prepared_q4_view key_proj = {};
  emel::kernel::cq::event::prepared_q4_view value_proj = {};
};

// CQ dispatch and deployment-numeric route. Both are selected once by explicit
// guarded init transitions and materialized as distinct step-chain states.
enum class route_kind : uint8_t {
  scalar = 0,
  prepared_avx2 = 1,
};
enum class activation_route_kind : uint8_t {
  f32 = 0,
  a8 = 1,
};
enum class projection_route_kind : uint8_t { serial = 0, parallel4 = 3 };

using projection_lane_pool = emel::policy::fork_join_lane_pool<3u, 256u>;

struct activation_payload {
  std::span<const float> values = {};
  float scale = 1.0f;
};

// Graph-owned runtime storage. ALL heap allocation happens here, in the
// constructor, sized from the bound contract's header geometry — dispatch
// (init/prefill/decode) never allocates. f32 KV storage is the parity
// contract: the reference decode path (configure_deploy kv_bits=8 ->
// KV_BITS=0 -> no KV quantization) also runs f32 caches, so kv_bits in the
// header stays a baked deployment hint, not a runtime obligation.
inline constexpr uint64_t k_max_graph_storage_bytes = uint64_t{1} << 30u;

struct storage_plan {
  uint64_t lanes_dim = 0u;
  uint64_t attn_dim = 0u;
  uint64_t kv_dim = 0u;
  uint64_t cache_floats = 0u;
  uint64_t max_order = 0u;
  uint64_t tables = 0u;
  uint64_t engram_e_dim = 0u;
  uint64_t hash_window = 0u;
  uint64_t cq_workspace = 0u;
  uint64_t prepared_indices = 0u;
  uint64_t prepared_norms = 0u;
  uint64_t prepared_norms_by_group32 = 0u;

  uint64_t total_bytes = 0u;
};

inline bool checked_add(const uint64_t lhs, const uint64_t rhs,
                        uint64_t &out) noexcept {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
    return false;
  out = lhs + rhs;
  return true;
}

inline bool checked_mul(const uint64_t lhs, const uint64_t rhs,
                        uint64_t &out) noexcept {
  if (lhs != 0u && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  out = lhs * rhs;
  return true;
}

inline bool add_storage(uint64_t elements, const uint64_t element_bytes,
                        uint64_t &total) noexcept {
  uint64_t bytes = 0u;
  return checked_mul(elements, element_bytes, bytes) &&
         checked_add(total, bytes, total) &&
         total <= k_max_graph_storage_bytes;
}

inline uint64_t compute_max_order(
    const emel::cact::loader::geometry &geo) noexcept {
  uint64_t max_order = 1u;
  for (uint32_t i = 0u; i < geo.num_engram_orders; ++i)
    max_order = geo.engram_orders[i] > max_order ? geo.engram_orders[i]
                                                   : max_order;
  return max_order;
}

inline uint64_t compute_in_pad(const tensor_view &view) noexcept {
  const uint64_t group = view.group != 0u ? view.group : 1u;
  return (static_cast<uint64_t>(view.shape[1]) + group - 1u) / group * group;
}
inline bool validate_fp16_payload(const tensor_view &view,
                                  const uint64_t elements) noexcept {
  uint64_t required = 0u;
  return checked_mul(elements, sizeof(uint16_t), required) &&
         view.data != nullptr && view.nbytes >= required &&
         (reinterpret_cast<uintptr_t>(view.data) % alignof(uint16_t)) == 0u;
}

inline bool validate_cq4_payload(const tensor_view &view,
                                 const needle::geometry &geo) noexcept {
  return emel::kernel::cq::detail::valid_packed_view<4u>(
      view, std::span<const float>{geo.codebook.data(),
                                  emel::cact::loader::k_codebook_len});
}

inline bool validate_layer_views(const needle::layer_views &layer,
                                 const needle::geometry &geo) noexcept {
  const std::array<tensor_view, needle::k_layer_tensor_count> views = {
      layer.norm_in,   layer.q_proj,   layer.k_proj,  layer.v_proj,
      layer.q_norm,    layer.k_norm,   layer.gate_proj,
      layer.out_proj,  layer.post_norm, layer.attn_gate,
      layer.pre_hada,  layer.d1,       layer.d2,      layer.d3,
  };
  needle::layer_views validated{};
  if (needle::detail::bind_layer(views, geo, validated) !=
      needle::detail::cast_needle_error(needle::error::none))
    return false;
  return validate_fp16_payload(layer.norm_in, geo.d_model) &&
         validate_cq4_payload(layer.q_proj, geo) &&
         validate_cq4_payload(layer.k_proj, geo) &&
         validate_cq4_payload(layer.v_proj, geo) &&
         validate_fp16_payload(layer.q_norm, geo.head_dim) &&
         validate_fp16_payload(layer.k_norm, geo.head_dim) &&
         validate_cq4_payload(layer.gate_proj, geo) &&
         validate_cq4_payload(layer.out_proj, geo) &&
         validate_fp16_payload(layer.post_norm, geo.d_model) &&
         validate_fp16_payload(layer.attn_gate, 1u) &&
         validate_fp16_payload(layer.pre_hada, geo.d_model) &&
         validate_fp16_payload(layer.d1, geo.hada_n) &&
         validate_fp16_payload(layer.d2, geo.hada_n) &&
         validate_fp16_payload(layer.d3, geo.hada_n);
}

inline bool validate_mhc_views(const needle::mhc_views &mhc,
                               const needle::geometry &geo) noexcept {
  const std::array<tensor_view, needle::k_mhc_tensor_count> views = {
      mhc.a_pre, mhc.a_post, mhc.a_res, mhc.b_pre, mhc.b_post,
      mhc.b_res, mhc.phi_pre, mhc.phi_post, mhc.phi_res,
  };
  needle::mhc_views validated{};
  if (needle::detail::bind_mhc(views, geo, validated) !=
      needle::detail::cast_needle_error(needle::error::none))
    return false;
  uint64_t layer_lanes = 0u;
  uint64_t layer_lane_pairs = 0u;
  if (!checked_mul(geo.num_layers, geo.mhc_lanes, layer_lanes) ||
      !checked_mul(layer_lanes, geo.mhc_lanes, layer_lane_pairs))
    return false;
  return validate_fp16_payload(mhc.a_pre, geo.num_layers) &&
         validate_fp16_payload(mhc.a_post, geo.num_layers) &&
         validate_fp16_payload(mhc.a_res, geo.num_layers) &&
         validate_fp16_payload(mhc.b_pre, layer_lanes) &&
         validate_fp16_payload(mhc.b_post, layer_lanes) &&
         validate_fp16_payload(mhc.b_res, layer_lane_pairs) &&
         validate_cq4_payload(mhc.phi_pre, geo) &&
         validate_cq4_payload(mhc.phi_post, geo) &&
         validate_cq4_payload(mhc.phi_res, geo);
}

inline bool validate_engram_views(const needle::engram_site_views &site,
                                  const needle::geometry &geo) noexcept {
  const std::array<tensor_view, needle::k_engram_site_tensor_count> views = {
      site.tables, site.key_proj, site.value_proj, site.taps};
  needle::engram_site_views validated{};
  if (needle::detail::bind_engram_site(views, geo, validated) !=
      needle::detail::cast_needle_error(needle::error::none))
    return false;
  uint64_t taps = 0u;
  return checked_mul(geo.engram_conv_taps, geo.d_model, taps) &&
         validate_cq4_payload(site.tables, geo) &&
         validate_cq4_payload(site.key_proj, geo) &&
         validate_cq4_payload(site.value_proj, geo) &&
         validate_fp16_payload(site.taps, taps);
}

inline bool validate_graph_contract(const needle::contract &bound) noexcept {
  const auto &geo = bound.geo;
  if (needle::detail::validate_geometry(geo) !=
          needle::detail::cast_needle_error(needle::error::none) ||
      bound.layer_count != geo.num_layers ||
      bound.engram_site_count != geo.num_engram_sites ||
      !std::isfinite(geo.rope_theta) || geo.rope_theta <= 0.0f)
    return false;
  for (const float value : geo.codebook)
    if (!std::isfinite(value))
      return false;
  const needle::detail::role_spec embedding_spec = {
      needle::detail::constants::dtype_cq, 2u,
      {geo.vocab_size, geo.d_model, 0u, 0u}};
  if (needle::detail::validate_role(bound.embedding, embedding_spec) !=
          needle::detail::cast_needle_error(needle::error::none) ||
      !validate_cq4_payload(bound.embedding, geo))
    return false;
  for (uint32_t layer = 0u; layer < bound.layer_count; ++layer)
    if (!validate_layer_views(bound.layers[layer], geo))
      return false;
  if (!validate_mhc_views(bound.mhc, geo))
    return false;
  for (uint32_t site = 0u; site < bound.engram_site_count; ++site)
    if (!validate_engram_views(bound.engram_sites[site], geo))
      return false;
  const needle::detail::role_spec final_norm_spec = {
      needle::detail::constants::dtype_fp16, 1u,
      {geo.d_model, 0u, 0u, 0u}};
  return needle::detail::validate_role(bound.final_norm, final_norm_spec) ==
             needle::detail::cast_needle_error(needle::error::none) &&
         validate_fp16_payload(bound.final_norm, geo.d_model);
}


inline bool compute_storage_plan(const needle::contract &bound,
                                 storage_plan &plan) noexcept {
  const auto &geo = bound.geo;
  if (!validate_graph_contract(bound))
    return false;

  uint64_t cache_layer = 0u;
  uint64_t rope_floats = 0u;
  uint64_t layer_model = 0u;
  uint64_t layer_head = 0u;
  uint64_t layer_lanes = 0u;
  uint64_t hash_extent = 0u;
  uint64_t hash_tables = 0u;
  uint64_t engram_tap_embed = 0u;
  uint64_t engram_tap_model = 0u;
  uint64_t engram_site_model = 0u;

  plan.max_order = compute_max_order(geo);
  plan.tables = geo.num_engram_tables;
  if (!checked_mul(geo.mhc_lanes, geo.d_model, plan.lanes_dim) ||
      !checked_mul(geo.num_heads, geo.head_dim, plan.attn_dim) ||
      !checked_mul(geo.num_kv_heads, geo.head_dim, plan.kv_dim) ||
      !checked_mul(plan.kv_dim, geo.kv_window, cache_layer) ||
      !checked_mul(cache_layer, geo.num_layers, plan.cache_floats) ||
      !checked_mul(geo.max_seq_len, geo.head_dim / 2u, rope_floats) ||
      !checked_mul(geo.num_layers, geo.d_model, layer_model) ||
      !checked_mul(geo.num_layers, geo.head_dim, layer_head) ||
      !checked_mul(geo.mhc_lanes, geo.mhc_lanes, layer_lanes) ||
      !checked_mul(geo.num_engram_tables, geo.engram_sub_dim,
                   plan.engram_e_dim) ||
      !checked_mul(geo.engram_conv_taps - 1u, geo.engram_conv_dilation,
                   hash_extent) ||
      !checked_mul(hash_extent, plan.max_order, hash_extent) ||
      !checked_add(hash_extent, 1u, plan.hash_window) ||
      !checked_mul(plan.hash_window, plan.tables, hash_tables) ||
      !checked_mul(geo.engram_conv_taps, plan.engram_e_dim,
                   engram_tap_embed) ||
      !checked_mul(geo.engram_conv_taps, geo.d_model, engram_tap_model) ||
      !checked_mul(geo.num_engram_sites, geo.d_model, engram_site_model))
    return false;

  uint64_t total = 0u;
  const auto floats = [&](const uint64_t count) noexcept {
    return add_storage(count, sizeof(float), total);
  };
  const auto bytes = [&](const uint64_t count) noexcept {
    return add_storage(count, sizeof(uint8_t), total);
  };
  const auto tokens = [&](const uint64_t count) noexcept {
    return add_storage(count, sizeof(int32_t), total);
  };
  const auto indices = [&](const uint64_t count) noexcept {
    return add_storage(count, sizeof(uint32_t), total);
  };
  if (!floats(plan.lanes_dim * 3u) || !floats(layer_lanes) ||
      !floats(8u * geo.d_model) || !floats(4u * plan.attn_dim) ||
      !floats(2u * plan.kv_dim) || !floats(geo.hada_n) ||
      !floats(2u * geo.kv_window) || !floats(2u * plan.cache_floats) ||
      !tokens(geo.max_seq_len) || !bytes(geo.max_seq_len) ||
      !floats(2u * rope_floats) || !floats(3u * layer_model) ||
      !floats(2u * layer_head) || !floats(geo.num_layers) ||
      !floats(geo.d_model) || !tokens(plan.hash_window) ||
      !bytes(plan.hash_window) || !indices(hash_tables) ||
      !floats(hash_tables) || !floats(engram_tap_embed) ||
      !floats(engram_tap_model) || !bytes(geo.engram_conv_taps) ||
      !floats(2u * engram_site_model))
    return false;

  const auto add_prepared = [&](const tensor_view &view) noexcept {
    const uint64_t in_pad = compute_in_pad(view);
    if (view.group == 0u || in_pad % view.group != 0u)
      return false;
    const uint64_t groups_per_row = in_pad / view.group;
    uint64_t indices_count = 0u;
    uint64_t norms_count = 0u;
    uint64_t group32_rows = static_cast<uint64_t>(view.shape[0] / 32u) * 32u;
    uint64_t group32_norms = 0u;
    return checked_mul(view.shape[0], in_pad, indices_count) &&
           checked_mul(view.shape[0], groups_per_row, norms_count) &&
           checked_mul(group32_rows, groups_per_row, group32_norms) &&
           checked_add(plan.prepared_indices, indices_count,
                       plan.prepared_indices) &&
           checked_add(plan.prepared_norms, norms_count, plan.prepared_norms) &&
           checked_add(plan.prepared_norms_by_group32, group32_norms,
                       plan.prepared_norms_by_group32);
  };
  bool prepared_ok = add_prepared(bound.embedding);
  for (uint32_t i = 0u; prepared_ok && i < bound.layer_count; ++i) {
    const auto &layer = bound.layers[i];
    prepared_ok = add_prepared(layer.q_proj) && add_prepared(layer.k_proj) &&
                  add_prepared(layer.v_proj) &&
                  add_prepared(layer.gate_proj) &&
                  add_prepared(layer.out_proj);
  }
  prepared_ok = prepared_ok && add_prepared(bound.mhc.phi_pre) &&
                add_prepared(bound.mhc.phi_post) &&
                add_prepared(bound.mhc.phi_res);
  for (uint32_t i = 0u; prepared_ok && i < bound.engram_site_count; ++i) {
    const auto &site = bound.engram_sites[i];
    prepared_ok = add_prepared(site.tables) && add_prepared(site.key_proj) &&
                  add_prepared(site.value_proj);
  }
  if (!prepared_ok || !bytes(plan.prepared_indices) ||
      !bytes(plan.prepared_indices) || !floats(plan.prepared_norms) ||
      !floats(plan.prepared_norms_by_group32))
    return false;

  plan.cq_workspace = compute_in_pad(bound.embedding);
  for (uint32_t i = 0u; i < bound.layer_count; ++i) {
    const uint64_t layer_pad = compute_in_pad(bound.layers[i].q_proj);
    const uint64_t out_pad = compute_in_pad(bound.layers[i].out_proj);
    plan.cq_workspace = layer_pad > plan.cq_workspace ? layer_pad
                                                      : plan.cq_workspace;
    plan.cq_workspace = out_pad > plan.cq_workspace ? out_pad
                                                     : plan.cq_workspace;
  }
  const uint64_t phi_pad = compute_in_pad(bound.mhc.phi_res);
  plan.cq_workspace = phi_pad > plan.cq_workspace ? phi_pad
                                                   : plan.cq_workspace;
  for (uint32_t s = 0u; s < bound.engram_site_count; ++s) {
    const uint64_t site_pad = compute_in_pad(bound.engram_sites[s].key_proj);
    plan.cq_workspace = site_pad > plan.cq_workspace ? site_pad
                                                      : plan.cq_workspace;
  }
  if (!floats(plan.cq_workspace) || !bytes(plan.cq_workspace) ||
      !floats(plan.cq_workspace))
    return false;
  plan.total_bytes = total;
  return true;
}
inline emel::error::type
validate_construction(const needle::contract &bound) noexcept {
  storage_plan plan{};
  return compute_storage_plan(bound, plan)
             ? emel::error::cast(error::none)
             : emel::error::cast(error::geometry_unsupported);
}

struct context {
  explicit context(const needle::contract &contract_in,
                   const bool parallel_projection_wave = false)
      : bound(contract_in),
        parallel_projection_wave(parallel_projection_wave) {
    storage_plan plan{};
    construction_error = compute_storage_plan(bound, plan)
                             ? emel::error::cast(error::none)
                             : emel::error::cast(error::geometry_unsupported);
    storage_valid = construction_error == emel::error::cast(error::none);
    if (!storage_valid)
      return;

    const auto &geo = bound.geo;
    const uint64_t d_model = geo.d_model;
    const uint64_t lanes_dim = plan.lanes_dim;
    const uint64_t attn_dim = plan.attn_dim;
    const uint64_t kv_dim = plan.kv_dim;
    const uint64_t layers = geo.num_layers;
    const uint64_t cache_floats = plan.cache_floats;
    const uint64_t half_dim = geo.head_dim / 2u;
    const uint64_t tables = plan.tables;
    const uint64_t engram_e_dim = plan.engram_e_dim;
    const uint64_t hash_window = plan.hash_window;
    uint64_t layer_model = 0u;
    uint64_t layer_head = 0u;
    uint64_t rope_floats = 0u;
    uint64_t hash_tables = 0u;
    uint64_t engram_tap_embed = 0u;
    uint64_t engram_tap_model = 0u;
    uint64_t engram_site_model = 0u;
    uint64_t lane_pairs = 0u;
    uint64_t attend_floats = 0u;
    checked_mul(geo.mhc_lanes, geo.mhc_lanes, lane_pairs);
    checked_mul(geo.kv_window, 2u, attend_floats);
    checked_mul(layers, d_model, layer_model);
    checked_mul(layers, geo.head_dim, layer_head);
    checked_mul(geo.max_seq_len, half_dim, rope_floats);
    checked_mul(hash_window, tables, hash_tables);
    checked_mul(geo.engram_conv_taps, engram_e_dim, engram_tap_embed);
    checked_mul(geo.engram_conv_taps, d_model, engram_tap_model);
    checked_mul(geo.num_engram_sites, d_model, engram_site_model);
    lanes.resize(lanes_dim);
    lanes_next.resize(lanes_dim);
    nx.resize(lanes_dim);
    pre_dots.resize(geo.mhc_lanes);
    post_dots.resize(geo.mhc_lanes);
    res_dots.resize(lane_pairs);
    u.resize(d_model);
    bx.resize(d_model);
    h_norm.resize(d_model);
    xb.resize(d_model);
    attn_norm.resize(d_model);
    block_out.resize(d_model);
    mean.resize(d_model);
    final_normed.resize(d_model);
    attn_proj.resize(d_model);
    q_rows.resize(attn_dim);
    k_rows.resize(kv_dim);
    v_rows.resize(kv_dim);
    attn_out.resize(attn_dim);
    gate_logits.resize(attn_dim);
    hada_workspace.resize(geo.hada_n);
    attend_workspace.resize(attend_floats);
    key_cache.resize(cache_floats);
    value_cache.resize(cache_floats);
    history_tokens.resize(geo.max_seq_len);
    history_valid.resize(geo.max_seq_len);
    rope_cos.resize(rope_floats);
    rope_sin.resize(rope_floats);
    norm_in_scale.resize(layer_model);
    post_norm_scale.resize(layer_model);
    pre_hada_scale.resize(layer_model);
    q_norm_scale.resize(layer_head);
    k_norm_scale.resize(layer_head);
    attn_gate_scale.resize(layers);
    final_norm_scale.resize(d_model);
    engram_hash_tokens.resize(hash_window);
    engram_hash_valid.resize(hash_window);
    engram_hash_indices.resize(hash_tables);
    engram_ngram_ok.resize(hash_tables);
    engram_e_rows.resize(engram_tap_embed);
    engram_v_taps.resize(engram_tap_model);
    engram_tap_valid.resize(geo.engram_conv_taps);
    engram_keys.resize(engram_site_model);
    engram_values.resize(engram_site_model);
    cq_workspace.resize(plan.cq_workspace);
    a8_quantized.resize(plan.cq_workspace);
    a8_integer_values.resize(plan.cq_workspace);
    prepared_index_bytes = static_cast<size_t>(plan.prepared_indices);
    prepared_input32_bytes = static_cast<size_t>(plan.prepared_indices);
    prepared_norm_bytes =
        static_cast<size_t>(plan.prepared_norms) * sizeof(float);
    prepared_group32_norm_bytes =
        static_cast<size_t>(plan.prepared_norms_by_group32) * sizeof(float);
    const auto make_prepared = [](const tensor_view &view) {
      return emel::kernel::cq::event::prepared_q4_view{
          view.shape[0], view.shape[1], view.group};
    };
    prepared_embedding = make_prepared(bound.embedding);
    for (uint32_t i = 0u; i < bound.layer_count; ++i) {
      const auto &layer = bound.layers[i];
      auto &prepared = prepared_layers[i];
      prepared.q_proj = make_prepared(layer.q_proj);
      prepared.k_proj = make_prepared(layer.k_proj);
      prepared.v_proj = make_prepared(layer.v_proj);
      prepared.gate_proj = make_prepared(layer.gate_proj);
      prepared.out_proj = make_prepared(layer.out_proj);
    }
    prepared_mhc.phi_pre = make_prepared(bound.mhc.phi_pre);
    prepared_mhc.phi_post = make_prepared(bound.mhc.phi_post);
    prepared_mhc.phi_res = make_prepared(bound.mhc.phi_res);
    for (uint32_t i = 0u; i < bound.engram_site_count; ++i) {
      const auto &site = bound.engram_sites[i];
      auto &prepared = prepared_engram_sites[i];
      prepared.tables = make_prepared(site.tables);
      prepared.key_proj = make_prepared(site.key_proj);
      prepared.value_proj = make_prepared(site.value_proj);
    }
    if (parallel_projection_wave)
      projection_pool.emplace(3u);
  }

  context(const context &) = delete;
  context &operator=(const context &) = delete;

  static uint64_t
  compute_max_order(const emel::cact::loader::geometry &geo) noexcept {
    return action::compute_max_order(geo);
  }
  static uint64_t compute_in_pad(const tensor_view &view) noexcept {
    return action::compute_in_pad(view);
  }


  // Immutable graph-owned metadata. Tensor payload bytes remain borrowed from
  // the mapped model image and must outlive this graph.
  const needle::contract bound;
  bool storage_valid = false;
  bool avx2_fma_available =
      emel::kernel::x86_64::detail::detect_avx2() &&
      emel::kernel::x86_64::detail::detect_fma();
  emel::error::type construction_error =
      emel::error::cast(error::geometry_unsupported);


  // Logical position of the next cache slot (persists across dispatches).
  uint32_t position = 0u;

  // 4-lane mHC hidden state and per-step scratch (preallocated; scratch
  // lives here because dispatch must stay allocation-free).
  std::vector<float> lanes;
  std::vector<float> lanes_next;
  std::vector<float> nx;
  std::vector<float> pre_dots;
  std::vector<float> post_dots;
  std::vector<float> res_dots;
  std::vector<float> u;
  std::vector<float> bx;
  std::vector<float> h_norm;
  std::vector<float> xb;
  std::vector<float> attn_norm;
  std::vector<float> block_out;
  std::vector<float> mean;
  std::vector<float> final_normed;
  std::vector<float> attn_proj;
  std::vector<float> q_rows;
  std::vector<float> k_rows;
  std::vector<float> v_rows;
  std::vector<float> attn_out;
  std::vector<float> gate_logits;
  std::vector<float> hada_workspace;
  std::vector<float> attend_workspace;
  std::vector<float> cq_workspace;
  std::vector<int8_t> a8_quantized;
  std::vector<float> a8_integer_values;
  size_t prepared_index_bytes = 0u;
  size_t prepared_input32_bytes = 0u;
  size_t prepared_norm_bytes = 0u;
  size_t prepared_group32_norm_bytes = 0u;
  emel::kernel::cq::event::prepared_codebook_q4 prepared_codebook = {};
  const bool parallel_projection_wave = false;
  std::array<uint64_t, 3u> worker_projection_calls = {};
  uint64_t projection_submitted = 0u;
  uint64_t projection_joined = 0u;
  std::atomic<uint64_t> projection_live{0u};
  uint64_t projection_cq_extra_nanoseconds = 0u;
  bool cq_timing_enabled = false;
  emel::kernel::cq::event::timestamp_now_fn cq_timing_now = nullptr;

  emel::kernel::cq::event::prepared_q4_view prepared_embedding = {};
  std::array<prepared_layer_views, needle::k_max_layers> prepared_layers = {};
  prepared_mhc_views prepared_mhc = {};
  std::array<prepared_engram_site_views, needle::k_max_engram_sites>
      prepared_engram_sites = {};

  bool timing_enabled = false;
  emel::model::needle::graph::event::timestamp_now_fn timing_now = nullptr;
  emel::model::needle::graph::event::timing_breakdown timing = {};
  uint64_t timing_step_begin = 0u;
  uint64_t timing_accounted_nanoseconds = 0u;
  uint64_t timing_cq_begin_nanoseconds = 0u;
  uint64_t swa_gqa2_calls = 0u;

  // Per-layer f32 KV ring caches over the kv_window sliding mask.
  std::vector<float> key_cache;
  std::vector<float> value_cache;

  // Token history consumed by the engram hash window.
  std::vector<int32_t> history_tokens;
  std::vector<uint8_t> history_valid;

  // RoPE tables precomputed at init (theta from the header).
  std::vector<float> rope_cos;
  std::vector<float> rope_sin;

  // fp16 scale tensors decoded once at init.
  std::vector<float> norm_in_scale;
  std::vector<float> post_norm_scale;
  std::vector<float> pre_hada_scale;
  std::vector<float> q_norm_scale;
  std::vector<float> k_norm_scale;
  std::vector<float> attn_gate_scale;
  std::vector<float> final_norm_scale;

  // Engram per-step scratch and per-site K/V for the current position.
  std::vector<int32_t> engram_hash_tokens;
  std::vector<uint8_t> engram_hash_valid;
  std::vector<uint32_t> engram_hash_indices;
  std::vector<float> engram_ngram_ok;
  std::vector<float> engram_e_rows;
  std::vector<float> engram_v_taps;
  std::vector<uint8_t> engram_tap_valid;
  std::vector<float> engram_keys;
  std::vector<float> engram_values;

  // Child kernel machines (parent-owned, dispatched via process_event only).
  emel::kernel::cq::sm cq;
  std::array<emel::kernel::cq::sm, 3u> worker_cq = {};
  emel::kernel::zcrms::sm zcrms;
  emel::kernel::rope::sm rope;
  emel::kernel::swa::sm swa;
  emel::kernel::hadamard::sm hadamard;
  emel::kernel::engram::sm engram;
  emel::kernel::mhc::sm mhc;

  // Last member: worker threads stop and join before actor/storage teardown.
  std::optional<projection_lane_pool> projection_pool = std::nullopt;
};

} // namespace emel::model::needle::graph::action
