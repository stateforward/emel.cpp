#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "emel/kernel/cq/sm.hpp"
#include "emel/kernel/engram/sm.hpp"
#include "emel/kernel/hadamard/sm.hpp"
#include "emel/kernel/mhc/sm.hpp"
#include "emel/kernel/rope/sm.hpp"
#include "emel/kernel/swa/sm.hpp"
#include "emel/kernel/zcrms/sm.hpp"
#include "emel/model/needle/graph/events.hpp"
#include "emel/model/needle/events.hpp"

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
struct context {
  explicit context(const needle::contract &contract_in) : bound(&contract_in) {
    const auto &geo = contract_in.geo;
    const uint64_t d_model = geo.d_model;
    const uint64_t lanes_dim = static_cast<uint64_t>(geo.mhc_lanes) * d_model;
    const uint64_t attn_dim =
        static_cast<uint64_t>(geo.num_heads) * geo.head_dim;
    const uint64_t kv_dim =
        static_cast<uint64_t>(geo.num_kv_heads) * geo.head_dim;
    const uint64_t layers = geo.num_layers;
    const uint64_t cache_floats =
        layers * kv_dim * static_cast<uint64_t>(geo.kv_window);
    const uint64_t half_dim = geo.head_dim / 2u;
    const uint64_t max_order = compute_max_order(geo);
    const uint64_t tables = geo.num_engram_tables;
    const uint64_t engram_e_dim = tables * geo.engram_sub_dim;
    // Engram hash window: ENGRAM_CONV_TAPS * max(orders) history positions
    // plus the current one (decode.py `_engram_window` + S=1).
    const uint64_t hash_window =
        static_cast<uint64_t>(geo.engram_conv_taps) * max_order + 1u;

    lanes.resize(lanes_dim);
    lanes_next.resize(lanes_dim);
    nx.resize(lanes_dim);
    pre_dots.resize(geo.mhc_lanes);
    post_dots.resize(geo.mhc_lanes);
    res_dots.resize(static_cast<uint64_t>(geo.mhc_lanes) * geo.mhc_lanes);
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
    attend_workspace.resize(static_cast<uint64_t>(geo.kv_window) * 2u);
    key_cache.resize(cache_floats);
    value_cache.resize(cache_floats);
    history_tokens.resize(geo.max_seq_len);
    history_valid.resize(geo.max_seq_len);
    rope_cos.resize(static_cast<uint64_t>(geo.max_seq_len) * half_dim);
    rope_sin.resize(static_cast<uint64_t>(geo.max_seq_len) * half_dim);
    norm_in_scale.resize(layers * d_model);
    post_norm_scale.resize(layers * d_model);
    pre_hada_scale.resize(layers * d_model);
    q_norm_scale.resize(layers * geo.head_dim);
    k_norm_scale.resize(layers * geo.head_dim);
    attn_gate_scale.resize(layers);
    final_norm_scale.resize(d_model);
    engram_hash_tokens.resize(hash_window);
    engram_hash_valid.resize(hash_window);
    engram_hash_indices.resize(hash_window * tables);
    engram_ngram_ok.resize(hash_window * tables);
    engram_e_rows.resize(static_cast<uint64_t>(geo.engram_conv_taps) *
                         engram_e_dim);
    engram_v_taps.resize(static_cast<uint64_t>(geo.engram_conv_taps) * d_model);
    engram_tap_valid.resize(geo.engram_conv_taps);
    engram_keys.resize(static_cast<uint64_t>(geo.num_engram_sites) * d_model);
    engram_values.resize(static_cast<uint64_t>(geo.num_engram_sites) * d_model);
    cq_workspace.resize(compute_cq_workspace(contract_in));
    a8_quantized.resize(cq_workspace.size());
    a8_integer_values.resize(cq_workspace.size());
    const auto prepared_sizes = compute_prepared_sizes(contract_in);
    prepared_indices.resize(prepared_sizes.indices);
    prepared_indices_by_input32.resize(prepared_sizes.indices);
    prepared_norms.resize(prepared_sizes.norms);
    prepared_norms_by_group32.resize(prepared_sizes.norms_by_group32);
  }

  context(const context &) = delete;
  context &operator=(const context &) = delete;

  static uint64_t
  compute_max_order(const emel::cact::loader::geometry &geo) noexcept {
    uint64_t max_order = 1u;
    for (uint32_t i = 0u; i < geo.num_engram_orders && i < 4u; ++i)
      max_order =
          geo.engram_orders[i] > max_order ? geo.engram_orders[i] : max_order;
    return max_order;
  }

  static uint64_t compute_in_pad(const tensor_view &view) noexcept {
    const uint64_t group = view.group != 0u ? view.group : 1u;
    return (static_cast<uint64_t>(view.shape[1]) + group - 1u) / group * group;
  }

  static uint64_t compute_cq_workspace(const needle::contract &bound) noexcept {
    uint64_t workspace = compute_in_pad(bound.embedding);
    for (uint32_t i = 0u; i < bound.layer_count; ++i) {
      const uint64_t layer_pad = compute_in_pad(bound.layers[i].q_proj);
      workspace = layer_pad > workspace ? layer_pad : workspace;
      const uint64_t out_pad = compute_in_pad(bound.layers[i].out_proj);
      workspace = out_pad > workspace ? out_pad : workspace;
    }
    const uint64_t phi_pad = compute_in_pad(bound.mhc.phi_res);
    workspace = phi_pad > workspace ? phi_pad : workspace;
    for (uint32_t s = 0u; s < bound.engram_site_count; ++s) {
      const uint64_t site_pad = compute_in_pad(bound.engram_sites[s].key_proj);
      workspace = site_pad > workspace ? site_pad : workspace;
    }
    return workspace;
  }

  struct prepared_storage_sizes {
    uint64_t indices = 0u;
    uint64_t norms = 0u;
    uint64_t norms_by_group32 = 0u;
  };

  static prepared_storage_sizes
  compute_prepared_sizes(const needle::contract &bound) noexcept {
    prepared_storage_sizes sizes{};
    const auto add = [&](const tensor_view &view) {
      const uint64_t in_pad = compute_in_pad(view);
      const uint64_t groups_per_row = in_pad / view.group;
      sizes.indices += static_cast<uint64_t>(view.shape[0]) * in_pad;
      sizes.norms += static_cast<uint64_t>(view.shape[0]) * groups_per_row;
      sizes.norms_by_group32 +=
          static_cast<uint64_t>(view.shape[0] / 32u * 32u) * groups_per_row;
    };
    add(bound.embedding);
    for (uint32_t i = 0u; i < bound.layer_count; ++i) {
      add(bound.layers[i].q_proj);
      add(bound.layers[i].k_proj);
      add(bound.layers[i].v_proj);
      add(bound.layers[i].gate_proj);
      add(bound.layers[i].out_proj);
    }
    add(bound.mhc.phi_pre);
    add(bound.mhc.phi_post);
    add(bound.mhc.phi_res);
    for (uint32_t i = 0u; i < bound.engram_site_count; ++i) {
      add(bound.engram_sites[i].tables);
      add(bound.engram_sites[i].key_proj);
      add(bound.engram_sites[i].value_proj);
    }
    return sizes;
  }

  // Bound contract (named views over the mmapped .cact); outlives the graph.
  const needle::contract *bound = nullptr;

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
  std::vector<uint8_t> prepared_indices;
  std::vector<uint8_t> prepared_indices_by_input32;
  std::vector<float> prepared_norms;
  std::vector<float> prepared_norms_by_group32;
  emel::kernel::cq::event::prepared_codebook_q4 prepared_codebook = {};

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
  emel::kernel::zcrms::sm zcrms;
  emel::kernel::rope::sm rope;
  emel::kernel::swa::sm swa;
  emel::kernel::hadamard::sm hadamard;
  emel::kernel::engram::sm engram;
  emel::kernel::mhc::sm mhc;
};

} // namespace emel::model::needle::graph::action
