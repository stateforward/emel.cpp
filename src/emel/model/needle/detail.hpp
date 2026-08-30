#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include "emel/error/error.hpp"
#include "emel/model/needle/errors.hpp"
#include "emel/model/needle/events.hpp"

namespace emel::model::needle::detail {

namespace constants {

inline constexpr uint32_t dtype_fp16 = 1u;
inline constexpr uint32_t dtype_fp32 = 2u;
inline constexpr uint32_t dtype_cq = 3u;
inline constexpr uint32_t dtype_raw = 4u;

} // namespace constants

inline emel::error::type cast_needle_error(const error err) noexcept {
  return emel::error::cast(err);
}

// Expected geometry of one positional tensor role. `shape` entries beyond
// `ndim` must be zero in the directory record; a zero entry inside `ndim`
// marks a free dimension (validated by a cross-check, not the table).
struct role_spec {
  uint32_t dtype = 0u;
  uint32_t ndim = 0u;
  std::array<uint32_t, 4> shape = {0u, 0u, 0u, 0u};
};

// Compares one loader view against its expected role spec. Pure data-plane
// comparison; the caller folds the outcome into a single error code consumed
// by guards, never into control-flow selection here.
inline emel::error::type validate_role(const tensor_view &view,
                                       const role_spec &spec) noexcept {
  if (view.dtype != spec.dtype) {
    return cast_needle_error(error::tensor_dtype_mismatch);
  }
  if (view.ndim != spec.ndim) {
    return cast_needle_error(error::tensor_shape_mismatch);
  }
  for (uint32_t d = 0u; d < 4u; ++d) {
    const bool free_dim = d < spec.ndim && spec.shape[d] == 0u;
    if (!free_dim && view.shape[d] != spec.shape[d]) {
      return cast_needle_error(error::tensor_shape_mismatch);
    }
  }
  return cast_needle_error(error::none);
}

// Decodes an IEEE fp16 value (little-endian bytes) to f32. Subnormals decode
// as zero: this is metadata decoding for the integer-valued head manifest,
// not numeric kernel work.
inline float compute_f16_to_f32(const uint8_t *bytes) noexcept {
  const uint16_t half = static_cast<uint16_t>(
      static_cast<uint16_t>(bytes[0]) |
      static_cast<uint16_t>(static_cast<uint16_t>(bytes[1]) << 8u));
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16u;
  const uint32_t exponent = (static_cast<uint32_t>(half) >> 10u) & 0x1Fu;
  const uint32_t mantissa = static_cast<uint32_t>(half) & 0x3FFu;
  uint32_t bits = sign;
  if (exponent == 31u) {
    bits = sign | 0x7F800000u | (mantissa << 13u);
  }
  if (exponent > 0u && exponent < 31u) {
    bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
  }
  return std::bit_cast<float>(bits);
}

// Validates the header geometry against the fixed contract capacities and the
// derived-dimension invariants the role table depends on.
inline emel::error::type validate_geometry(const geometry &geo) noexcept {
  if (geo.d_model == 0u || geo.vocab_size == 0u || geo.num_heads == 0u ||
      geo.num_kv_heads == 0u || geo.head_dim == 0u || geo.hada_n == 0u ||
      geo.mhc_lanes == 0u) {
    return cast_needle_error(error::geometry_invalid);
  }
  if (geo.num_layers == 0u || geo.num_layers > k_max_layers) {
    return cast_needle_error(error::geometry_invalid);
  }
  if (geo.num_engram_sites > k_max_engram_sites) {
    return cast_needle_error(error::geometry_invalid);
  }
  if (geo.num_engram_sites > 0u &&
      (geo.engram_slots == 0u || geo.engram_sub_dim == 0u ||
       geo.num_engram_tables == 0u || geo.engram_conv_taps == 0u)) {
    return cast_needle_error(error::geometry_invalid);
  }
  return cast_needle_error(error::none);
}

// Number of positional tensors through final_norm (before optional heads and
// the trailing RAW tokenizer blob).
inline uint64_t compute_base_tensor_count(const geometry &geo) noexcept {
  return 1u + static_cast<uint64_t>(geo.num_layers) * k_layer_tensor_count +
         k_mhc_tensor_count +
         static_cast<uint64_t>(geo.num_engram_sites) *
             k_engram_site_tensor_count +
         1u;
}

// Maps and validates the per-layer tensor run [norm_in .. d3] starting at
// `views[0]`, writing named views into `layer_out`. Emission order is the
// exact `export.py _tensors()` per-layer order.
inline emel::error::type bind_layer(const std::span<const tensor_view> views,
                                    const geometry &geo,
                                    layer_views &layer_out) noexcept {
  const uint32_t d = geo.d_model;
  const uint32_t attn_dim = geo.num_heads * geo.head_dim;
  const uint32_t kv_dim = geo.num_kv_heads * geo.head_dim;
  const std::array<role_spec, k_layer_tensor_count> specs = {{
      {constants::dtype_fp16, 1u, {d, 0u, 0u, 0u}},            // norm_in
      {constants::dtype_cq, 2u, {attn_dim, d, 0u, 0u}},        // q_proj
      {constants::dtype_cq, 2u, {kv_dim, d, 0u, 0u}},          // k_proj
      {constants::dtype_cq, 2u, {kv_dim, d, 0u, 0u}},          // v_proj
      {constants::dtype_fp16, 1u, {geo.head_dim, 0u, 0u, 0u}}, // q_norm
      {constants::dtype_fp16, 1u, {geo.head_dim, 0u, 0u, 0u}}, // k_norm
      {constants::dtype_cq, 2u, {attn_dim, d, 0u, 0u}},        // gate_proj
      {constants::dtype_cq, 2u, {d, attn_dim, 0u, 0u}},        // out_proj
      {constants::dtype_fp16, 1u, {d, 0u, 0u, 0u}},            // post_norm
      {constants::dtype_fp16, 1u, {1u, 0u, 0u, 0u}},           // attn_gate
      {constants::dtype_fp16, 1u, {d, 0u, 0u, 0u}},            // pre_hada
      {constants::dtype_fp16, 1u, {geo.hada_n, 0u, 0u, 0u}},   // d1
      {constants::dtype_fp16, 1u, {geo.hada_n, 0u, 0u, 0u}},   // d2
      {constants::dtype_fp16, 1u, {geo.hada_n, 0u, 0u, 0u}},   // d3
  }};

  std::array<tensor_view *, k_layer_tensor_count> slots = {
      &layer_out.norm_in,   &layer_out.q_proj,   &layer_out.k_proj,
      &layer_out.v_proj,    &layer_out.q_norm,   &layer_out.k_norm,
      &layer_out.gate_proj, &layer_out.out_proj, &layer_out.post_norm,
      &layer_out.attn_gate, &layer_out.pre_hada, &layer_out.d1,
      &layer_out.d2,        &layer_out.d3,
  };

  for (uint32_t t = 0u; t < k_layer_tensor_count; ++t) {
    const emel::error::type err = validate_role(views[t], specs[t]);
    if (err != cast_needle_error(error::none)) {
      return err;
    }
    *slots[t] = views[t];
  }
  return cast_needle_error(error::none);
}

// Maps and validates the 9-tensor mHC stack starting at `views[0]`:
// a_pre/a_post/a_res (L,), b_pre/b_post (L,n), b_res (L,n,n), then the
// quantized phi blocks reshaped to [L*n, n*d] / [L*n*n, n*d].
inline emel::error::type bind_mhc(const std::span<const tensor_view> views,
                                  const geometry &geo,
                                  mhc_views &mhc_out) noexcept {
  const uint32_t layers = geo.num_layers;
  const uint32_t lanes = geo.mhc_lanes;
  const uint32_t nc = lanes * geo.d_model;
  const std::array<role_spec, k_mhc_tensor_count> specs = {{
      {constants::dtype_fp16, 1u, {layers, 0u, 0u, 0u}},       // a_pre
      {constants::dtype_fp16, 1u, {layers, 0u, 0u, 0u}},       // a_post
      {constants::dtype_fp16, 1u, {layers, 0u, 0u, 0u}},       // a_res
      {constants::dtype_fp16, 2u, {layers, lanes, 0u, 0u}},    // b_pre
      {constants::dtype_fp16, 2u, {layers, lanes, 0u, 0u}},    // b_post
      {constants::dtype_fp16, 3u, {layers, lanes, lanes, 0u}}, // b_res
      {constants::dtype_cq, 2u, {layers * lanes, nc, 0u, 0u}}, // phi_pre
      {constants::dtype_cq, 2u, {layers * lanes, nc, 0u, 0u}}, // phi_post
      {constants::dtype_cq,
       2u,
       {layers * lanes * lanes, nc, 0u, 0u}}, // phi_res
  }};

  std::array<tensor_view *, k_mhc_tensor_count> slots = {
      &mhc_out.a_pre,   &mhc_out.a_post,   &mhc_out.a_res,
      &mhc_out.b_pre,   &mhc_out.b_post,   &mhc_out.b_res,
      &mhc_out.phi_pre, &mhc_out.phi_post, &mhc_out.phi_res,
  };

  for (uint32_t t = 0u; t < k_mhc_tensor_count; ++t) {
    const emel::error::type err = validate_role(views[t], specs[t]);
    if (err != cast_needle_error(error::none)) {
      return err;
    }
    *slots[t] = views[t];
  }
  return cast_needle_error(error::none);
}

// Maps and validates one engram site's 4-tensor run starting at `views[0]`:
// tables [tables*slots, sub_dim], key_proj [d, tables*sub_dim],
// value_proj [d, tables*sub_dim], taps (conv_taps, d).
inline emel::error::type
bind_engram_site(const std::span<const tensor_view> views, const geometry &geo,
                 engram_site_views &site_out) noexcept {
  const uint32_t d = geo.d_model;
  const uint32_t table_rows = geo.num_engram_tables * geo.engram_slots;
  const uint32_t embed_dim = geo.num_engram_tables * geo.engram_sub_dim;
  const std::array<role_spec, k_engram_site_tensor_count> specs = {{
      {constants::dtype_cq, 2u, {table_rows, geo.engram_sub_dim, 0u, 0u}},
      {constants::dtype_cq, 2u, {d, embed_dim, 0u, 0u}},
      {constants::dtype_cq, 2u, {d, embed_dim, 0u, 0u}},
      {constants::dtype_fp16, 2u, {geo.engram_conv_taps, d, 0u, 0u}},
  }};

  std::array<tensor_view *, k_engram_site_tensor_count> slots = {
      &site_out.tables,
      &site_out.key_proj,
      &site_out.value_proj,
      &site_out.taps,
  };

  for (uint32_t t = 0u; t < k_engram_site_tensor_count; ++t) {
    const emel::error::type err = validate_role(views[t], specs[t]);
    if (err != cast_needle_error(error::none)) {
      return err;
    }
    *slots[t] = views[t];
  }
  return cast_needle_error(error::none);
}

// Maps and validates the optional probe-head section: a `heads.manifest`
// FP16 vector of canonical head codes followed by [probes, proj, bias]
// triples. `views` spans exactly the head tensors (manifest included).
inline emel::error::type bind_heads(const std::span<const tensor_view> views,
                                    const geometry &geo,
                                    contract &contract_out) noexcept {
  const uint64_t extra = views.size();
  if (extra == 0u) {
    contract_out.head_count = 0u;
    return cast_needle_error(error::none);
  }
  if ((extra - 1u) % k_head_tensor_count != 0u) {
    return cast_needle_error(error::head_manifest_invalid);
  }
  const uint64_t head_count = (extra - 1u) / k_head_tensor_count;
  if (head_count == 0u || head_count > k_max_heads) {
    return cast_needle_error(error::head_manifest_invalid);
  }

  const role_spec manifest_spec = {
      constants::dtype_fp16,
      1u,
      {static_cast<uint32_t>(head_count), 0u, 0u, 0u}};
  const emel::error::type manifest_err = validate_role(views[0], manifest_spec);
  if (manifest_err != cast_needle_error(error::none)) {
    return manifest_err;
  }
  if (views[0].data == nullptr ||
      views[0].nbytes < head_count * sizeof(uint16_t)) {
    return cast_needle_error(error::head_manifest_invalid);
  }
  contract_out.head_manifest = views[0];

  const uint32_t d = geo.d_model;
  for (uint64_t h = 0u; h < head_count; ++h) {
    const float code_value =
        compute_f16_to_f32(views[0].data + h * sizeof(uint16_t));
    const uint32_t code = static_cast<uint32_t>(code_value);
    if (static_cast<float>(code) != code_value ||
        (code != k_head_code_contrastive && code != k_head_code_confidence)) {
      return cast_needle_error(error::head_manifest_invalid);
    }

    const tensor_view &probes = views[1u + h * k_head_tensor_count];
    const tensor_view &proj = views[2u + h * k_head_tensor_count];
    const tensor_view &bias = views[3u + h * k_head_tensor_count];

    const role_spec probes_spec = {constants::dtype_fp16, 2u, {0u, d, 0u, 0u}};
    const emel::error::type probes_err = validate_role(probes, probes_spec);
    if (probes_err != cast_needle_error(error::none)) {
      return probes_err;
    }
    if (probes.shape[0] == 0u) {
      return cast_needle_error(error::tensor_shape_mismatch);
    }

    const role_spec proj_spec = {
        constants::dtype_fp16, 2u, {0u, probes.shape[0] * d, 0u, 0u}};
    const emel::error::type proj_err = validate_role(proj, proj_spec);
    if (proj_err != cast_needle_error(error::none)) {
      return proj_err;
    }
    if (proj.shape[0] == 0u) {
      return cast_needle_error(error::tensor_shape_mismatch);
    }

    const role_spec bias_spec = {
        constants::dtype_fp16, 1u, {proj.shape[0], 0u, 0u, 0u}};
    const emel::error::type bias_err = validate_role(bias, bias_spec);
    if (bias_err != cast_needle_error(error::none)) {
      return bias_err;
    }

    head_views &head_out = contract_out.heads[h];
    head_out.code = code;
    head_out.probes = probes;
    head_out.proj = proj;
    head_out.bias = bias;
  }
  contract_out.head_count = static_cast<uint32_t>(head_count);
  return cast_needle_error(error::none);
}

// Full positional binding pass: reproduces `export.py _tensors()` emission
// order (embedding; per-layer runs; mHC stack; per-site engram runs;
// final_norm; optional heads; trailing RAW tokenizer) and validates every
// tensor's dtype/rank/shape against the geometry-derived role table. Bulk
// data-plane iteration folding each outcome into a single error code; the
// owning machine's guards dispatch on that code.
inline emel::error::type
bind_contract(const geometry &geo, const std::span<const tensor_view> tensors,
              contract &contract_out) noexcept {
  contract_out = {};

  const emel::error::type geo_err = validate_geometry(geo);
  if (geo_err != cast_needle_error(error::none)) {
    return geo_err;
  }

  if (tensors.size() != geo.num_tensors) {
    return cast_needle_error(error::tensor_count_mismatch);
  }

  const uint64_t base = compute_base_tensor_count(geo);
  if (tensors.size() < base) {
    return cast_needle_error(error::tensor_count_mismatch);
  }

  contract_out.geo = geo;

  uint64_t index = 0u;

  const role_spec embedding_spec = {
      constants::dtype_cq, 2u, {geo.vocab_size, geo.d_model, 0u, 0u}};
  const emel::error::type embedding_err =
      validate_role(tensors[index], embedding_spec);
  if (embedding_err != cast_needle_error(error::none)) {
    return embedding_err;
  }
  contract_out.embedding = tensors[index];
  index += 1u;

  for (uint32_t layer = 0u; layer < geo.num_layers; ++layer) {
    const emel::error::type layer_err =
        bind_layer(tensors.subspan(index, k_layer_tensor_count), geo,
                   contract_out.layers[layer]);
    if (layer_err != cast_needle_error(error::none)) {
      return layer_err;
    }
    index += k_layer_tensor_count;
  }
  contract_out.layer_count = geo.num_layers;

  const emel::error::type mhc_err = bind_mhc(
      tensors.subspan(index, k_mhc_tensor_count), geo, contract_out.mhc);
  if (mhc_err != cast_needle_error(error::none)) {
    return mhc_err;
  }
  index += k_mhc_tensor_count;

  for (uint32_t site = 0u; site < geo.num_engram_sites; ++site) {
    const emel::error::type site_err =
        bind_engram_site(tensors.subspan(index, k_engram_site_tensor_count),
                         geo, contract_out.engram_sites[site]);
    if (site_err != cast_needle_error(error::none)) {
      return site_err;
    }
    index += k_engram_site_tensor_count;
  }
  contract_out.engram_site_count = geo.num_engram_sites;

  const role_spec final_norm_spec = {
      constants::dtype_fp16, 1u, {geo.d_model, 0u, 0u, 0u}};
  const emel::error::type final_norm_err =
      validate_role(tensors[index], final_norm_spec);
  if (final_norm_err != cast_needle_error(error::none)) {
    return final_norm_err;
  }
  contract_out.final_norm = tensors[index];
  index += 1u;

  // The trailing RAW tokenizer blob is present iff the last tensor is RAW;
  // everything between final_norm and it is the optional head section.
  uint64_t head_end = tensors.size();
  const bool has_tokenizer =
      head_end > index && tensors[head_end - 1u].dtype == constants::dtype_raw;
  if (has_tokenizer) {
    const role_spec tokenizer_spec = {
        constants::dtype_raw, 0u, {0u, 0u, 0u, 0u}};
    const emel::error::type tokenizer_err =
        validate_role(tensors[head_end - 1u], tokenizer_spec);
    if (tokenizer_err != cast_needle_error(error::none)) {
      return tokenizer_err;
    }
    contract_out.tokenizer_blob = tensors[head_end - 1u];
    contract_out.has_tokenizer = true;
    head_end -= 1u;
  }

  const emel::error::type heads_err =
      bind_heads(tensors.subspan(index, head_end - index), geo, contract_out);
  if (heads_err != cast_needle_error(error::none)) {
    return heads_err;
  }

  return cast_needle_error(error::none);
}

} // namespace emel::model::needle::detail
