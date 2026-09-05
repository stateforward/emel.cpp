#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "emel/cact/loader/events.hpp"
#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/model/needle/errors.hpp"

namespace emel::model::needle {

using tensor_view = emel::cact::loader::tensor_view;
using geometry = emel::cact::loader::geometry;

// Contract capacities. The `.cact` header carries four engram-site slots, the
// probe-head manifest defines exactly two canonical head codes, and the layer
// bound caps the fixed-size contract storage (fixture ships 27 layers).
inline constexpr uint32_t k_max_layers = 64u;
inline constexpr uint32_t k_max_engram_sites = 4u;
inline constexpr uint32_t k_max_heads = 2u;
inline constexpr uint32_t k_layer_tensor_count = 14u;
inline constexpr uint32_t k_mhc_tensor_count = 9u;
inline constexpr uint32_t k_engram_site_tensor_count = 4u;
inline constexpr uint32_t k_head_tensor_count = 3u;
inline constexpr uint32_t k_head_code_contrastive = 1u;
inline constexpr uint32_t k_head_code_confidence = 2u;

// Named views over one transformer layer's positional tensor run, in exact
// `export.py _tensors()` emission order.
struct layer_views {
  tensor_view norm_in = {};
  tensor_view q_proj = {};
  tensor_view k_proj = {};
  tensor_view v_proj = {};
  tensor_view q_norm = {};
  tensor_view k_norm = {};
  tensor_view gate_proj = {};
  tensor_view out_proj = {};
  tensor_view post_norm = {};
  tensor_view attn_gate = {};
  tensor_view pre_hada = {};
  tensor_view d1 = {};
  tensor_view d2 = {};
  tensor_view d3 = {};
};

// The 9 mHC stack tensors (6 FP16 then 3 quantized phi blocks).
struct mhc_views {
  tensor_view a_pre = {};
  tensor_view a_post = {};
  tensor_view a_res = {};
  tensor_view b_pre = {};
  tensor_view b_post = {};
  tensor_view b_res = {};
  tensor_view phi_pre = {};
  tensor_view phi_post = {};
  tensor_view phi_res = {};
};

struct engram_site_views {
  tensor_view tables = {};
  tensor_view key_proj = {};
  tensor_view value_proj = {};
  tensor_view taps = {};
};

struct head_views {
  uint32_t code = 0u;
  tensor_view probes = {};
  tensor_view proj = {};
  tensor_view bias = {};
};

// Named needle model contract: every positional `.cact` tensor bound to its
// architectural role, validated against the header geometry. Phase 3/4 consume
// this instead of raw positional indices.
struct contract {
  geometry geo = {};
  tensor_view embedding = {};
  uint32_t layer_count = 0u;
  std::array<layer_views, k_max_layers> layers = {};
  mhc_views mhc = {};
  uint32_t engram_site_count = 0u;
  std::array<engram_site_views, k_max_engram_sites> engram_sites = {};
  tensor_view final_norm = {};
  tensor_view head_manifest = {};
  uint32_t head_count = 0u;
  std::array<head_views, k_max_heads> heads = {};
  bool has_tokenizer = false;
  tensor_view tokenizer_blob = {};
};

namespace events {

struct bind_done;
struct bind_error;

} // namespace events

namespace event {

using bind_done_fn = emel::callback<void(const events::bind_done &)>;
using bind_error_fn = emel::callback<void(const events::bind_error &)>;

// Binds the loader's positional tensor table to the named needle contract.
// The geometry and tensor views come straight from a completed
// `emel::cact::loader::sm` probe/bind/parse pass on the same file image.
struct bind {
  const geometry &geo;
  std::span<const tensor_view> tensors = {};
  contract &contract_out;
  const bind_done_fn &on_done;
  const bind_error_fn &on_error;

  bind(const geometry &geo_in, std::span<const tensor_view> tensors_in,
       contract &contract_out_in, const bind_done_fn &on_done_in,
       const bind_error_fn &on_error_in) noexcept
      : geo(geo_in), tensors(tensors_in), contract_out(contract_out_in),
        on_done(on_done_in), on_error(on_error_in) {}
};

struct bind_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct bind_runtime {
  const bind &request;
  bind_ctx &ctx;
};

} // namespace event

namespace events {

struct bind_done {
  const event::bind &request;
  const contract &contract_out;
};

struct bind_error {
  const event::bind &request;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace events

} // namespace emel::model::needle
