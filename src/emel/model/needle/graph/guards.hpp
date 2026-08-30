#pragma once

#include "emel/model/needle/graph/context.hpp"
#include "emel/model/needle/graph/events.hpp"

namespace emel::model::needle::graph::guard {

// Compile-time CQ route availability: the AVX2 chain exists in the table on
// every host, but its entry guard is constant-false unless the build actually
// carries AVX2+FMA codegen (mirrors emel::kernel::cq::guard::avx2_supported).
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
inline constexpr bool k_avx2_route_available = true;
#else
inline constexpr bool k_avx2_route_available = false;
#endif

inline bool is_power_of_two(const uint32_t value) noexcept {
  return value != 0u && (value & (value - 1u)) == 0u;
}
inline bool cq_group128(const emel::model::needle::contract &bound) noexcept {
  const auto supported = [](const tensor_view &view) noexcept {
    return view.group == 128u;
  };
  bool ok = supported(bound.embedding) && supported(bound.mhc.phi_pre) &&
            supported(bound.mhc.phi_post) && supported(bound.mhc.phi_res);
  for (uint32_t i = 0u; i < bound.layer_count; ++i) {
    const auto &layer = bound.layers[i];
    ok = ok && supported(layer.q_proj) && supported(layer.k_proj) &&
         supported(layer.v_proj) && supported(layer.gate_proj) &&
         supported(layer.out_proj);
  }
  for (uint32_t i = 0u; i < bound.engram_site_count; ++i) {
    const auto &site = bound.engram_sites[i];
    ok = ok && supported(site.tables) && supported(site.key_proj) &&
         supported(site.value_proj);
  }
  return ok;
}


inline bool layer_is_engram_site(const emel::cact::loader::geometry &geo,
                                 const uint32_t layer_index) noexcept {
  bool match = false;
  for (uint32_t s = 0u; s < geo.num_engram_sites && s < 4u; ++s)
    match = match || geo.engram_sites[s] == layer_index;
  return match;
}

// Compile-time route split for the completion chain: exactly one of the two
// passes on any given build.
struct guard_route_avx2 {
  bool operator()(const event::step_run &,
                  const action::context &ctx) const noexcept {
    return k_avx2_route_available && cq_group128(*ctx.bound);
  }
};

struct guard_route_scalar {
  bool operator()(const event::step_run &,
                  const action::context &ctx) const noexcept {
    return !k_avx2_route_available || !cq_group128(*ctx.bound);
  }
};

// The `.cact` deployment header carries kv_bits=8 for the W4A8 artifact. The
// legacy f32 parity route remains explicit for synthetic/legacy contracts;
// production route selection never hides this numeric mode inside actions.
struct guard_deployment_a8 {
  bool operator()(const event::step_run &ev,
                  const action::context &) const noexcept {
    return ev.ctx.activation_quant;
  }
};

struct guard_deployment_f32 {
  bool operator()(const event::step_run &ev,
                  const action::context &) const noexcept {
    return !ev.ctx.activation_quant;
  }
};

// Geometry the graph storage layout supports: derived buffer sizes must be
// coherent and within the fixed kernel capacities. The context constructor
// binds `bound` from a reference, so it is never null. The explicit CQ
// routes are 4-bit: the pinned deployment format (route-w4-qat) packs every
// quantized tensor as q4, checked here so the guarded q4 GEMV chain is the
// only dispatch path.
struct guard_init_supported {
  bool operator()(const event::init_run &,
                  const action::context &ctx) const noexcept {
    const auto &geo = ctx.bound->geo;
    return geo.d_model > 0u && geo.num_heads > 0u && geo.num_kv_heads > 0u &&
           (geo.num_heads % geo.num_kv_heads) == 0u && geo.head_dim >= 2u &&
           (geo.head_dim % 2u) == 0u &&
           geo.num_heads * geo.head_dim >= geo.d_model && geo.num_layers > 0u &&
           ctx.bound->layer_count == geo.num_layers &&
           is_power_of_two(geo.hada_n) && geo.hada_n >= geo.d_model &&
           geo.mhc_lanes > 0u &&
           geo.mhc_lanes <= emel::kernel::mhc::event::k_max_lanes &&
           geo.num_engram_orders <= emel::kernel::engram::event::k_max_orders &&
           (geo.num_engram_sites == 0u ||
            (geo.num_engram_orders > 0u && geo.num_engram_tables > 0u &&
             (geo.num_engram_tables % geo.num_engram_orders) == 0u &&
             geo.engram_conv_taps > 0u && geo.engram_conv_dilation > 0u &&
             geo.engram_slots > 0u)) &&
           ctx.bound->engram_site_count == geo.num_engram_sites &&
           geo.kv_window > 0u && geo.max_seq_len > 0u && geo.vocab_size > 0u &&
           geo.rope_theta > 0.0f && ctx.bound->embedding.bits == 4u;
  }
};

struct guard_init_unsupported {
  bool operator()(const event::init_run &ev,
                  const action::context &ctx) const noexcept {
    return !guard_init_supported{}(ev, ctx);
  }
};

struct guard_init_ok {
  bool operator()(const event::init_run &ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct guard_init_failed {
  bool operator()(const event::init_run &ev,
                  const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

// A step is valid when the token id is in vocab, a cache slot remains, and a
// logits request carries a large enough destination span.
struct guard_step_valid {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    const auto &geo = ctx.bound->geo;
    const bool logits_ok =
        !ev.ctx.want_logits || ev.ctx.logits_out.size() >= geo.vocab_size;
    return ev.ctx.token >= 0 &&
           static_cast<uint32_t>(ev.ctx.token) < geo.vocab_size &&
           ctx.position < geo.max_seq_len && logits_ok;
  }
};

struct guard_step_invalid {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return !guard_step_valid{}(ev, ctx);
  }
};

struct guard_step_ok {
  bool operator()(const event::step_run &ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct guard_step_failed {
  bool operator()(const event::step_run &ev,
                  const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

struct guard_engram_present {
  bool operator()(const event::step_run &,
                  const action::context &ctx) const noexcept {
    return ctx.bound->geo.num_engram_sites > 0u;
  }
};

struct guard_engram_absent {
  bool operator()(const event::step_run &,
                  const action::context &ctx) const noexcept {
    return ctx.bound->geo.num_engram_sites == 0u;
  }
};

struct guard_layer_engram_site {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_step_ok{}(ev, ctx) &&
           layer_is_engram_site(ctx.bound->geo, ev.ctx.layer_index);
  }
};

struct guard_layer_plain {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_step_ok{}(ev, ctx) &&
           !layer_is_engram_site(ctx.bound->geo, ev.ctx.layer_index);
  }
};

// Sliding mask states: growing while fewer than kv_window positions exist,
// full once the window saturates and the oldest positions fall out.
struct guard_window_growing {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_step_ok{}(ev, ctx) &&
           ctx.position + 1u <= ctx.bound->geo.kv_window;
  }
};

struct guard_window_full {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_step_ok{}(ev, ctx) &&
           ctx.position + 1u > ctx.bound->geo.kv_window;
  }
};

struct guard_more_layers {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_step_ok{}(ev, ctx) &&
           ev.ctx.layer_index + 1u < ctx.bound->geo.num_layers;
  }
};

struct guard_layers_done_want_logits {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_step_ok{}(ev, ctx) &&
           ev.ctx.layer_index + 1u >= ctx.bound->geo.num_layers &&
           ev.ctx.want_logits;
  }
};

struct guard_layers_done_no_logits {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_step_ok{}(ev, ctx) &&
           ev.ctx.layer_index + 1u >= ctx.bound->geo.num_layers &&
           !ev.ctx.want_logits;
  }
};

//------------------------------------------------------------------------------//
// Composed transition guards (single functors per SML row).
//------------------------------------------------------------------------------//

struct guard_step_valid_avx2 {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_step_valid{}(ev, ctx) && k_avx2_route_available;
  }
};

struct guard_step_valid_scalar {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_step_valid{}(ev, ctx) && !k_avx2_route_available;
  }
};

struct guard_engram_present_ok {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_step_ok{}(ev, ctx) && guard_engram_present{}(ev, ctx);
  }
};

struct guard_engram_absent_ok {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_step_ok{}(ev, ctx) && guard_engram_absent{}(ev, ctx);
  }
};

struct guard_layer_engram_growing {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_layer_engram_site{}(ev, ctx) &&
           guard_window_growing{}(ev, ctx);
  }
};

struct guard_layer_engram_full {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_layer_engram_site{}(ev, ctx) && guard_window_full{}(ev, ctx);
  }
};

struct guard_layer_plain_growing {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_layer_plain{}(ev, ctx) && guard_window_growing{}(ev, ctx);
  }
};

struct guard_layer_plain_full {
  bool operator()(const event::step_run &ev,
                  const action::context &ctx) const noexcept {
    return guard_layer_plain{}(ev, ctx) && guard_window_full{}(ev, ctx);
  }
};

} // namespace emel::model::needle::graph::guard
