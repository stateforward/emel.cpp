#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "emel/kernel/detail.hpp"
#include "emel/kernel/f32_matvec/context.hpp"
#include "emel/kernel/f32_matvec/events.hpp"

namespace emel::kernel::f32_matvec::action {

template <class source_type, class load_type>
inline void compute_pack_x4(const std::span<const source_type> source,
                            const std::span<float> destination,
                            const uint64_t inner, const uint64_t rows,
                            const load_type load) noexcept {
  const uint64_t full_rows = rows - rows % 4u;
  for (uint64_t row = 0u; row < full_rows; row += 4u) {
    float *packed = destination.data() + row * inner;
    for (uint64_t column = 0u; column < inner; ++column) {
      packed[column * 4u + 0u] = load(source[(row + 0u) * inner + column]);
      packed[column * 4u + 1u] = load(source[(row + 1u) * inner + column]);
      packed[column * 4u + 2u] = load(source[(row + 2u) * inner + column]);
      packed[column * 4u + 3u] = load(source[(row + 3u) * inner + column]);
    }
  }
  for (uint64_t row = full_rows; row < rows; ++row) {
    for (uint64_t column = 0u; column < inner; ++column) {
      destination[row * inner + column] = load(source[row * inner + column]);
    }
  }
}

inline float compute_reference_row(const float *weights, const float *input,
                                   const uint64_t inner) noexcept {
  double sum = 0.0;
  for (uint64_t column = 0u; column < inner; ++column) {
    sum += static_cast<double>(weights[column] * input[column]);
  }
  return static_cast<float>(sum);
}

struct effect_prepare_f32 {
  void operator()(const event::prepare_f32 &ev, context &ctx) const noexcept {
    compute_pack_x4(ev.request.source, ev.request.destination, ev.request.inner,
                    ev.request.rows,
                    [](const float value) noexcept { return value; });
    ++ctx.prepare_calls;
    ctx.prepared_floats += ev.request.inner * ev.request.rows;
    ev.result.accepted = true;
  }
};

struct effect_prepare_f16 {
  void operator()(const event::prepare_f16 &ev, context &ctx) const noexcept {
    compute_pack_x4(ev.request.source, ev.request.destination, ev.request.inner,
                    ev.request.rows, [](const uint16_t value) noexcept {
                      return emel::kernel::detail::quant::fp16_to_fp32(value);
                    });
    ++ctx.prepare_calls;
    ctx.prepared_floats += ev.request.inner * ev.request.rows;
    ev.result.accepted = true;
  }
};

struct effect_execute_reference {
  void operator()(const event::execute_reference &ev,
                  context &ctx) const noexcept {
    for (uint64_t row = 0u; row < ev.request.rows; ++row) {
      ev.request.output[row] = compute_reference_row(
          ev.request.weights.data() + row * ev.request.inner,
          ev.request.input.data(), ev.request.inner);
    }
    ++ctx.reference_calls;
    ev.result.accepted = true;
  }
};

struct effect_execute_exact_x4 {
  void operator()(const event::execute_exact_x4 &ev,
                  context &ctx) const noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    const uint64_t full_rows = ev.request.rows - ev.request.rows % 4u;
    for (uint64_t row = 0u; row < full_rows; row += 4u) {
      const float *weights = ev.request.weights.data() + row * ev.request.inner;
      float64x2_t sums_low = vdupq_n_f64(0.0);
      float64x2_t sums_high = vdupq_n_f64(0.0);
      for (uint64_t column = 0u; column < ev.request.inner; ++column) {
        const float32x4_t products = vmulq_n_f32(
            vld1q_f32(weights + column * 4u), ev.request.input[column]);
        sums_low = vaddq_f64(sums_low, vcvt_f64_f32(vget_low_f32(products)));
        sums_high = vaddq_f64(sums_high, vcvt_f64_f32(vget_high_f32(products)));
      }
      const float32x2_t output_low = vcvt_f32_f64(sums_low);
      const float32x2_t output_high = vcvt_f32_f64(sums_high);
      vst1_f32(ev.request.output.data() + row, output_low);
      vst1_f32(ev.request.output.data() + row + 2u, output_high);
    }
    for (uint64_t row = full_rows; row < ev.request.rows; ++row) {
      ev.request.output[row] = compute_reference_row(
          ev.request.weights.data() + row * ev.request.inner,
          ev.request.input.data(), ev.request.inner);
    }
    ++ctx.exact_x4_calls;
    ev.result.accepted = true;
#else
    (void)ev;
    (void)ctx;
#endif
  }
};

template <class event_type> struct effect_reject {
  void operator()(const event_type &ev, context &) const noexcept {
    ev.result.accepted = false;
  }
};

template <class event_type> struct effect_accept {
  void operator()(const event_type &, context &) const noexcept {}
};

template <class event_type> struct effect_emit_done {
  void operator()(const event_type &ev, context &) const noexcept {
    ev.on_done(events::dispatch_done<event_type>{.request = ev});
  }
};

template <class event_type> struct effect_emit_error {
  void operator()(const event_type &ev, context &) const noexcept {
    ev.on_error(events::dispatch_error<event_type>{.request = ev});
  }
};

struct effect_reject_prepare_f32 : effect_reject<event::prepare_f32> {};
struct effect_accept_prepare_f32 : effect_accept<event::prepare_f32> {};
struct effect_emit_prepare_f32_done : effect_emit_done<event::prepare_f32> {};
struct effect_emit_prepare_f32_error : effect_emit_error<event::prepare_f32> {};
struct effect_reject_prepare_f16 : effect_reject<event::prepare_f16> {};
struct effect_accept_prepare_f16 : effect_accept<event::prepare_f16> {};
struct effect_emit_prepare_f16_done : effect_emit_done<event::prepare_f16> {};
struct effect_emit_prepare_f16_error : effect_emit_error<event::prepare_f16> {};
struct effect_reject_execute_reference
    : effect_reject<event::execute_reference> {};
struct effect_accept_execute_reference
    : effect_accept<event::execute_reference> {};
struct effect_emit_execute_reference_done
    : effect_emit_done<event::execute_reference> {};
struct effect_emit_execute_reference_error
    : effect_emit_error<event::execute_reference> {};
struct effect_reject_execute_exact_x4 : effect_reject<event::execute_exact_x4> {
};
struct effect_accept_execute_exact_x4 : effect_accept<event::execute_exact_x4> {
};
struct effect_emit_execute_exact_x4_done
    : effect_emit_done<event::execute_exact_x4> {};
struct effect_emit_execute_exact_x4_error
    : effect_emit_error<event::execute_exact_x4> {};

struct effect_capture_diagnostics {
  void operator()(const event::capture_diagnostics &ev,
                  context &ctx) const noexcept {
    ev.out = event::diagnostics{
        .prepare_calls = ctx.prepare_calls,
        .prepared_floats = ctx.prepared_floats,
        .reference_calls = ctx.reference_calls,
        .exact_x4_calls = ctx.exact_x4_calls,
    };
    ev.result.accepted = true;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.result.accepted; }) {
      ev.result.accepted = false;
    }
  }
};

} // namespace emel::kernel::f32_matvec::action
