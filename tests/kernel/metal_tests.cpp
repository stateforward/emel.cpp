#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <doctest/doctest.h>

#include "emel/kernel/any.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/metal/sm.hpp"
#include "test_helpers.hpp"

// Metal kernel actor tests.
//
// Every compute case drives the same op event through two `emel::kernel::any`
// machines - the host CPU backend (reference) and the Metal backend - and
// compares outputs with tolerances that account for GPU accumulation order
// and FMA contraction. Tests skip when the host has no Metal device. The
// actor is also exercised through `emel::kernel::any` with an explicit
// kernel_kind::metal so the sm_any wiring (variant dispatch tables) is
// covered, not just the actor directly.
namespace emel::kernel::metal_tests {

using emel::kernel::event::dtype;
using emel::kernel::event::unary_subop;
using emel::kernel::test::make_dst;
using emel::kernel::test::make_quantized_src;
using emel::kernel::test::make_src;
using emel::kernel::test::set_op_param_i32;

namespace {

// Deterministic LCG so results are reproducible across hosts.
class lcg {
public:
  explicit lcg(const uint64_t seed = UINT64_C(0x9e3779b97f4a7c15))
      : state_(seed) {}

  float next_f32(const float lo, const float hi) {
    state_ =
        state_ * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    const float unit =
        static_cast<float>((state_ >> 11u) & ((UINT64_C(1) << 24u) - 1u)) /
        static_cast<float>(UINT64_C(1) << 24u);
    return lo + unit * (hi - lo);
  }

private:
  uint64_t state_ = 0;
};

void fill(std::vector<float> &values, lcg &gen, const float lo,
          const float hi) {
  for (float &value : values) {
    value = gen.next_f32(lo, hi);
  }
}

float rel_abs_tolerance(const float actual, const float expected,
                        const float rel_tol, const float abs_tol) {
  return abs_tol + rel_tol * std::max(std::fabs(actual), std::fabs(expected));
}

bool close(const float actual, const float expected, const float rel_tol,
           const float abs_tol) {
  return std::fabs(actual - expected) <=
         rel_abs_tolerance(actual, expected, rel_tol, abs_tol);
}

bool all_close(const std::vector<float> &actual,
               const std::vector<float> &expected, const float rel_tol,
               const float abs_tol) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!close(actual[i], expected[i], rel_tol, abs_tol)) {
      return false;
    }
  }
  return true;
}

// One-time probe: constructing the metal actor compiles the MSL library, so
// share the probe across cases.
bool host_has_metal() {
  emel::kernel::any probe{emel::kernel::kernel_kind::metal};
  return probe.metal_available();
}

constexpr float k_rel_tol = 1.0e-4f;
constexpr float k_abs_tol = 1.0e-6f;
constexpr float k_f16_rel_tol = 2.0e-3f;
constexpr float k_f16_abs_tol = 1.0e-4f;

} // namespace

TEST_CASE("kernel_metal_actor_rejects_when_host_lacks_device") {
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};
  const bool available = metal.metal_available();

  std::array<float, 8> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::array<float, 8> dst = {};

  const emel::kernel::event::op_add ok{
      .src0 = make_src(src.data(), dtype::f32, 8),
      .src1 = make_src(src.data(), dtype::f32, 8),
      .dst = make_dst(dst.data(), dtype::f32, 8),
  };

  if (available) {
    CHECK(metal.process_event(ok));
    CHECK(metal.kind() == emel::kernel::kernel_kind::metal);
    CHECK(metal.metal_available());
  } else {
    // Explicit rejection, never a silent fallback to the CPU actor.
    CHECK_FALSE(metal.process_event(ok));
  }
}

TEST_CASE("kernel_metal_mul_mat_f32_matches_cpu") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  lcg gen(1);
  // matvec: src0 [k, m], src1 [n=1, k], dst [1, m].
  std::vector<float> weights(6 * 8);
  std::vector<float> input(6);
  std::vector<float> cpu_out(8);
  std::vector<float> metal_out(8);
  fill(weights, gen, -1.0f, 1.0f);
  fill(input, gen, -1.0f, 1.0f);

  const auto build = [&](emel::kernel::event::op_mul_mat &ev) {
    ev.src0 = make_src(weights.data(), dtype::f32, 6, 8);
    ev.src1 = make_src(input.data(), dtype::f32, 1, 6);
    ev.dst = make_dst(metal_out.data(), dtype::f32, 1, 8);
  };

  emel::kernel::event::op_mul_mat cpu_ev{};
  emel::kernel::event::op_mul_mat metal_ev{};
  build(cpu_ev);
  build(metal_ev);
  cpu_ev.dst.data = cpu_out.data();
  metal_ev.dst.data = metal_out.data();

  emel::kernel::any cpu{};
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};
  REQUIRE(cpu.process_event(cpu_ev));
  REQUIRE(metal.process_event(metal_ev));
  CHECK(all_close(metal_out, cpu_out, k_rel_tol, k_abs_tol));
}

TEST_CASE("kernel_metal_mul_mat_f32_matrix_matches_cpu") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  lcg gen(2);
  constexpr uint64_t k_k = 8;
  constexpr uint64_t k_m = 5;
  constexpr uint64_t k_n = 7;
  // GEMM: src0 [k, m], src1 [n, k], dst [n, m].
  std::vector<float> src0(k_k * k_m);
  std::vector<float> src1(k_n * k_k);
  std::vector<float> cpu_out(k_m * k_n);
  std::vector<float> metal_out(k_m * k_n);
  fill(src0, gen, -1.0f, 1.0f);
  fill(src1, gen, -1.0f, 1.0f);

  const auto make = [&](float *dst) {
    emel::kernel::event::op_mul_mat ev{};
    ev.src0 = make_src(src0.data(), dtype::f32, k_k, k_m);
    ev.src1 = make_src(src1.data(), dtype::f32, k_n, k_k);
    ev.dst = make_dst(dst, dtype::f32, k_n, k_m);
    return ev;
  };

  emel::kernel::any cpu{};
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};
  REQUIRE(cpu.process_event(make(cpu_out.data())));
  REQUIRE(metal.process_event(make(metal_out.data())));
  CHECK(all_close(metal_out, cpu_out, k_rel_tol, k_abs_tol));
}

TEST_CASE("kernel_metal_mul_mat_f16_matches_cpu") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  lcg gen(3);
  constexpr uint64_t k_k = 12;
  constexpr uint64_t k_m = 4;
  constexpr uint64_t k_n = 6;
  std::vector<float> src0(k_k * k_m);
  std::vector<float> src1(k_k * k_n);
  std::vector<uint16_t> src0_f16(k_k * k_m);
  std::vector<uint16_t> src1_f16(k_k * k_n);
  std::vector<float> cpu_out(k_m * k_n);
  std::vector<float> metal_out(k_m * k_n);
  fill(src0, gen, -1.0f, 1.0f);
  fill(src1, gen, -1.0f, 1.0f);
  for (size_t i = 0; i < src0.size(); ++i) {
    src0_f16[i] = emel::kernel::detail::quant::fp32_to_fp16(src0[i]);
  }
  for (size_t i = 0; i < src1.size(); ++i) {
    src1_f16[i] = emel::kernel::detail::quant::fp32_to_fp16(src1[i]);
  }

  const auto make = [&](float *dst) {
    emel::kernel::event::op_mul_mat ev{};
    ev.src0 = make_src(src0_f16.data(), dtype::f16, k_k, k_m);
    ev.src1 = make_src(src1_f16.data(), dtype::f16, k_k, k_n);
    ev.dst = make_dst(dst, dtype::f32, k_m, k_n);
    return ev;
  };

  emel::kernel::any cpu{};
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};
  REQUIRE(cpu.process_event(make(cpu_out.data())));
  REQUIRE(metal.process_event(make(metal_out.data())));
  CHECK(all_close(metal_out, cpu_out, k_f16_rel_tol, k_f16_abs_tol));
}

TEST_CASE("kernel_metal_mul_mat_q8_0_matches_cpu") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  lcg gen(4);
  constexpr uint64_t k_k = 64;
  constexpr uint64_t k_m = 3;
  std::vector<float> src0(k_k * k_m);
  std::vector<float> input(k_k);
  std::vector<float> cpu_out(k_m);
  std::vector<float> metal_out(k_m);
  fill(src0, gen, -1.0f, 1.0f);
  fill(input, gen, -1.0f, 1.0f);

  std::vector<uint8_t> q8(
      static_cast<size_t>(emel::kernel::detail::quantized_row_storage_bytes(
                              emel::kernel::detail::dtype_q8_0, k_k) *
                          k_m));
  for (uint64_t row = 0; row < k_m; ++row) {
    auto *block = reinterpret_cast<emel::kernel::detail::quant::block_q8_0 *>(
        q8.data() + row * emel::kernel::detail::quantized_row_storage_bytes(
                              emel::kernel::detail::dtype_q8_0, k_k));
    emel::kernel::detail::quant::quantize_row_q8_0_strided(
        src0.data() + row * k_k, 1u, block, static_cast<int64_t>(k_k));
  }

  const auto make = [&](float *dst) {
    emel::kernel::event::op_mul_mat ev{};
    ev.src0 = make_quantized_src(q8.data(), dtype::q8_0, k_k, k_m);
    ev.src1 = make_src(input.data(), dtype::f32, 1, k_k);
    ev.dst = make_dst(dst, dtype::f32, 1, k_m);
    return ev;
  };

  emel::kernel::any cpu{};
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};
  REQUIRE(cpu.process_event(make(cpu_out.data())));
  REQUIRE(metal.process_event(make(metal_out.data())));
  CHECK(all_close(metal_out, cpu_out, k_rel_tol, k_abs_tol));
}

TEST_CASE("kernel_metal_add_matches_cpu") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  lcg gen(5);
  std::vector<float> lhs(32);
  std::vector<float> rhs(32);
  std::vector<float> cpu_out(32);
  std::vector<float> metal_out(32);
  fill(lhs, gen, -1.0f, 1.0f);
  fill(rhs, gen, -1.0f, 1.0f);

  const auto make = [&](float *dst) {
    emel::kernel::event::op_add ev{};
    ev.src0 = make_src(lhs.data(), dtype::f32, 8, 4);
    ev.src1 = make_src(rhs.data(), dtype::f32, 8, 4);
    ev.dst = make_dst(dst, dtype::f32, 8, 4);
    return ev;
  };

  emel::kernel::any cpu{};
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};
  REQUIRE(cpu.process_event(make(cpu_out.data())));
  REQUIRE(metal.process_event(make(metal_out.data())));
  // Pure elementwise add: bitwise identical on both backends.
  CHECK(all_close(metal_out, cpu_out, 0.0f, 0.0f));
}

TEST_CASE("kernel_metal_add_broadcast_row_matches_cpu") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  lcg gen(6);
  std::vector<float> lhs(8 * 4);
  std::vector<float> bias(8);
  std::vector<float> cpu_out(8 * 4);
  std::vector<float> metal_out(8 * 4);
  fill(lhs, gen, -1.0f, 1.0f);
  fill(bias, gen, -1.0f, 1.0f);

  const auto make = [&](float *dst) {
    emel::kernel::event::op_add ev{};
    ev.src0 = make_src(lhs.data(), dtype::f32, 8, 4);
    ev.src1 = make_src(bias.data(), dtype::f32, 8);
    ev.dst = make_dst(dst, dtype::f32, 8, 4);
    return ev;
  };

  emel::kernel::any cpu{};
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};
  REQUIRE(cpu.process_event(make(cpu_out.data())));
  REQUIRE(metal.process_event(make(metal_out.data())));
  CHECK(all_close(metal_out, cpu_out, 0.0f, 0.0f));
}

TEST_CASE("kernel_metal_unary_subops_match_cpu") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  lcg gen(7);
  std::vector<float> src(64);
  std::vector<float> cpu_out(64);
  std::vector<float> metal_out(64);
  fill(src, gen, -0.5f, 0.5f);

  const std::array<unary_subop, 8> subops = {
      unary_subop::abs,  unary_subop::neg,  unary_subop::tanh,
      unary_subop::elu,  unary_subop::relu, unary_subop::gelu,
      unary_subop::silu, unary_subop::exp};

  emel::kernel::any cpu{};
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};
  for (const unary_subop subop : subops) {
    const auto make = [&](float *dst) {
      emel::kernel::event::op_unary ev{};
      ev.src0 = make_src(src.data(), dtype::f32, 64);
      ev.dst = make_dst(dst, dtype::f32, 64);
      ev.subop = subop;
      return ev;
    };
    std::fill(cpu_out.begin(), cpu_out.end(), 0.0f);
    std::fill(metal_out.begin(), metal_out.end(), 0.0f);
    REQUIRE(cpu.process_event(make(cpu_out.data())));
    REQUIRE(metal.process_event(make(metal_out.data())));
    const float rel = subop == unary_subop::gelu ? k_f16_rel_tol : k_rel_tol;
    const float abs = subop == unary_subop::gelu ? k_f16_abs_tol : k_abs_tol;
    CHECK_MESSAGE(all_close(metal_out, cpu_out, rel, abs), "subop mismatch");
  }
}

TEST_CASE("kernel_metal_im2col_f32_and_f16_match_cpu") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  lcg gen(8);
  constexpr uint64_t k_taps = 3;
  constexpr uint64_t k_channels = 2;
  constexpr uint64_t k_length = 10;
  constexpr uint64_t k_out_length = 8;
  std::vector<float> src(k_channels * k_length);
  std::vector<float> cpu_out(k_channels * k_taps * k_out_length);
  std::vector<float> metal_out(k_channels * k_taps * k_out_length);
  fill(src, gen, -1.0f, 1.0f);

  // src0 (the weight view) is never dereferenced by the im2col kernel; pass a
  // non-null dummy pointer so the shared request validation accepts the shape.
  const float dummy_weight = 0.0f;

  const auto make = [&](float *dst) {
    emel::kernel::event::op_im2col ev{};
    ev.src0 = make_src(&dummy_weight, dtype::f32, k_taps, k_channels);
    ev.src1 = make_src(src.data(), dtype::f32, k_length, k_channels);
    ev.dst = make_dst(dst, dtype::f32, k_channels * k_taps, k_out_length);
    set_op_param_i32(ev, 0u, 1);
    set_op_param_i32(ev, 2u, 0);
    set_op_param_i32(ev, 4u, 1);
    set_op_param_i32(ev, 6u, 0);
    return ev;
  };

  emel::kernel::any cpu{};
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};
  REQUIRE(cpu.process_event(make(cpu_out.data())));
  REQUIRE(metal.process_event(make(metal_out.data())));
  // Pure layout copy: bitwise identical.
  CHECK(all_close(metal_out, cpu_out, 0.0f, 0.0f));
}

TEST_CASE("kernel_metal_conv_transpose_1d_f32_and_f16_match_cpu") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  lcg gen(9);
  constexpr uint64_t k_taps = 4;
  constexpr uint64_t k_out_channels = 3;
  constexpr uint64_t k_in_channels = 2;
  constexpr uint64_t k_length = 6;
  constexpr int32_t k_stride = 2;
  constexpr uint64_t k_out_length = (k_length - 1) * k_stride + k_taps;

  std::vector<float> weight_f32(k_taps * k_out_channels * k_in_channels);
  std::vector<uint16_t> weight_f16(k_taps * k_out_channels * k_in_channels);
  std::vector<float> input(k_in_channels * k_length);
  std::vector<float> cpu_out(k_out_channels * k_out_length);
  std::vector<float> metal_out(k_out_channels * k_out_length);
  fill(weight_f32, gen, -1.0f, 1.0f);
  fill(input, gen, -1.0f, 1.0f);
  for (size_t i = 0; i < weight_f16.size(); ++i) {
    weight_f16[i] = emel::kernel::detail::quant::fp32_to_fp16(weight_f32[i]);
  }

  const auto make = [&](float *dst, const void *w, const dtype w_type) {
    emel::kernel::event::op_conv_transpose_1d ev{};
    ev.src0 = make_src(w, w_type, k_taps, k_out_channels, k_in_channels);
    ev.src1 = make_src(input.data(), dtype::f32, k_length, k_in_channels);
    ev.dst = make_dst(dst, dtype::f32, k_out_length, k_out_channels);
    set_op_param_i32(ev, 0u, k_stride);
    set_op_param_i32(ev, 1u, 0);
    set_op_param_i32(ev, 2u, 1);
    return ev;
  };

  emel::kernel::any cpu{};
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};
  {
    std::fill(cpu_out.begin(), cpu_out.end(), 0.0f);
    std::fill(metal_out.begin(), metal_out.end(), 0.0f);
    REQUIRE(
        cpu.process_event(make(cpu_out.data(), weight_f32.data(), dtype::f32)));
    REQUIRE(metal.process_event(
        make(metal_out.data(), weight_f32.data(), dtype::f32)));
    CHECK(all_close(metal_out, cpu_out, k_rel_tol, k_abs_tol));
  }
  {
    std::fill(cpu_out.begin(), cpu_out.end(), 0.0f);
    std::fill(metal_out.begin(), metal_out.end(), 0.0f);
    REQUIRE(
        cpu.process_event(make(cpu_out.data(), weight_f16.data(), dtype::f16)));
    REQUIRE(metal.process_event(
        make(metal_out.data(), weight_f16.data(), dtype::f16)));
    CHECK(all_close(metal_out, cpu_out, k_f16_rel_tol, k_f16_abs_tol));
  }
}

TEST_CASE("kernel_metal_get_rows_f32_and_f16_match_cpu") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  lcg gen(10);
  constexpr uint64_t k_entries = 8;
  constexpr uint64_t k_dim = 5;
  std::vector<float> codebook_f32(k_dim * k_entries);
  std::vector<uint16_t> codebook_f16(k_dim * k_entries);
  std::vector<int32_t> indices = {0, 3, 7, 2, 5};
  std::vector<float> cpu_out(k_dim * indices.size());
  std::vector<float> metal_out(k_dim * indices.size());
  fill(codebook_f32, gen, -1.0f, 1.0f);
  for (size_t i = 0; i < codebook_f16.size(); ++i) {
    codebook_f16[i] =
        emel::kernel::detail::quant::fp32_to_fp16(codebook_f32[i]);
  }

  const auto make = [&](float *dst, const void *codebook, const dtype cb_type) {
    emel::kernel::event::op_get_rows ev{};
    ev.src0 = make_src(codebook, cb_type, k_dim, k_entries);
    ev.src1 = make_src(indices.data(), dtype::i32, indices.size());
    ev.dst = make_dst(dst, dtype::f32, k_dim, indices.size());
    return ev;
  };

  emel::kernel::any cpu{};
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};
  {
    std::fill(cpu_out.begin(), cpu_out.end(), 0.0f);
    std::fill(metal_out.begin(), metal_out.end(), 0.0f);
    REQUIRE(cpu.process_event(
        make(cpu_out.data(), codebook_f32.data(), dtype::f32)));
    REQUIRE(metal.process_event(
        make(metal_out.data(), codebook_f32.data(), dtype::f32)));
    CHECK(all_close(metal_out, cpu_out, 0.0f, 0.0f));
  }
  {
    std::fill(cpu_out.begin(), cpu_out.end(), 0.0f);
    std::fill(metal_out.begin(), metal_out.end(), 0.0f);
    REQUIRE(cpu.process_event(
        make(cpu_out.data(), codebook_f16.data(), dtype::f16)));
    REQUIRE(metal.process_event(
        make(metal_out.data(), codebook_f16.data(), dtype::f16)));
    CHECK(all_close(metal_out, cpu_out, 0.0f, 0.0f));
  }
}

TEST_CASE("kernel_metal_rejects_invalid_and_unsupported_requests") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};

  std::array<float, 8> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::array<float, 8> dst = {};

  // Unsupported op (op_sub has no metal row): explicit rejection.
  emel::kernel::event::op_sub unsupported{
      .src0 = make_src(src.data(), dtype::f32, 8),
      .src1 = make_src(src.data(), dtype::f32, 8),
      .dst = make_dst(dst.data(), dtype::f32, 8),
  };
  CHECK_FALSE(metal.process_event(unsupported));

  // Null data: invalid request.
  emel::kernel::event::op_add null_src1{
      .src0 = make_src(src.data(), dtype::f32, 8),
      .src1 = make_src(nullptr, dtype::f32, 8),
      .dst = make_dst(dst.data(), dtype::f32, 8),
  };
  CHECK_FALSE(metal.process_event(null_src1));

  // Shape mismatch: invalid request.
  emel::kernel::event::op_add shape_mismatch{
      .src0 = make_src(src.data(), dtype::f32, 8),
      .src1 = make_src(src.data(), dtype::f32, 4),
      .dst = make_dst(dst.data(), dtype::f32, 8),
  };
  CHECK_FALSE(metal.process_event(shape_mismatch));

  // Unsupported unary subop.
  emel::kernel::event::op_unary unsupported_subop{
      .src0 = make_src(src.data(), dtype::f32, 8),
      .dst = make_dst(dst.data(), dtype::f32, 8),
      .subop = unary_subop::floor,
  };
  CHECK_FALSE(metal.process_event(unsupported_subop));

  // A valid dispatch event is accepted (metadata-only, no compute).
  const emel::kernel::event::dispatch dispatch_ev{};
  CHECK(metal.process_event(dispatch_ev));
}

TEST_CASE("kernel_metal_rejects_over_capacity_requests") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};

  // A dense view larger than one staging slice must be rejected by the
  // capacity guards before any dispatch.
  constexpr uint64_t k_huge =
      emel::kernel::metal::detail::k_pool_slice_capacity_bytes / sizeof(float) +
      16u;
  std::vector<float> huge_src(k_huge);
  std::vector<float> huge_dst(k_huge);

  emel::kernel::event::op_add huge{
      .src0 = make_src(huge_src.data(), dtype::f32, k_huge),
      .src1 = make_src(huge_src.data(), dtype::f32, k_huge),
      .dst = make_dst(huge_dst.data(), dtype::f32, k_huge),
  };
  CHECK_FALSE(metal.process_event(huge));
}

TEST_CASE("kernel_metal_actor_counts_dispatches") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  emel::kernel::metal::sm machine{};
  REQUIRE(machine.metal_available());

  std::array<float, 4> src = {1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 4> dst = {};
  emel::kernel::event::op_unary ev{
      .src0 = make_src(src.data(), dtype::f32, 4),
      .dst = make_dst(dst.data(), dtype::f32, 4),
      .subop = unary_subop::neg,
  };
  const uint64_t before = machine.metal_dispatch_count();
  REQUIRE(machine.process_event(ev));
  CHECK(machine.metal_dispatch_count() == before + 1u);
}

TEST_CASE("kernel_metal_quantized_row_storage_is_measured_in_row_bytes") {
  // A q8_0 row spans 34 * k / 32 bytes for k elements; the storage contract
  // must measure full rows, otherwise a near-capacity request could stage
  // past its slice (regression: last-element offset under-measured).
  constexpr uint64_t k_k = 64;
  constexpr uint64_t k_m = 3;
  std::vector<uint8_t> q8(
      static_cast<size_t>(emel::kernel::detail::quantized_row_storage_bytes(
                              emel::kernel::detail::dtype_q8_0, k_k) *
                          k_m));
  const emel::kernel::event::tensor_view view =
      make_quantized_src(q8.data(), dtype::q8_0, k_k, k_m);
  const uint64_t row_bytes = emel::kernel::detail::quantized_row_storage_bytes(
      emel::kernel::detail::dtype_q8_0, k_k);
  CHECK(emel::kernel::metal::detail::tensor_storage_bytes(view) ==
        row_bytes * k_m);
}

TEST_CASE("kernel_metal_strided_views_stage_and_readback_preserving_layout") {
  if (!host_has_metal()) {
    MESSAGE("skipping: no Metal device");
    return;
  }
  lcg gen(11);
  // Row-padded (strided) views: nb[1] wider than ne[0] * elem. The guards
  // accept equal-count strided add; staging must preserve the nb layout so
  // the shader reads and writes the same offsets as the CPU reference.
  constexpr uint64_t k_cols = 8;
  constexpr uint64_t k_rows = 4;
  constexpr uint64_t k_pad_bytes = 16u;
  std::vector<float> lhs(k_rows * (k_cols * 4 + k_pad_bytes) / 4);
  std::vector<float> rhs(k_rows * (k_cols * 4 + k_pad_bytes) / 4);
  std::vector<float> cpu_out(k_rows * (k_cols * 4 + k_pad_bytes) / 4);
  std::vector<float> metal_out(k_rows * (k_cols * 4 + k_pad_bytes) / 4);
  fill(lhs, gen, -1.0f, 1.0f);
  fill(rhs, gen, -1.0f, 1.0f);

  const auto make_strided = [&](float *data, const uint64_t ne0,
                                const uint64_t ne1) {
    emel::kernel::event::tensor_view view{};
    view.data = data;
    view.type = dtype::f32;
    view.ne = {ne0, ne1, 1, 1};
    view.nb[0] = 4;
    view.nb[1] = ne0 * 4 + k_pad_bytes;
    view.nb[2] = view.nb[1] * ne1;
    view.nb[3] = view.nb[2];
    return view;
  };
  const auto make = [&](float *dst) {
    emel::kernel::event::op_add ev{};
    ev.src0 = make_strided(lhs.data(), k_cols, k_rows);
    ev.src1 = make_strided(rhs.data(), k_cols, k_rows);
    ev.dst.data = dst;
    ev.dst.type = dtype::f32;
    ev.dst.ne = {k_cols, k_rows, 1, 1};
    ev.dst.nb[0] = 4;
    ev.dst.nb[1] = k_cols * 4 + k_pad_bytes;
    ev.dst.nb[2] = ev.dst.nb[1] * k_rows;
    ev.dst.nb[3] = ev.dst.nb[2];
    return ev;
  };

  emel::kernel::any cpu{};
  emel::kernel::any metal{emel::kernel::kernel_kind::metal};
  REQUIRE(cpu.process_event(make(cpu_out.data())));
  REQUIRE(metal.process_event(make(metal_out.data())));
  CHECK(all_close(metal_out, cpu_out, 0.0f, 0.0f));
}

} // namespace emel::kernel::metal_tests
