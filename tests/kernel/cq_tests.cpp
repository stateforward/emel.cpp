#include "emel/kernel/cq/detail.hpp"
#include "emel/kernel/cq/sm.hpp"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <limits>
#include <doctest/doctest.h>
#include <vector>
namespace {
using emel::cact::loader::tensor_view;
using emel::kernel::cq::event::gemv_request;
template <uint32_t Bits>
std::vector<uint8_t> blob(const std::vector<uint32_t> &ix, uint32_t out,
                          uint32_t in, uint32_t group,
                          const std::vector<uint16_t> &ns) {
  const uint32_t pad = (in + group - 1u) / group * group;
  const size_t rb = emel::kernel::cq::detail::packed_row_bytes<Bits>(pad);
  const size_t nc = static_cast<size_t>(out) * pad / group;
  std::vector<uint8_t> b(static_cast<size_t>(out) * rb + nc * 2u);
  for (uint32_t r = 0; r < out; ++r)
    for (uint32_t i = 0; i < pad; ++i) {
      uint8_t *p = b.data() + static_cast<size_t>(r) * rb;
      const uint32_t v = ix[static_cast<size_t>(r) * pad + i];
      if constexpr (Bits == 5u) {
        const uint32_t c = v == 0u ? 3u : v - 1u;
        p[i >> 2u] |= static_cast<uint8_t>(c << ((i & 3u) * 2u));
      } else {
        const size_t bit = static_cast<size_t>(i) * Bits;
        p[bit >> 3u] |= static_cast<uint8_t>(v << (bit & 7u));
        if ((bit & 7u) + Bits > 8u)
          p[(bit >> 3u) + 1u] |= static_cast<uint8_t>(v >> (8u - (bit & 7u)));
      }
    }
  const size_t po = static_cast<size_t>(out) * rb;
  for (size_t i = 0; i < nc; ++i) {
    b[po + i * 2u] = static_cast<uint8_t>(ns[i]);
    b[po + i * 2u + 1u] = static_cast<uint8_t>(ns[i] >> 8u);
  }
  return b;
}
tensor_view view(const std::vector<uint8_t> &b, uint32_t out, uint32_t in,
                 uint32_t group, uint32_t bits) {
  return tensor_view{.dtype = 3u,
                     .ndim = 2u,
                     .shape = {out, in, 0u, 0u},
                     .nbytes = b.size(),
                     .group = group,
                     .bits = bits,
                     .data = b.data()};
}
size_t blocked_norm_count(const tensor_view &view) {
  const uint32_t in_pad =
      (view.shape[1] + view.group - 1u) / view.group * view.group;
  return static_cast<size_t>(view.shape[0] / 32u * 32u) * in_pad / view.group;
}
#if defined(__AVX2__) && defined(__FMA__)
void execute_block32_canonical_reference(
    const emel::kernel::cq::event::prepared_q4_view &prepared,
    const emel::kernel::cq::event::prepared_codebook_q4 &codebook,
    const std::span<const float> activation_fwht,
    const std::span<float> output) {
  __m256i byte0;
  __m256i byte1;
  __m256i byte2;
  __m256i byte3;
  emel::kernel::cq::action::q4_codebook_byte_tables(codebook, byte0, byte1,
                                                    byte2, byte3);
  const uint32_t groups_per_row = prepared.in_pad / prepared.group;
  for (uint32_t row = 0u; row < prepared.out; row += 32u) {
    const uint8_t *selectors = prepared.indices_by_input32.data() +
                               static_cast<size_t>(row) * prepared.in_pad;
    __m256 row_total0 = _mm256_setzero_ps();
    __m256 row_total1 = _mm256_setzero_ps();
    __m256 row_total2 = _mm256_setzero_ps();
    __m256 row_total3 = _mm256_setzero_ps();
    for (uint32_t begin = 0u, group_index = 0u; begin < prepared.in_pad;
         begin += prepared.group, ++group_index) {
      __m256 group_accum0 = _mm256_setzero_ps();
      __m256 group_accum1 = _mm256_setzero_ps();
      __m256 group_accum2 = _mm256_setzero_ps();
      __m256 group_accum3 = _mm256_setzero_ps();
      for (uint32_t i = 0u; i < prepared.group; ++i) {
        const auto values = emel::kernel::cq::action::lookup_codebook32_pshufb(
            emel::kernel::cq::action::load_selector32(
                selectors + static_cast<size_t>(begin + i) * 32u),
            byte0, byte1, byte2, byte3);
        const __m256 activation = _mm256_set1_ps(activation_fwht[begin + i]);
        group_accum0 =
            _mm256_fmadd_ps(values.values0, activation, group_accum0);
        group_accum1 =
            _mm256_fmadd_ps(values.values1, activation, group_accum1);
        group_accum2 =
            _mm256_fmadd_ps(values.values2, activation, group_accum2);
        group_accum3 =
            _mm256_fmadd_ps(values.values3, activation, group_accum3);
      }
      alignas(32) std::array<float, 32u> canonical_norms{};
      for (uint32_t lane = 0u; lane < canonical_norms.size(); ++lane)
        canonical_norms[lane] =
            prepared.norms[static_cast<size_t>(row + lane) * groups_per_row +
                           group_index];
      row_total0 = _mm256_fmadd_ps(group_accum0,
                                   _mm256_load_ps(canonical_norms.data()),
                                   row_total0);
      row_total1 = _mm256_fmadd_ps(group_accum1,
                                   _mm256_load_ps(canonical_norms.data() + 8u),
                                   row_total1);
      row_total2 = _mm256_fmadd_ps(group_accum2,
                                   _mm256_load_ps(canonical_norms.data() + 16u),
                                   row_total2);
      row_total3 = _mm256_fmadd_ps(group_accum3,
                                   _mm256_load_ps(canonical_norms.data() + 24u),
                                   row_total3);
    }
    _mm256_storeu_ps(output.data() + row, row_total0);
    _mm256_storeu_ps(output.data() + row + 8u, row_total1);
    _mm256_storeu_ps(output.data() + row + 16u, row_total2);
    _mm256_storeu_ps(output.data() + row + 24u, row_total3);
  }
}
#endif
template <uint32_t Bits>
void run_route(uint32_t in, const std::array<float, 28u> &cb) {
  constexpr uint32_t group = 8u;
  const uint32_t levels = 1u << Bits;
  std::vector<uint32_t> ix;
  for (uint32_t i = 0; i < 4u * in; ++i)
    ix.push_back(i % levels);
  const auto b = blob<Bits>(
      ix, 4u, in, group,
      {0x3c00, 0x4000, 0x3c00, 0x4000, 0x3c00, 0x4000, 0x3c00, 0x4000});
  const auto v = view(b, 4u, in, group, Bits);
  std::array<float, 32u> a{};
  for (uint32_t i = 0; i < in; ++i)
    a[i] = static_cast<float>(i + 1u) / 8.f;
  std::array<float, 32u> w{};
  std::array<float, 4u> so{};
  gemv_request q{v, cb, a, so, w};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result sr{};
  REQUIRE(
      sm.process_event(emel::kernel::cq::event::execute_scalar<Bits>{q, sr}));
  CHECK(sr.accepted);
#if defined(__AVX2__) && defined(__FMA__)
  std::array<float, 4u> ao{};
  gemv_request aq{v, cb, a, ao, w};
  emel::kernel::cq::sm am;
  emel::kernel::cq::event::dispatch_result ar{};
  REQUIRE(
      am.process_event(emel::kernel::cq::event::execute_avx2<Bits>{aq, ar}));
  for (uint32_t r = 0; r < 4u; ++r)
    CHECK(ao[r] == doctest::Approx(so[r]));
  uint64_t scalar_calls = 0u;
  uint64_t avx2_calls = 0u;
  REQUIRE(am.process_event(
      emel::kernel::cq::event::capture_diagnostics{scalar_calls, avx2_calls}));
  CHECK(scalar_calls == 0u);
  CHECK(avx2_calls == 1u);
#endif
}
} // namespace
TEST_CASE("CQ4 prepared route rejects a zero group without evaluating counts") {
  emel::kernel::cq::event::prepared_q4_view prepared{};
  emel::kernel::cq::event::prepared_codebook_q4 prepared_codebook{};
  std::array<float, 1u> activation{};
  std::array<float, 1u> output{};
  std::array<float, 1u> workspace{};
  const emel::kernel::cq::event::prepared_gemv_request request{
      prepared, prepared_codebook, activation, output, workspace};
  emel::kernel::cq::event::dispatch_result result{};
  emel::kernel::cq::sm sm;
  CHECK_FALSE(sm.process_event(
      emel::kernel::cq::event::execute_prepared_avx2_q4{request, result}));
  CHECK_FALSE(result.accepted);
}

TEST_CASE("CQ4 prepared route rejects non-canonical padded geometry") {
  constexpr uint32_t in = 128u;
  constexpr uint32_t in_pad = 384u;
  constexpr uint32_t out = 64u;
  constexpr uint32_t group = 128u;
  std::vector<uint8_t> indices(static_cast<size_t>(out) * in_pad);
  std::vector<uint8_t> indices_by_input32(indices.size());
  std::vector<float> norms(static_cast<size_t>(out) * in_pad / group, 1.0f);
  std::vector<float> norms_by_group32(norms.size(), 1.0f);
  std::array<float, 28u> codebook_values{};
  emel::kernel::cq::event::prepared_codebook_q4 prepared_codebook{};
  prepared_codebook.values = codebook_values;
  emel::kernel::cq::event::prepared_q4_view prepared{
      .source = reinterpret_cast<const uint8_t *>(1u),
      .out = out,
      .in = in,
      .group = group,
      .in_pad = in_pad,
      .indices = indices,
      .indices_by_input32 = indices_by_input32,
      .norms = norms,
      .norms_by_group32 = norms_by_group32,
  };
  std::array<float, in> activation{};
  std::array<float, out> output{};
  std::array<float, in_pad> workspace{};
  const emel::kernel::cq::event::prepared_gemv_request request{
      prepared, prepared_codebook, activation, output, workspace};
  emel::kernel::cq::event::dispatch_result result{};
  emel::kernel::cq::sm sm;
  CHECK_FALSE(sm.process_event(
      emel::kernel::cq::event::execute_prepared_avx2_q4{request, result}));
  CHECK_FALSE(result.accepted);
}

TEST_CASE("CQ A8 fake quant matches JAX ties zero and signed boundary") {
  REQUIRE(std::fesetround(FE_TONEAREST) == 0);
  emel::kernel::cq::sm sm;

  {
    const std::array<float, 4u> input{0.0f, 0.0f, -0.0f, 0.0f};
    std::array<int8_t, 4u> quantized{};
    std::array<float, 4u> integer_values{};
    float scale = 0.0f;
    const emel::kernel::cq::event::quantize_a8_request request{
        input, quantized, integer_values, scale};
    emel::kernel::cq::event::dispatch_result result{};
    REQUIRE(sm.process_event(
        emel::kernel::cq::event::quantize_a8{request, result}));
    CHECK(scale == 1.0f);
    for (uint32_t i = 0u; i < input.size(); ++i) {
      CHECK(quantized[i] == 0);
      CHECK(integer_values[i] == 0.0f);
    }
  }

  {
    // absmax=127 => scale=1. JAX round is ties-to-even and signed A8 clamps
    // to [-128, 127]; the negative endpoint is representable but absmax/qmax
    // makes ordinary finite inputs reach -127, not -128.
    const std::array<float, 9u> input{-127.0f, -126.5f, -1.5f,  -0.5f, 0.0f,
                                      0.5f,    1.5f,    126.5f, 127.0f};
    const std::array<int8_t, 9u> expected{-127, -126, -2, 0, 0, 0, 2, 126, 127};
    std::array<int8_t, 9u> quantized{};
    std::array<float, 9u> integer_values{};
    float scale = 0.0f;
    const emel::kernel::cq::event::quantize_a8_request request{
        input, quantized, integer_values, scale};
    emel::kernel::cq::event::dispatch_result result{};
    REQUIRE(sm.process_event(
        emel::kernel::cq::event::quantize_a8{request, result}));
    CHECK(scale == 1.0f);
    for (uint32_t i = 0u; i < input.size(); ++i) {
      CHECK(quantized[i] == expected[i]);
      CHECK(integer_values[i] == static_cast<float>(expected[i]));
    }
    CHECK(quantized.front() > INT8_MIN);
  }

  uint64_t quantize_calls = 0u;
  REQUIRE(sm.process_event(
      emel::kernel::cq::event::capture_a8_diagnostics{quantize_calls}));
  CHECK(quantize_calls == 2u);
}

TEST_CASE("CQ A8 guard rejects incomplete caller scratch") {
  const std::array<float, 2u> input{1.0f, -1.0f};
  std::array<int8_t, 1u> quantized{};
  std::array<float, 2u> integer_values{};
  float scale = 0.0f;
  const emel::kernel::cq::event::quantize_a8_request request{
      input, quantized, integer_values, scale};
  emel::kernel::cq::event::dispatch_result result{};
  emel::kernel::cq::sm sm;
  CHECK_FALSE(
      sm.process_event(emel::kernel::cq::event::quantize_a8{request, result}));
  CHECK(sm.is(stateforward::sml::state<emel::kernel::cq::state_ready>));
}

TEST_CASE("CQ A8 guard rejects non-finite activation values") {
  emel::kernel::cq::sm sm;
  for (const float nonfinite : {std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity()}) {
    const std::array<float, 2u> input{1.0f, nonfinite};
    std::array<int8_t, 2u> quantized{};
    std::array<float, 2u> integer_values{};
    float scale = 0.0f;
    const emel::kernel::cq::event::quantize_a8_request request{
        input, quantized, integer_values, scale};
    emel::kernel::cq::event::dispatch_result result{};
    CHECK_FALSE(
        sm.process_event(emel::kernel::cq::event::quantize_a8{request, result}));
    CHECK(sm.is(stateforward::sml::state<emel::kernel::cq::state_ready>));
  }
}

TEST_CASE("CQ AVX2 FWHT128 matches scalar on random zeros tails and A8") {
#if defined(__AVX2__) && defined(__FMA__)
  const auto check = [](const std::array<float, 128u> &input) {
    auto expected = input;
    auto actual = input;
    emel::kernel::cq::detail::fwht(expected.data(), expected.size());
    emel::kernel::cq::sm sm;
    emel::kernel::cq::event::dispatch_result result{};
    const emel::kernel::cq::event::fwht_request request{actual};
    REQUIRE(sm.process_event(
        emel::kernel::cq::event::execute_fwht_avx2{request, result}));
    for (uint32_t i = 0u; i < actual.size(); ++i)
      CHECK(actual[i] == doctest::Approx(expected[i]).epsilon(2.0e-6));
  };

  check({});
  std::array<float, 128u> a8{};
  for (uint32_t i = 0u; i < a8.size(); ++i)
    a8[i] = static_cast<float>(static_cast<int32_t>(i % 255u) - 127);
  check(a8);
  std::array<float, 128u> tail{};
  for (uint32_t i = 0u; i < 113u; ++i)
    tail[i] = std::sin(static_cast<float>(i) * 0.19f);
  check(tail);
  std::mt19937 random{0x12345678u};
  std::uniform_real_distribution<float> values{-4.0f, 4.0f};
  for (uint32_t sample = 0u; sample < 64u; ++sample) {
    std::array<float, 128u> input{};
    for (float &value : input)
      value = values(random);
    check(input);
  }
#endif
}

TEST_CASE("CQ A8 scale hoist matches dequantized scalar projection") {
  std::array<float, 28u> cb{};
  for (uint32_t i = 0u; i < 16u; ++i)
    cb[12u + i] = (static_cast<float>(i) - 7.5f) / 8.0f;
  constexpr uint32_t group = 128u, in = 128u, out = 3u;
  std::vector<uint32_t> ix(static_cast<size_t>(out) * in);
  for (size_t i = 0u; i < ix.size(); ++i)
    ix[i] = static_cast<uint32_t>((i * 13u + 5u) & 15u);
  const auto b = blob<4u>(ix, out, in, group, {0x3c00u, 0x3800u, 0x4000u});
  const auto v = view(b, out, in, group, 4u);
  std::array<float, in> input{};
  for (uint32_t i = 0u; i < in; ++i)
    input[i] = std::sin(static_cast<float>(i + 1u) * 0.07f) * 3.25f;
  std::array<int8_t, in> quantized{};
  std::array<float, in> integer_values{};
  float scale = 0.0f;
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result quantize_result{};
  const emel::kernel::cq::event::quantize_a8_request quantize_request{
      input, quantized, integer_values, scale};
  REQUIRE(sm.process_event(emel::kernel::cq::event::quantize_a8{
      quantize_request, quantize_result}));
  std::array<float, in> dequantized{};
  for (uint32_t i = 0u; i < in; ++i)
    dequantized[i] = integer_values[i] * scale;
  std::array<float, in> baseline_workspace{};
  std::array<float, in> hoisted_workspace{};
  std::array<float, out> baseline{};
  std::array<float, out> hoisted{};
  emel::kernel::cq::event::dispatch_result baseline_result{};
  const gemv_request baseline_request{v, cb, dequantized, baseline,
                                      baseline_workspace};
  REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_q4{
      baseline_request, baseline_result}));
  emel::kernel::cq::event::dispatch_result hoisted_result{};
  const gemv_request hoisted_request{v, cb, integer_values, hoisted,
                                     hoisted_workspace, scale};
  REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_q4{
      hoisted_request, hoisted_result}));
  for (uint32_t row = 0u; row < out; ++row)
    CHECK(hoisted[row] == doctest::Approx(baseline[row]).epsilon(2.0e-6));
}

TEST_CASE("CQ2 scalar parity and normalized FWHT") {
  std::array<float, 28u> cb{};
  cb[0] = -.5f;
  cb[1] = -.1f;
  cb[2] = .1f;
  cb[3] = .5f;
  const std::vector<uint32_t> ix{0, 1, 2, 3, 0, 1, 2, 3};
  const auto b = blob<2u>(ix, 1, 8, 8, {0x3c00});
  const auto v = view(b, 1, 8, 8, 2);
  const std::array<float, 8u> a{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<float, 8u> w{};
  std::array<float, 1u> o{};
  gemv_request q{v, cb, a, o, w};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result r{};
  REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_q2{q, r}));
  std::array<float, 8u> t = a;
  emel::kernel::cq::detail::fwht(t.data(), 8);
  float e = 0;
  for (uint32_t i = 0; i < 8; ++i)
    e += cb[ix[i]] * t[i];
  CHECK(o[0] == doctest::Approx(e));
}
TEST_CASE("CQ4 symmetry guard matches values independent of ordering") {
  std::array<float, 28u> cb{};
  const std::array<float, 8u> magnitudes{0.015625f, 0.03125f, 0.0625f, 0.125f,
                                         0.25f,     0.5f,     1.0f,    2.0f};
  for (uint32_t i = 0u; i < magnitudes.size(); ++i) {
    cb[12u + i * 2u] = magnitudes[7u - i];
    cb[13u + i * 2u] = -magnitudes[7u - i];
  }
  CHECK(emel::kernel::cq::detail::q4_codebook_is_symmetric(cb));
  cb[12u] = std::nextafter(cb[12u], INFINITY);
  CHECK_FALSE(emel::kernel::cq::detail::q4_codebook_is_symmetric(cb));
}

TEST_CASE("pinned Needle CQ4 codebook rejects lossless sign-rank mapping") {
  constexpr std::array<float, 16u> pinned{
      -0.239531547f,  -0.180706054f,  -0.140957937f,  -0.109012552f,
      -0.0815129653f, -0.0565276518f, -0.0329391509f, -0.0101161096f,
      0.0127108004f,  0.0357079171f,  0.0594582558f,  0.0849770233f,
      0.112741619f,   0.144696921f,   0.184239f,      0.242273703f};
  std::array<float, 28u> cb{};
  for (uint32_t i = 0u; i < pinned.size(); ++i)
    cb[12u + i] = pinned[i];
  CHECK_FALSE(emel::kernel::cq::detail::q4_codebook_is_symmetric(cb));
}
TEST_CASE("CQ3 and CQ4 explicit routes preserve parity") {
  std::array<float, 28u> cb{};
  for (uint32_t i = 0; i < 8; ++i)
    cb[4 + i] = static_cast<float>(i + 1) / 10.f;
  run_route<3u>(16u, cb);
  for (uint32_t i = 0; i < 16; ++i)
    cb[12 + i] = static_cast<float>(i + 1) / 10.f;
  run_route<4u>(16u, cb);
}
TEST_CASE("CQ ternary crumbs use analytic centroid") {
  const std::vector<uint32_t> ix{0, 1, 2, 0, 2, 1, 0, 2};
  const auto b = blob<5u>(ix, 1, 8, 8, {0x3c00});
  const auto v = view(b, 1, 8, 8, 5);
  const std::array<float, 8u> a{1, 0, 0, 0, 0, 0, 0, 0};
  std::array<float, 8u> w{};
  std::array<float, 1u> o{};
  std::array<float, 28u> cb{};
  gemv_request q{v, cb, a, o, w};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result r{};
  REQUIRE(
      sm.process_event(emel::kernel::cq::event::execute_scalar_ternary{q, r}));
}
TEST_CASE("CQ guard rejects incomplete padded workspace") {
  const auto b =
      blob<4u>(std::vector<uint32_t>(16, 0), 1, 13, 8, {0x3c00, 0x3c00});
  const auto v = view(b, 1, 13, 8, 4);
  std::array<float, 28u> cb{};
  std::array<float, 13u> a{};
  std::array<float, 1u> o{};
  std::array<float, 13u> w{};
  gemv_request q{v, cb, a, o, w};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result r{};
  CHECK_FALSE(
      sm.process_event(emel::kernel::cq::event::execute_scalar_q4{q, r}));
  CHECK(sm.is(stateforward::sml::state<emel::kernel::cq::state_ready>));
}
TEST_CASE("CQ row-range GEMV matches the full-view route") {
  std::array<float, 28u> cb{};
  for (uint32_t i = 0; i < 16; ++i)
    cb[12 + i] = static_cast<float>(i + 1) / 10.f;
  constexpr uint32_t group = 8u, in = 16u, out = 4u;
  std::vector<uint32_t> ix;
  for (uint32_t i = 0; i < out * in; ++i)
    ix.push_back(i % 16u);
  const auto b = blob<4u>(
      ix, out, in, group,
      {0x3c00, 0x4000, 0x3c00, 0x4000, 0x3c00, 0x4000, 0x3c00, 0x4000});
  const auto v = view(b, out, in, group, 4u);
  std::array<float, in> a{};
  for (uint32_t i = 0; i < in; ++i)
    a[i] = static_cast<float>(i + 1u) / 8.f;
  std::array<float, in> w{};
  std::array<float, out> full{};
  gemv_request fq{v, cb, a, full, w};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result fr{};
  REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_q4{fq, fr}));
  std::array<float, 2u> part{};
  emel::kernel::cq::event::gemv_rows_request rq{v, cb, a, 1u, 2u, part, w};
  emel::kernel::cq::event::dispatch_result rr{};
  REQUIRE(sm.process_event(
      emel::kernel::cq::event::execute_scalar_rows_q4{rq, rr}));
  CHECK(part[0] == doctest::Approx(full[1]));
  CHECK(part[1] == doctest::Approx(full[2]));
  emel::kernel::cq::event::gemv_rows_request oob{v, cb, a, 3u, 2u, part, w};
  emel::kernel::cq::event::dispatch_result orr{};
  CHECK_FALSE(sm.process_event(
      emel::kernel::cq::event::execute_scalar_rows_q4{oob, orr}));
}
TEST_CASE("CQ row dequant reconstructs exporter unpack values") {
  std::array<float, 28u> cb{};
  for (uint32_t i = 0; i < 16; ++i)
    cb[12 + i] = static_cast<float>(i + 1) / 10.f;
  constexpr uint32_t group = 8u, in = 8u, out = 2u;
  std::vector<uint32_t> ix;
  for (uint32_t i = 0; i < out * in; ++i)
    ix.push_back(i % 16u);
  const auto b = blob<4u>(ix, out, in, group, {0x3c00, 0x4000});
  const auto v = view(b, out, in, group, 4u);
  std::array<float, in> row{};
  emel::kernel::cq::event::dequant_rows_request dq{v, cb, 1u, 1u, 2.f, row};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result dr{};
  REQUIRE(sm.process_event(
      emel::kernel::cq::event::execute_scalar_dequant_q4{dq, dr}));
  // expectation: unit=codebook[idx], rot=unit*norm(=2.0),
  // values=fwht(rot)*scale(=2)
  std::array<float, in> expect{};
  for (uint32_t i = 0; i < in; ++i)
    expect[i] = cb[12u + ((in + i) % 16u)] * 2.f;
  emel::kernel::cq::detail::fwht(expect.data(), in);
  for (uint32_t i = 0; i < in; ++i)
    CHECK(row[i] == doctest::Approx(expect[i] * 2.f));
  // dot-product consistency: dequant row . activation == rows GEMV output
  std::array<float, in> act{};
  for (uint32_t i = 0; i < in; ++i)
    act[i] = static_cast<float>(i + 1u);
  std::array<float, in> w{};
  std::array<float, 1u> g{};
  emel::kernel::cq::event::gemv_rows_request rq{v, cb, act, 1u, 1u, g, w};
  emel::kernel::cq::event::dispatch_result rr{};
  REQUIRE(sm.process_event(
      emel::kernel::cq::event::execute_scalar_rows_q4{rq, rr}));
  float dot = 0.f;
  for (uint32_t i = 0; i < in; ++i)
    dot += row[i] / 2.f * act[i];
  CHECK(g[0] == doctest::Approx(dot));
}

TEST_CASE("CQ4 thirty-two-selector byte shuffle lookup is bit exact") {
#if defined(__AVX2__) && defined(__FMA__)
  std::array<float, 28u> cb{};
  constexpr std::array<uint32_t, 16u> codebook_bits{
      0xbe755c29u, 0xbe39077fu, 0xbe10570bu, 0xbddf3ccdu,
      0xbda6f255u, 0xbd67859du, 0xbd06e8acu, 0xbc25be9du,
      0x3c503f89u, 0x3d12418eu, 0x3d7387f2u, 0x3dae087bu,
      0x3de6e864u, 0x3e1428c2u, 0x3e3ca578u, 0x3e781d43u};
  for (uint32_t i = 0u; i < 16u; ++i)
    std::memcpy(&cb[12u + i], &codebook_bits[i], sizeof(float));

  emel::kernel::cq::event::prepared_codebook_q4 prepared_codebook{};
  emel::kernel::cq::action::prepare_codebook_q4({cb, prepared_codebook});
  __m256i byte0;
  __m256i byte1;
  __m256i byte2;
  __m256i byte3;
  emel::kernel::cq::action::q4_codebook_byte_tables(
      prepared_codebook, byte0, byte1, byte2, byte3);
  const auto check = [&](const std::array<uint8_t, 32u> &selectors) {
    const auto values = emel::kernel::cq::action::lookup_codebook32_pshufb(
        emel::kernel::cq::action::load_selector32(selectors.data()), byte0,
        byte1, byte2, byte3);
    alignas(32) std::array<uint32_t, 32u> actual{};
    _mm256_store_si256(reinterpret_cast<__m256i *>(actual.data()),
                       _mm256_castps_si256(values.values0));
    _mm256_store_si256(reinterpret_cast<__m256i *>(actual.data() + 8u),
                       _mm256_castps_si256(values.values1));
    _mm256_store_si256(reinterpret_cast<__m256i *>(actual.data() + 16u),
                       _mm256_castps_si256(values.values2));
    _mm256_store_si256(reinterpret_cast<__m256i *>(actual.data() + 24u),
                       _mm256_castps_si256(values.values3));
    for (uint32_t lane = 0u; lane < selectors.size(); ++lane)
      CHECK(actual[lane] == codebook_bits[selectors[lane]]);
    const auto raw = emel::kernel::cq::action::lookup_codebook32_raw(
        emel::kernel::cq::action::load_selector32(selectors.data()), byte0,
        byte1, byte2, byte3);
    alignas(32) std::array<uint32_t, 32u> raw_actual{};
    _mm256_store_si256(reinterpret_cast<__m256i *>(raw_actual.data()),
                       _mm256_castps_si256(raw.values0));
    _mm256_store_si256(reinterpret_cast<__m256i *>(raw_actual.data() + 8u),
                       _mm256_castps_si256(raw.values1));
    _mm256_store_si256(reinterpret_cast<__m256i *>(raw_actual.data() + 16u),
                       _mm256_castps_si256(raw.values2));
    _mm256_store_si256(reinterpret_cast<__m256i *>(raw_actual.data() + 24u),
                       _mm256_castps_si256(raw.values3));
    constexpr std::array<uint32_t, 32u> raw_rows{
        0u,  1u,  2u,  3u,  16u, 17u, 18u, 19u,
        4u,  5u,  6u,  7u,  20u, 21u, 22u, 23u,
        8u,  9u,  10u, 11u, 24u, 25u, 26u, 27u,
        12u, 13u, 14u, 15u, 28u, 29u, 30u, 31u};
    for (uint32_t lane = 0u; lane < raw_rows.size(); ++lane)
      CHECK(raw_actual[lane] == codebook_bits[selectors[raw_rows[lane]]]);
  };

  std::array<uint8_t, 32u> lane_boundaries{};
  for (uint32_t lane = 0u; lane < lane_boundaries.size(); ++lane)
    lane_boundaries[lane] = static_cast<uint8_t>(lane & 15u);
  check(lane_boundaries);

  for (uint32_t selector = 0u; selector < 16u; ++selector) {
    std::array<uint8_t, 32u> exhaustive{};
    exhaustive.fill(static_cast<uint8_t>(selector));
    check(exhaustive);
  }

  uint32_t random = 0x9e3779b9u;
  for (uint32_t sequence = 0u; sequence < 256u; ++sequence) {
    std::array<uint8_t, 32u> selectors{};
    for (uint8_t &selector : selectors) {
      random = random * 1664525u + 1013904223u;
      selector = static_cast<uint8_t>(random >> 28u);
    }
    check(selectors);
  }
#endif
}

TEST_CASE("CQ4 preparation preserves selectors norms and numerical identity") {
  std::array<float, 28u> cb{};
  for (uint32_t i = 0u; i < 16u; ++i)
    cb[12u + i] = static_cast<float>(i + 1u) / 10.f;
  constexpr uint32_t group = 8u, in = 16u, out = 4u;
  std::vector<uint32_t> ix;
  for (uint32_t i = 0u; i < out * in; ++i)
    ix.push_back((i * 7u + 3u) % 16u);
  const auto b = blob<4u>(
      ix, out, in, group,
      {0x3c00, 0x4000, 0x4200, 0x4400, 0x3c00, 0x4000, 0x4200, 0x4400});
  const auto v = view(b, out, in, group, 4u);
  std::vector<uint8_t> prepared_indices(out * in);
  std::vector<uint8_t> prepared_indices_by_input32(out * in);
  std::vector<float> prepared_norms(out * in / group);
  std::vector<float> prepared_norms_by_group32(blocked_norm_count(v));
  emel::kernel::cq::event::prepared_q4_view prepared{};
  const emel::kernel::cq::event::prepare_q4_request prepare_request{
      v, prepared_indices, prepared_indices_by_input32, prepared_norms,
      prepared_norms_by_group32, prepared};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result prepare_result{};
  REQUIRE(sm.process_event(
      emel::kernel::cq::event::prepare_q4{prepare_request, prepare_result}));
  emel::kernel::cq::event::prepared_codebook_q4 prepared_codebook{};
  emel::kernel::cq::event::dispatch_result codebook_result{};
  const emel::kernel::cq::event::prepare_codebook_q4_request codebook_request{
      cb, prepared_codebook};
  REQUIRE(sm.process_event(emel::kernel::cq::event::prepare_codebook_q4{
      codebook_request, codebook_result}));
  CHECK(prepared.source == v.data);
  CHECK(prepared.indices.size() == out * in);
  CHECK(prepared.norms.size() == out * in / group);
  for (size_t i = 0u; i < ix.size(); ++i)
    CHECK(prepared.indices[i] == ix[i]);

  std::array<float, in> activation{};
  for (uint32_t i = 0u; i < in; ++i)
    activation[i] = static_cast<float>(i + 1u) / 8.f;
  std::array<float, in> scalar_workspace{};
  std::array<float, in> prepared_workspace{};
  std::array<float, out> scalar_output{};
  std::array<float, out> prepared_output{};
  const gemv_request scalar_request{v, cb, activation, scalar_output,
                                    scalar_workspace};
  emel::kernel::cq::event::dispatch_result scalar_result{};
  REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_q4{
      scalar_request, scalar_result}));
  const emel::kernel::cq::event::prepared_gemv_request prepared_request{
      prepared, prepared_codebook, activation, prepared_output,
      prepared_workspace};
  emel::kernel::cq::event::dispatch_result prepared_result{};
  REQUIRE(sm.process_event(emel::kernel::cq::event::execute_prepared_avx2_q4{
      prepared_request, prepared_result}));
  for (uint32_t row = 0u; row < out; ++row)
    CHECK(prepared_output[row] == doctest::Approx(scalar_output[row]));

  uint64_t prepare_calls = 0u;
  uint64_t prepared_calls = 0u;
  REQUIRE(
      sm.process_event(emel::kernel::cq::event::capture_prepared_diagnostics{
          prepare_calls, prepared_calls}));
  CHECK(prepare_calls == 1u);
  CHECK(prepared_calls == 1u);
}

TEST_CASE(
    "CQ4 batch4 shares one transform and matches separate prepared routes") {
  std::array<float, 28u> cb{};
  for (uint32_t i = 0u; i < 16u; ++i)
    cb[12u + i] = static_cast<float>(i + 1u) / 10.f;
  constexpr uint32_t group = 8u, in = 16u, out = 4u;
  std::vector<uint32_t> ix;
  for (uint32_t i = 0u; i < out * in; ++i)
    ix.push_back((i * 5u + 1u) % 16u);
  const auto b = blob<4u>(
      ix, out, in, group,
      {0x3c00, 0x4000, 0x4200, 0x4400, 0x3c00, 0x4000, 0x4200, 0x4400});
  const auto v = view(b, out, in, group, 4u);
  std::vector<uint8_t> indices(out * in);
  std::vector<float> norms(out * in / group);
  std::vector<float> norms_by_group32(blocked_norm_count(v));
  std::vector<uint8_t> indices_by_input32(out * in);
  emel::kernel::cq::event::prepared_q4_view prepared{};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result prepare_result{};
  const emel::kernel::cq::event::prepare_q4_request prepare_request{
      v, indices, indices_by_input32, norms, norms_by_group32, prepared};
  REQUIRE(sm.process_event(
      emel::kernel::cq::event::prepare_q4{prepare_request, prepare_result}));
  emel::kernel::cq::event::prepared_codebook_q4 prepared_codebook{};
  emel::kernel::cq::event::dispatch_result codebook_result{};
  const emel::kernel::cq::event::prepare_codebook_q4_request codebook_request{
      cb, prepared_codebook};
  REQUIRE(sm.process_event(emel::kernel::cq::event::prepare_codebook_q4{
      codebook_request, codebook_result}));
  std::array<float, in> activation{};
  for (uint32_t i = 0u; i < in; ++i)
    activation[i] = static_cast<float>(i + 1u) / 8.f;
  std::array<float, in> separate_workspace{};
  std::array<float, in> batch_workspace{};
  std::array<float, out> expected{};
  const emel::kernel::cq::event::prepared_gemv_request separate_request{
      prepared, prepared_codebook, activation, expected, separate_workspace};
  emel::kernel::cq::event::dispatch_result separate_result{};
  REQUIRE(sm.process_event(emel::kernel::cq::event::execute_prepared_avx2_q4{
      separate_request, separate_result}));
  std::array<std::array<float, out>, 4u> outputs{};
  const emel::kernel::cq::event::prepared_gemv_batch4_request batch_request{
      .targets = {{{&prepared, outputs[0]},
                   {&prepared, outputs[1]},
                   {&prepared, outputs[2]},
                   {&prepared, outputs[3]}}},
      .codebook = prepared_codebook,
      .activation = activation,
      .workspace = batch_workspace};
  emel::kernel::cq::event::dispatch_result batch_result{};
  REQUIRE(
      sm.process_event(emel::kernel::cq::event::execute_prepared_avx2_batch4_q4{
          batch_request, batch_result}));
  for (const auto &output : outputs)
    for (uint32_t row = 0u; row < out; ++row)
      CHECK(output[row] == doctest::Approx(expected[row]));
}

TEST_CASE("CQ4 realistic batch4 matches independent scalar and prepared calls") {
#if defined(__AVX2__) && defined(__FMA__)
  constexpr uint32_t group = 128u;
  constexpr uint32_t in = 512u;
  constexpr std::array<uint32_t, 4u> outs{512u, 256u, 256u, 512u};
  std::array<float, 28u> codebook{};
  uint32_t random = 0x6a09e667u;
  for (uint32_t i = 0u; i < 16u; ++i) {
    random = random * 1664525u + 1013904223u;
    codebook[12u + i] =
        (static_cast<float>(static_cast<int32_t>(random >> 8u)) /
         8388608.0f) -
        1.0f;
  }
  std::array<float, in> activation{};
  for (float &value : activation) {
    random = random * 1664525u + 1013904223u;
    value = (static_cast<float>(static_cast<int32_t>(random >> 8u)) /
             8388608.0f) -
            1.0f;
  }

  struct target_fixture {
    std::vector<uint8_t> blob;
    tensor_view source{};
    std::vector<uint8_t> indices;
    std::vector<uint8_t> indices_by_input32;
    std::vector<float> norms;
    std::vector<float> norms_by_group32;
    emel::kernel::cq::event::prepared_q4_view prepared{};
    std::vector<float> scalar;
    std::vector<float> separate;
    std::vector<float> batch;
    std::array<float, in> scalar_workspace{};
    std::array<float, in> prepared_workspace{};
  };
  std::array<target_fixture, 4u> targets{};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::prepared_codebook_q4 prepared_codebook{};
  emel::kernel::cq::action::prepare_codebook_q4(
      {codebook, prepared_codebook});

  for (uint32_t target_index = 0u; target_index < targets.size();
       ++target_index) {
    auto &target = targets[target_index];
    const uint32_t out = outs[target_index];
    std::vector<uint32_t> selectors(static_cast<size_t>(out) * in);
    for (uint32_t &selector : selectors) {
      random = random * 1664525u + 1013904223u;
      selector = (random >> 28u) & 15u;
    }
    std::vector<uint16_t> norm_bits(static_cast<size_t>(out) * in / group);
    for (uint16_t &bits : norm_bits) {
      random = random * 1664525u + 1013904223u;
      bits = static_cast<uint16_t>(0x3000u + ((random >> 24u) & 15u) * 0x80u);
    }
    target.blob = blob<4u>(selectors, out, in, group, norm_bits);
    target.source = view(target.blob, out, in, group, 4u);
    target.indices.resize(static_cast<size_t>(out) * in);
    target.indices_by_input32.resize(static_cast<size_t>(out) * in);
    target.norms.resize(static_cast<size_t>(out) * in / group);
    emel::kernel::cq::event::dispatch_result prepare_result{};
    const size_t blocked_rows = out / 32u * 32u;
    target.norms_by_group32.resize(blocked_rows * in / group);
    const emel::kernel::cq::event::prepare_q4_request prepare_request{
        target.source, target.indices, target.indices_by_input32, target.norms,
        target.norms_by_group32, target.prepared};
    REQUIRE(sm.process_event(
        emel::kernel::cq::event::prepare_q4{prepare_request, prepare_result}));

    target.scalar.resize(out);
    target.separate.resize(out);
    target.batch.resize(out);
    const gemv_request scalar_request{target.source, codebook, activation,
                                      target.scalar, target.scalar_workspace};
    emel::kernel::cq::event::dispatch_result scalar_result{};
    REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_q4{
        scalar_request, scalar_result}));
    const emel::kernel::cq::event::prepared_gemv_request prepared_request{
        target.prepared, prepared_codebook, activation, target.separate,
        target.prepared_workspace};
    emel::kernel::cq::event::dispatch_result prepared_result{};
    REQUIRE(sm.process_event(emel::kernel::cq::event::execute_prepared_avx2_q4{
        prepared_request, prepared_result}));
  }

  std::array<float, in> batch_workspace{};
  const emel::kernel::cq::event::prepared_gemv_batch4_request batch_request{
      .targets = {{{&targets[0].prepared, targets[0].batch},
                   {&targets[1].prepared, targets[1].batch},
                   {&targets[2].prepared, targets[2].batch},
                   {&targets[3].prepared, targets[3].batch}}},
      .codebook = prepared_codebook,
      .activation = activation,
      .workspace = batch_workspace};
  emel::kernel::cq::event::dispatch_result batch_result{};
  REQUIRE(
      sm.process_event(emel::kernel::cq::event::execute_prepared_avx2_batch4_q4{
          batch_request, batch_result}));
  for (const auto &target : targets)
    for (uint32_t row = 0u; row < target.batch.size(); ++row) {
      CHECK(target.batch[row] == target.separate[row]);
      CHECK(target.batch[row] ==
            doctest::Approx(target.scalar[row]).epsilon(1.0e-5));
    }
#endif
}

TEST_CASE("CQ4 prepared lookup preserves vector and scalar group tails") {
#if defined(__AVX2__) && defined(__FMA__)
  for (const uint32_t group : {2u, 4u, 8u, 16u, 32u, 64u, 128u}) {
    const uint32_t in = group * 2u;
    constexpr uint32_t out = 3u;
    std::array<float, 28u> cb{};
    for (uint32_t i = 0u; i < 16u; ++i)
      cb[12u + i] = std::sin(static_cast<float>(i + 1u) * 0.37f);
    std::vector<uint32_t> ix(static_cast<size_t>(out) * in);
    for (size_t i = 0u; i < ix.size(); ++i)
      ix[i] = static_cast<uint32_t>((i * 11u + i / in * 5u + 7u) & 15u);
    std::vector<uint16_t> source_norms(static_cast<size_t>(out) * 2u);
    constexpr std::array<uint16_t, 4u> norm_bits{0x3800u, 0x3c00u, 0x4000u,
                                                 0x4200u};
    for (size_t i = 0u; i < source_norms.size(); ++i)
      source_norms[i] = norm_bits[(i * 3u + i / 2u) & 3u];
    const auto b = blob<4u>(ix, out, in, group, source_norms);
    const auto v = view(b, out, in, group, 4u);
    std::vector<uint8_t> indices(static_cast<size_t>(out) * in);
    std::vector<uint8_t> indices_by_input32(static_cast<size_t>(out) * in);
    std::vector<float> norms(static_cast<size_t>(out) * 2u);
    std::vector<float> norms_by_group32(blocked_norm_count(v));
    emel::kernel::cq::event::prepared_q4_view prepared{};
    emel::kernel::cq::sm sm;
    emel::kernel::cq::event::dispatch_result prepare_result{};
    const emel::kernel::cq::event::prepare_q4_request prepare_request{
        v, indices, indices_by_input32, norms, norms_by_group32, prepared};
    REQUIRE(sm.process_event(
        emel::kernel::cq::event::prepare_q4{prepare_request, prepare_result}));
    emel::kernel::cq::event::prepared_codebook_q4 prepared_codebook{};
    emel::kernel::cq::event::dispatch_result codebook_result{};
    const emel::kernel::cq::event::prepare_codebook_q4_request codebook_request{
        cb, prepared_codebook};
    REQUIRE(sm.process_event(emel::kernel::cq::event::prepare_codebook_q4{
        codebook_request, codebook_result}));
    std::vector<float> activation(in);
    for (uint32_t i = 0u; i < in; ++i)
      activation[i] = std::cos(static_cast<float>(i + 1u) * 0.0625f);
    std::vector<float> scalar_workspace(in);
    std::vector<float> prepared_workspace(in);
    std::array<float, out> scalar{};
    std::array<float, out> actual{};
    const gemv_request scalar_request{v, cb, activation, scalar,
                                      scalar_workspace};
    emel::kernel::cq::event::dispatch_result scalar_result{};
    REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_q4{
        scalar_request, scalar_result}));
    const emel::kernel::cq::event::prepared_gemv_request prepared_request{
        prepared, prepared_codebook, activation, actual, prepared_workspace};
    emel::kernel::cq::event::dispatch_result prepared_result{};
    REQUIRE(sm.process_event(emel::kernel::cq::event::execute_prepared_avx2_q4{
        prepared_request, prepared_result}));
    for (uint32_t row = 0u; row < out; ++row)
      CHECK(actual[row] == doctest::Approx(scalar[row]).epsilon(1.0e-5));
  }
#endif
}

TEST_CASE("CQ4 prepared norm hoist matches scalar for realistic random tensors") {
#if defined(__AVX2__) && defined(__FMA__)
  constexpr uint32_t group = 128u, in = 512u, out = 512u;
  std::array<float, 28u> cb{};
  uint32_t random = 0x243f6a88u;
  for (uint32_t i = 0u; i < 16u; ++i) {
    random = random * 1664525u + 1013904223u;
    cb[12u + i] =
        (static_cast<float>(static_cast<int32_t>(random >> 9u)) /
         8388608.0f) -
        1.0f;
  }
  std::vector<uint32_t> source_indices(static_cast<size_t>(out) * in);
  for (uint32_t &index : source_indices) {
    random = random * 1664525u + 1013904223u;
    index = random >> 28u;
  }
  std::vector<uint16_t> norm_bits(static_cast<size_t>(out) * in / group);
  for (uint16_t &bits : norm_bits) {
    random = random * 1664525u + 1013904223u;
    bits = static_cast<uint16_t>(0x3000u + ((random >> 23u) & 31u) * 0x40u);
  }
  const auto b = blob<4u>(source_indices, out, in, group, norm_bits);
  const auto v = view(b, out, in, group, 4u);
  std::vector<uint8_t> indices(static_cast<size_t>(out) * in);
  std::vector<uint8_t> indices_by_input32(static_cast<size_t>(out) * in);
  std::vector<float> norms(static_cast<size_t>(out) * in / group);
  std::vector<float> norms_by_group32(static_cast<size_t>(out) * in / group);
  emel::kernel::cq::event::prepared_q4_view prepared{};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result prepare_result{};
  const emel::kernel::cq::event::prepare_q4_request prepare_request{
      v, indices, indices_by_input32, norms, norms_by_group32, prepared};
  REQUIRE(sm.process_event(
      emel::kernel::cq::event::prepare_q4{prepare_request, prepare_result}));
  emel::kernel::cq::event::prepared_codebook_q4 prepared_codebook{};
  emel::kernel::cq::action::prepare_codebook_q4({cb, prepared_codebook});
  std::array<float, in> activation{};
  for (float &value : activation) {
    random = random * 1664525u + 1013904223u;
    value = (static_cast<float>(static_cast<int32_t>(random >> 8u)) /
             8388608.0f) -
            1.0f;
  }
  std::array<float, in> scalar_workspace{};
  std::array<float, in> prepared_workspace{};
  std::array<float, out> scalar{};
  std::array<float, out> block32{};
  const gemv_request scalar_request{v, cb, activation, scalar,
                                    scalar_workspace};
  emel::kernel::cq::event::dispatch_result scalar_result{};
  REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_q4{
      scalar_request, scalar_result}));
  const emel::kernel::cq::event::prepared_gemv_request prepared_request{
      prepared, prepared_codebook, activation, block32, prepared_workspace};
  emel::kernel::cq::event::dispatch_result prepared_result{};
  REQUIRE(sm.process_event(emel::kernel::cq::event::execute_prepared_avx2_q4{
      prepared_request, prepared_result}));
  for (uint32_t row = 0u; row < out; ++row)
    CHECK(block32[row] == doctest::Approx(scalar[row]).epsilon(1.0e-5));
#endif
}

TEST_CASE("CQ4 prepared block32 input-major route preserves exact output") {
#if defined(__AVX2__) && defined(__FMA__)
  constexpr uint32_t group = 128u, in = 512u, out = 512u;
  std::array<float, 28u> cb{};
  for (uint32_t i = 0u; i < 16u; ++i)
    cb[12u + i] = (static_cast<float>(i) - 7.5f) / 8.0f;
  std::vector<uint32_t> source_indices(static_cast<size_t>(out) * in);
  std::vector<uint8_t> indices(static_cast<size_t>(out) * in);
  std::vector<uint8_t> indices_by_input32(static_cast<size_t>(out) * in);
  std::vector<float> norms(static_cast<size_t>(out) * in / group);
  std::vector<float> norms_by_group32(static_cast<size_t>(out) * in / group);
  std::array<float, in> activation{};
  for (size_t i = 0u; i < source_indices.size(); ++i) {
    const size_t row = i / in;
    const size_t column = i % in;
    source_indices[i] = static_cast<uint32_t>(
        (column * 13u + row * 7u + (row >> 4u) * 8u + 3u) & 15u);
  }
  for (uint32_t i = 0u; i < in; ++i)
    activation[i] = std::sin(static_cast<float>(i + 1u) * 0.03125f);
  std::vector<uint16_t> norm_bits(norms.size());
  for (size_t i = 0u; i < norm_bits.size(); ++i)
    norm_bits[i] = static_cast<uint16_t>(0x3000u + i);
  const auto b = blob<4u>(source_indices, out, in, group, norm_bits);
  const auto v = view(b, out, in, group, 4u);
  emel::kernel::cq::event::prepared_q4_view prepared{};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result prepare_result{};
  const emel::kernel::cq::event::prepare_q4_request prepare_request{
      v, indices, indices_by_input32, norms, norms_by_group32, prepared};
  REQUIRE(sm.process_event(
      emel::kernel::cq::event::prepare_q4{prepare_request, prepare_result}));
  emel::kernel::cq::event::prepared_codebook_q4 prepared_codebook{};
  REQUIRE(prepared.norms_by_group32.size() == norms_by_group32.size());
  const uint32_t groups_per_row = in / group;
  constexpr std::array<uint32_t, 32u> raw_rows{
      0u,  1u,  2u,  3u,  16u, 17u, 18u, 19u,
      4u,  5u,  6u,  7u,  20u, 21u, 22u, 23u,
      8u,  9u,  10u, 11u, 24u, 25u, 26u, 27u,
      12u, 13u, 14u, 15u, 28u, 29u, 30u, 31u};
  for (uint32_t row = 0u; row < out; row += 32u)
    for (uint32_t group_index = 0u; group_index < groups_per_row;
         ++group_index)
      for (uint32_t lane = 0u; lane < raw_rows.size(); ++lane)
        CHECK(prepared.norms_by_group32[static_cast<size_t>(row) *
                                            groups_per_row +
                                        static_cast<size_t>(group_index) * 32u +
                                        lane] ==
              prepared.norms[static_cast<size_t>(row + raw_rows[lane]) *
                                 groups_per_row +
                             group_index]);
  emel::kernel::cq::action::prepare_codebook_q4({cb, prepared_codebook});
  REQUIRE(prepared.indices_by_input32.size() == indices_by_input32.size());
  for (uint32_t row = 0u; row < out; row += 32u)
    for (uint32_t i = 0u; i < in; ++i)
      for (uint32_t lane = 0u; lane < 32u; ++lane)
        CHECK(prepared.indices_by_input32[static_cast<size_t>(row) * in +
                                          static_cast<size_t>(i) * 32u + lane] ==
              source_indices[static_cast<size_t>(row + lane) * in + i]);
  std::array<float, out> canonical_block32{};
  execute_block32_canonical_reference(prepared, prepared_codebook, activation,
                                      canonical_block32);
  std::array<float, out> block32{};
  std::array<float, out> block32_repeat{};
  emel::kernel::cq::action::execute_prepared_avx2_dot_block32_loaded(
      prepared, prepared_codebook, activation, block32,
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
          prepared_codebook.byte_planes[0].data())),
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
          prepared_codebook.byte_planes[1].data())),
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
          prepared_codebook.byte_planes[2].data())),
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
          prepared_codebook.byte_planes[3].data())));
  emel::kernel::cq::action::execute_prepared_avx2_dot_block32_loaded(
      prepared, prepared_codebook, activation, block32_repeat,
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
          prepared_codebook.byte_planes[0].data())),
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
          prepared_codebook.byte_planes[1].data())),
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
          prepared_codebook.byte_planes[2].data())),
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
          prepared_codebook.byte_planes[3].data())));
  for (uint32_t row = 0u; row < out; ++row) {
    CHECK(block32[row] == block32_repeat[row]);
    CHECK(block32[row] == canonical_block32[row]);
  }
#endif
}

TEST_CASE("CQ4 prepared block64 matches block32 bitwise including tails and scale") {
#if defined(__AVX2__) && defined(__FMA__)
  constexpr uint32_t group = 128u;
  constexpr uint32_t in = 512u;
  for (const uint32_t out : {64u, 65u, 96u, 128u, 160u, 256u, 512u,
                             8192u}) {
    uint32_t random = 0x9e3779b9u + out;
    std::array<float, 28u> codebook{};
    for (uint32_t i = 0u; i < 16u; ++i) {
      random = random * 1664525u + 1013904223u;
      const uint32_t bits = 0x3d000000u | (random & 0x007fffffu);
      std::memcpy(&codebook[12u + i], &bits, sizeof(float));
      if ((i & 1u) != 0u)
        codebook[12u + i] = -codebook[12u + i];
    }
    std::vector<uint32_t> selectors(static_cast<size_t>(out) * in);
    for (uint32_t &selector : selectors) {
      random = random * 1664525u + 1013904223u;
      selector = random >> 28u;
    }
    std::vector<uint16_t> norm_bits(static_cast<size_t>(out) * in / group);
    for (uint16_t &bits : norm_bits) {
      random = random * 1664525u + 1013904223u;
      bits = static_cast<uint16_t>(0x3000u + ((random >> 24u) & 15u) * 0x80u);
    }
    const auto b = blob<4u>(selectors, out, in, group, norm_bits);
    const auto v = view(b, out, in, group, 4u);
    std::vector<uint8_t> indices(static_cast<size_t>(out) * in);
    std::vector<uint8_t> indices_by_input32(
        static_cast<size_t>(out / 32u * 32u) * in);
    std::vector<float> norms(static_cast<size_t>(out) * in / group);
    std::vector<float> norms_by_group32(
        static_cast<size_t>(out / 32u * 32u) * in / group);
    emel::kernel::cq::event::prepared_q4_view prepared{};
    emel::kernel::cq::action::prepare_q4(
        {v, indices, indices_by_input32, norms, norms_by_group32, prepared});
    emel::kernel::cq::event::prepared_codebook_q4 prepared_codebook{};
    emel::kernel::cq::action::prepare_codebook_q4(
        {codebook, prepared_codebook});
    std::array<float, in> activation{};
    for (float &value : activation) {
      random = random * 1664525u + 1013904223u;
      value = (static_cast<float>(static_cast<int32_t>(random >> 8u)) /
               8388608.0f) -
              1.0f;
    }
    std::vector<float> block32(out);
    std::vector<float> block64(out);
    __m256i codebook_byte0;
    __m256i codebook_byte1;
    __m256i codebook_byte2;
    __m256i codebook_byte3;
    emel::kernel::cq::action::q4_codebook_byte_tables(
        prepared_codebook, codebook_byte0, codebook_byte1, codebook_byte2,
        codebook_byte3);
    emel::kernel::cq::action::execute_prepared_avx2_dot_block32_loaded(
        prepared, prepared_codebook, activation, block32, codebook_byte0,
        codebook_byte1, codebook_byte2, codebook_byte3);
    emel::kernel::cq::action::execute_prepared_avx2_dot_block64(
        prepared, prepared_codebook, activation, block64);
    for (uint32_t row = 0u; row < out; ++row)
      CHECK(block64[row] == block32[row]);

    constexpr float output_scale = 0.03125f;
    std::array<float, in> transformed{};
    emel::kernel::cq::detail::compute_fwht128_groups_avx2(
        activation, in, transformed);
    std::vector<float> transformed_block32(out);
    emel::kernel::cq::action::execute_prepared_avx2_dot_block32_loaded(
        prepared, prepared_codebook, transformed, transformed_block32,
        codebook_byte0, codebook_byte1, codebook_byte2, codebook_byte3);
    std::array<float, in> workspace{};
    std::vector<float> scaled(out);
    const emel::kernel::cq::event::prepared_gemv_request request{
        prepared, prepared_codebook, activation, scaled, workspace,
        output_scale};
    emel::kernel::cq::event::dispatch_result result{};
    emel::kernel::cq::sm sm;
    REQUIRE(sm.process_event(
        emel::kernel::cq::event::execute_prepared_avx2_q4{request, result}));
    for (uint32_t row = 0u; row < out; ++row)
      CHECK(scaled[row] == transformed_block32[row] * output_scale);
  }
#endif
}

TEST_CASE("CQ4 prepared dot-only dispatch matches full route and rejects malformed spans without writes") {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
  constexpr uint32_t out = 65u;
  constexpr uint32_t in = 128u;
  constexpr uint32_t group = 128u;
  std::vector<uint32_t> selectors(static_cast<size_t>(out) * in, 3u);
  std::vector<uint16_t> norm_bits(out, 0x3c00u);
  const auto b = blob<4u>(selectors, out, in, group, norm_bits);
  const auto v = view(b, out, in, group, 4u);
  std::vector<uint8_t> indices(static_cast<size_t>(out) * in);
  std::vector<uint8_t> indices_by_input32(static_cast<size_t>(out / 32u * 32u) * in);
  std::vector<float> norms(out);
  std::vector<float> norms_by_group32(out / 32u * 32u);
  emel::kernel::cq::event::prepared_q4_view prepared{};
  emel::kernel::cq::action::prepare_q4(
      {v, indices, indices_by_input32, norms, norms_by_group32, prepared});
  std::array<float, emel::cact::loader::k_codebook_len> codebook{};
  for (size_t i = 0u; i < codebook.size(); ++i)
    codebook[i] = static_cast<float>(static_cast<int32_t>(i) - 14) * 0.125f;
  emel::kernel::cq::event::prepared_codebook_q4 prepared_codebook{};
  emel::kernel::cq::action::prepare_codebook_q4({codebook, prepared_codebook});
  std::array<float, in> activation{};
  for (size_t i = 0u; i < activation.size(); ++i)
    activation[i] = static_cast<float>(static_cast<int32_t>(i % 17u) - 8) * 0.25f;
  std::array<float, in> transformed{};
  emel::kernel::cq::detail::compute_fwht128_groups_avx2(activation, in,
                                                        transformed);
  std::vector<float> full(out);
  std::vector<float> dot(out);
  std::array<float, in> workspace{};
  constexpr float scale = 0.03125f;
  emel::kernel::cq::sm machine;
  emel::kernel::cq::event::dispatch_result full_result{};
  const emel::kernel::cq::event::prepared_gemv_request full_request{
      prepared, prepared_codebook, activation, full, workspace, scale};
  REQUIRE(machine.process_event(
      emel::kernel::cq::event::execute_prepared_avx2_q4{full_request,
                                                        full_result}));
  emel::kernel::cq::event::dispatch_result dot_result{};
  const emel::kernel::cq::event::prepared_dot_q4_request dot_request{
      prepared, prepared_codebook, transformed, dot, scale};
  REQUIRE(machine.process_event(
      emel::kernel::cq::event::execute_prepared_avx2_dot_q4{dot_request,
                                                            dot_result}));
  CHECK(dot == full);
  std::vector<float> rejected(out, 123.0f);
  const emel::kernel::cq::event::prepared_dot_q4_request malformed{
      prepared, prepared_codebook,
      std::span<const float>{transformed}.first(in - 1u), rejected, scale};
  emel::kernel::cq::event::dispatch_result rejected_result{};
  CHECK_FALSE(machine.process_event(
      emel::kernel::cq::event::execute_prepared_avx2_dot_q4{malformed,
                                                            rejected_result}));
  for (const float value : rejected)
    CHECK(value == 123.0f);
#endif
}
