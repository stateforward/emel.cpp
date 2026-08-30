#include "../allocation_tracker.hpp"
#include "emel/kernel/cq/detail.hpp"
#include "emel/kernel/cq/sm.hpp"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
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
TEST_CASE("CQ A8 fake quant matches JAX ties zero and signed boundary") {
  REQUIRE(std::fesetround(FE_TONEAREST) == 0);
  emel::kernel::cq::sm sm;

  {
    const std::array<float, 4u> input{0.0f, 0.0f, -0.0f, 0.0f};
    std::array<int8_t, 4u> quantized{};
    std::array<float, 4u> dequantized{};
    float scale = 0.0f;
    const emel::kernel::cq::event::quantize_a8_request request{
        input, quantized, dequantized, scale};
    emel::kernel::cq::event::dispatch_result result{};
    REQUIRE(sm.process_event(
        emel::kernel::cq::event::quantize_a8{request, result}));
    CHECK(scale == 1.0f);
    for (uint32_t i = 0u; i < input.size(); ++i) {
      CHECK(quantized[i] == 0);
      CHECK(dequantized[i] == 0.0f);
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
    std::array<float, 9u> dequantized{};
    float scale = 0.0f;
    const emel::kernel::cq::event::quantize_a8_request request{
        input, quantized, dequantized, scale};
    emel::kernel::cq::event::dispatch_result result{};
    REQUIRE(sm.process_event(
        emel::kernel::cq::event::quantize_a8{request, result}));
    CHECK(scale == 1.0f);
    for (uint32_t i = 0u; i < input.size(); ++i) {
      CHECK(quantized[i] == expected[i]);
      CHECK(dequantized[i] == static_cast<float>(expected[i]));
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
  std::array<float, 2u> dequantized{};
  float scale = 0.0f;
  const emel::kernel::cq::event::quantize_a8_request request{
      input, quantized, dequantized, scale};
  emel::kernel::cq::event::dispatch_result result{};
  emel::kernel::cq::sm sm;
  CHECK_FALSE(
      sm.process_event(emel::kernel::cq::event::quantize_a8{request, result}));
  CHECK(sm.is(stateforward::sml::state<emel::kernel::cq::state_ready>));
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

TEST_CASE("CQ4 preparation borrows packed selectors and preserves identity") {
  std::array<float, 28u> cb{};
  for (uint32_t i = 0u; i < 16u; ++i)
    cb[12u + i] = static_cast<float>(i + 1u) / 10.f;
  constexpr uint32_t group = 8u, in = 13u, in_pad = 16u, out = 5u;
  std::vector<uint32_t> ix;
  for (uint32_t i = 0u; i < out * in_pad; ++i)
    ix.push_back((i * 7u + 3u) % 16u);
  const auto b = blob<4u>(ix, out, in, group,
                          {0x3c00, 0x4000, 0x4200, 0x4400, 0x3c00, 0x4000,
                           0x4200, 0x4400, 0x3c00, 0x4000});
  const auto v = view(b, out, in, group, 4u);
  std::vector<float> prepared_norms(out * in_pad / group);
  emel::kernel::cq::event::prepared_q4_view prepared{};
  const emel::kernel::cq::event::prepare_q4_request prepare_request{
      v, prepared_norms, prepared};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result prepare_result{};
  {
    emel::test::allocation::allocation_scope allocations{};
    REQUIRE(sm.process_event(
        emel::kernel::cq::event::prepare_q4{prepare_request, prepare_result}));
    CHECK(allocations.allocations() == 0u);
  }
  CHECK(prepared.source == v.data);
  CHECK(prepared.norms.size() == out * in_pad / group);
#if defined(__AVX2__) && defined(__FMA__)
  __m128i decoded0;
  __m128i decoded1;
  __m128i decoded2;
  __m128i decoded3;
  emel::kernel::cq::action::q4_unpack_32(b.data(), decoded0, decoded1, decoded2,
                                         decoded3);
  alignas(16) std::array<uint8_t, 32u> decoded{};
  _mm_storel_epi64(reinterpret_cast<__m128i *>(decoded.data()), decoded0);
  _mm_storel_epi64(reinterpret_cast<__m128i *>(decoded.data() + 8u), decoded1);
  _mm_storel_epi64(reinterpret_cast<__m128i *>(decoded.data() + 16u), decoded2);
  _mm_storel_epi64(reinterpret_cast<__m128i *>(decoded.data() + 24u), decoded3);
  for (uint32_t i = 0u; i < decoded.size(); ++i)
    CHECK(decoded[i] == ix[i]);
#endif

  std::array<float, in> activation{};
  for (uint32_t i = 0u; i < in; ++i)
    activation[i] = static_cast<float>(i + 1u) / 8.f;
  std::array<float, in_pad> scalar_workspace{};
  std::array<float, in_pad> prepared_workspace{};
  std::array<float, out> scalar_output{};
  std::array<float, out> prepared_output{};
  const gemv_request scalar_request{v, cb, activation, scalar_output,
                                    scalar_workspace};
  emel::kernel::cq::event::dispatch_result scalar_result{};
  REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_q4{
      scalar_request, scalar_result}));
  const emel::kernel::cq::event::prepared_gemv_request prepared_request{
      prepared, cb, activation, prepared_output, prepared_workspace};
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
  std::vector<float> norms(out * in / group);
  emel::kernel::cq::event::prepared_q4_view prepared{};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result prepare_result{};
  const emel::kernel::cq::event::prepare_q4_request prepare_request{v, norms,
                                                                    prepared};
  REQUIRE(sm.process_event(
      emel::kernel::cq::event::prepare_q4{prepare_request, prepare_result}));
  std::array<float, in> activation{};
  for (uint32_t i = 0u; i < in; ++i)
    activation[i] = static_cast<float>(i + 1u) / 8.f;
  std::array<float, in> separate_workspace{};
  std::array<float, in> batch_workspace{};
  std::array<float, out> expected{};
  const emel::kernel::cq::event::prepared_gemv_request separate_request{
      prepared, cb, activation, expected, separate_workspace};
  emel::kernel::cq::event::dispatch_result separate_result{};
  REQUIRE(sm.process_event(emel::kernel::cq::event::execute_prepared_avx2_q4{
      separate_request, separate_result}));
  std::array<std::array<float, out>, 4u> outputs{};
  const emel::kernel::cq::event::prepared_gemv_batch4_request batch_request{
      .targets = {{{&prepared, outputs[0]},
                   {&prepared, outputs[1]},
                   {&prepared, outputs[2]},
                   {&prepared, outputs[3]}}},
      .codebook = cb,
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

TEST_CASE("CQ4 packed 512x512 group128 row blocks preserve exact output") {
#if defined(__AVX2__) && defined(__FMA__)
  constexpr uint32_t group = 128u, in = 512u, out = 512u;
  std::array<float, 28u> cb{};
  for (uint32_t i = 0u; i < 16u; ++i)
    cb[12u + i] = (static_cast<float>(i) - 7.5f) / 8.0f;
  std::vector<uint32_t> selectors(static_cast<size_t>(out) * in);
  for (size_t i = 0u; i < selectors.size(); ++i)
    selectors[i] = static_cast<uint32_t>((i * 13u + i / in * 7u + 3u) & 15u);
  std::vector<uint16_t> norm_bits(static_cast<size_t>(out) * in / group);
  for (size_t i = 0u; i < norm_bits.size(); ++i)
    norm_bits[i] = static_cast<uint16_t>(0x3000u + ((i * 5u) & 0x03ffu));
  const auto packed = blob<4u>(selectors, out, in, group, norm_bits);
  const auto weights = view(packed, out, in, group, 4u);
  std::vector<float> norms(norm_bits.size());
  emel::kernel::cq::event::prepared_q4_view prepared{};
  const emel::kernel::cq::event::prepare_q4_request prepare_request{
      weights, norms, prepared};
  emel::kernel::cq::sm sm;
  emel::kernel::cq::event::dispatch_result prepare_result{};
  REQUIRE(sm.process_event(
      emel::kernel::cq::event::prepare_q4{prepare_request, prepare_result}));
  std::array<float, in> activation{};
  for (uint32_t i = 0u; i < in; ++i)
    activation[i] = std::sin(static_cast<float>(i + 1u) * 0.03125f);
  std::array<float, out> single{};
  std::array<float, out> block4{};
  std::array<float, out> block8{};
  emel::kernel::cq::action::execute_prepared_avx2_dot(prepared, cb, activation,
                                                      0u, out, single);
  emel::kernel::cq::action::execute_prepared_avx2_dot_blocked4(
      prepared, cb, activation, block4);
  emel::kernel::cq::action::execute_prepared_avx2_dot_blocked8(
      prepared, cb, activation, block8);
  for (uint32_t row = 0u; row < out; ++row) {
    CHECK(block4[row] == doctest::Approx(single[row]).epsilon(1.0e-5));
    CHECK(block8[row] == doctest::Approx(single[row]).epsilon(1.0e-5));
  }
#endif
}
