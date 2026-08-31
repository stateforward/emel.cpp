#include <array>
#include <cstring>
#include <vector>
#include <limits>
#include <span>

#include <doctest/doctest.h>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/swa/sm.hpp"

namespace {

using emel::kernel::swa::event::dispatch_result;


void fill_gqa2_fixture(std::vector<float> &query, std::vector<float> &key_cache,
                       std::vector<float> &value_cache,
                       const uint32_t capacity) {
  constexpr uint32_t heads = 8u;
  constexpr uint32_t kv_heads = 4u;
  constexpr uint32_t head_dim = 64u;
  for (uint32_t head = 0u; head < heads; ++head) {
    for (uint32_t col = 0u; col < head_dim; ++col) {
      const int32_t signed_value =
          static_cast<int32_t>((head * 37u + col * 13u) % 61u) - 30;
      query[static_cast<size_t>(head) * head_dim + col] =
          static_cast<float>(signed_value) * 0.03125f;
    }
  }
  for (uint32_t head = 0u; head < kv_heads; ++head) {
    for (uint32_t slot = 0u; slot < capacity; ++slot) {
      for (uint32_t col = 0u; col < head_dim; ++col) {
        const size_t index =
            (static_cast<size_t>(head) * capacity + slot) * head_dim + col;
        const int32_t key_value = static_cast<int32_t>(
                                      (head * 29u + slot * 7u + col * 17u) %
                                      83u) -
                                  41;
        const int32_t value_value = static_cast<int32_t>(
                                        (head * 31u + slot * 19u + col * 5u) %
                                        97u) -
                                    48;
        key_cache[index] = static_cast<float>(key_value) * 0.015625f;
        value_cache[index] = static_cast<float>(value_value) * 0.0078125f;
      }
    }
  }
}

void check_gqa2_matches_generic(const uint32_t capacity,
                                const uint32_t window_begin,
                                const uint32_t position) {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
  constexpr uint32_t heads = 8u;
  constexpr uint32_t kv_heads = 4u;
  constexpr uint32_t head_dim = 64u;
  const uint32_t span_len = position - window_begin + 1u;
  std::vector<float> query(static_cast<size_t>(heads) * head_dim);
  std::vector<float> key_cache(static_cast<size_t>(kv_heads) * capacity *
                               head_dim);
  std::vector<float> value_cache(static_cast<size_t>(kv_heads) * capacity *
                                 head_dim);
  fill_gqa2_fixture(query, key_cache, value_cache, capacity);
  std::vector<float> generic_workspace(span_len);
  std::vector<float> fused_workspace(static_cast<size_t>(span_len) * 2u);
  std::vector<float> generic_output(static_cast<size_t>(heads) * head_dim);
  std::vector<float> fused_output(static_cast<size_t>(heads) * head_dim);
  const emel::kernel::swa::event::attend_request generic_request{
      .query = query,
      .key_cache = key_cache,
      .value_cache = value_cache,
      .position = position,
      .window_begin = window_begin,
      .capacity = capacity,
      .heads = heads,
      .kv_heads = kv_heads,
      .head_dim = head_dim,
      .workspace = generic_workspace,
      .output = generic_output};
  const emel::kernel::swa::event::attend_request fused_request{
      .query = query,
      .key_cache = key_cache,
      .value_cache = value_cache,
      .position = position,
      .window_begin = window_begin,
      .capacity = capacity,
      .heads = heads,
      .kv_heads = kv_heads,
      .head_dim = head_dim,
      .workspace = fused_workspace,
      .output = fused_output};
  emel::kernel::swa::sm generic_machine;
  emel::kernel::swa::sm fused_machine;
  dispatch_result generic_result{};
  dispatch_result fused_result{};
  REQUIRE(generic_machine.process_event(
      emel::kernel::swa::event::execute_attend{generic_request,
                                               generic_result}));
  REQUIRE(fused_machine.process_event(
      emel::kernel::swa::event::execute_attend_gqa2_avx2{fused_request,
                                                         fused_result}));
  CHECK(std::memcmp(generic_output.data(), fused_output.data(),
                    generic_output.size() * sizeof(float)) == 0);
#else
  (void)capacity;
  (void)window_begin;
  (void)position;
#endif
}

} // namespace

TEST_CASE("swa attend computes grouped sliding-window softmax attention") {
  // heads=2, kv_heads=1, head_dim=2, capacity=4, positions 0..2 valid.
  std::array<float, 8> key_cache{};
  std::array<float, 8> value_cache{};
  const auto put = [](std::array<float, 8> &cache, const uint32_t position,
                      const float a, const float b) {
    cache[position * 2u] = a;
    cache[position * 2u + 1u] = b;
  };
  put(key_cache, 0u, 1.0f, 1.0f);
  put(key_cache, 1u, 0.0f, 2.0f);
  put(key_cache, 2u, -1.0f, 0.5f);
  put(value_cache, 0u, 1.0f, 0.0f);
  put(value_cache, 1u, 0.0f, 1.0f);
  put(value_cache, 2u, 2.0f, 2.0f);

  const std::array<float, 4> query{1.0f, 0.0f, 0.5f, -0.5f};
  std::array<float, 4> workspace{};
  std::array<float, 4> output{};
  const emel::kernel::swa::event::attend_request request{.query = query,
                                                         .key_cache = key_cache,
                                                         .value_cache =
                                                             value_cache,
                                                         .position = 2u,
                                                         .window_begin = 0u,
                                                         .capacity = 4u,
                                                         .heads = 2u,
                                                         .kv_heads = 1u,
                                                         .head_dim = 2u,
                                                         .workspace = workspace,
                                                         .output = output};
  emel::kernel::swa::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::swa::event::execute_attend{request, result}));
  CHECK(output[0] == doctest::Approx(0.8560338020324707f));
  CHECK(output[1] == doctest::Approx(0.5640538930892944f));
  CHECK(output[2] == doctest::Approx(1.0458049774169922f));
  CHECK(output[3] == doctest::Approx(0.8022611737251282f));
}

TEST_CASE("swa GQA2 AVX2 matches generic attention bitwise") {
  SUBCASE("span one") { check_gqa2_matches_generic(704u, 17u, 17u); }
  SUBCASE("ring wrap") { check_gqa2_matches_generic(8u, 6u, 10u); }
  SUBCASE("full pinned window") {
    check_gqa2_matches_generic(704u, 701u, 1404u);
  }
}

TEST_CASE("swa GQA2 route rejects reps mismatch and short workspace") {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
  constexpr uint32_t capacity = 4u;
  constexpr uint32_t head_dim = 64u;
  std::vector<float> query(6u * head_dim);
  std::vector<float> key_cache(2u * capacity * head_dim);
  std::vector<float> value_cache(2u * capacity * head_dim);
  std::vector<float> valid_gqa2_workspace(8u);
  std::vector<float> generic_workspace(4u);
  std::vector<float> output(6u * head_dim);
  const emel::kernel::swa::event::attend_request mismatch_request{
      .query = query,
      .key_cache = key_cache,
      .value_cache = value_cache,
      .position = 3u,
      .window_begin = 0u,
      .capacity = capacity,
      .heads = 6u,
      .kv_heads = 2u,
      .head_dim = head_dim,
      .workspace = valid_gqa2_workspace,
      .output = output};
  emel::kernel::swa::sm machine;
  dispatch_result result{};
  CHECK_FALSE(machine.process_event(
      emel::kernel::swa::event::execute_attend_gqa2_avx2{mismatch_request,
                                                         result}));

  auto generic_request = mismatch_request;
  generic_request.workspace = generic_workspace;
  dispatch_result generic_result{};
  REQUIRE(machine.process_event(emel::kernel::swa::event::execute_attend{
      generic_request, generic_result}));

  std::vector<float> query_gqa2(4u * head_dim);
  std::vector<float> short_workspace(7u);
  std::vector<float> output_gqa2(4u * head_dim);
  const emel::kernel::swa::event::attend_request short_request{
      .query = query_gqa2,
      .key_cache = key_cache,
      .value_cache = value_cache,
      .position = 3u,
      .window_begin = 0u,
      .capacity = capacity,
      .heads = 4u,
      .kv_heads = 2u,
      .head_dim = head_dim,
      .workspace = short_workspace,
      .output = output_gqa2};
  dispatch_result short_result{};
  CHECK_FALSE(machine.process_event(
      emel::kernel::swa::event::execute_attend_gqa2_avx2{short_request,
                                                         short_result}));
#endif
}

TEST_CASE("swa attend rejects overflowing geometry before writes") {
  if constexpr (sizeof(size_t) >= sizeof(uint64_t)) {
    std::array<float, 2> storage{17.0f, 19.0f};
    const auto before = storage;
    constexpr uint32_t extent = uint32_t{1} << 31u;
    const size_t query_elements = static_cast<size_t>(uint64_t{4} * extent);
    const emel::kernel::swa::event::attend_request request{
        .query = std::span<const float>{storage.data(), query_elements},
        .key_cache = std::span<const float>{storage.data(), 0u},
        .value_cache = std::span<const float>{storage.data(), 0u},
        .position = 0u,
        .window_begin = 0u,
        .capacity = extent,
        .heads = 4u,
        .kv_heads = 4u,
        .head_dim = extent,
        .workspace = std::span<float>{storage.data(), 1u},
        .output = std::span<float>{storage.data(), query_elements}};
    emel::kernel::swa::sm machine;
    dispatch_result result{};
    CHECK_FALSE(machine.process_event(
        emel::kernel::swa::event::execute_attend{request, result}));
    CHECK(storage == before);
  }
}

TEST_CASE("swa GQA2 rejects inclusive span beyond uint32 before writes") {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
  std::array<float, 4> query{1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 1> cache{5.0f};
  std::array<float, 4> output{7.0f, 11.0f, 13.0f, 17.0f};
  const auto output_before = output;
  const size_t cache_elements =
      static_cast<size_t>(uint64_t{2} * std::numeric_limits<uint32_t>::max());
  const emel::kernel::swa::event::attend_request request{
      .query = query,
      .key_cache = std::span<const float>{cache.data(), cache_elements},
      .value_cache = std::span<const float>{cache.data(), cache_elements},
      .position = std::numeric_limits<uint32_t>::max(),
      .window_begin = 0u,
      .capacity = std::numeric_limits<uint32_t>::max(),
      .heads = 2u,
      .kv_heads = 1u,
      .head_dim = 2u,
      .workspace = {},
      .output = output};
  emel::kernel::swa::sm machine;
  dispatch_result result{};
  CHECK_FALSE(machine.process_event(
      emel::kernel::swa::event::execute_attend_gqa2_avx2{request, result}));
  CHECK(output == output_before);
#endif
}

TEST_CASE("swa GQA2 rejects writable range aliasing before writes") {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
  const std::array<float, 4> query{1.0f, 2.0f, 3.0f, 4.0f};
  const std::array<float, 4> key_cache{0.5f, 0.25f, -0.5f, -0.25f};
  const std::array<float, 4> value_cache{2.0f, 3.0f, 5.0f, 7.0f};

  SUBCASE("workspace and output exactly overlap") {
    std::array<float, 4> writable{17.0f, 19.0f, 23.0f, 29.0f};
    const auto before = writable;
    const emel::kernel::swa::event::attend_request request{
        .query = query,
        .key_cache = key_cache,
        .value_cache = value_cache,
        .position = 1u,
        .window_begin = 0u,
        .capacity = 2u,
        .heads = 2u,
        .kv_heads = 1u,
        .head_dim = 2u,
        .workspace = writable,
        .output = writable};
    emel::kernel::swa::sm machine;
    dispatch_result result{};
    CHECK_FALSE(machine.process_event(
        emel::kernel::swa::event::execute_attend_gqa2_avx2{request, result}));
    CHECK(writable == before);
  }

  SUBCASE("workspace and output partially overlap") {
    std::array<float, 6> writable{17.0f, 19.0f, 23.0f,
                                  29.0f, 31.0f, 37.0f};
    const auto before = writable;
    const emel::kernel::swa::event::attend_request request{
        .query = query,
        .key_cache = key_cache,
        .value_cache = value_cache,
        .position = 1u,
        .window_begin = 0u,
        .capacity = 2u,
        .heads = 2u,
        .kv_heads = 1u,
        .head_dim = 2u,
        .workspace = std::span<float>{writable.data(), 4u},
        .output = std::span<float>{writable.data() + 2u, 4u}};
    emel::kernel::swa::sm machine;
    dispatch_result result{};
    CHECK_FALSE(machine.process_event(
        emel::kernel::swa::event::execute_attend_gqa2_avx2{request, result}));
    CHECK(writable == before);
  }

  SUBCASE("output overlaps query input") {
    std::array<float, 4> query_and_output{1.0f, 2.0f, 3.0f, 4.0f};
    const auto before = query_and_output;
    std::array<float, 4> workspace{};
    const emel::kernel::swa::event::attend_request request{
        .query = query_and_output,
        .key_cache = key_cache,
        .value_cache = value_cache,
        .position = 1u,
        .window_begin = 0u,
        .capacity = 2u,
        .heads = 2u,
        .kv_heads = 1u,
        .head_dim = 2u,
        .workspace = workspace,
        .output = query_and_output};
    emel::kernel::swa::sm machine;
    dispatch_result result{};
    CHECK_FALSE(machine.process_event(
        emel::kernel::swa::event::execute_attend_gqa2_avx2{request, result}));
    CHECK(query_and_output == before);
  }

  SUBCASE("workspace overlaps value input") {
    std::array<float, 6> value_and_workspace{2.0f, 3.0f, 5.0f,
                                             7.0f, 11.0f, 13.0f};
    const auto before = value_and_workspace;
    std::array<float, 4> output{};
    const emel::kernel::swa::event::attend_request request{
        .query = query,
        .key_cache = key_cache,
        .value_cache = std::span<const float>{value_and_workspace.data(), 4u},
        .position = 1u,
        .window_begin = 0u,
        .capacity = 2u,
        .heads = 2u,
        .kv_heads = 1u,
        .head_dim = 2u,
        .workspace = std::span<float>{value_and_workspace.data() + 2u, 4u},
        .output = output};
    emel::kernel::swa::sm machine;
    dispatch_result result{};
    CHECK_FALSE(machine.process_event(
        emel::kernel::swa::event::execute_attend_gqa2_avx2{request, result}));
    CHECK(value_and_workspace == before);
  }
#endif
}

TEST_CASE("swa cache write lands rows at position modulo capacity") {
  std::array<float, 8> key_cache{};
  std::array<float, 8> value_cache{};
  const std::array<float, 2> key_rows{5.0f, 6.0f};
  const std::array<float, 2> value_rows{7.0f, 8.0f};
  const emel::kernel::swa::event::cache_write_request request{
      .key_rows = key_rows,
      .value_rows = value_rows,
      .position = 6u, // capacity 4 -> physical slot 2
      .capacity = 4u,
      .kv_heads = 1u,
      .head_dim = 2u,
      .key_cache = key_cache,
      .value_cache = value_cache};
  emel::kernel::swa::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::swa::event::execute_cache_write{request, result}));
  CHECK(key_cache[4] == 5.0f);
  CHECK(key_cache[5] == 6.0f);
  CHECK(value_cache[4] == 7.0f);
  CHECK(value_cache[5] == 8.0f);
}

TEST_CASE("swa gate mul applies elementwise sigmoid gating") {
  std::array<float, 4> values{1.0f, 2.0f, 3.0f, 4.0f};
  const std::array<float, 4> gate_logits{0.3f, -1.2f, 2.0f, 0.0f};
  const emel::kernel::swa::event::gate_mul_request request{
      .values = values, .gate_logits = gate_logits, .dim = 4u};
  emel::kernel::swa::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::swa::event::execute_gate_mul{request, result}));
  CHECK(values[0] == doctest::Approx(0.5744425058364868f));
  CHECK(values[1] == doctest::Approx(0.4629504382610321f));
  CHECK(values[2] == doctest::Approx(2.6423912048339844f));
  CHECK(values[3] == doctest::Approx(2.0f));
}

TEST_CASE("swa residual gate adds sigmoid-scaled values onto skip") {
  // gate = fp16(0.7) decoded to f32.
  const float gate = emel::kernel::detail::quant::fp16_to_fp32(0x399au);
  const std::array<float, 2> skip{1.0f, -1.0f};
  const std::array<float, 2> values{2.0f, 4.0f};
  std::array<float, 2> output{};
  const emel::kernel::swa::event::residual_gate_request request{
      .skip = skip,
      .gate = gate,
      .values = values,
      .dim = 2u,
      .output = output};
  emel::kernel::swa::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::swa::event::execute_residual_gate{request, result}));
  CHECK(output[0] == doctest::Approx(2.3364620208740234f));
  CHECK(output[1] == doctest::Approx(1.672924280166626f));
}

TEST_CASE("swa attend guard rejects windows wider than the ring capacity") {
  const std::array<float, 4> query{};
  const std::array<float, 8> key_cache{};
  const std::array<float, 8> value_cache{};
  std::array<float, 8> workspace{};
  std::array<float, 4> output{};
  const emel::kernel::swa::event::attend_request request{
      .query = query,
      .key_cache = key_cache,
      .value_cache = value_cache,
      .position = 5u,
      .window_begin = 0u, // span 6 > capacity 4
      .capacity = 4u,
      .heads = 2u,
      .kv_heads = 1u,
      .head_dim = 2u,
      .workspace = workspace,
      .output = output};
  emel::kernel::swa::sm machine;
  dispatch_result result{};
  CHECK_FALSE(machine.process_event(
      emel::kernel::swa::event::execute_attend{request, result}));
  CHECK(machine.is(stateforward::sml::state<emel::kernel::swa::state_ready>));
}
