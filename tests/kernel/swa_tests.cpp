#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

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
        const int32_t key_value =
            static_cast<int32_t>((head * 29u + slot * 7u + col * 17u) % 83u) -
            41;
        const int32_t value_value =
            static_cast<int32_t>((head * 31u + slot * 19u + col * 5u) % 97u) -
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
  REQUIRE(
      generic_machine.process_event(emel::kernel::swa::event::execute_attend{
          generic_request, generic_result}));
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

void check_vector_exp_matches_scalar(const uint32_t capacity,
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
  std::vector<float> value_cache(key_cache.size());
  fill_gqa2_fixture(query, key_cache, value_cache, capacity);
  std::vector<float> scalar_workspace(static_cast<size_t>(span_len) * 2u);
  std::vector<float> vector_workspace(static_cast<size_t>(span_len) * 2u);
  std::vector<float> scalar_output(static_cast<size_t>(heads) * head_dim);
  std::vector<float> vector_output(scalar_output.size());
  const auto make_request = [&](std::span<float> workspace,
                                std::span<float> output) {
    return emel::kernel::swa::event::attend_request{.query = query,
                                                    .key_cache = key_cache,
                                                    .value_cache = value_cache,
                                                    .position = position,
                                                    .window_begin =
                                                        window_begin,
                                                    .capacity = capacity,
                                                    .heads = heads,
                                                    .kv_heads = kv_heads,
                                                    .head_dim = head_dim,
                                                    .workspace = workspace,
                                                    .output = output};
  };
  const auto scalar_request = make_request(scalar_workspace, scalar_output);
  const auto vector_request = make_request(vector_workspace, vector_output);
  emel::kernel::swa::sm scalar_machine;
  emel::kernel::swa::sm vector_machine;
  dispatch_result scalar_result{};
  dispatch_result vector_result{};
  REQUIRE(scalar_machine.process_event(
      emel::kernel::swa::event::execute_attend_gqa2_avx2{scalar_request,
                                                         scalar_result}));
  REQUIRE(vector_machine.process_event(
      emel::kernel::swa::event::execute_attend_gqa2_avx2_vector_exp{
          vector_request, vector_result}));
  for (size_t i = 0u; i < scalar_output.size(); ++i)
    CHECK(vector_output[i] == doctest::Approx(scalar_output[i]).epsilon(3e-5));
#else
  (void)capacity;
  (void)window_begin;
  (void)position;
#endif
}

TEST_CASE("swa vector exp approximation is finite monotonic and accurate") {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
  std::array<float, 808u> values{};
  for (size_t i = 0u; i < 801u; ++i)
    values[i] = -100.0f + static_cast<float>(i) * 0.125f;
  values[801] = -100.0f;
  values[802] = std::nextafter(-100.0f, 0.0f);
  values[803] = -std::numeric_limits<float>::min();
  values[804] = -std::numeric_limits<float>::denorm_min();
  values[805] = -0.0f;
  values[806] = 0.0f;
  values[807] = std::nextafter(0.0f, -1.0f);
  for (size_t base = 0u; base < values.size(); base += 8u) {
    alignas(32) float output[8];
    _mm256_store_ps(output, emel::kernel::swa::detail::expf8_approx_avx2(
                                _mm256_loadu_ps(values.data() + base)));
    for (size_t lane = 0u; lane < 8u; ++lane) {
      const float reference = std::exp(values[base + lane]);
      CHECK(std::isfinite(output[lane]));
      CHECK(output[lane] >= 0.0f);
      CHECK(std::abs(output[lane] - reference) <=
            std::max(1e-7f, reference * 3e-5f));
    }
  }
  std::sort(values.begin(), values.end());
  float previous = -1.0f;
  for (size_t base = 0u; base < values.size(); base += 8u) {
    alignas(32) float output[8];
    _mm256_store_ps(output, emel::kernel::swa::detail::expf8_approx_avx2(
                                _mm256_loadu_ps(values.data() + base)));
    for (const float value : output) {
      CHECK(value >= previous);
      previous = value;
    }
  }

  std::array<float, 17u> scores{
      -100.0f,        -31.0f,     -8.0f,       -4.0f,        -2.0f,
      -1.0f,          -0.5f,      -0.25f,      -0.125f,      -0.0625f,
      -0.03125f,      -0.015625f, -0.0078125f, -0.00390625f, -0.001953125f,
      -0.0009765625f, 0.0f};
  const float sum = emel::kernel::swa::detail::exp_sum_avx2(
      scores.data(), static_cast<uint32_t>(scores.size()), 0.0f);
  float normalized_sum = 0.0f;
  for (const float weight : scores)
    normalized_sum += weight / sum;
  CHECK(normalized_sum == doctest::Approx(1.0f).epsilon(2e-6));
#endif
}

TEST_CASE("swa vector exp GQA2 route tracks scalar stable attention") {
  SUBCASE("span 128") { check_vector_exp_matches_scalar(704u, 0u, 127u); }
  SUBCASE("span 512") { check_vector_exp_matches_scalar(704u, 0u, 511u); }
  SUBCASE("span 704") { check_vector_exp_matches_scalar(704u, 0u, 703u); }
  SUBCASE("ring wrap") { check_vector_exp_matches_scalar(8u, 6u, 10u); }
}

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
  CHECK_FALSE(
      machine.process_event(emel::kernel::swa::event::execute_attend_gqa2_avx2{
          mismatch_request, result}));

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
  CHECK_FALSE(
      machine.process_event(emel::kernel::swa::event::execute_attend_gqa2_avx2{
          short_request, short_result}));
  dispatch_result vector_short_result{};
  CHECK_FALSE(machine.process_event(
      emel::kernel::swa::event::execute_attend_gqa2_avx2_vector_exp{
          short_request, vector_short_result}));
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
    std::array<float, 6> writable{17.0f, 19.0f, 23.0f, 29.0f, 31.0f, 37.0f};
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
    std::array<float, 6> value_and_workspace{2.0f, 3.0f,  5.0f,
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

TEST_CASE("swa cache write rejects aliased ranges before writes") {
  const auto reject = [](const std::span<const float> key_rows,
                         const std::span<const float> value_rows,
                         const std::span<float> key_cache,
                         const std::span<float> value_cache) {
    const emel::kernel::swa::event::cache_write_request request{
        .key_rows = key_rows,
        .value_rows = value_rows,
        .position = 0u,
        .capacity = 2u,
        .kv_heads = 1u,
        .head_dim = 2u,
        .key_cache = key_cache,
        .value_cache = value_cache};
    emel::kernel::swa::sm machine;
    dispatch_result result{};
    CHECK_FALSE(machine.process_event(
        emel::kernel::swa::event::execute_cache_write{request, result}));
  };

  SUBCASE("key and value caches exactly overlap") {
    std::array<float, 4> cache{11.0f, 13.0f, 17.0f, 19.0f};
    const auto before = cache;
    const std::array<float, 2> key_rows{5.0f, 6.0f};
    const std::array<float, 2> value_rows{7.0f, 8.0f};
    reject(key_rows, value_rows, cache, cache);
    CHECK(cache == before);
  }

  SUBCASE("key and value caches partially overlap") {
    std::array<float, 6> caches{11.0f, 13.0f, 17.0f, 19.0f, 23.0f, 29.0f};
    const auto before = caches;
    const std::array<float, 2> key_rows{5.0f, 6.0f};
    const std::array<float, 2> value_rows{7.0f, 8.0f};
    reject(key_rows, value_rows, std::span<float>{caches.data(), 4u},
           std::span<float>{caches.data() + 2u, 4u});
    CHECK(caches == before);
  }

  SUBCASE("key cache overlaps key rows") {
    std::array<float, 6> key_storage{5.0f, 6.0f, 11.0f, 13.0f, 17.0f, 19.0f};
    const auto before = key_storage;
    const std::array<float, 2> value_rows{7.0f, 8.0f};
    std::array<float, 4> value_cache{23.0f, 29.0f, 31.0f, 37.0f};
    const auto value_before = value_cache;
    reject(std::span<const float>{key_storage.data(), 2u}, value_rows,
           std::span<float>{key_storage.data(), 4u}, value_cache);
    CHECK(key_storage == before);
    CHECK(value_cache == value_before);
  }

  SUBCASE("value cache overlaps value rows") {
    const std::array<float, 2> key_rows{5.0f, 6.0f};
    std::array<float, 6> value_storage{7.0f, 8.0f, 11.0f, 13.0f, 17.0f, 19.0f};
    const auto before = value_storage;
    std::array<float, 4> key_cache{23.0f, 29.0f, 31.0f, 37.0f};
    const auto key_before = key_cache;
    reject(key_rows, std::span<const float>{value_storage.data(), 2u},
           key_cache, std::span<float>{value_storage.data(), 4u});
    CHECK(value_storage == before);
    CHECK(key_cache == key_before);
  }

  SUBCASE("key cache overlaps value rows") {
    const std::array<float, 2> key_rows{5.0f, 6.0f};
    std::array<float, 6> key_storage{7.0f, 8.0f, 11.0f, 13.0f, 17.0f, 19.0f};
    const auto before = key_storage;
    std::array<float, 4> value_cache{23.0f, 29.0f, 31.0f, 37.0f};
    const auto value_before = value_cache;
    reject(key_rows, std::span<const float>{key_storage.data(), 2u},
           std::span<float>{key_storage.data(), 4u}, value_cache);
    CHECK(key_storage == before);
    CHECK(value_cache == value_before);
  }

  SUBCASE("value cache overlaps key rows") {
    std::array<float, 6> value_storage{5.0f, 6.0f, 11.0f, 13.0f, 17.0f, 19.0f};
    const auto before = value_storage;
    const std::array<float, 2> value_rows{7.0f, 8.0f};
    std::array<float, 4> key_cache{23.0f, 29.0f, 31.0f, 37.0f};
    const auto key_before = key_cache;
    reject(std::span<const float>{value_storage.data(), 2u}, value_rows,
           key_cache, std::span<float>{value_storage.data(), 4u});
    CHECK(value_storage == before);
    CHECK(key_cache == key_before);
  }

  SUBCASE("cache endpoint exceeds uintptr") {
    const std::array<float, 2> key_rows{5.0f, 6.0f};
    const std::array<float, 2> value_rows{7.0f, 8.0f};
    std::array<float, 4> real_cache{11.0f, 13.0f, 17.0f, 19.0f};
    const auto before = real_cache;
    auto *near_max = reinterpret_cast<float *>(
        std::numeric_limits<std::uintptr_t>::max() - (sizeof(float) - 1u));
    reject(key_rows, value_rows, std::span<float>{near_max, 4u}, real_cache);
    CHECK(real_cache == before);
  }
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
