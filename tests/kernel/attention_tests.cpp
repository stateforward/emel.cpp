#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

#include <doctest/doctest.h>

#include "../allocation_tracker.hpp"
#include "emel/kernel/attention/sm.hpp"
#include "emel/kernel/detail.hpp"

namespace {

namespace attention = emel::kernel::attention;

inline constexpr int32_t k_heads = 8;
inline constexpr int32_t k_head_dim = 16;
inline constexpr int32_t k_hidden_dim = k_heads * k_head_dim;
inline constexpr int32_t k_capacity = 5;

struct attention_fixture {
  attention_fixture() {
    for (std::size_t index = 0u; index < query.size(); ++index) {
      query[index] =
          static_cast<float>(static_cast<int32_t>(index) - 7) / 16.0f;
    }
    for (std::size_t index = 0u; index < key.size(); ++index) {
      const float key_value =
          static_cast<float>(static_cast<int32_t>(index % 19u) - 9) / 32.0f;
      const float value_value =
          static_cast<float>(static_cast<int32_t>(index % 23u) - 11) / 24.0f;
      key[index] = emel::kernel::detail::fp32_to_bf16(key_value);
      value[index] = emel::kernel::detail::fp32_to_bf16(value_value);
    }
  }

  attention::event::head_range_request
  request(const int32_t head_begin, const int32_t head_end,
          std::span<float> output) const noexcept {
    return attention::event::head_range_request{
        .query = std::span<const float>{query},
        .key_cache = std::span<const uint16_t>{key},
        .value_cache = std::span<const uint16_t>{value},
        .output = output,
        .layer_offset = 0u,
        .hidden_dim = k_hidden_dim,
        .head_dim = k_head_dim,
        .head_begin = head_begin,
        .head_end = head_end,
        .position_capacity = k_capacity,
        .physical_begin = 3,
        .valid_positions = 4,
    };
  }

  std::array<float, static_cast<std::size_t>(k_hidden_dim)> query = {};
  std::array<uint16_t, static_cast<std::size_t>(k_capacity *k_hidden_dim)> key =
      {};
  std::array<uint16_t, static_cast<std::size_t>(k_capacity *k_hidden_dim)>
      value = {};
};

struct outcome_counts {
  int done = 0;
  int error = 0;
};

void count_done(void *object, const attention::events::execute_done &) {
  ++static_cast<outcome_counts *>(object)->done;
}

void count_error(void *object, const attention::events::execute_error &) {
  ++static_cast<outcome_counts *>(object)->error;
}

std::array<float, static_cast<std::size_t>(k_hidden_dim)>
compute_legacy_attention(const attention::event::head_range_request &request) {
  std::array<float, static_cast<std::size_t>(k_hidden_dim)> output = {};
  std::array<uint16_t, static_cast<std::size_t>(k_head_dim)> q_bf16 = {};
  std::array<float, static_cast<std::size_t>(k_capacity)> scores = {};
  std::array<uint16_t, static_cast<std::size_t>(k_capacity)> weights = {};
  const float scale = 1.0f / std::sqrt(static_cast<float>(request.head_dim));

  for (int32_t head = request.head_begin; head < request.head_end; ++head) {
    const int32_t head_offset = head * request.head_dim;
    const float *q_head = request.query.data() + head_offset;
    for (int32_t dim = 0; dim < request.head_dim; ++dim) {
      q_bf16[static_cast<std::size_t>(dim)] =
          emel::kernel::detail::fp32_to_bf16(q_head[dim]);
    }
    scores.fill(-std::numeric_limits<float>::infinity());
    for (int32_t position = 0; position < request.valid_positions; ++position) {
      const int32_t unwrapped = request.physical_begin + position;
      const int32_t physical =
          unwrapped -
          static_cast<int32_t>(unwrapped >= request.position_capacity) *
              request.position_capacity;
      const std::size_t cache_begin =
          request.layer_offset +
          static_cast<std::size_t>(physical * request.hidden_dim + head_offset);
      scores[static_cast<std::size_t>(physical)] =
          emel::kernel::detail::vec_dot_bf16_ggml(
              request.head_dim, request.key_cache.data() + cache_begin,
              q_bf16.data()) *
          scale;
    }
    emel::kernel::detail::soft_max_row_ggml(request.position_capacity,
                                            scores.data());
    for (int32_t physical = 0; physical < request.position_capacity;
         ++physical) {
      weights[static_cast<std::size_t>(physical)] =
          emel::kernel::detail::fp32_to_bf16(
              scores[static_cast<std::size_t>(physical)]);
    }
    for (int32_t dim = 0; dim < request.head_dim; ++dim) {
      double sum = 0.0;
      for (int32_t position = 0; position < request.valid_positions;
           ++position) {
        const int32_t unwrapped = request.physical_begin + position;
        const int32_t physical =
            unwrapped -
            static_cast<int32_t>(unwrapped >= request.position_capacity) *
                request.position_capacity;
        const std::size_t value_index =
            request.layer_offset +
            static_cast<std::size_t>(physical * request.hidden_dim +
                                     head_offset + dim);
        sum += static_cast<double>(
            emel::kernel::detail::bf16_to_fp32(
                request.value_cache[value_index]) *
            emel::kernel::detail::bf16_to_fp32(
                weights[static_cast<std::size_t>(physical)]));
      }
      output[static_cast<std::size_t>(head_offset + dim)] =
          static_cast<float>(sum);
    }
  }
  return output;
}

} // namespace

TEST_CASE(
    "kernel attention head actors preserve exact contiguous head output") {
  attention_fixture fixture{};
  auto serial_actor = std::make_unique<attention::sm>();
  auto lane_actors = std::make_unique<std::array<attention::sm, k_heads>>();
  std::array<float, static_cast<std::size_t>(k_hidden_dim)> serial = {};
  std::array<float, static_cast<std::size_t>(k_hidden_dim)> partitioned = {};

  auto serial_request = fixture.request(0, k_heads, std::span<float>{serial});
  attention::event::dispatch_result serial_result{};
  REQUIRE(serial_actor->process_event(
      attention::event::execute{serial_request, serial_result}));
  REQUIRE(serial_result.accepted);

  for (int32_t head = 0; head < k_heads; ++head) {
    const std::size_t begin = static_cast<std::size_t>(head * k_head_dim);
    auto request =
        fixture.request(head, head + 1,
                        std::span<float>{partitioned.data() + begin,
                                         static_cast<std::size_t>(k_head_dim)});
    attention::event::dispatch_result result{};
    REQUIRE((*lane_actors)[static_cast<std::size_t>(head)].process_event(
        attention::event::execute{request, result}));
    REQUIRE(result.accepted);
  }

  CHECK(std::equal(serial.begin(), serial.end(), partitioned.begin()));
}

TEST_CASE("kernel attention value traversal preserves legacy output bits") {
  attention_fixture fixture{};
  auto actor = std::make_unique<attention::sm>();
  std::array<float, static_cast<std::size_t>(k_hidden_dim)> output = {};
  auto request = fixture.request(0, k_heads, std::span<float>{output});
  const auto legacy = compute_legacy_attention(request);
  attention::event::dispatch_result result{};

  REQUIRE(actor->process_event(attention::event::execute{request, result}));
  REQUIRE(result.accepted);
  CHECK(std::equal(output.begin(), output.end(), legacy.begin()));
}

TEST_CASE("kernel attention dispatch reuses actor scratch without allocation") {
  attention_fixture fixture{};
  auto actor = std::make_unique<attention::sm>();
  std::array<float, static_cast<std::size_t>(k_hidden_dim)> output = {};
  auto request = fixture.request(0, k_heads, std::span<float>{output});
  attention::event::dispatch_result result{};

  std::size_t allocations = 0u;
  {
    emel::test::allocation::allocation_scope allocation_scope{};
    REQUIRE(actor->process_event(attention::event::execute{request, result}));
    allocations = allocation_scope.allocations();
  }
  CHECK(result.accepted);
  CHECK(allocations == 0u);
}

TEST_CASE("kernel attention publishes explicit done and error outcomes") {
  attention_fixture fixture{};
  auto actor = std::make_unique<attention::sm>();
  std::array<float, static_cast<std::size_t>(k_hidden_dim)> output = {};
  auto request = fixture.request(0, k_heads, std::span<float>{output});
  attention::event::dispatch_result result{};
  outcome_counts counts{};
  attention::event::execute valid{request, result};
  valid.on_done = {&counts, count_done};
  valid.on_error = {&counts, count_error};

  REQUIRE(actor->process_event(valid));
  CHECK(counts.done == 1);
  CHECK(counts.error == 0);

  request.output = request.output.first(1u);
  attention::event::execute invalid{request, result};
  invalid.on_done = {&counts, count_done};
  invalid.on_error = {&counts, count_error};
  CHECK_FALSE(actor->process_event(invalid));
  CHECK(counts.done == 1);
  CHECK(counts.error == 1);
}

TEST_CASE("kernel attention rejects an undersized disjoint output span") {
  attention_fixture fixture{};
  auto actor = std::make_unique<attention::sm>();
  std::array<float, 1u> output = {};
  auto request = fixture.request(0, 2, std::span<float>{output});
  attention::event::dispatch_result result{};

  CHECK_FALSE(actor->process_event(attention::event::execute{request, result}));
  CHECK_FALSE(result.accepted);
}

TEST_CASE("kernel attention rejects invalid public request shapes and spans") {
  using request_type = attention::event::head_range_request;
  using mutate_type = void (*)(request_type &);
  struct invalid_case {
    const char *name;
    mutate_type mutate;
  };
  const std::array cases{
      invalid_case{"nonpositive hidden dim",
                   [](request_type &request) { request.hidden_dim = 0; }},
      invalid_case{"nonpositive head dim",
                   [](request_type &request) { request.head_dim = 0; }},
      invalid_case{"negative head begin",
                   [](request_type &request) { request.head_begin = -1; }},
      invalid_case{
          "empty head range",
          [](request_type &request) { request.head_end = request.head_begin; }},
      invalid_case{
          "nonpositive capacity",
          [](request_type &request) { request.position_capacity = 0; }},
      invalid_case{"negative physical begin",
                   [](request_type &request) { request.physical_begin = -1; }},
      invalid_case{"physical begin at capacity",
                   [](request_type &request) {
                     request.physical_begin = request.position_capacity;
                   }},
      invalid_case{"nonpositive valid positions",
                   [](request_type &request) { request.valid_positions = 0; }},
      invalid_case{"valid positions exceed capacity",
                   [](request_type &request) {
                     request.valid_positions = request.position_capacity + 1;
                   }},
      invalid_case{"layer offset overflows cache requirement",
                   [](request_type &request) {
                     request.layer_offset =
                         std::numeric_limits<std::size_t>::max();
                   }},
      invalid_case{
          "head end exceeds hidden heads",
          [](request_type &request) { request.head_end = k_heads + 1; }},
      invalid_case{"head dim exceeds scratch capacity",
                   [](request_type &request) {
                     request.head_dim = attention::action::k_max_head_dim + 1;
                     request.hidden_dim = request.head_dim;
                     request.head_end = 1;
                   }},
      invalid_case{"context exceeds scratch capacity",
                   [](request_type &request) {
                     request.position_capacity =
                         attention::action::k_max_context + 1;
                     request.physical_begin = 0;
                     request.valid_positions = 1;
                   }},
      invalid_case{"query span is undersized",
                   [](request_type &request) {
                     request.query =
                         request.query.first(request.query.size() - 1u);
                   }},
      invalid_case{"key span is undersized",
                   [](request_type &request) {
                     request.key_cache =
                         request.key_cache.first(request.key_cache.size() - 1u);
                   }},
      invalid_case{"value span is undersized",
                   [](request_type &request) {
                     request.value_cache = request.value_cache.first(
                         request.value_cache.size() - 1u);
                   }},
      invalid_case{"output span is undersized",
                   [](request_type &request) {
                     request.output = request.output.first(1u);
                   }},
  };

  for (const auto &test_case : cases) {
    CAPTURE(test_case.name);
    attention_fixture fixture{};
    auto actor = std::make_unique<attention::sm>();
    std::array<float, static_cast<std::size_t>(k_hidden_dim)> output = {};
    auto request = fixture.request(0, k_heads, std::span<float>{output});
    test_case.mutate(request);
    attention::event::dispatch_result result{.accepted = true};
    CHECK_FALSE(
        actor->process_event(attention::event::execute{request, result}));
    CHECK_FALSE(result.accepted);
  }
}

TEST_CASE("kernel attention rejects unsafe public span storage") {
  attention_fixture fixture{};
  auto actor = std::make_unique<attention::sm>();
  std::array<float, static_cast<std::size_t>(k_hidden_dim)> output = {};
  auto request = fixture.request(0, k_heads, std::span<float>{output});
  attention::event::dispatch_result result{.accepted = true};

  SUBCASE("sized null query") {
    request.query = std::span<const float>{static_cast<const float *>(nullptr),
                                           fixture.query.size()};
  }
  SUBCASE("misaligned output") {
    alignas(float) std::array<std::byte, sizeof(output) + 1u> storage = {};
    request.output = std::span<float>{
        reinterpret_cast<float *>(storage.data() + 1u), output.size()};
  }
  SUBCASE("output aliases query") {
    request.output = std::span<float>{const_cast<float *>(request.query.data()),
                                      output.size()};
  }
  SUBCASE("output aliases key cache") {
    request.output = std::span<float>{
        reinterpret_cast<float *>(fixture.key.data()), output.size()};
  }
  SUBCASE("output aliases value cache") {
    request.output = std::span<float>{
        reinterpret_cast<float *>(fixture.value.data()), output.size()};
  }

  CHECK_FALSE(actor->process_event(attention::event::execute{request, result}));
  CHECK_FALSE(result.accepted);
}

TEST_CASE("kernel attention rejects unexpected result-bearing dispatches") {
  struct unexpected_dispatch {
    attention::event::dispatch_result &result;
  };

  attention::action::context ctx{};
  attention::event::dispatch_result result{.accepted = true};
  attention::action::effect_on_unexpected{}(unexpected_dispatch{result}, ctx);

  CHECK_FALSE(result.accepted);
}
