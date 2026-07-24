#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

#include <doctest/doctest.h>

#include "../allocation_tracker.hpp"
#include "emel/kernel/f32_matvec/sm.hpp"

namespace {

namespace f32_matvec = emel::kernel::f32_matvec;

struct unknown_with_result {
  f32_matvec::event::dispatch_result &result;
};

struct unknown_without_result {};

struct outcome_counts {
  int done = 0;
  int error = 0;
  const f32_matvec::event::prepare_f32 *done_request = nullptr;
  const f32_matvec::event::prepare_f32 *error_request = nullptr;
};

void count_done(void *object, const f32_matvec::events::dispatch_done<
                                  f32_matvec::event::prepare_f32> &outcome) {
  auto &counts = *static_cast<outcome_counts *>(object);
  ++counts.done;
  counts.done_request = &outcome.request;
}

void count_error(void *object, const f32_matvec::events::dispatch_error<
                                   f32_matvec::event::prepare_f32> &outcome) {
  auto &counts = *static_cast<outcome_counts *>(object);
  ++counts.error;
  counts.error_request = &outcome.request;
}

template <std::size_t inner, std::size_t rows>
void check_exact_case(const std::array<float, inner * rows> &weights,
                      const std::array<float, inner> &input) {
  REQUIRE(std::fesetround(FE_TONEAREST) == 0);
  REQUIRE(std::fegetround() == FE_TONEAREST);
  auto actor = std::make_unique<f32_matvec::sm>();
  std::array<float, inner * rows> packed{};
  std::array<float, rows> reference{};
  std::array<float, rows> exact{};

  const f32_matvec::event::prepare_f32_request prepare_request{
      .source = std::span<const float>{weights},
      .destination = std::span<float>{packed},
      .inner = inner,
      .rows = rows,
  };
  f32_matvec::event::dispatch_result prepare_result{};
  REQUIRE(actor->process_event(
      f32_matvec::event::prepare_f32{prepare_request, prepare_result}));

  const f32_matvec::event::execute_request reference_request{
      .weights = std::span<const float>{weights},
      .input = std::span<const float>{input},
      .output = std::span<float>{reference},
      .inner = inner,
      .rows = rows,
  };
  f32_matvec::event::dispatch_result reference_result{};
  REQUIRE(actor->process_event(f32_matvec::event::execute_reference{
      reference_request, reference_result}));

  const f32_matvec::event::execute_request exact_request{
      .weights = std::span<const float>{packed},
      .input = std::span<const float>{input},
      .output = std::span<float>{exact},
      .inner = inner,
      .rows = rows,
  };
  f32_matvec::event::dispatch_result exact_result{};
#if defined(__aarch64__) || defined(_M_ARM64)
  REQUIRE(actor->process_event(
      f32_matvec::event::execute_exact_x4{exact_request, exact_result}));
  for (std::size_t row = 0u; row < rows; ++row) {
    if (std::isnan(reference[row])) {
      CHECK(std::isnan(exact[row]));
    } else {
      CHECK(std::bit_cast<uint32_t>(exact[row]) ==
            std::bit_cast<uint32_t>(reference[row]));
    }
  }
#else
  CHECK_FALSE(actor->process_event(
      f32_matvec::event::execute_exact_x4{exact_request, exact_result}));
  CHECK_FALSE(exact_result.accepted);
#endif
}

} // namespace

TEST_CASE("f32 matvec x4 preserves independent row arithmetic and tails") {
  constexpr std::size_t inner = 13u;
  constexpr std::size_t rows = 7u;
  std::array<float, inner * rows> weights{};
  std::array<float, inner> input{};
  for (std::size_t index = 0u; index < weights.size(); ++index) {
    weights[index] =
        static_cast<float>(static_cast<int32_t>(index % 29u) - 14) / 32.0f;
  }
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int32_t>(index) - 6) / 16.0f;
  }
  check_exact_case<inner, rows>(weights, input);
}

TEST_CASE("f32 matvec packs f16 rows without changing values") {
  constexpr std::size_t inner = 3u;
  constexpr std::size_t rows = 5u;
  std::array<uint16_t, inner * rows> source{};
  for (std::size_t index = 0u; index < source.size(); ++index) {
    source[index] = emel::kernel::detail::quant::fp32_to_fp16(
        static_cast<float>(static_cast<int32_t>(index) - 7) / 8.0f);
  }
  std::array<float, inner * rows> packed{};
  auto actor = std::make_unique<f32_matvec::sm>();
  const f32_matvec::event::prepare_f16_request request{
      .source = std::span<const uint16_t>{source},
      .destination = std::span<float>{packed},
      .inner = inner,
      .rows = rows,
  };
  f32_matvec::event::dispatch_result result{};
  REQUIRE(
      actor->process_event(f32_matvec::event::prepare_f16{request, result}));
  REQUIRE(result.accepted);
  for (std::size_t column = 0u; column < inner; ++column) {
    for (std::size_t row = 0u; row < 4u; ++row) {
      CHECK(packed[column * 4u + row] ==
            emel::kernel::detail::quant::fp16_to_fp32(
                source[row * inner + column]));
    }
  }
  for (std::size_t column = 0u; column < inner; ++column) {
    CHECK(
        packed[4u * inner + column] ==
        emel::kernel::detail::quant::fp16_to_fp32(source[4u * inner + column]));
  }
}

TEST_CASE("f32 matvec publishes explicit done and error outcomes") {
  std::array<float, 4u> source{1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 4u> destination{};
  auto actor = std::make_unique<f32_matvec::sm>();
  outcome_counts counts{};
  f32_matvec::event::dispatch_result result{};
  const f32_matvec::event::prepare_f32_request request{
      .source = source,
      .destination = destination,
      .inner = 2u,
      .rows = 2u,
  };
  f32_matvec::event::prepare_f32 valid{request, result};
  valid.on_done = {&counts, count_done};
  valid.on_error = {&counts, count_error};

  REQUIRE(actor->process_event(valid));
  CHECK(counts.done == 1);
  CHECK(counts.error == 0);
  CHECK(counts.done_request == &valid);

  const f32_matvec::event::prepare_f32_request bad_request{
      .source = source,
      .destination = std::span<float>{destination}.first(1u),
      .inner = 2u,
      .rows = 2u,
  };
  f32_matvec::event::prepare_f32 invalid{bad_request, result};
  invalid.on_done = {&counts, count_done};
  invalid.on_error = {&counts, count_error};
  CHECK_FALSE(actor->process_event(invalid));
  CHECK(counts.done == 1);
  CHECK(counts.error == 1);
  CHECK(counts.error_request == &invalid);
}

TEST_CASE("f32 matvec x4 preserves NaN classification") {
  constexpr std::size_t inner = 1u;
  constexpr std::size_t rows = 4u;
  const float quiet_nan = std::numeric_limits<float>::quiet_NaN();
  const std::array<float, inner * rows> weights{1.0f, -1.0f, 2.0f, -2.0f};
  const std::array<float, inner> input{quiet_nan};
  check_exact_case<inner, rows>(weights, input);
}

TEST_CASE("f32 matvec x4 preserves subnormal results") {
  constexpr std::size_t inner = 1u;
  constexpr std::size_t rows = 4u;
  const float denorm = std::numeric_limits<float>::denorm_min();
  const std::array<float, inner * rows> weights{denorm, -denorm, 2.0f * denorm,
                                                -2.0f * denorm};
  const std::array<float, inner> input{1.0f};
  check_exact_case<inner, rows>(weights, input);
  CHECK(std::fpclassify(weights[0] * input[0]) == FP_SUBNORMAL);
}

TEST_CASE("f32 matvec x4 preserves non-NaN infinity results") {
  constexpr std::size_t inner = 1u;
  constexpr std::size_t rows = 4u;
  const float infinity = std::numeric_limits<float>::infinity();
  const std::array<float, inner * rows> weights{infinity, -infinity, infinity,
                                                -infinity};
  const std::array<float, inner> input{1.0f};
  check_exact_case<inner, rows>(weights, input);
  CHECK(std::isinf(weights[0] * input[0]));
  CHECK_FALSE(std::isnan(weights[0] * input[0]));
}

TEST_CASE("f32 matvec x4 preserves the reference signed zero result") {
  constexpr std::size_t inner = 1u;
  constexpr std::size_t rows = 4u;
  const std::array<float, inner * rows> weights{-0.0f, 0.0f, -0.0f, 0.0f};
  const std::array<float, inner> input{1.0f};
  check_exact_case<inner, rows>(weights, input);

  double sum = 0.0;
  sum += static_cast<double>(weights[0] * input[0]);
  const float reference = static_cast<float>(sum);
  CHECK(reference == 0.0f);
  CHECK_FALSE(std::signbit(reference));
}

TEST_CASE("f32 matvec prepare rejection never writes caller storage") {
  auto actor = std::make_unique<f32_matvec::sm>();
  std::array<float, 8u> storage{};
  storage.fill(7.0f);
  const f32_matvec::event::prepare_f32_request undersized{
      .source = std::span<const float>{storage},
      .destination = std::span<float>{storage}.first(7u),
      .inner = 2u,
      .rows = 4u,
  };
  f32_matvec::event::dispatch_result result{};
  CHECK_FALSE(
      actor->process_event(f32_matvec::event::prepare_f32{undersized, result}));
  CHECK_FALSE(result.accepted);
  for (const float value : storage) {
    CHECK(value == 7.0f);
  }

  const f32_matvec::event::prepare_f32_request overlapping{
      .source = std::span<const float>{storage},
      .destination = std::span<float>{storage},
      .inner = 2u,
      .rows = 4u,
  };
  CHECK_FALSE(actor->process_event(
      f32_matvec::event::prepare_f32{overlapping, result}));
  for (const float value : storage) {
    CHECK(value == 7.0f);
  }

  const f32_matvec::event::prepare_f32_request overflow{
      .source = std::span<const float>{storage},
      .destination = std::span<float>{storage},
      .inner = std::numeric_limits<uint64_t>::max(),
      .rows = 2u,
  };
  CHECK_FALSE(
      actor->process_event(f32_matvec::event::prepare_f32{overflow, result}));

  alignas(float) std::array<std::byte, sizeof(float) * 8u + 1u> bytes{};
  auto *misaligned = reinterpret_cast<float *>(bytes.data() + 1u);
  const f32_matvec::event::prepare_f32_request bad_alignment{
      .source = std::span<const float>{misaligned, 8u},
      .destination = std::span<float>{storage},
      .inner = 2u,
      .rows = 4u,
  };
  CHECK_FALSE(actor->process_event(
      f32_matvec::event::prepare_f32{bad_alignment, result}));
}

TEST_CASE("f32 matvec rejects aliased execute spans without writes") {
  auto actor = std::make_unique<f32_matvec::sm>();
  std::array<float, 64u> weights{};
  std::array<float, 8u> input_and_output{};
  for (std::size_t index = 0u; index < weights.size(); ++index) {
    weights[index] = static_cast<float>(index + 1u);
  }
  input_and_output.fill(1.0f);
  const auto before = input_and_output;
  const f32_matvec::event::execute_request request{
      .weights = std::span<const float>{weights},
      .input = std::span<const float>{input_and_output},
      .output = std::span<float>{input_and_output},
      .inner = 8u,
      .rows = 8u,
  };
  f32_matvec::event::dispatch_result result{.accepted = true};
  CHECK_FALSE(actor->process_event(
      f32_matvec::event::execute_exact_x4{request, result}));
  CHECK_FALSE(result.accepted);
  CHECK(input_and_output == before);

  std::array<float, 64u> weights_and_output{};
  for (std::size_t index = 0u; index < weights_and_output.size(); ++index) {
    weights_and_output[index] = static_cast<float>(index + 1u);
  }
  const auto weights_before = weights_and_output;
  const f32_matvec::event::execute_request weight_alias_request{
      .weights = std::span<const float>{weights_and_output},
      .input = std::span<const float>{input_and_output},
      .output = std::span<float>{weights_and_output}.first(8u),
      .inner = 8u,
      .rows = 8u,
  };
  result.accepted = true;
  CHECK_FALSE(actor->process_event(
      f32_matvec::event::execute_reference{weight_alias_request, result}));
  CHECK_FALSE(result.accepted);
  CHECK(weights_and_output == weights_before);
}

TEST_CASE("f32 matvec rejects unexpected events and stale results") {
  auto actor = std::make_unique<f32_matvec::sm>();
  f32_matvec::event::dispatch_result result{.accepted = true};
  CHECK_FALSE(actor->process_event(unknown_with_result{result}));
  CHECK_FALSE(result.accepted);
  CHECK_FALSE(actor->process_event(unknown_without_result{}));
}

TEST_CASE("f32 matvec dispatch is allocation-free and reports deltas") {
  auto actor = std::make_unique<f32_matvec::sm>();
  const std::array<float, 8u> weights{1.0f, 2.0f, 3.0f, 4.0f,
                                      5.0f, 6.0f, 7.0f, 8.0f};
  const std::array<float, 2u> input{0.5f, -0.25f};
  std::array<float, 8u> packed{};
  std::array<float, 4u> output{};
  f32_matvec::event::dispatch_result result{};
  const f32_matvec::event::prepare_f32_request prepare_request{
      .source = std::span<const float>{weights},
      .destination = std::span<float>{packed},
      .inner = 2u,
      .rows = 4u,
  };
  const f32_matvec::event::execute_request execute_request{
      .weights = std::span<const float>{packed},
      .input = std::span<const float>{input},
      .output = std::span<float>{output},
      .inner = 2u,
      .rows = 4u,
  };

  std::size_t allocations = 0u;
  {
    emel::test::allocation::allocation_scope allocation_scope{};
    REQUIRE(actor->process_event(
        f32_matvec::event::prepare_f32{prepare_request, result}));
#if defined(__aarch64__) || defined(_M_ARM64)
    for (std::size_t iteration = 0u; iteration < 64u; ++iteration) {
      REQUIRE(actor->process_event(
          f32_matvec::event::execute_exact_x4{execute_request, result}));
    }
#endif
    allocations = allocation_scope.allocations();
  }
  CHECK(allocations == 0u);

  f32_matvec::event::diagnostics diagnostics{};
  REQUIRE(actor->process_event(
      f32_matvec::event::capture_diagnostics{diagnostics, result}));
  CHECK(diagnostics.prepare_calls == 1u);
  CHECK(diagnostics.prepared_floats == 8u);
#if defined(__aarch64__) || defined(_M_ARM64)
  CHECK(diagnostics.exact_x4_calls == 64u);
#else
  CHECK(diagnostics.exact_x4_calls == 0u);
#endif
}
