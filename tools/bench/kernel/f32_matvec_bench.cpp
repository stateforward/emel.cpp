#include <algorithm>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

#include "emel/kernel/f32_matvec/sm.hpp"

namespace {

namespace f32_matvec = emel::kernel::f32_matvec;

struct geometry {
  uint64_t inner = 0u;
  uint64_t rows = 0u;
};

uint32_t next_bits(uint32_t &state) noexcept {
  state = state * 1664525u + 1013904223u;
  return state;
}

float next_value(uint32_t &state) noexcept {
  const int32_t centered = static_cast<int32_t>(next_bits(state) >> 8u) -
                           static_cast<int32_t>(1u << 23u);
  return static_cast<float>(centered) / static_cast<float>(1u << 23u);
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2u];
}

bool run_geometry(const geometry shape, const int samples) {
  const size_t elements = static_cast<size_t>(shape.inner * shape.rows);
  std::vector<float> weights(elements);
  std::vector<float> packed(elements);
  std::vector<float> input(static_cast<size_t>(shape.inner));
  std::vector<float> reference(static_cast<size_t>(shape.rows));
  std::vector<float> exact(static_cast<size_t>(shape.rows));
  uint32_t seed = 0x5a17c9e3u;
  for (float &value : weights) {
    value = next_value(seed);
  }
  for (float &value : input) {
    value = next_value(seed);
  }

  f32_matvec::sm actor{};
  f32_matvec::event::dispatch_result result{};
  const f32_matvec::event::prepare_f32_request prepare_request{
      .source = std::span<const float>{weights},
      .destination = std::span<float>{packed},
      .inner = shape.inner,
      .rows = shape.rows,
  };
  if (!actor.process_event(
          f32_matvec::event::prepare_f32{prepare_request, result})) {
    return false;
  }
  const f32_matvec::event::execute_request reference_request{
      .weights = std::span<const float>{weights},
      .input = std::span<const float>{input},
      .output = std::span<float>{reference},
      .inner = shape.inner,
      .rows = shape.rows,
  };
  const f32_matvec::event::execute_request exact_request{
      .weights = std::span<const float>{packed},
      .input = std::span<const float>{input},
      .output = std::span<float>{exact},
      .inner = shape.inner,
      .rows = shape.rows,
  };

#if defined(__aarch64__) || defined(_M_ARM64)
  (void)actor.process_event(
      f32_matvec::event::execute_reference{reference_request, result});
  (void)actor.process_event(
      f32_matvec::event::execute_exact_x4{exact_request, result});
  for (size_t row = 0u; row < exact.size(); ++row) {
    if (std::bit_cast<uint32_t>(reference[row]) !=
        std::bit_cast<uint32_t>(exact[row])) {
      std::fprintf(stderr,
                   "mismatch inner=%llu rows=%llu row=%zu ref=%08x x4=%08x\n",
                   static_cast<unsigned long long>(shape.inner),
                   static_cast<unsigned long long>(shape.rows), row,
                   std::bit_cast<uint32_t>(reference[row]),
                   std::bit_cast<uint32_t>(exact[row]));
      return false;
    }
  }

  std::vector<double> reference_ms{};
  std::vector<double> exact_ms{};
  reference_ms.reserve(static_cast<size_t>(samples));
  exact_ms.reserve(static_cast<size_t>(samples));
  for (int sample = 0; sample < samples; ++sample) {
    auto begin = std::chrono::steady_clock::now();
    (void)actor.process_event(
        f32_matvec::event::execute_reference{reference_request, result});
    auto end = std::chrono::steady_clock::now();
    reference_ms.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());

    begin = std::chrono::steady_clock::now();
    (void)actor.process_event(
        f32_matvec::event::execute_exact_x4{exact_request, result});
    end = std::chrono::steady_clock::now();
    exact_ms.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double reference_median = median(reference_ms);
  const double exact_median = median(exact_ms);
  std::printf(
      "inner=%llu rows=%llu reference_ms=%.6f exact_x4_ms=%.6f speedup=%.6f "
      "mismatch=0\n",
      static_cast<unsigned long long>(shape.inner),
      static_cast<unsigned long long>(shape.rows), reference_median,
      exact_median, reference_median / exact_median);
  return true;
#else
  (void)samples;
  std::fprintf(stderr, "exact x4 is compile-time unavailable off AArch64\n");
  return false;
#endif
}

} // namespace

int main(int argc, char **argv) {
  const int samples = argc > 1 ? std::max(5, std::atoi(argv[1])) : 7;
  if (std::fesetround(FE_TONEAREST) != 0 || std::fegetround() != FE_TONEAREST) {
    std::fprintf(stderr, "failed to establish FE_TONEAREST\n");
    return 1;
  }
  std::printf("samples=%d warmup_samples=1 rounding=FE_TONEAREST\n", samples);
  constexpr geometry shapes[] = {
      {512u, 1536u},
      {512u, 512u},
      {512u, 2048u},
      {2048u, 512u},
  };
  for (const geometry shape : shapes) {
    if (!run_geometry(shape, samples)) {
      return 1;
    }
  }
  return 0;
}
