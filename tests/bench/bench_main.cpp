#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench/bench_registry.hpp"

namespace emel::bench {
constexpr std::uint64_t k_default_iterations = 100000;
constexpr std::size_t k_default_runs = 5;
constexpr std::uint64_t k_default_warmup_iterations = 1000;
constexpr std::size_t k_default_warmup_runs = 1;
constexpr std::size_t k_max_runs = 25;

std::uint64_t read_env_u64(const char * name, std::uint64_t fallback) {
  const char * value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char * end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value) {
    return fallback;
  }
  return static_cast<std::uint64_t>(parsed);
}

std::size_t read_env_size(const char * name, std::size_t fallback) {
  const auto parsed = read_env_u64(name, static_cast<std::uint64_t>(fallback));
  if (parsed == 0) {
    return fallback;
  }
  if (parsed > static_cast<std::uint64_t>(k_max_runs)) {
    return k_max_runs;
  }
  return static_cast<std::size_t>(parsed);
}

struct result {
  std::string name;
  double ns_per_op = 0.0;
  std::uint64_t iterations = 0;
  std::size_t runs = 0;
};

result measure_case(const case_entry & entry,
                    std::uint64_t iterations,
                    std::size_t runs,
                    std::uint64_t warmup_iterations,
                    std::size_t warmup_runs) {
  std::vector<double> samples;
  samples.reserve(runs);

  for (std::size_t run = 0; run < warmup_runs; ++run) {
    entry.run(warmup_iterations);
  }

  for (std::size_t run = 0; run < runs; ++run) {
    const auto start = std::chrono::steady_clock::now();
    entry.run(iterations);
    const auto end = std::chrono::steady_clock::now();
    const auto duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    samples.push_back(static_cast<double>(duration_ns) / static_cast<double>(iterations));
  }

  std::sort(samples.begin(), samples.end());
  const double median = samples[samples.size() / 2];

  result out;
  out.name = entry.name;
  out.ns_per_op = median;
  out.iterations = iterations;
  out.runs = runs;
  return out;
}

} // namespace emel::bench

int main() {
  const auto iterations = emel::bench::read_env_u64("EMEL_BENCH_ITERS",
                                                    emel::bench::k_default_iterations);
  const auto runs = emel::bench::read_env_size("EMEL_BENCH_RUNS",
                                               emel::bench::k_default_runs);
  const auto warmup_iterations = emel::bench::read_env_u64(
    "EMEL_BENCH_WARMUP_ITERS",
    std::min(iterations, emel::bench::k_default_warmup_iterations));
  const auto warmup_runs = emel::bench::read_env_size("EMEL_BENCH_WARMUP_RUNS",
                                                      emel::bench::k_default_warmup_runs);

  const auto * registry = emel::bench::cases();
  const auto count = emel::bench::case_count();

  std::vector<emel::bench::result> results;
  results.reserve(count);

  for (std::size_t i = 0; i < count; ++i) {
    results.push_back(emel::bench::measure_case(
      registry[i], iterations, runs, warmup_iterations, warmup_runs));
  }

  std::sort(results.begin(), results.end(),
            [](const emel::bench::result & a, const emel::bench::result & b) {
              return a.name < b.name;
            });

  for (const auto & entry : results) {
    std::printf("%s ns_per_op=%.3f iter=%" PRIu64 " runs=%zu\n",
                entry.name.c_str(),
                entry.ns_per_op,
                entry.iterations,
                entry.runs);
  }

  return 0;
}
