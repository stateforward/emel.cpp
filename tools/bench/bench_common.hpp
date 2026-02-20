#pragma once

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace emel::bench {

struct config {
  std::uint64_t iterations = 0;
  std::size_t runs = 0;
  std::uint64_t warmup_iterations = 0;
  std::size_t warmup_runs = 0;
};

struct result {
  std::string name;
  double ns_per_op = 0.0;
  std::uint64_t iterations = 0;
  std::size_t runs = 0;
};

template <class Fn>
result measure_case(const char * name, const config & cfg, Fn && fn) {
  std::vector<double> samples;
  samples.reserve(cfg.runs);

  for (std::size_t run = 0; run < cfg.warmup_runs; ++run) {
    for (std::uint64_t i = 0; i < cfg.warmup_iterations; ++i) {
      fn();
    }
  }

  for (std::size_t run = 0; run < cfg.runs; ++run) {
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < cfg.iterations; ++i) {
      fn();
    }
    const auto end = std::chrono::steady_clock::now();
    const auto duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    samples.push_back(static_cast<double>(duration_ns) / static_cast<double>(cfg.iterations));
  }

  std::sort(samples.begin(), samples.end());
  const double median = samples[samples.size() / 2];

  result out;
  out.name = name;
  out.ns_per_op = median;
  out.iterations = cfg.iterations;
  out.runs = cfg.runs;
  return out;
}

}  // namespace emel::bench
