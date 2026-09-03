#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace emel::bench::needle_request {

struct run_sample {
  double wall_ns = 0.0;
  double prefill_ns = 0.0;
  double decode_ns = 0.0;
  std::uint64_t prompt_tokens = 0u;
  std::uint64_t decode_tokens = 0u;
  std::vector<std::string> envelopes;
};

inline double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2u;
  if (values.size() % 2u != 0u) return values[middle];
  return (values[middle - 1u] + values[middle]) / 2.0;
}

inline bool aggregate_runs(const std::span<const run_sample> samples,
                           run_sample &out) {
  if (samples.empty()) return false;
  std::vector<double> wall;
  std::vector<double> prefill;
  std::vector<double> decode;
  wall.reserve(samples.size());
  prefill.reserve(samples.size());
  decode.reserve(samples.size());
  const run_sample &expected = samples.front();
  for (const run_sample &sample : samples) {
    if (sample.prompt_tokens != expected.prompt_tokens ||
        sample.decode_tokens != expected.decode_tokens ||
        sample.envelopes != expected.envelopes)
      return false;
    wall.push_back(sample.wall_ns);
    prefill.push_back(sample.prefill_ns);
    decode.push_back(sample.decode_ns);
  }
  out = expected;
  out.wall_ns = median(std::move(wall));
  out.prefill_ns = median(std::move(prefill));
  out.decode_ns = median(std::move(decode));
  return true;
}

} // namespace emel::bench::needle_request
