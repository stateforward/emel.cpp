#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

namespace emel::diarization::sortformer::cache::detail {

inline constexpr int32_t k_speaker_count = 4;
inline constexpr int32_t k_cache_len = 188;
inline constexpr int32_t k_hidden_dim = 192;

struct state {
  std::array<float, static_cast<size_t>(k_cache_len * k_hidden_dim)> speaker = {};
  int32_t frame_count = 0;
};

inline void reset(state & cache) noexcept {
  std::fill(cache.speaker.begin(), cache.speaker.end(), 0.0f);
  cache.frame_count = 0;
}

inline bool write_frame(state & cache,
                        const int32_t frame,
                        std::span<const float, k_hidden_dim> hidden) noexcept {
  if (frame < 0 || frame >= k_cache_len) {
    return false;
  }

  const size_t offset = static_cast<size_t>(frame) * static_cast<size_t>(k_hidden_dim);
  std::copy(hidden.begin(), hidden.end(), cache.speaker.begin() + offset);
  cache.frame_count = std::max(cache.frame_count, frame + 1);
  return true;
}

inline bool read_frame(const state & cache,
                       const int32_t frame,
                       std::span<float, k_hidden_dim> hidden_out) noexcept {
  if (frame < 0 || frame >= cache.frame_count || frame >= k_cache_len) {
    return false;
  }

  const size_t offset = static_cast<size_t>(frame) * static_cast<size_t>(k_hidden_dim);
  std::copy(cache.speaker.begin() + offset,
            cache.speaker.begin() + offset + static_cast<size_t>(k_hidden_dim),
            hidden_out.begin());
  return true;
}

}  // namespace emel::diarization::sortformer::cache::detail
