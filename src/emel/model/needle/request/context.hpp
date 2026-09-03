#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "emel/model/data.hpp"
#include "emel/model/needle/events.hpp"
#include "emel/model/needle/graph/sm.hpp"
#include "emel/text/detokenizer/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

namespace emel::model::needle::request::action {

inline constexpr size_t k_max_system_bytes = 64u * 1024u;
inline constexpr size_t k_max_tools_bytes = 256u * 1024u;
inline constexpr size_t k_max_query_bytes = 1024u * 1024u;
inline constexpr size_t k_max_prompt_bytes =
    k_max_system_bytes + k_max_tools_bytes + k_max_query_bytes + 128u;
inline constexpr size_t k_max_request_tokens = 4096u;
inline constexpr size_t k_max_response_bytes = 64u * 1024u;
inline constexpr size_t k_detokenize_piece_bytes = 4096u;

using timestamp_now_fn = uint64_t (*)() noexcept;

inline uint64_t steady_timestamp_now() noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

struct dependencies {
  const needle::contract &bound;
  timestamp_now_fn timestamp_now = &steady_timestamp_now;
};

struct context {
  explicit context(const dependencies &deps)
      : bound(deps.bound), timestamp_now(deps.timestamp_now),
        vocab(std::make_unique<emel::model::data::vocab>()),
        tokenizer(std::make_unique<emel::text::tokenizer::sm>()),
        detokenizer(std::make_unique<emel::text::detokenizer::sm>()),
        graph(std::make_unique<needle::graph::serial_sm>(deps.bound)),
        prompt_ids(std::min<size_t>(deps.bound.geo.max_seq_len,
                                    k_max_request_tokens)),
        logits(deps.bound.geo.vocab_size) {
    system_storage.resize(k_max_system_bytes);
    tools_storage.resize(k_max_tools_bytes);
    prompt_storage.resize(k_max_prompt_bytes);
    generated_ids.resize(std::min<size_t>(deps.bound.geo.max_seq_len,
                                          k_max_request_tokens));
    generated_text.resize(k_max_response_bytes);
    normalized_envelope.resize(k_max_response_bytes);
  }

  context(const context &) = delete;
  context &operator=(const context &) = delete;

  const needle::contract &bound;
  timestamp_now_fn timestamp_now = nullptr;
  std::unique_ptr<emel::model::data::vocab> vocab;
  std::unique_ptr<emel::text::tokenizer::sm> tokenizer;
  std::unique_ptr<emel::text::detokenizer::sm> detokenizer;
  std::unique_ptr<needle::graph::serial_sm> graph;

  std::vector<char> system_storage;
  std::vector<char> tools_storage;
  std::vector<char> prompt_storage;
  std::vector<int32_t> prompt_ids;
  std::vector<int32_t> generated_ids;
  std::vector<float> logits;
  std::vector<char> generated_text;
  std::vector<char> normalized_envelope;
  std::array<char, k_detokenize_piece_bytes> detokenize_piece = {};
  std::array<uint8_t, 4> detokenize_pending = {};

  size_t system_size = 0u;
  size_t tools_size = 0u;
  size_t prompt_size = 0u;
  size_t prompt_id_count = 0u;
  size_t generated_id_count = 0u;
  size_t generated_text_size = 0u;
  size_t normalized_envelope_size = 0u;
  uint64_t prefill_nanoseconds = 0u;
  uint64_t decode_nanoseconds = 0u;

  bool assets_ready = false;
  bool configured = false;
  bool reset_ready = false;
};

} // namespace emel::model::needle::request::action
