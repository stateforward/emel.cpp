#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "emel/text/encoders/context.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/text/encoders/types.hpp"
#include "emel/model/data.hpp"
#include "emel/text/unicode.hpp"

namespace emel::text::encoders::detail {

template <class value_type>
inline void write_optional(value_type * destination, value_type & sink,
                           const value_type value) noexcept {
  value_type * destinations[2] = {&sink, destination};
  *destinations[static_cast<size_t>(destination != nullptr)] = value;
}

inline void dispatch_done_noop(const event::encode &, const int32_t) noexcept {
}

inline void dispatch_done_call(const event::encode & request,
                               const int32_t token_count) noexcept {
  request.dispatch_done(request.owner_sm, events::encoding_done{request, token_count});
}

inline void dispatch_done_if_bound(const event::encode & request,
                                   const int32_t token_count) noexcept {
  using dispatch_fn = void (*)(const event::encode &, int32_t);
  const std::array<dispatch_fn, 2> dispatchers{&dispatch_done_noop, &dispatch_done_call};
  const bool can_dispatch = request.dispatch_done != nullptr && request.owner_sm != nullptr;
  dispatchers[static_cast<size_t>(can_dispatch)](request, token_count);
}

inline void dispatch_error_noop(const event::encode &, const int32_t) noexcept {
}

inline void dispatch_error_call(const event::encode & request, const int32_t err) noexcept {
  request.dispatch_error(request.owner_sm, events::encoding_error{request, err});
}

inline void dispatch_error_if_bound(const event::encode & request, const int32_t err) noexcept {
  using dispatch_fn = void (*)(const event::encode &, int32_t);
  const std::array<dispatch_fn, 2> dispatchers{&dispatch_error_noop, &dispatch_error_call};
  const bool can_dispatch = request.dispatch_error != nullptr && request.owner_sm != nullptr;
  dispatchers[static_cast<size_t>(can_dispatch)](request, err);
}

inline void publish_error(const event::encode & request, const event::encode_ctx & ctx) noexcept {
  dispatch_error_if_bound(request, ctx.err);
}

inline void publish_done(const event::encode & request, const event::encode_ctx & ctx) noexcept {
  dispatch_done_if_bound(request, ctx.token_count);
}

inline void publish_result(const event::encode & request,
                           const event::encode_ctx & ctx) noexcept {
  using publish_fn = void (*)(const event::encode &, const event::encode_ctx &);
  const std::array<publish_fn, 2> publishers{&publish_error, &publish_done};
  publishers[static_cast<size_t>(ctx.err == EMEL_OK)](request, ctx);
}

inline int32_t select_final_error(const bool accepted,
                                  const int32_t runtime_error) noexcept {
  const std::array<int32_t, 2> accepted_errors{EMEL_ERR_INVALID_ARGUMENT, runtime_error};
  const std::array<int32_t, 2> final_errors{
      accepted_errors[static_cast<size_t>(accepted)],
      EMEL_OK,
  };
  const bool succeeded = accepted && runtime_error == EMEL_OK;
  return final_errors[static_cast<size_t>(succeeded)];
}

template <size_t N>
inline std::string_view string_view_from_array(const std::array<char, N> &data) {
  size_t len = 0;
  while (len < N && data[len] != '\0') {
    ++len;
  }
  return std::string_view(data.data(), len);
}

inline size_t utf8_len(const char byte) {
  return emel::text::unicode_len_utf8(byte);
}

inline bool is_chinese_char(const uint32_t cpt) {
  return emel::text::unicode_cpt_is_han(cpt);
}

inline std::string cpt_to_utf8(const uint32_t cpt) {
  return emel::text::unicode_cpt_to_utf8(cpt);
}

inline std::string_view token_text(const emel::model::data::vocab &vocab,
                                   const int32_t id) {
    {
    const size_t emel_branch_1 = static_cast<size_t>(id < 0 || static_cast<uint32_t>(id) >= vocab.n_tokens);
    for (size_t emel_case_1 = emel_branch_1; emel_case_1 == 1u; emel_case_1 = 2u) {
            return {};
    }
    for (size_t emel_case_1 = emel_branch_1; emel_case_1 == 0u; emel_case_1 = 2u) {

    }
  }
  const auto &entry = vocab.entries[static_cast<uint32_t>(id)];
    {
    const size_t emel_branch_2 = static_cast<size_t>(entry.text_length == 0);
    for (size_t emel_case_2 = emel_branch_2; emel_case_2 == 1u; emel_case_2 = 2u) {
            return {};
    }
    for (size_t emel_case_2 = emel_branch_2; emel_case_2 == 0u; emel_case_2 = 2u) {

    }
  }
  return std::string_view(vocab.token_storage.data() + entry.text_offset,
                          entry.text_length);
}

inline bool is_token_type(const emel::model::data::vocab &vocab,
                          const int32_t id,
                          const int32_t type) {
    {
    const size_t emel_branch_3 = static_cast<size_t>(id < 0 || static_cast<uint32_t>(id) >= vocab.n_tokens);
    for (size_t emel_case_3 = emel_branch_3; emel_case_3 == 1u; emel_case_3 = 2u) {
            return false;
    }
    for (size_t emel_case_3 = emel_branch_3; emel_case_3 == 0u; emel_case_3 = 2u) {

    }
  }
  return vocab.entries[static_cast<uint32_t>(id)].type == type;
}

constexpr uint32_t k_fnv_offset = 2166136261u;
constexpr uint32_t k_fnv_prime = 16777619u;

inline uint32_t hash_bytes(const uint32_t seed, const std::string_view data) {
  uint32_t hash = seed;
  for (const unsigned char byte : data) {
    hash ^= byte;
    hash *= k_fnv_prime;
  }
  const std::array<uint32_t, 2> hash_candidates = {hash, 1u};
  return hash_candidates[static_cast<size_t>(hash == 0)];
}

inline uint32_t hash_sv(const std::string_view data) {
  return hash_bytes(k_fnv_offset, data);
}

inline uint32_t hash_concat(const std::string_view left,
                            const std::string_view right) {
  return hash_bytes(hash_bytes(k_fnv_offset, left), right);
}

inline uint32_t hash_pair(const std::string_view left,
                          const std::string_view right) {
  const uint32_t h1 = hash_sv(left);
  const uint32_t h2 = hash_sv(right);
  const uint32_t combined = h1 ^ (h2 + 0x9e3779b9u + (h1 << 6u) + (h1 >> 2u));
  const std::array<uint32_t, 2> combined_candidates = {combined, 1u};
  return combined_candidates[static_cast<size_t>(combined == 0)];
}

inline std::string_view merge_text(const emel::model::data::vocab &vocab,
                                   const int32_t idx) {
    {
    const size_t emel_branch_4 = static_cast<size_t>(idx < 0 || static_cast<uint32_t>(idx) >= vocab.n_merges);
    for (size_t emel_case_4 = emel_branch_4; emel_case_4 == 1u; emel_case_4 = 2u) {
            return {};
    }
    for (size_t emel_case_4 = emel_branch_4; emel_case_4 == 0u; emel_case_4 = 2u) {

    }
  }
  const uint32_t offset = vocab.merge_offsets[static_cast<uint32_t>(idx)];
  const uint32_t length = vocab.merge_lengths[static_cast<uint32_t>(idx)];
    {
    const size_t emel_branch_5 = static_cast<size_t>(offset + length > vocab.merge_storage.size());
    for (size_t emel_case_5 = emel_branch_5; emel_case_5 == 1u; emel_case_5 = 2u) {
            return {};
    }
    for (size_t emel_case_5 = emel_branch_5; emel_case_5 == 0u; emel_case_5 = 2u) {

    }
  }
  return std::string_view(vocab.merge_storage.data() + offset, length);
}

inline bool merge_match(const std::string_view merge,
                        const std::string_view left,
                        const std::string_view right) {
    {
    const size_t emel_branch_6 = static_cast<size_t>(merge.empty());
    for (size_t emel_case_6 = emel_branch_6; emel_case_6 == 1u; emel_case_6 = 2u) {
            return false;
    }
    for (size_t emel_case_6 = emel_branch_6; emel_case_6 == 0u; emel_case_6 = 2u) {

    }
  }
  const size_t pos = merge.find(' ');
    {
    const size_t emel_branch_7 = static_cast<size_t>(pos == std::string_view::npos);
    for (size_t emel_case_7 = emel_branch_7; emel_case_7 == 1u; emel_case_7 = 2u) {
            return false;
    }
    for (size_t emel_case_7 = emel_branch_7; emel_case_7 == 0u; emel_case_7 = 2u) {

    }
  }
    {
    const size_t emel_branch_8 = static_cast<size_t>(merge.size() != left.size() + right.size() + 1);
    for (size_t emel_case_8 = emel_branch_8; emel_case_8 == 1u; emel_case_8 = 2u) {
            return false;
    }
    for (size_t emel_case_8 = emel_branch_8; emel_case_8 == 0u; emel_case_8 = 2u) {

    }
  }
    {
    const size_t emel_branch_9 = static_cast<size_t>(merge.substr(0, pos) != left);
    for (size_t emel_case_9 = emel_branch_9; emel_case_9 == 1u; emel_case_9 = 2u) {
            return false;
    }
    for (size_t emel_case_9 = emel_branch_9; emel_case_9 == 0u; emel_case_9 = 2u) {

    }
  }
  return merge.substr(pos + 1) == right;
}

inline bool insert_token_map(token_map &map,
                             const emel::model::data::vocab &vocab,
                             const std::string_view text,
                             const int32_t id) {
    {
    const size_t emel_branch_10 = static_cast<size_t>(text.empty());
    for (size_t emel_case_10 = emel_branch_10; emel_case_10 == 1u; emel_case_10 = 2u) {
            return true;
    }
    for (size_t emel_case_10 = emel_branch_10; emel_case_10 == 0u; emel_case_10 = 2u) {

    }
  }
  const uint32_t hash = hash_sv(text);
  const uint32_t mask = k_token_hash_size - 1;
  uint32_t slot = hash & mask;
  for (uint32_t probes = 0; probes < k_token_hash_size; ++probes) {
    const uint32_t slot_hash = map.hashes[slot];
        {
      const size_t emel_branch_11 = static_cast<size_t>(slot_hash == 0);
      for (size_t emel_case_11 = emel_branch_11; emel_case_11 == 1u; emel_case_11 = 2u) {
                map.hashes[slot] = hash;
                map.values[slot] = id;
                map.count += 1;
                return true;
      }
      for (size_t emel_case_11 = emel_branch_11; emel_case_11 == 0u; emel_case_11 = 2u) {

      }
    }
        {
      const size_t emel_branch_12 = static_cast<size_t>(slot_hash == hash);
      for (size_t emel_case_12 = emel_branch_12; emel_case_12 == 1u; emel_case_12 = 2u) {
         {
                const int32_t existing = map.values[slot];
                const std::string_view existing_text = token_text(vocab, existing);
                {
                  const size_t emel_branch_existing_match =
                      static_cast<size_t>(existing_text == text);
                  for (size_t emel_case_existing_match = emel_branch_existing_match;
                       emel_case_existing_match == 1u;
                       emel_case_existing_match = 2u) {
                    map.values[slot] = id;
                    return true;
                  }
                  for (size_t emel_case_existing_match = emel_branch_existing_match;
                       emel_case_existing_match == 0u;
                       emel_case_existing_match = 2u) {

                  }
                }
                break;
              }
      }
      for (size_t emel_case_12 = emel_branch_12; emel_case_12 == 0u; emel_case_12 = 2u) {

      }
    }
    slot = (slot + 1) & mask;
  }
  return false;
}

inline bool insert_merge_map(merge_map &map,
                             const std::string_view left,
                             const std::string_view right,
                             const int32_t rank,
                             const emel::model::data::vocab &vocab) {
    {
    const size_t emel_branch_13 = static_cast<size_t>(left.empty() || right.empty());
    for (size_t emel_case_13 = emel_branch_13; emel_case_13 == 1u; emel_case_13 = 2u) {
            return false;
    }
    for (size_t emel_case_13 = emel_branch_13; emel_case_13 == 0u; emel_case_13 = 2u) {

    }
  }
  const uint32_t hash = hash_pair(left, right);
  const uint32_t mask = k_merge_hash_size - 1;
  uint32_t slot = hash & mask;
  for (uint32_t probes = 0; probes < k_merge_hash_size; ++probes) {
    const uint32_t slot_hash = map.hashes[slot];
        {
      const size_t emel_branch_14 = static_cast<size_t>(slot_hash == 0);
      for (size_t emel_case_14 = emel_branch_14; emel_case_14 == 1u; emel_case_14 = 2u) {
                map.hashes[slot] = hash;
                map.values[slot] = rank;
                map.count += 1;
                return true;
      }
      for (size_t emel_case_14 = emel_branch_14; emel_case_14 == 0u; emel_case_14 = 2u) {

      }
    }
        {
      const size_t emel_branch_15 = static_cast<size_t>(slot_hash == hash);
      for (size_t emel_case_15 = emel_branch_15; emel_case_15 == 1u; emel_case_15 = 2u) {
         {
                const int32_t existing = map.values[slot];
                const std::string_view merge = merge_text(vocab, existing);
                {
                  const size_t emel_branch_merge_match =
                      static_cast<size_t>(merge_match(merge, left, right));
                  for (size_t emel_case_merge_match = emel_branch_merge_match;
                       emel_case_merge_match == 1u;
                       emel_case_merge_match = 2u) {
                    return true;
                  }
                  for (size_t emel_case_merge_match = emel_branch_merge_match;
                       emel_case_merge_match == 0u;
                       emel_case_merge_match = 2u) {

                  }
                }
                break;
              }
      }
      for (size_t emel_case_15 = emel_branch_15; emel_case_15 == 0u; emel_case_15 = 2u) {

      }
    }
    slot = (slot + 1) & mask;
  }
  return false;
}

inline int32_t lookup_token(const action::context &ctx,
                            const std::string_view text) {
    {
    const size_t emel_branch_16 = static_cast<size_t>(text.empty());
    for (size_t emel_case_16 = emel_branch_16; emel_case_16 == 1u; emel_case_16 = 2u) {
            return k_token_null;
    }
    for (size_t emel_case_16 = emel_branch_16; emel_case_16 == 0u; emel_case_16 = 2u) {

    }
  }
  const uint32_t hash = hash_sv(text);
  const uint32_t mask = k_token_hash_size - 1;
  uint32_t slot = hash & mask;
  for (uint32_t probes = 0; probes < k_token_hash_size; ++probes) {
    const uint32_t entry = ctx.token_to_id.hashes[slot];
        {
      const size_t emel_branch_17 = static_cast<size_t>(entry == 0);
      for (size_t emel_case_17 = emel_branch_17; emel_case_17 == 1u; emel_case_17 = 2u) {
                return k_token_null;
      }
      for (size_t emel_case_17 = emel_branch_17; emel_case_17 == 0u; emel_case_17 = 2u) {

      }
    }
        {
      const size_t emel_branch_18 = static_cast<size_t>(entry == hash);
      for (size_t emel_case_18 = emel_branch_18; emel_case_18 == 1u; emel_case_18 = 2u) {
         {
                const int32_t id = ctx.token_to_id.values[slot];
                {
                  const size_t emel_branch_token_match =
                      static_cast<size_t>(token_text(*ctx.vocab, id) == text);
                  for (size_t emel_case_token_match = emel_branch_token_match;
                       emel_case_token_match == 1u;
                       emel_case_token_match = 2u) {
                    return id;
                  }
                  for (size_t emel_case_token_match = emel_branch_token_match;
                       emel_case_token_match == 0u;
                       emel_case_token_match = 2u) {

                  }
                }
                break;
              }
      }
      for (size_t emel_case_18 = emel_branch_18; emel_case_18 == 0u; emel_case_18 = 2u) {

      }
    }
    slot = (slot + 1) & mask;
  }
  return k_token_null;
}

inline int32_t lookup_token_concat(const action::context &ctx,
                                   const std::string_view left,
                                   const std::string_view right) {
  const uint32_t hash = hash_concat(left, right);
  const uint32_t mask = k_token_hash_size - 1;
  const size_t combined_len = left.size() + right.size();
  uint32_t slot = hash & mask;
  for (uint32_t probes = 0; probes < k_token_hash_size; ++probes) {
    const uint32_t entry = ctx.token_to_id.hashes[slot];
        {
      const size_t emel_branch_19 = static_cast<size_t>(entry == 0);
      for (size_t emel_case_19 = emel_branch_19; emel_case_19 == 1u; emel_case_19 = 2u) {
                return k_token_null;
      }
      for (size_t emel_case_19 = emel_branch_19; emel_case_19 == 0u; emel_case_19 = 2u) {

      }
    }
    {
      const size_t emel_branch_entry_match = static_cast<size_t>(entry == hash);
      for (size_t emel_case_entry_match = emel_branch_entry_match;
           emel_case_entry_match == 1u;
           emel_case_entry_match = 2u) {
        const int32_t id = ctx.token_to_id.values[slot];
        const std::string_view token = token_text(*ctx.vocab, id);
        const bool size_mismatch = token.size() != combined_len;
        const bool left_mismatch =
            !left.empty() && std::memcmp(token.data(), left.data(), left.size()) != 0;
        const bool right_mismatch =
            !right.empty() &&
            std::memcmp(token.data() + left.size(), right.data(), right.size()) != 0;
        const size_t emel_branch_token_match =
            static_cast<size_t>(!(size_mismatch || left_mismatch || right_mismatch));
        for (size_t emel_case_token_match = emel_branch_token_match;
             emel_case_token_match == 1u;
             emel_case_token_match = 2u) {
          return id;
        }
        for (size_t emel_case_token_match = emel_branch_token_match;
             emel_case_token_match == 0u;
             emel_case_token_match = 2u) {

        }
      }
      for (size_t emel_case_entry_match = emel_branch_entry_match;
           emel_case_entry_match == 0u;
           emel_case_entry_match = 2u) {

      }
    }
    slot = (slot + 1) & mask;
  }
  return k_token_null;
}

inline int32_t lookup_merge_rank(const action::context &ctx,
                                 const emel::model::data::vocab &vocab,
                                 const std::string_view left,
                                 const std::string_view right) {
    {
    const size_t emel_branch_20 = static_cast<size_t>(left.empty() || right.empty());
    for (size_t emel_case_20 = emel_branch_20; emel_case_20 == 1u; emel_case_20 = 2u) {
            return k_token_null;
    }
    for (size_t emel_case_20 = emel_branch_20; emel_case_20 == 0u; emel_case_20 = 2u) {

    }
  }
  const uint32_t hash = hash_pair(left, right);
  const uint32_t mask = k_merge_hash_size - 1;
  uint32_t slot = hash & mask;
  for (uint32_t probes = 0; probes < k_merge_hash_size; ++probes) {
    const uint32_t entry = ctx.bpe_ranks.hashes[slot];
        {
      const size_t emel_branch_21 = static_cast<size_t>(entry == 0);
      for (size_t emel_case_21 = emel_branch_21; emel_case_21 == 1u; emel_case_21 = 2u) {
                return k_token_null;
      }
      for (size_t emel_case_21 = emel_branch_21; emel_case_21 == 0u; emel_case_21 = 2u) {

      }
    }
        {
      const size_t emel_branch_22 = static_cast<size_t>(entry == hash);
      for (size_t emel_case_22 = emel_branch_22; emel_case_22 == 1u; emel_case_22 = 2u) {
         {
                const int32_t rank = ctx.bpe_ranks.values[slot];
                const std::string_view merge = merge_text(vocab, rank);
                {
                  const size_t emel_branch_merge_match =
                      static_cast<size_t>(merge_match(merge, left, right));
                  for (size_t emel_case_merge_match = emel_branch_merge_match;
                       emel_case_merge_match == 1u;
                       emel_case_merge_match = 2u) {
                    return rank;
                  }
                  for (size_t emel_case_merge_match = emel_branch_merge_match;
                       emel_case_merge_match == 0u;
                       emel_case_merge_match = 2u) {

                  }
                }
                break;
              }
      }
      for (size_t emel_case_22 = emel_branch_22; emel_case_22 == 0u; emel_case_22 = 2u) {

      }
    }
    slot = (slot + 1) & mask;
  }
  return k_token_null;
}

inline bool push_token(const event::encode &ev, const int32_t token, int32_t &count) {
    {
    const size_t emel_branch_23 = static_cast<size_t>(token < 0 || ev.token_ids.empty());
    for (size_t emel_case_23 = emel_branch_23; emel_case_23 == 1u; emel_case_23 = 2u) {
            return false;
    }
    for (size_t emel_case_23 = emel_branch_23; emel_case_23 == 0u; emel_case_23 = 2u) {

    }
  }
    {
    const size_t emel_branch_24 = static_cast<size_t>(static_cast<size_t>(count) >= ev.token_ids.size());
    for (size_t emel_case_24 = emel_branch_24; emel_case_24 == 1u; emel_case_24 = 2u) {
            return false;
    }
    for (size_t emel_case_24 = emel_branch_24; emel_case_24 == 0u; emel_case_24 = 2u) {

    }
  }
  ev.token_ids[static_cast<size_t>(count++)] = token;
  return true;
}

inline const std::array<uint32_t, 256> &byte_to_codepoint_table() {
  static const std::array<uint32_t, 256> table = [] {
    std::array<uint32_t, 256> map = {};
    std::array<bool, 256> used = {};
    for (size_t idx = 0; idx < 256; ++idx) {
      used[idx] = false;
      map[idx] = 0;
    }
    for (uint32_t c = 33; c <= 126; ++c) {
      const uint8_t idx = static_cast<uint8_t>(c);
      used[idx] = true;
      map[idx] = c;
    }
    for (uint32_t c = 161; c <= 172; ++c) {
      const uint8_t idx = static_cast<uint8_t>(c);
      used[idx] = true;
      map[idx] = c;
    }
    for (uint32_t c = 174; c <= 255; ++c) {
      const uint8_t idx = static_cast<uint8_t>(c);
      used[idx] = true;
      map[idx] = c;
    }
    uint32_t n = 0;
    for (int ch = 0; ch < 256; ++ch) {
            {
        const size_t emel_branch_25 = static_cast<size_t>(!used[static_cast<size_t>(ch)]);
        for (size_t emel_case_25 = emel_branch_25; emel_case_25 == 1u; emel_case_25 = 2u) {
                    map[static_cast<size_t>(ch)] = 256u + n;
                    n += 1;
        }
        for (size_t emel_case_25 = emel_branch_25; emel_case_25 == 0u; emel_case_25 = 2u) {

        }
      }
    }
    return map;
  }();
  return table;
}

inline uint8_t encode_cpt_utf8(const uint32_t cpt, char out[4]) {
    {
    const size_t emel_branch_26 = static_cast<size_t>(cpt <= 0x7F);
    for (size_t emel_case_26 = emel_branch_26; emel_case_26 == 1u; emel_case_26 = 2u) {
            out[0] = static_cast<char>(cpt);
            return 1;
    }
    for (size_t emel_case_26 = emel_branch_26; emel_case_26 == 0u; emel_case_26 = 2u) {

    }
  }
    {
    const size_t emel_branch_27 = static_cast<size_t>(cpt <= 0x7FF);
    for (size_t emel_case_27 = emel_branch_27; emel_case_27 == 1u; emel_case_27 = 2u) {
            out[0] = static_cast<char>(0xC0 | ((cpt >> 6) & 0x1F));
            out[1] = static_cast<char>(0x80 | (cpt & 0x3F));
            return 2;
    }
    for (size_t emel_case_27 = emel_branch_27; emel_case_27 == 0u; emel_case_27 = 2u) {

    }
  }
    {
    const size_t emel_branch_28 = static_cast<size_t>(cpt <= 0xFFFF);
    for (size_t emel_case_28 = emel_branch_28; emel_case_28 == 1u; emel_case_28 = 2u) {
            out[0] = static_cast<char>(0xE0 | ((cpt >> 12) & 0x0F));
            out[1] = static_cast<char>(0x80 | ((cpt >> 6) & 0x3F));
            out[2] = static_cast<char>(0x80 | (cpt & 0x3F));
            return 3;
    }
    for (size_t emel_case_28 = emel_branch_28; emel_case_28 == 0u; emel_case_28 = 2u) {

    }
  }
  out[0] = static_cast<char>(0xF0 | ((cpt >> 18) & 0x07));
  out[1] = static_cast<char>(0x80 | ((cpt >> 12) & 0x3F));
  out[2] = static_cast<char>(0x80 | ((cpt >> 6) & 0x3F));
  out[3] = static_cast<char>(0x80 | (cpt & 0x3F));
  return 4;
}

inline const std::array<std::string, 256> &byte_to_utf8_table() {
  static const std::array<std::string, 256> table = [] {
    std::array<std::string, 256> map = {};
    const auto &codepoints = byte_to_codepoint_table();
    for (size_t idx = 0; idx < map.size(); ++idx) {
      map[idx] = cpt_to_utf8(codepoints[idx]);
    }
    return map;
  }();
  return table;
}

inline int32_t byte_to_token(const action::context &ctx,
                             const emel::model::data::vocab &vocab,
                             const uint8_t byte,
                             const emel::model::data::tokenizer_model model) {
  (void)vocab;
  const bool none_model = model == emel::model::data::tokenizer_model::NONE;
    {
    const size_t emel_branch_29 = static_cast<size_t>(none_model);
    for (size_t emel_case_29 = emel_branch_29; emel_case_29 == 1u; emel_case_29 = 2u) {
            return k_token_null;
    }
    for (size_t emel_case_29 = emel_branch_29; emel_case_29 == 0u; emel_case_29 = 2u) {

    }
  }

  const bool piece_model = model == emel::model::data::tokenizer_model::SPM ||
      model == emel::model::data::tokenizer_model::UGM ||
      model == emel::model::data::tokenizer_model::PLAMO2;
    {
    const size_t emel_branch_30 = static_cast<size_t>(piece_model);
    for (size_t emel_case_30 = emel_branch_30; emel_case_30 == 1u; emel_case_30 = 2u) {
       {
          char hex[7] = {};
          static const char *digits = "0123456789ABCDEF";
          hex[0] = '<';
          hex[1] = '0';
          hex[2] = 'x';
          hex[3] = digits[(byte >> 4) & 0x0F];
          hex[4] = digits[byte & 0x0F];
          hex[5] = '>';
          hex[6] = '\0';
          const int32_t hex_token = lookup_token(ctx, std::string_view(hex, 6));
          {
            const size_t emel_branch_has_hex = static_cast<size_t>(hex_token != k_token_null);
            for (size_t emel_case_has_hex = emel_branch_has_hex; emel_case_has_hex == 1u;
                 emel_case_has_hex = 2u) {
              return hex_token;
            }
            for (size_t emel_case_has_hex = emel_branch_has_hex; emel_case_has_hex == 0u;
                 emel_case_has_hex = 2u) {

            }
          }
            const char raw = static_cast<char>(byte);
            return lookup_token(ctx, std::string_view(&raw, 1));
          }
    }
    for (size_t emel_case_30 = emel_branch_30; emel_case_30 == 0u; emel_case_30 = 2u) {

    }
  }

  const bool bpe_model = model == emel::model::data::tokenizer_model::BPE ||
      model == emel::model::data::tokenizer_model::WPM ||
      model == emel::model::data::tokenizer_model::RWKV;
    {
    const size_t emel_branch_31 = static_cast<size_t>(bpe_model);
    for (size_t emel_case_31 = emel_branch_31; emel_case_31 == 1u; emel_case_31 = 2u) {
          const uint32_t cpt = byte_to_codepoint_table()[byte];
          char utf8[4] = {};
          const uint8_t len = encode_cpt_utf8(cpt, utf8);
          return lookup_token(ctx, std::string_view(utf8, len));
    }
    for (size_t emel_case_31 = emel_branch_31; emel_case_31 == 0u; emel_case_31 = 2u) {

    }
  }

  const char raw = static_cast<char>(byte);
  return lookup_token(ctx, std::string_view(&raw, 1));
}

inline bool ensure_tables(action::context &ctx) {
    {
    const size_t emel_branch_32 = static_cast<size_t>(ctx.vocab == nullptr);
    for (size_t emel_case_32 = emel_branch_32; emel_case_32 == 1u; emel_case_32 = 2u) {
            return false;
    }
    for (size_t emel_case_32 = emel_branch_32; emel_case_32 == 0u; emel_case_32 = 2u) {

    }
  }
    {
    const size_t emel_branch_33 = static_cast<size_t>(ctx.tables_ready);
    for (size_t emel_case_33 = emel_branch_33; emel_case_33 == 1u; emel_case_33 = 2u) {
            return true;
    }
    for (size_t emel_case_33 = emel_branch_33; emel_case_33 == 0u; emel_case_33 = 2u) {

    }
  }

  ctx.token_to_id.clear();
  ctx.bpe_ranks.clear();
  ctx.max_token_len = 0;

  const emel::model::data::vocab &vocab = *ctx.vocab;
  for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
    const std::string_view text = token_text(vocab, static_cast<int32_t>(id));
        {
      const size_t emel_branch_34 = static_cast<size_t>(
        !insert_token_map(ctx.token_to_id, vocab, text, static_cast<int32_t>(id)));
      for (size_t emel_case_34 = emel_branch_34; emel_case_34 == 1u; emel_case_34 = 2u) {
                return false;
      }
      for (size_t emel_case_34 = emel_branch_34; emel_case_34 == 0u; emel_case_34 = 2u) {

      }
    }
        {
      const size_t emel_branch_35 = static_cast<size_t>(text.size() > static_cast<size_t>(ctx.max_token_len));
      for (size_t emel_case_35 = emel_branch_35; emel_case_35 == 1u; emel_case_35 = 2u) {
                ctx.max_token_len = static_cast<int32_t>(text.size());
      }
      for (size_t emel_case_35 = emel_branch_35; emel_case_35 == 0u; emel_case_35 = 2u) {

      }
    }
  }

  for (uint32_t idx = 0; idx < vocab.n_merges; ++idx) {
    const std::string_view merge = merge_text(vocab, static_cast<int32_t>(idx));
    const size_t pos = merge.find(' ');
    const bool has_merge = !merge.empty();
    const bool has_separator = pos != std::string_view::npos;
    const size_t emel_branch_insert_merge = static_cast<size_t>(has_merge && has_separator);
    for (size_t emel_case_insert_merge = emel_branch_insert_merge;
         emel_case_insert_merge == 1u;
         emel_case_insert_merge = 2u) {
      const std::string_view left = merge.substr(0, pos);
      const std::string_view right = merge.substr(pos + 1);
      insert_merge_map(ctx.bpe_ranks, left, right, static_cast<int32_t>(idx), vocab);
    }
    for (size_t emel_case_insert_merge = emel_branch_insert_merge;
         emel_case_insert_merge == 0u;
         emel_case_insert_merge = 2u) {

    }
  }

  ctx.ugm_ready = vocab.precompiled_charsmap_size > 0;
  ctx.tables_ready = true;
  return true;
}

inline void split_whitespace(const std::string_view text,
                             std::vector<std::string_view> &parts) {
  parts.clear();
  size_t start = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
        {
      const size_t emel_branch_36 = static_cast<size_t>(std::isspace(c) != 0);
      for (size_t emel_case_36 = emel_branch_36; emel_case_36 == 1u; emel_case_36 = 2u) {
                parts.emplace_back(text.substr(start, i - start));
                start = i + 1;
      }
      for (size_t emel_case_36 = emel_branch_36; emel_case_36 == 0u; emel_case_36 = 2u) {

      }
    }
  }
  parts.emplace_back(text.substr(start));
}

inline bool build_symbols(const std::string_view text,
                          encode_scratch &scratch,
                          encode_result &result) {
  scratch.symbol_count = 0;
  size_t offset = 0;
  while (offset < text.size()) {
        {
      const size_t emel_branch_37 = static_cast<size_t>(scratch.symbol_count >= scratch.offsets.size());
      for (size_t emel_case_37 = emel_branch_37; emel_case_37 == 1u; emel_case_37 = 2u) {
                result.error = EMEL_ERR_INVALID_ARGUMENT;
                return false;
      }
      for (size_t emel_case_37 = emel_branch_37; emel_case_37 == 0u; emel_case_37 = 2u) {

      }
    }
    const size_t len = std::min(text.size() - offset, utf8_len(text[offset]));
    scratch.offsets[scratch.symbol_count] = static_cast<uint32_t>(offset);
    scratch.lengths[scratch.symbol_count] = static_cast<uint32_t>(len);
    scratch.prev[scratch.symbol_count] = static_cast<int32_t>(scratch.symbol_count) - 1;
    const size_t has_next = static_cast<size_t>(offset + len < text.size());
    const std::array<int32_t, 2> next_candidates = {
        -1,
        static_cast<int32_t>(scratch.symbol_count) + 1,
    };
    scratch.next[scratch.symbol_count] = next_candidates[has_next];
    scratch.symbol_count += 1;
    offset += len;
  }
    {
    const size_t emel_branch_38 = static_cast<size_t>(scratch.symbol_count > 0);
    for (size_t emel_case_38 = emel_branch_38; emel_case_38 == 1u; emel_case_38 = 2u) {
            scratch.prev[0] = -1;
    }
    for (size_t emel_case_38 = emel_branch_38; emel_case_38 == 0u; emel_case_38 = 2u) {

    }
  }
  return true;
}

inline void merge_symbols(encode_scratch &scratch,
                          const int32_t left,
                          const int32_t right) {
  scratch.lengths[static_cast<size_t>(left)] += scratch.lengths[static_cast<size_t>(right)];
  const int32_t right_next = scratch.next[static_cast<size_t>(right)];
  scratch.next[static_cast<size_t>(left)] = right_next;
    {
    const size_t emel_branch_39 = static_cast<size_t>(right_next >= 0);
    for (size_t emel_case_39 = emel_branch_39; emel_case_39 == 1u; emel_case_39 = 2u) {
            scratch.prev[static_cast<size_t>(right_next)] = left;
    }
    for (size_t emel_case_39 = emel_branch_39; emel_case_39 == 0u; emel_case_39 = 2u) {

    }
  }
  scratch.lengths[static_cast<size_t>(right)] = 0;
}

inline bool encode_bytes(const event::encode &ev,
                         action::context &ctx,
                         const emel::model::data::vocab &vocab,
                         const emel::model::data::tokenizer_model model,
                         encode_result &result) {
  (void)vocab;
  int32_t count = 0;
  for (const unsigned char c : ev.text) {
    const int32_t token = byte_to_token(ctx, vocab, c, model);
    const bool failed = token == k_token_null || !push_token(ev, token, count);
        {
      const size_t emel_branch_40 = static_cast<size_t>(failed);
      for (size_t emel_case_40 = emel_branch_40; emel_case_40 == 1u; emel_case_40 = 2u) {
                result.error = EMEL_ERR_BACKEND;
                return false;
      }
      for (size_t emel_case_40 = emel_branch_40; emel_case_40 == 0u; emel_case_40 = 2u) {

      }
    }
  }
  result.token_count = count;
  result.error = EMEL_OK;
  return true;
}

}  // namespace emel::text::encoders::detail
