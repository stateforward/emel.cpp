#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "emel/model/data.hpp"
#include "emel/text/tokenizer/preprocessor/events.hpp"
#include "emel/text/tokenizer/preprocessor/types.hpp"

namespace emel::text::tokenizer::preprocessor::detail {

template <class runtime_event_type>
constexpr decltype(auto)
unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return (ev.event_);
  } else {
    return (ev);
  }
}

template <class value_type>
inline void write_optional(value_type * destination,
                           value_type & sink,
                           const value_type value) noexcept {
  value_type * destinations[2] = {&sink, destination};
  value_type * const target =
      destinations[static_cast<size_t>(destination != nullptr)];
  *target = value;
}

template <class event_type>
inline bool ignore_callback(void *, const event_type &) noexcept {
  return true;
}

template <class event_type>
inline void dispatch_optional_callback(void * owner,
                                       bool (*callback)(void * owner,
                                                        const event_type &),
                                       const event_type & payload) noexcept {
  const size_t callback_ready = static_cast<size_t>(callback != nullptr);
  const size_t owner_ready = static_cast<size_t>(owner != nullptr);
  const size_t valid = callback_ready & owner_ready;
  bool (*callbacks[2])(void *, const event_type &) = {
      ignore_callback<event_type>, callback};
  void * owners[2] = {nullptr, owner};
  callbacks[valid](owners[valid], payload);
}

inline preprocessor::error select_error(const bool ok,
                                        const preprocessor::error runtime_error) noexcept {
  return preprocessor::select_result_error(ok, runtime_error);
}

template <class request_type, class done_event_type, class error_event_type>
inline void dispatch_result_callback(
    const bool ok, const request_type & request, const done_event_type & done_ev,
    const error_event_type & error_ev,
    void (*on_done)(const request_type &, const done_event_type &,
                    const error_event_type &) noexcept,
    void (*on_error)(const request_type &, const done_event_type &,
                     const error_event_type &) noexcept) noexcept {
  using dispatch_fn_type =
      void (*)(const request_type &, const done_event_type &,
               const error_event_type &) noexcept;
  const std::array<dispatch_fn_type, 2> dispatchers = {on_error, on_done};
  dispatchers[static_cast<size_t>(ok)](request, done_ev, error_ev);
}

inline void dispatch_preprocess_done(const event::preprocess & request,
                                     const events::preprocess_done & done_ev,
                                     const events::preprocess_error &) noexcept {
  dispatch_optional_callback(request.owner_sm, request.dispatch_done, done_ev);
}

inline void
dispatch_preprocess_error(const event::preprocess & request,
                          const events::preprocess_done &,
                          const events::preprocess_error & error_ev) noexcept {
  dispatch_optional_callback(request.owner_sm, request.dispatch_error, error_ev);
}

constexpr int32_t k_token_type_unknown = 2;
constexpr int32_t k_token_type_control = 3;
constexpr int32_t k_token_type_user_defined = 4;

inline bool token_type_is_special(const int32_t type) noexcept {
  return type == k_token_type_control || type == k_token_type_user_defined ||
         type == k_token_type_unknown;
}

inline bool token_type_skip_when_no_parse(const int32_t type) noexcept {
  return type == k_token_type_control || type == k_token_type_unknown;
}

inline std::string_view token_text(const emel::model::data::vocab & vocab,
                                   const uint32_t id) {
  if (id >= vocab.n_tokens) {
    return {};
  }
  const auto & entry = vocab.entries[id];
  if (entry.text_length == 0) {
    return {};
  }
  return std::string_view(vocab.token_storage.data() + entry.text_offset,
                          entry.text_length);
}

inline bool flag_set(
    const emel::model::data::vocab & vocab,
    const std::array<uint8_t, emel::model::data::vocab::k_attr_flag_bytes> & flags,
    const uint32_t id) noexcept {
  if (id >= vocab.n_tokens) {
    return false;
  }
  const uint32_t byte = id >> 3;
  const uint8_t mask = static_cast<uint8_t>(1u << (id & 7u));
  return (flags[byte] & mask) != 0;
}

inline bool has_lstrip(const emel::model::data::vocab & vocab,
                       const uint32_t id) noexcept {
  return flag_set(vocab, vocab.lstrip_flags, id);
}

inline bool has_rstrip(const emel::model::data::vocab & vocab,
                       const uint32_t id) noexcept {
  return flag_set(vocab, vocab.rstrip_flags, id);
}

inline bool is_special_type(const emel::model::data::vocab & vocab,
                            const uint32_t id) noexcept {
  if (id >= vocab.n_tokens) {
    return false;
  }
  return token_type_is_special(vocab.entries[id].type);
}

inline bool build_special_tokens(special_token_cache & cache,
                                 const emel::model::data::vocab & vocab) {
  if (cache.vocab == &vocab) {
    return true;
  }
  cache.vocab = &vocab;
  cache.count = 0;
  for (uint32_t i = 0; i < vocab.n_tokens; ++i) {
    const bool include_token = is_special_type(vocab, i);
    const std::string_view text = token_text(vocab, i);
    if (include_token && !text.empty()) {
      if (cache.count >= cache.tokens.size()) {
        return false;
      }
      special_token & entry = cache.tokens[cache.count];
      entry.text = text;
      entry.token = static_cast<int32_t>(i);
      entry.type = vocab.entries[i].type;
      entry.lstrip = has_lstrip(vocab, i);
      entry.rstrip = has_rstrip(vocab, i);
      cache.count += 1;
    }
  }
  std::sort(cache.tokens.begin(),
            cache.tokens.begin() + static_cast<std::ptrdiff_t>(cache.count),
            [](const special_token & a, const special_token & b) {
              return a.text.size() > b.text.size();
            });
  return true;
}

inline bool push_raw_fragment(fragment * out, const size_t capacity,
                              size_t & count, const std::string_view text) {
  if (text.empty()) {
    return true;
  }
  if (count >= capacity) {
    return false;
  }
  fragment & entry = out[count];
  entry.kind = fragment_kind::raw_text;
  entry.text = text;
  entry.token = -1;
  count += 1;
  return true;
}

inline bool push_token_fragment(fragment * out, const size_t capacity,
                                size_t & count, const int32_t token) {
  if (token < 0) {
    return false;
  }
  if (count >= capacity) {
    return false;
  }
  fragment & entry = out[count];
  entry.kind = fragment_kind::token;
  entry.text = {};
  entry.token = token;
  count += 1;
  return true;
}

inline bool special_token_allowed_parse_enabled(const special_token & token) noexcept {
  return !token.text.empty();
}

inline bool special_token_allowed_parse_disabled(const special_token & token) noexcept {
  return !token.text.empty() && !token_type_skip_when_no_parse(token.type);
}

using special_token_allowed_fn = bool (*)(const special_token &) noexcept;

inline bool partition_with_specials_filtered(const std::string_view text,
                                             const special_token_cache & cache,
                                             const std::span<fragment> fragments_out,
                                             size_t & fragment_count_out,
                                             const special_token_allowed_fn token_allowed) {
  fragment_count_out = 0;
  const size_t fragment_capacity = fragments_out.size();
  const bool invalid_output =
      fragments_out.data() == nullptr || fragment_capacity == 0 ||
      fragment_capacity > k_max_fragments;
  if (invalid_output) {
    return false;
  }

  if (cache.count == 0) {
    size_t count = 0;
    if (!push_raw_fragment(fragments_out.data(), fragment_capacity, count, text)) {
      return false;
    }
    fragment_count_out = count;
    return true;
  }

  std::array<fragment, k_max_fragments> current_fragments = {};
  size_t current_count = 0;
  if (!push_raw_fragment(current_fragments.data(), fragment_capacity, current_count, text)) {
    return false;
  }

  std::array<fragment, k_max_fragments> next_fragments = {};
  for (size_t token_idx = 0; token_idx < cache.count; ++token_idx) {
    const special_token & token = cache.tokens[token_idx];
    const bool process_token = token_allowed(token);
    if (process_token) {
      size_t next_count = 0;
      for (size_t frag_idx = 0; frag_idx < current_count; ++frag_idx) {
        const fragment & frag = current_fragments[frag_idx];
        if (frag.kind != fragment_kind::raw_text) {
          if (!push_token_fragment(next_fragments.data(), fragment_capacity, next_count,
                                   frag.token)) {
            return false;
          }
          continue;
        }

        const std::string_view raw = frag.text;
        size_t base_offset = 0;
        while (base_offset < raw.size()) {
          const size_t match = raw.find(token.text, base_offset);
          if (match != std::string_view::npos) {
            size_t left_len = match - base_offset;
            if (token.lstrip) {
              while (left_len > 0 &&
                     std::isspace(
                         static_cast<unsigned char>(raw[base_offset + left_len - 1])) != 0) {
                left_len -= 1;
              }
            }
            if (left_len > 0) {
              if (!push_raw_fragment(next_fragments.data(), fragment_capacity, next_count,
                                     raw.substr(base_offset, left_len))) {
                return false;
              }
            }

            if (!push_token_fragment(next_fragments.data(), fragment_capacity, next_count,
                                     token.token)) {
              return false;
            }

            size_t right_offset = match + token.text.size();
            if (token.rstrip) {
              while (right_offset < raw.size() &&
                     std::isspace(static_cast<unsigned char>(raw[right_offset])) != 0) {
                right_offset += 1;
              }
            }
            base_offset = right_offset;
          } else {
            if (!push_raw_fragment(next_fragments.data(), fragment_capacity, next_count,
                                   raw.substr(base_offset))) {
              return false;
            }
            base_offset = raw.size();
          }
        }
      }

      current_fragments = next_fragments;
      current_count = next_count;
    }
  }

  for (size_t i = 0; i < current_count; ++i) {
    fragments_out[i] = current_fragments[i];
  }
  fragment_count_out = current_count;
  return true;
}

inline bool partition_with_specials_parse_enabled(const std::string_view text,
                                                  const special_token_cache & cache,
                                                  const std::span<fragment> fragments_out,
                                                  size_t & fragment_count_out) {
  return partition_with_specials_filtered(text, cache, fragments_out, fragment_count_out,
                                          special_token_allowed_parse_enabled);
}

inline bool partition_with_specials_parse_disabled(const std::string_view text,
                                                   const special_token_cache & cache,
                                                   const std::span<fragment> fragments_out,
                                                   size_t & fragment_count_out) {
  return partition_with_specials_filtered(text, cache, fragments_out, fragment_count_out,
                                          special_token_allowed_parse_disabled);
}

}  // namespace emel::text::tokenizer::preprocessor::detail
