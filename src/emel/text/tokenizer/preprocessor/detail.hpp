#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "emel/model/data.hpp"
#include "emel/text/tokenizer/bpe/split.hpp"
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
    {
    const size_t emel_branch_1 = static_cast<size_t>(id >= vocab.n_tokens);
    for (size_t emel_case_1 = emel_branch_1; emel_case_1 == 1u; emel_case_1 = 2u) {
            return {};
    }
    for (size_t emel_case_1 = emel_branch_1; emel_case_1 == 0u; emel_case_1 = 2u) {

    }
  }
  const auto & entry = vocab.entries[id];
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

inline bool flag_set(
    const emel::model::data::vocab & vocab,
    const std::array<uint8_t, emel::model::data::vocab::k_attr_flag_bytes> & flags,
    const uint32_t id) noexcept {
    {
    const size_t emel_branch_3 = static_cast<size_t>(id >= vocab.n_tokens);
    for (size_t emel_case_3 = emel_branch_3; emel_case_3 == 1u; emel_case_3 = 2u) {
            return false;
    }
    for (size_t emel_case_3 = emel_branch_3; emel_case_3 == 0u; emel_case_3 = 2u) {

    }
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
    {
    const size_t emel_branch_4 = static_cast<size_t>(id >= vocab.n_tokens);
    for (size_t emel_case_4 = emel_branch_4; emel_case_4 == 1u; emel_case_4 = 2u) {
            return false;
    }
    for (size_t emel_case_4 = emel_branch_4; emel_case_4 == 0u; emel_case_4 = 2u) {

    }
  }
  return token_type_is_special(vocab.entries[id].type);
}

inline bool build_special_tokens(special_token_cache & cache,
                                 const emel::model::data::vocab & vocab) {
    {
    const size_t emel_branch_5 = static_cast<size_t>(cache.vocab == &vocab);
    for (size_t emel_case_5 = emel_branch_5; emel_case_5 == 1u; emel_case_5 = 2u) {
            return true;
    }
    for (size_t emel_case_5 = emel_branch_5; emel_case_5 == 0u; emel_case_5 = 2u) {

    }
  }
  cache.vocab = &vocab;
  cache.count = 0;
  for (uint32_t i = 0; i < vocab.n_tokens; ++i) {
    const bool include_token = is_special_type(vocab, i);
    const std::string_view text = token_text(vocab, i);
    const size_t emel_branch_include =
        static_cast<size_t>(include_token && !text.empty());
    for (size_t emel_case_include = emel_branch_include; emel_case_include == 1u;
         emel_case_include = 2u) {
      {
        const size_t emel_branch_full = static_cast<size_t>(cache.count >= cache.tokens.size());
        for (size_t emel_case_full = emel_branch_full; emel_case_full == 1u;
             emel_case_full = 2u) {
          return false;
        }
        for (size_t emel_case_full = emel_branch_full; emel_case_full == 0u;
             emel_case_full = 2u) {

        }
      }
      special_token & entry = cache.tokens[cache.count];
      entry.text = text;
      entry.token = static_cast<int32_t>(i);
      entry.type = vocab.entries[i].type;
      entry.lstrip = has_lstrip(vocab, i);
      entry.rstrip = has_rstrip(vocab, i);
      cache.count += 1;
    }
    for (size_t emel_case_include = emel_branch_include; emel_case_include == 0u;
         emel_case_include = 2u) {

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
    {
    const size_t emel_branch_6 = static_cast<size_t>(text.empty());
    for (size_t emel_case_6 = emel_branch_6; emel_case_6 == 1u; emel_case_6 = 2u) {
            return true;
    }
    for (size_t emel_case_6 = emel_branch_6; emel_case_6 == 0u; emel_case_6 = 2u) {

    }
  }
    {
    const size_t emel_branch_7 = static_cast<size_t>(count >= capacity);
    for (size_t emel_case_7 = emel_branch_7; emel_case_7 == 1u; emel_case_7 = 2u) {
            return false;
    }
    for (size_t emel_case_7 = emel_branch_7; emel_case_7 == 0u; emel_case_7 = 2u) {

    }
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
    {
    const size_t emel_branch_8 = static_cast<size_t>(token < 0);
    for (size_t emel_case_8 = emel_branch_8; emel_case_8 == 1u; emel_case_8 = 2u) {
            return false;
    }
    for (size_t emel_case_8 = emel_branch_8; emel_case_8 == 0u; emel_case_8 = 2u) {

    }
  }
    {
    const size_t emel_branch_9 = static_cast<size_t>(count >= capacity);
    for (size_t emel_case_9 = emel_branch_9; emel_case_9 == 1u; emel_case_9 = 2u) {
            return false;
    }
    for (size_t emel_case_9 = emel_branch_9; emel_case_9 == 0u; emel_case_9 = 2u) {

    }
  }
  fragment & entry = out[count];
  entry.kind = fragment_kind::token;
  entry.text = {};
  entry.token = token;
  count += 1;
  return true;
}

inline bool partition_with_specials(const std::string_view text,
                                    const special_token_cache & cache,
                                    const bool parse_special,
                                    const std::span<fragment> fragments_out,
                                    size_t & fragment_count_out) {
  fragment_count_out = 0;
  const size_t fragment_capacity = fragments_out.size();
  const bool invalid_output =
      fragments_out.data() == nullptr || fragment_capacity == 0 ||
      fragment_capacity > k_max_fragments;
    {
    const size_t emel_branch_10 = static_cast<size_t>(invalid_output);
    for (size_t emel_case_10 = emel_branch_10; emel_case_10 == 1u; emel_case_10 = 2u) {
            return false;
    }
    for (size_t emel_case_10 = emel_branch_10; emel_case_10 == 0u; emel_case_10 = 2u) {

    }
  }

    {
    const size_t emel_branch_11 = static_cast<size_t>(cache.count == 0);
    for (size_t emel_case_11 = emel_branch_11; emel_case_11 == 1u; emel_case_11 = 2u) {
       {
            size_t count = 0;
            {
              const size_t emel_branch_push = static_cast<size_t>(
                  !push_raw_fragment(fragments_out.data(), fragment_capacity, count, text));
              for (size_t emel_case_push = emel_branch_push; emel_case_push == 1u;
                   emel_case_push = 2u) {
                return false;
              }
              for (size_t emel_case_push = emel_branch_push; emel_case_push == 0u;
                   emel_case_push = 2u) {

              }
            }
            fragment_count_out = count;
            return true;
          }
    }
    for (size_t emel_case_11 = emel_branch_11; emel_case_11 == 0u; emel_case_11 = 2u) {

    }
  }

  std::array<fragment, k_max_fragments> current_fragments = {};
  size_t current_count = 0;
    {
    const size_t emel_branch_12 = static_cast<size_t>(
      !push_raw_fragment(current_fragments.data(), fragment_capacity, current_count, text));
    for (size_t emel_case_12 = emel_branch_12; emel_case_12 == 1u; emel_case_12 = 2u) {
            return false;
    }
    for (size_t emel_case_12 = emel_branch_12; emel_case_12 == 0u; emel_case_12 = 2u) {

    }
  }

  std::array<fragment, k_max_fragments> next_fragments = {};
  for (size_t token_idx = 0; token_idx < cache.count; ++token_idx) {
    const special_token & token = cache.tokens[token_idx];
    const bool skip_without_parse = !parse_special && token_type_skip_when_no_parse(token.type);
    const size_t emel_branch_process_token =
        static_cast<size_t>(!token.text.empty() && !skip_without_parse);
    for (size_t emel_case_process_token = emel_branch_process_token;
         emel_case_process_token == 1u;
         emel_case_process_token = 2u) {
      size_t next_count = 0;
      for (size_t frag_idx = 0; frag_idx < current_count; ++frag_idx) {
        const fragment & frag = current_fragments[frag_idx];
        const bool is_raw = frag.kind == fragment_kind::raw_text;
        {
          const size_t emel_branch_copy_token = static_cast<size_t>(!is_raw);
          for (size_t emel_case_copy_token = emel_branch_copy_token;
               emel_case_copy_token == 1u;
               emel_case_copy_token = 2u) {
            {
              const size_t emel_branch_push_token = static_cast<size_t>(
                  !push_token_fragment(next_fragments.data(), fragment_capacity, next_count,
                                       frag.token));
              for (size_t emel_case_push_token = emel_branch_push_token;
                   emel_case_push_token == 1u;
                   emel_case_push_token = 2u) {
                return false;
              }
              for (size_t emel_case_push_token = emel_branch_push_token;
                   emel_case_push_token == 0u;
                   emel_case_push_token = 2u) {

              }
            }
          }
          for (size_t emel_case_copy_token = emel_branch_copy_token;
               emel_case_copy_token == 0u;
               emel_case_copy_token = 2u) {
            const std::string_view raw = frag.text;
            size_t base_offset = 0;
            while (base_offset < raw.size()) {
              const size_t match = raw.find(token.text, base_offset);
              const size_t emel_branch_has_match =
                  static_cast<size_t>(match != std::string_view::npos);
              for (size_t emel_case_has_match = emel_branch_has_match;
                   emel_case_has_match == 1u;
                   emel_case_has_match = 2u) {
                size_t left_len = match - base_offset;
                {
                  const size_t emel_branch_13 = static_cast<size_t>(token.lstrip);
                  for (size_t emel_case_13 = emel_branch_13; emel_case_13 == 1u;
                       emel_case_13 = 2u) {
                    while (left_len > 0 &&
                           std::isspace(static_cast<unsigned char>(
                               raw[base_offset + left_len - 1])) != 0) {
                      left_len -= 1;
                    }
                  }
                  for (size_t emel_case_13 = emel_branch_13; emel_case_13 == 0u;
                       emel_case_13 = 2u) {

                  }
                }
                {
                  const size_t emel_branch_14 = static_cast<size_t>(left_len > 0);
                  for (size_t emel_case_14 = emel_branch_14; emel_case_14 == 1u;
                       emel_case_14 = 2u) {
                    {
                      const size_t emel_branch_push_left = static_cast<size_t>(
                          !push_raw_fragment(next_fragments.data(), fragment_capacity, next_count,
                                             raw.substr(base_offset, left_len)));
                      for (size_t emel_case_push_left = emel_branch_push_left;
                           emel_case_push_left == 1u;
                           emel_case_push_left = 2u) {
                        return false;
                      }
                      for (size_t emel_case_push_left = emel_branch_push_left;
                           emel_case_push_left == 0u;
                           emel_case_push_left = 2u) {

                      }
                    }
                  }
                  for (size_t emel_case_14 = emel_branch_14; emel_case_14 == 0u;
                       emel_case_14 = 2u) {

                  }
                }

                {
                  const size_t emel_branch_15 = static_cast<size_t>(
                      !push_token_fragment(next_fragments.data(), fragment_capacity, next_count,
                                           token.token));
                  for (size_t emel_case_15 = emel_branch_15; emel_case_15 == 1u;
                       emel_case_15 = 2u) {
                    return false;
                  }
                  for (size_t emel_case_15 = emel_branch_15; emel_case_15 == 0u;
                       emel_case_15 = 2u) {

                  }
                }

                size_t right_offset = match + token.text.size();
                {
                  const size_t emel_branch_16 = static_cast<size_t>(token.rstrip);
                  for (size_t emel_case_16 = emel_branch_16; emel_case_16 == 1u;
                       emel_case_16 = 2u) {
                    while (right_offset < raw.size() &&
                           std::isspace(static_cast<unsigned char>(raw[right_offset])) != 0) {
                      right_offset += 1;
                    }
                  }
                  for (size_t emel_case_16 = emel_branch_16; emel_case_16 == 0u;
                       emel_case_16 = 2u) {

                  }
                }
                base_offset = right_offset;
              }
              for (size_t emel_case_has_match = emel_branch_has_match;
                   emel_case_has_match == 0u;
                   emel_case_has_match = 2u) {
                {
                  const size_t emel_branch_push_tail = static_cast<size_t>(
                      !push_raw_fragment(next_fragments.data(), fragment_capacity, next_count,
                                         raw.substr(base_offset)));
                  for (size_t emel_case_push_tail = emel_branch_push_tail;
                       emel_case_push_tail == 1u;
                       emel_case_push_tail = 2u) {
                    return false;
                  }
                  for (size_t emel_case_push_tail = emel_branch_push_tail;
                       emel_case_push_tail == 0u;
                       emel_case_push_tail = 2u) {

                  }
                }
                base_offset = raw.size();
              }
            }
          }
        }
      }

      current_fragments = next_fragments;
      current_count = next_count;
    }
    for (size_t emel_case_process_token = emel_branch_process_token;
         emel_case_process_token == 0u;
         emel_case_process_token = 2u) {

    }
  }

  for (size_t i = 0; i < current_count; ++i) {
    fragments_out[i] = current_fragments[i];
  }
  fragment_count_out = current_count;
  return true;
}

inline bool
partition_bpe_no_specials(const event::preprocess & request,
                          emel::text::tokenizer::bpe::detail::split_scratch & scratch,
                          size_t & fragment_count_out) {
  fragment_count_out = 0;
  scratch.reset();

  emel::text::tokenizer::bpe::detail::split_view view = {};
    {
    const size_t emel_branch_17 = static_cast<size_t>(
      !emel::text::tokenizer::bpe::detail::split_and_encode_append(
          request.text, request.vocab, scratch, view));
    for (size_t emel_case_17 = emel_branch_17; emel_case_17 == 1u; emel_case_17 = 2u) {
            return false;
    }
    for (size_t emel_case_17 = emel_branch_17; emel_case_17 == 0u; emel_case_17 = 2u) {

    }
  }

  size_t out_count = 0;
  for (size_t idx = 0; idx < view.count; ++idx) {
    const std::string_view word = view.words[idx];
    {
      const size_t emel_branch_emit_word = static_cast<size_t>(!word.empty());
      for (size_t emel_case_emit_word = emel_branch_emit_word; emel_case_emit_word == 1u;
           emel_case_emit_word = 2u) {
            {
          const size_t emel_branch_18 = static_cast<size_t>(
            !push_raw_fragment(request.fragments_out.data(), request.fragments_out.size(), out_count,
                               word));
          for (size_t emel_case_18 = emel_branch_18; emel_case_18 == 1u; emel_case_18 = 2u) {
                    return false;
          }
          for (size_t emel_case_18 = emel_branch_18; emel_case_18 == 0u; emel_case_18 = 2u) {

          }
        }
      }
      for (size_t emel_case_emit_word = emel_branch_emit_word; emel_case_emit_word == 0u;
           emel_case_emit_word = 2u) {

      }
    }
  }

  fragment_count_out = out_count;
  return true;
}

inline bool partition_bpe_with_specials(
    const event::preprocess & request, const special_token_cache & cache,
    emel::text::tokenizer::bpe::detail::split_scratch & scratch,
    size_t & fragment_count_out) {
  fragment_count_out = 0;

  std::array<fragment, k_max_fragments> partitions = {};
  size_t partition_count = 0;
    {
    const size_t emel_branch_19 = static_cast<size_t>(
      !partition_with_specials(
          request.text, cache, request.parse_special,
          std::span<fragment>(partitions.data(), request.fragments_out.size()),
          partition_count));
    for (size_t emel_case_19 = emel_branch_19; emel_case_19 == 1u; emel_case_19 = 2u) {
            return false;
    }
    for (size_t emel_case_19 = emel_branch_19; emel_case_19 == 0u; emel_case_19 = 2u) {

    }
  }

  scratch.reset();
  size_t out_count = 0;
  for (size_t idx = 0; idx < partition_count; ++idx) {
    const fragment & frag = partitions[idx];
    {
      const size_t emel_branch_token = static_cast<size_t>(frag.kind == fragment_kind::token);
      for (size_t emel_case_token = emel_branch_token; emel_case_token == 1u;
           emel_case_token = 2u) {
        {
          const size_t emel_branch_push = static_cast<size_t>(
              !push_token_fragment(request.fragments_out.data(), request.fragments_out.size(),
                                   out_count, frag.token));
          for (size_t emel_case_push = emel_branch_push; emel_case_push == 1u;
               emel_case_push = 2u) {
            return false;
          }
          for (size_t emel_case_push = emel_branch_push; emel_case_push == 0u;
               emel_case_push = 2u) {

          }
        }
      }
      for (size_t emel_case_token = emel_branch_token; emel_case_token == 0u;
           emel_case_token = 2u) {
        {
          const size_t emel_branch_text = static_cast<size_t>(!frag.text.empty());
          for (size_t emel_case_text = emel_branch_text; emel_case_text == 1u;
               emel_case_text = 2u) {
            emel::text::tokenizer::bpe::detail::split_view view = {};
                {
              const size_t emel_branch_20 = static_cast<size_t>(
                !emel::text::tokenizer::bpe::detail::split_and_encode_append(
                    frag.text, request.vocab, scratch, view));
              for (size_t emel_case_20 = emel_branch_20; emel_case_20 == 1u;
                   emel_case_20 = 2u) {
                        return false;
              }
              for (size_t emel_case_20 = emel_branch_20; emel_case_20 == 0u;
                   emel_case_20 = 2u) {

              }
            }
            for (size_t word_idx = 0; word_idx < view.count; ++word_idx) {
              const std::string_view word = view.words[word_idx];
              {
                const size_t emel_branch_emit_word = static_cast<size_t>(!word.empty());
                for (size_t emel_case_emit_word = emel_branch_emit_word;
                     emel_case_emit_word == 1u;
                     emel_case_emit_word = 2u) {
                        {
                    const size_t emel_branch_21 = static_cast<size_t>(
                      !push_raw_fragment(request.fragments_out.data(), request.fragments_out.size(),
                                         out_count, word));
                    for (size_t emel_case_21 = emel_branch_21; emel_case_21 == 1u;
                         emel_case_21 = 2u) {
                                return false;
                    }
                    for (size_t emel_case_21 = emel_branch_21; emel_case_21 == 0u;
                         emel_case_21 = 2u) {

                    }
                  }
                }
                for (size_t emel_case_emit_word = emel_branch_emit_word;
                     emel_case_emit_word == 0u;
                     emel_case_emit_word = 2u) {

                }
              }
            }
          }
          for (size_t emel_case_text = emel_branch_text; emel_case_text == 0u;
               emel_case_text = 2u) {

          }
        }
      }
    }
  }

  fragment_count_out = out_count;
  return true;
}

}  // namespace emel::text::tokenizer::preprocessor::detail
