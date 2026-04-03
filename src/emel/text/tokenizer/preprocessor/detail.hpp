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

inline size_t select_size(const bool choose_true,
                          const size_t true_value,
                          const size_t false_value) noexcept {
  const std::array<size_t, 2> values = {false_value, true_value};
  return values[static_cast<size_t>(choose_true)];
}

inline uintptr_t select_uptr(const bool choose_true,
                             const uintptr_t true_value,
                             const uintptr_t false_value) noexcept {
  const std::array<uintptr_t, 2> values = {false_value, true_value};
  return values[static_cast<size_t>(choose_true)];
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

inline emel::model::data::tokenizer_pre tokenizer_pre_profile_from_name(
    const std::string_view name) noexcept {
  using tokenizer_pre = emel::model::data::tokenizer_pre;

  if (name.empty() || name == "default") {
    return tokenizer_pre::DEFAULT;
  }
  if (name == "llama3" || name == "llama-v3" || name == "llama-bpe" ||
      name == "falcon3" || name == "falcon-h1" || name == "pixtral" ||
      name == "midm-2.0" || name == "lfm2" || name == "jina-v5-nano") {
    return tokenizer_pre::LLAMA3;
  }
  if (name == "jais2") {
    return tokenizer_pre::JAIS2;
  }
  if (name == "dbrx") {
    return tokenizer_pre::DBRX;
  }
  if (name == "smaug") {
    return tokenizer_pre::SMAUG;
  }
  if (name == "deepseek-llm") {
    return tokenizer_pre::DEEPSEEK_LLM;
  }
  if (name == "deepseek-coder") {
    return tokenizer_pre::DEEPSEEK_CODER;
  }
  if (name == "deepseek-v3") {
    return tokenizer_pre::DEEPSEEK3_LLM;
  }
  if (name == "youtu") {
    return tokenizer_pre::YOUTU;
  }
  if (name == "falcon") {
    return tokenizer_pre::FALCON;
  }
  if (name == "mpt") {
    return tokenizer_pre::MPT;
  }
  if (name == "starcoder") {
    return tokenizer_pre::STARCODER;
  }
  if (name == "gpt2" || name == "gpt-2") {
    return tokenizer_pre::GPT2;
  }
  if (name == "jais") {
    return tokenizer_pre::JAIS;
  }
  if (name == "refact") {
    return tokenizer_pre::REFACT;
  }
  if (name == "command-r") {
    return tokenizer_pre::COMMAND_R;
  }
  if (name == "qwen2") {
    return tokenizer_pre::QWEN2;
  }
  if (name == "qwen2.5" || name == "qwen35") {
    return tokenizer_pre::QWEN35;
  }
  if (name == "stablelm2") {
    return tokenizer_pre::STABLELM2;
  }
  if (name == "olmo") {
    return tokenizer_pre::OLMO;
  }
  if (name == "poro") {
    return tokenizer_pre::PORO;
  }
  if (name == "chatglm4") {
    return tokenizer_pre::CHATGLM4;
  }
  if (name == "viking") {
    return tokenizer_pre::VIKING;
  }
  if (name == "tekken") {
    return tokenizer_pre::TEKKEN;
  }
  if (name == "smollm") {
    return tokenizer_pre::SMOLLM;
  }
  if (name == "codeshell") {
    return tokenizer_pre::CODESHELL;
  }
  if (name == "bloom") {
    return tokenizer_pre::BLOOM;
  }
  if (name == "gpt3-finnish") {
    return tokenizer_pre::GPT3_FINNISH;
  }
  if (name == "exaone") {
    return tokenizer_pre::EXAONE;
  }
  if (name == "exaone4") {
    return tokenizer_pre::EXAONE4;
  }
  if (name == "exaone-moe") {
    return tokenizer_pre::EXAONE_MOE;
  }
  if (name == "chameleon") {
    return tokenizer_pre::CHAMELEON;
  }
  if (name == "minerva") {
    return tokenizer_pre::MINERVA;
  }
  if (name == "megrez") {
    return tokenizer_pre::MEGREZ;
  }
  if (name == "gpt4o" || name == "gpt-4o") {
    return tokenizer_pre::GPT4O;
  }
  if (name == "tiny-aya") {
    return tokenizer_pre::TINY_AYA;
  }
  if (name == "superbpe") {
    return tokenizer_pre::SUPERBPE;
  }
  if (name == "trillion") {
    return tokenizer_pre::TRILLION;
  }
  if (name == "granite-docling") {
    return tokenizer_pre::GRANITE_DOCLING;
  }
  if (name == "bailingmoe") {
    return tokenizer_pre::BAILINGMOE;
  }
  if (name == "seed-coder") {
    return tokenizer_pre::SEED_CODER;
  }
  if (name == "hunyuan") {
    return tokenizer_pre::HUNYUAN;
  }
  if (name == "hunyuan-dense") {
    return tokenizer_pre::HUNYUAN_DENSE;
  }
  if (name == "joyai-llm") {
    return tokenizer_pre::JOYAI_LLM;
  }
  if (name == "kimi-k2") {
    return tokenizer_pre::KIMI_K2;
  }
  if (name == "grok-2") {
    return tokenizer_pre::GROK_2;
  }
  if (name == "afmoe") {
    return tokenizer_pre::AFMOE;
  }
  if (name == "minimax-m2") {
    return tokenizer_pre::MINIMAX_M2;
  }
  if (name == "solar-open") {
    return tokenizer_pre::SOLAR_OPEN;
  }
  return tokenizer_pre::UNKNOWN;
}

inline void apply_tokenizer_pre_defaults(
    const std::string_view name,
    emel::model::data::vocab & vocab) noexcept {
  if (name == "llama3" || name == "llama-v3" || name == "llama-bpe" ||
      name == "falcon3" || name == "falcon-h1" || name == "pixtral" ||
      name == "midm-2.0" || name == "lfm2" || name == "jina-v5-nano") {
    vocab.ignore_merges = true;
    vocab.add_bos = true;
    return;
  }

  if (name == "youtu") {
    vocab.ignore_merges = true;
  }
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
  static constexpr char k_zero = '\0';
  const bool id_valid = id < vocab.n_tokens;
  const uint32_t safe_id = static_cast<uint32_t>(select_size(id_valid, id, 0u));
  const auto & entry = vocab.entries[safe_id];
  const bool has_text = id_valid && entry.text_length != 0;
  const uintptr_t data_addr = select_uptr(
      has_text,
      reinterpret_cast<uintptr_t>(vocab.token_storage.data() + entry.text_offset),
      reinterpret_cast<uintptr_t>(&k_zero));
  const std::array<std::string_view, 2> texts = {
      std::string_view{},
      std::string_view(reinterpret_cast<const char *>(data_addr), entry.text_length),
  };
  return texts[static_cast<size_t>(has_text)];
}

inline bool flag_set(
    const emel::model::data::vocab & vocab,
    const std::array<uint8_t, emel::model::data::vocab::k_attr_flag_bytes> & flags,
    const uint32_t id) noexcept {
  const bool id_valid = id < vocab.n_tokens;
  const uint32_t safe_id = static_cast<uint32_t>(select_size(id_valid, id, 0u));
  const uint32_t byte = safe_id >> 3;
  const uint8_t mask = static_cast<uint8_t>(1u << (safe_id & 7u));
  const bool bit_set = (flags[byte] & mask) != 0;
  const std::array<bool, 2> values = {false, bit_set};
  return values[static_cast<size_t>(id_valid)];
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
  const bool id_valid = id < vocab.n_tokens;
  const uint32_t safe_id = static_cast<uint32_t>(select_size(id_valid, id, 0u));
  const bool is_special = token_type_is_special(vocab.entries[safe_id].type);
  const std::array<bool, 2> values = {false, is_special};
  return values[static_cast<size_t>(id_valid)];
}

inline bool keep_special_token(special_token_cache &,
                               const emel::model::data::vocab &,
                               const uint32_t,
                               const std::string_view) noexcept {
  return true;
}

inline bool overflow_special_token(special_token_cache &,
                                   const emel::model::data::vocab &,
                                   const uint32_t,
                                   const std::string_view) noexcept {
  return false;
}

inline bool write_special_token(special_token_cache & cache,
                                const emel::model::data::vocab & vocab,
                                const uint32_t id,
                                const std::string_view text) noexcept {
  special_token & entry = cache.tokens[cache.count];
  entry.text = text;
  entry.token = static_cast<int32_t>(id);
  entry.type = vocab.entries[id].type;
  entry.lstrip = has_lstrip(vocab, id);
  entry.rstrip = has_rstrip(vocab, id);
  cache.count += 1;
  return true;
}

inline bool add_special_token_entry(special_token_cache & cache,
                                    const emel::model::data::vocab & vocab,
                                    const uint32_t id,
                                    const std::string_view text) noexcept {
  const bool has_capacity = cache.count < cache.tokens.size();
  using add_fn = bool (*)(special_token_cache &, const emel::model::data::vocab &,
                          uint32_t, std::string_view) noexcept;
  constexpr std::array<add_fn, 2> adders = {
      overflow_special_token,
      write_special_token,
  };
  return adders[static_cast<size_t>(has_capacity)](cache, vocab, id, text);
}

inline bool scan_special_token_entry(special_token_cache & cache,
                                     const emel::model::data::vocab & vocab,
                                     const uint32_t id) noexcept {
  const bool include_token = is_special_type(vocab, id);
  const std::string_view text = token_text(vocab, id);
  const bool include = include_token && !text.empty();
  using scan_fn = bool (*)(special_token_cache &, const emel::model::data::vocab &,
                           uint32_t, std::string_view) noexcept;
  constexpr std::array<scan_fn, 2> scanners = {
      keep_special_token,
      add_special_token_entry,
  };
  return scanners[static_cast<size_t>(include)](cache, vocab, id, text);
}

inline bool scan_special_token_range(special_token_cache & cache,
                                     const emel::model::data::vocab & vocab,
                                     const uint32_t begin,
                                     const uint32_t end) noexcept;

inline bool scan_special_token_range_done(special_token_cache &,
                                          const emel::model::data::vocab &,
                                          const uint32_t,
                                          const uint32_t) noexcept {
  return true;
}

inline bool scan_special_token_range_active(special_token_cache & cache,
                                            const emel::model::data::vocab & vocab,
                                            const uint32_t begin,
                                            const uint32_t end) noexcept {
  const uint32_t span = end - begin;
  const uint32_t mid = begin + (span >> 1u);
  const bool left_ok = scan_special_token_range(cache, vocab, begin, mid);
  const bool center_ok = scan_special_token_entry(cache, vocab, mid);
  const bool right_ok = scan_special_token_range(cache, vocab, mid + 1u, end);
  return left_ok && center_ok && right_ok;
}

inline bool scan_special_token_range(special_token_cache & cache,
                                     const emel::model::data::vocab & vocab,
                                     const uint32_t begin,
                                     const uint32_t end) noexcept {
  using scan_fn = bool (*)(special_token_cache &, const emel::model::data::vocab &,
                           uint32_t, uint32_t) noexcept;
  constexpr std::array<scan_fn, 2> scanners = {
      scan_special_token_range_done,
      scan_special_token_range_active,
  };
  const bool has_range = begin < end;
  return scanners[static_cast<size_t>(has_range)](cache, vocab, begin, end);
}

inline bool finish_build_special_tokens_error(special_token_cache &) noexcept {
  return false;
}

inline bool finish_build_special_tokens_ok(special_token_cache & cache) {
  std::sort(cache.tokens.begin(),
            cache.tokens.begin() + static_cast<std::ptrdiff_t>(cache.count),
            [](const special_token & a, const special_token & b) {
              return a.text.size() > b.text.size();
            });
  return true;
}

inline bool build_special_tokens_cached(special_token_cache &,
                                        const emel::model::data::vocab &) noexcept {
  return true;
}

inline bool build_special_tokens_rebuild(special_token_cache & cache,
                                         const emel::model::data::vocab & vocab) {
  cache.vocab = &vocab;
  cache.count = 0;
  const bool scanned = scan_special_token_range(cache, vocab, 0u, vocab.n_tokens);
  using finish_fn = bool (*)(special_token_cache &);
  const std::array<finish_fn, 2> finishers = {
      finish_build_special_tokens_error,
      finish_build_special_tokens_ok,
  };
  return finishers[static_cast<size_t>(scanned)](cache);
}

inline bool build_special_tokens(special_token_cache & cache,
                                 const emel::model::data::vocab & vocab) {
  const bool cache_matches = cache.vocab == &vocab;
  using build_fn = bool (*)(special_token_cache &, const emel::model::data::vocab &);
  const std::array<build_fn, 2> builders = {
      build_special_tokens_rebuild,
      build_special_tokens_cached,
  };
  return builders[static_cast<size_t>(cache_matches)](cache, vocab);
}

inline void write_raw_fragment_noop(fragment *,
                                    size_t &,
                                    const std::string_view) noexcept {}

inline void write_raw_fragment_active(fragment * out,
                                      size_t & count,
                                      const std::string_view text) noexcept {
  fragment & entry = out[count];
  entry.kind = fragment_kind::raw_text;
  entry.text = text;
  entry.token = -1;
  count += 1;
}

inline bool push_raw_fragment(fragment * out, const size_t capacity,
                              size_t & count, const std::string_view text) {
  const bool has_text = !text.empty();
  const bool has_capacity = count < capacity;
  const size_t state = (static_cast<size_t>(has_text) << 1u) |
                       static_cast<size_t>(has_capacity);
  using write_fn = void (*)(fragment *, size_t &, std::string_view) noexcept;
  constexpr std::array<write_fn, 4> writers = {
      write_raw_fragment_noop,
      write_raw_fragment_noop,
      write_raw_fragment_noop,
      write_raw_fragment_active,
  };
  constexpr std::array<bool, 4> results = {
      true,
      true,
      false,
      true,
  };
  writers[state](out, count, text);
  return results[state];
}

inline void write_token_fragment_noop(fragment *,
                                      size_t &,
                                      const int32_t) noexcept {}

inline void write_token_fragment_active(fragment * out,
                                        size_t & count,
                                        const int32_t token) noexcept {
  fragment & entry = out[count];
  entry.kind = fragment_kind::token;
  entry.text = {};
  entry.token = token;
  count += 1;
}

inline bool push_token_fragment(fragment * out, const size_t capacity,
                                size_t & count, const int32_t token) {
  const bool token_valid = token >= 0;
  const bool has_capacity = count < capacity;
  const size_t state = (static_cast<size_t>(token_valid) << 1u) |
                       static_cast<size_t>(has_capacity);
  using write_fn = void (*)(fragment *, size_t &, int32_t) noexcept;
  constexpr std::array<write_fn, 4> writers = {
      write_token_fragment_noop,
      write_token_fragment_noop,
      write_token_fragment_noop,
      write_token_fragment_active,
  };
  constexpr std::array<bool, 4> results = {
      false,
      false,
      false,
      true,
  };
  writers[state](out, count, token);
  return results[state];
}

inline bool special_token_allowed_parse_enabled(const special_token & token) noexcept {
  return !token.text.empty();
}

inline bool special_token_allowed_parse_disabled(const special_token & token) noexcept {
  return !token.text.empty() && !token_type_skip_when_no_parse(token.type);
}

using special_token_allowed_fn = bool (*)(const special_token &) noexcept;

inline void trim_left_noop(const std::string_view,
                           const size_t,
                           size_t &) noexcept {}

inline void trim_left_active(const std::string_view raw,
                             const size_t base_offset,
                             size_t & left_len) noexcept;

inline void trim_left_step_stop(const std::string_view,
                                const size_t,
                                size_t &) noexcept {}

inline void trim_left_step_continue(const std::string_view raw,
                                    const size_t base_offset,
                                    size_t & left_len) noexcept {
  left_len -= 1;
  trim_left_active(raw, base_offset, left_len);
}

inline void trim_left_active(const std::string_view raw,
                             const size_t base_offset,
                             size_t & left_len) noexcept {
  const bool can_trim =
      left_len > 0 &&
      std::isspace(static_cast<unsigned char>(raw[base_offset + left_len - 1u])) != 0;
  using step_fn = void (*)(std::string_view, size_t, size_t &) noexcept;
  constexpr std::array<step_fn, 2> steppers = {
      trim_left_step_stop,
      trim_left_step_continue,
  };
  steppers[static_cast<size_t>(can_trim)](raw, base_offset, left_len);
}

inline void trim_right_noop(const std::string_view,
                            size_t &) noexcept {}

inline void trim_right_active(const std::string_view raw,
                              size_t & right_offset) noexcept;

inline void trim_right_step_stop(const std::string_view,
                                 size_t &) noexcept {}

inline void trim_right_step_continue(const std::string_view raw,
                                     size_t & right_offset) noexcept {
  right_offset += 1u;
  trim_right_active(raw, right_offset);
}

inline void trim_right_active(const std::string_view raw,
                              size_t & right_offset) noexcept {
  const bool can_trim =
      right_offset < raw.size() &&
      std::isspace(static_cast<unsigned char>(raw[right_offset])) != 0;
  using step_fn = void (*)(std::string_view, size_t &) noexcept;
  constexpr std::array<step_fn, 2> steppers = {
      trim_right_step_stop,
      trim_right_step_continue,
  };
  steppers[static_cast<size_t>(can_trim)](raw, right_offset);
}

inline void partition_raw_scan_recursive(const std::string_view raw,
                                         const special_token & token,
                                         fragment * out,
                                         const size_t capacity,
                                         size_t & next_count,
                                         size_t & base_offset,
                                         bool & ok) noexcept;

inline void partition_raw_scan_stop(const std::string_view,
                                    const special_token &,
                                    fragment *,
                                    const size_t,
                                    size_t &,
                                    size_t &,
                                    bool &) noexcept {}

inline void partition_raw_scan_no_match(const std::string_view raw,
                                        const special_token &,
                                        fragment * out,
                                        const size_t capacity,
                                        size_t & next_count,
                                        size_t & base_offset,
                                        bool & ok,
                                        const size_t) {
  const bool push_ok = push_raw_fragment(out, capacity, next_count, raw.substr(base_offset));
  ok = ok && push_ok;
  base_offset = raw.size();
}

inline void partition_raw_scan_match(const std::string_view raw,
                                     const special_token & token,
                                     fragment * out,
                                     const size_t capacity,
                                     size_t & next_count,
                                     size_t & base_offset,
                                     bool & ok,
                                     const size_t match) {
  size_t left_len = match - base_offset;
  using trim_left_fn = void (*)(std::string_view, size_t, size_t &) noexcept;
  constexpr std::array<trim_left_fn, 2> trim_left_handlers = {
      trim_left_noop,
      trim_left_active,
  };
  trim_left_handlers[static_cast<size_t>(token.lstrip)](raw, base_offset, left_len);

  const bool left_ok =
      push_raw_fragment(out, capacity, next_count, raw.substr(base_offset, left_len));
  const bool token_ok = push_token_fragment(out, capacity, next_count, token.token);
  ok = ok && left_ok && token_ok;

  size_t right_offset = match + token.text.size();
  using trim_right_fn = void (*)(std::string_view, size_t &) noexcept;
  constexpr std::array<trim_right_fn, 2> trim_right_handlers = {
      trim_right_noop,
      trim_right_active,
  };
  trim_right_handlers[static_cast<size_t>(token.rstrip)](raw, right_offset);
  base_offset = right_offset;
}

inline void partition_raw_scan_continue(const std::string_view raw,
                                        const special_token & token,
                                        fragment * out,
                                        const size_t capacity,
                                        size_t & next_count,
                                        size_t & base_offset,
                                        bool & ok) {
  const size_t match = raw.find(token.text, base_offset);
  const bool has_match = match != std::string_view::npos;
  using match_fn = void (*)(std::string_view, const special_token &, fragment *,
                            size_t, size_t &, size_t &, bool &, size_t);
  constexpr std::array<match_fn, 2> match_handlers = {
      partition_raw_scan_no_match,
      partition_raw_scan_match,
  };
  match_handlers[static_cast<size_t>(has_match)](raw, token, out, capacity, next_count,
                                                 base_offset, ok, match);
  partition_raw_scan_recursive(raw, token, out, capacity, next_count, base_offset,
                               ok);
}

inline void partition_raw_scan_recursive(const std::string_view raw,
                                         const special_token & token,
                                         fragment * out,
                                         const size_t capacity,
                                         size_t & next_count,
                                         size_t & base_offset,
                                         bool & ok) noexcept {
  const bool continue_scan = ok && base_offset < raw.size();
  using scan_fn = void (*)(std::string_view, const special_token &, fragment *,
                           size_t, size_t &, size_t &, bool &);
  constexpr std::array<scan_fn, 2> scanners = {
      partition_raw_scan_stop,
      partition_raw_scan_continue,
  };
  scanners[static_cast<size_t>(continue_scan)](raw, token, out, capacity, next_count,
                                               base_offset, ok);
}

inline void partition_fragment_token(const fragment & frag,
                                     const special_token &,
                                     fragment * out,
                                     const size_t capacity,
                                     size_t & next_count,
                                     bool & ok) {
  const bool push_ok = push_token_fragment(out, capacity, next_count, frag.token);
  ok = ok && push_ok;
}

inline void partition_fragment_raw(const fragment & frag,
                                   const special_token & token,
                                   fragment * out,
                                   const size_t capacity,
                                   size_t & next_count,
                                   bool & ok) {
  size_t base_offset = 0;
  partition_raw_scan_recursive(frag.text, token, out, capacity, next_count,
                               base_offset, ok);
}

inline void partition_fragments_recursive(
    const std::array<fragment, k_max_fragments> & current_fragments,
    const size_t current_count,
    const special_token & token,
    fragment * out,
    const size_t capacity,
    size_t & next_count,
    const size_t frag_idx,
    bool & ok) noexcept;

inline void partition_fragments_stop(
    const std::array<fragment, k_max_fragments> &,
    const size_t,
    const special_token &,
    fragment *,
    const size_t,
    size_t &,
    const size_t,
    bool &) noexcept {}

inline void partition_fragments_continue(
    const std::array<fragment, k_max_fragments> & current_fragments,
    const size_t current_count,
    const special_token & token,
    fragment * out,
    const size_t capacity,
    size_t & next_count,
    const size_t frag_idx,
    bool & ok) {
  const fragment & frag = current_fragments[frag_idx];
  using partition_fn = void (*)(const fragment &, const special_token &, fragment *,
                                size_t, size_t &, bool &);
  constexpr std::array<partition_fn, 2> partitioners = {
      partition_fragment_raw,
      partition_fragment_token,
  };
  const size_t token_fragment =
      static_cast<size_t>(frag.kind == fragment_kind::token);
  partitioners[token_fragment](frag, token, out, capacity, next_count, ok);
  partition_fragments_recursive(current_fragments, current_count, token, out, capacity,
                                next_count, frag_idx + 1u, ok);
}

inline void partition_fragments_recursive(
    const std::array<fragment, k_max_fragments> & current_fragments,
    const size_t current_count,
    const special_token & token,
    fragment * out,
    const size_t capacity,
    size_t & next_count,
    const size_t frag_idx,
    bool & ok) noexcept {
  const bool continue_partition = ok && frag_idx < current_count;
  using partition_fn = void (*)(const std::array<fragment, k_max_fragments> &,
                                size_t, const special_token &, fragment *, size_t,
                                size_t &, size_t, bool &);
  constexpr std::array<partition_fn, 2> partitioners = {
      partition_fragments_stop,
      partition_fragments_continue,
  };
  partitioners[static_cast<size_t>(continue_partition)](
      current_fragments, current_count, token, out, capacity, next_count, frag_idx,
      ok);
}

inline void apply_token_skip(
    std::array<fragment, k_max_fragments> &,
    size_t &,
    std::array<fragment, k_max_fragments> &,
    const size_t,
    const special_token &,
    bool &) noexcept {}

inline void apply_token_partition(
    std::array<fragment, k_max_fragments> & current_fragments,
    size_t & current_count,
    std::array<fragment, k_max_fragments> & next_fragments,
    const size_t capacity,
    const special_token & token,
    bool & ok) {
  size_t next_count = 0;
  partition_fragments_recursive(current_fragments, current_count, token,
                                next_fragments.data(), capacity, next_count, 0u, ok);
  current_fragments = next_fragments;
  current_count = next_count;
}

inline void partition_tokens_recursive(
    const special_token_cache & cache,
    const special_token_allowed_fn token_allowed,
    std::array<fragment, k_max_fragments> & current_fragments,
    size_t & current_count,
    std::array<fragment, k_max_fragments> & next_fragments,
    const size_t capacity,
    const size_t token_idx,
    bool & ok) noexcept;

inline void partition_tokens_stop(
    const special_token_cache &,
    const special_token_allowed_fn,
    std::array<fragment, k_max_fragments> &,
    size_t &,
    std::array<fragment, k_max_fragments> &,
    const size_t,
    const size_t,
    bool &) noexcept {}

inline void partition_tokens_continue(
    const special_token_cache & cache,
    const special_token_allowed_fn token_allowed,
    std::array<fragment, k_max_fragments> & current_fragments,
    size_t & current_count,
    std::array<fragment, k_max_fragments> & next_fragments,
    const size_t capacity,
    const size_t token_idx,
    bool & ok) {
  const special_token & token = cache.tokens[token_idx];
  const bool process_token = token_allowed(token);
  using token_fn = void (*)(std::array<fragment, k_max_fragments> &, size_t &,
                            std::array<fragment, k_max_fragments> &, size_t,
                            const special_token &, bool &);
  constexpr std::array<token_fn, 2> token_handlers = {
      apply_token_skip,
      apply_token_partition,
  };
  token_handlers[static_cast<size_t>(process_token)](
      current_fragments, current_count, next_fragments, capacity, token, ok);
  partition_tokens_recursive(cache, token_allowed, current_fragments, current_count,
                             next_fragments, capacity, token_idx + 1u, ok);
}

inline void partition_tokens_recursive(
    const special_token_cache & cache,
    const special_token_allowed_fn token_allowed,
    std::array<fragment, k_max_fragments> & current_fragments,
    size_t & current_count,
    std::array<fragment, k_max_fragments> & next_fragments,
    const size_t capacity,
    const size_t token_idx,
    bool & ok) noexcept {
  const bool continue_partition = ok && token_idx < cache.count;
  using token_fn = void (*)(const special_token_cache &, special_token_allowed_fn,
                            std::array<fragment, k_max_fragments> &, size_t &,
                            std::array<fragment, k_max_fragments> &, size_t, size_t,
                            bool &);
  constexpr std::array<token_fn, 2> token_handlers = {
      partition_tokens_stop,
      partition_tokens_continue,
  };
  token_handlers[static_cast<size_t>(continue_partition)](
      cache, token_allowed, current_fragments, current_count, next_fragments,
      capacity, token_idx, ok);
}

inline void copy_fragments_recursive(
    fragment * out,
    const std::array<fragment, k_max_fragments> & current_fragments,
    const size_t current_count,
    const size_t idx) noexcept;

inline void copy_fragments_stop(fragment *,
                                const std::array<fragment, k_max_fragments> &,
                                const size_t,
                                const size_t) noexcept {}

inline void copy_fragments_continue(
    fragment * out,
    const std::array<fragment, k_max_fragments> & current_fragments,
    const size_t current_count,
    const size_t idx) noexcept {
  out[idx] = current_fragments[idx];
  copy_fragments_recursive(out, current_fragments, current_count, idx + 1u);
}

inline void copy_fragments_recursive(
    fragment * out,
    const std::array<fragment, k_max_fragments> & current_fragments,
    const size_t current_count,
    const size_t idx) noexcept {
  const bool continue_copy = idx < current_count;
  using copy_fn = void (*)(fragment *, const std::array<fragment, k_max_fragments> &,
                           size_t, size_t) noexcept;
  constexpr std::array<copy_fn, 2> copiers = {
      copy_fragments_stop,
      copy_fragments_continue,
  };
  copiers[static_cast<size_t>(continue_copy)](out, current_fragments, current_count,
                                              idx);
}

inline bool partition_invalid_output(const std::string_view,
                                     const special_token_cache &,
                                     const std::span<fragment>,
                                     size_t & fragment_count_out,
                                     const special_token_allowed_fn) noexcept {
  fragment_count_out = 0;
  return false;
}

inline bool partition_empty_cache(const std::string_view text,
                                  const special_token_cache &,
                                  const std::span<fragment> fragments_out,
                                  size_t & fragment_count_out,
                                  const special_token_allowed_fn) {
  size_t count = 0;
  const bool ok =
      push_raw_fragment(fragments_out.data(), fragments_out.size(), count, text);
  fragment_count_out = count;
  return ok;
}

inline bool partition_with_cache(const std::string_view text,
                                 const special_token_cache & cache,
                                 const std::span<fragment> fragments_out,
                                 size_t & fragment_count_out,
                                 const special_token_allowed_fn token_allowed) {
  std::array<fragment, k_max_fragments> current_fragments = {};
  size_t current_count = 0;
  bool ok = push_raw_fragment(current_fragments.data(), fragments_out.size(),
                              current_count, text);

  std::array<fragment, k_max_fragments> next_fragments = {};
  partition_tokens_recursive(cache, token_allowed, current_fragments, current_count,
                             next_fragments, fragments_out.size(), 0u, ok);

  using copy_fn = void (*)(fragment *,
                           const std::array<fragment, k_max_fragments> &,
                           size_t,
                           size_t) noexcept;
  constexpr std::array<copy_fn, 2> copiers = {
      copy_fragments_stop,
      copy_fragments_recursive,
  };
  copiers[static_cast<size_t>(ok)](fragments_out.data(), current_fragments,
                                   current_count, 0u);

  const std::array<size_t, 2> counts = {
      0,
      current_count,
  };
  fragment_count_out = counts[static_cast<size_t>(ok)];
  return ok;
}

inline bool partition_valid_output(const std::string_view text,
                                   const special_token_cache & cache,
                                   const std::span<fragment> fragments_out,
                                   size_t & fragment_count_out,
                                   const special_token_allowed_fn token_allowed) {
  using partition_fn = bool (*)(std::string_view, const special_token_cache &,
                                std::span<fragment>, size_t &,
                                special_token_allowed_fn);
  const std::array<partition_fn, 2> partitions = {
      partition_empty_cache,
      partition_with_cache,
  };
  const bool has_specials = cache.count != 0;
  return partitions[static_cast<size_t>(has_specials)](text, cache, fragments_out,
                                                       fragment_count_out,
                                                       token_allowed);
}

inline bool partition_with_specials_filtered(const std::string_view text,
                                             const special_token_cache & cache,
                                             const std::span<fragment> fragments_out,
                                             size_t & fragment_count_out,
                                             const special_token_allowed_fn token_allowed) {
  fragment_count_out = 0;
  const size_t fragment_capacity = fragments_out.size();
  const bool output_valid =
      fragments_out.data() != nullptr && fragment_capacity != 0 &&
      fragment_capacity <= k_max_fragments;
  using partition_fn = bool (*)(std::string_view, const special_token_cache &,
                                std::span<fragment>, size_t &,
                                special_token_allowed_fn);
  const std::array<partition_fn, 2> partitions = {
      partition_invalid_output,
      partition_valid_output,
  };
  return partitions[static_cast<size_t>(output_valid)](text, cache, fragments_out,
                                                        fragment_count_out,
                                                        token_allowed);
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
