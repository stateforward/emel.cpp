#pragma once

#include <cstdint>
#include <string>

#include "emel/text/encoders/rwkv/context.hpp"
#include "emel/text/encoders/detail.hpp"
#include "emel/model/data.hpp"

namespace emel::text::encoders::rwkv::detail {

using emel::text::encoders::detail::k_token_null;

inline int32_t select_i32(const bool choose_true,
                          const int32_t true_value,
                          const int32_t false_value) noexcept {
  const int32_t mask = -static_cast<int32_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline uint32_t select_u32(const bool choose_true,
                           const uint32_t true_value,
                           const uint32_t false_value) noexcept {
  const uint32_t mask = static_cast<uint32_t>(0) - static_cast<uint32_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline uint8_t select_u8(const bool choose_true,
                         const uint8_t true_value,
                         const uint8_t false_value) noexcept {
  const uint8_t mask = static_cast<uint8_t>(0) - static_cast<uint8_t>(choose_true);
  return static_cast<uint8_t>((false_value & static_cast<uint8_t>(~mask)) | (true_value & mask));
}

inline std::string_view rwkv_token_text(const emel::model::data::vocab &vocab,
                                        const int32_t id) noexcept {
  const bool valid_id = id >= 0 && static_cast<uint32_t>(id) < vocab.n_tokens;
  const uint32_t idx = select_u32(valid_id, static_cast<uint32_t>(id), 0u);
  const auto &entry = vocab.entries[idx];
  const bool has_text = valid_id && entry.text_length > 0u;
  const uint32_t offset = select_u32(has_text, entry.text_offset, 0u);
  const uint32_t length = select_u32(has_text, entry.text_length, 0u);
  return std::string_view(
    vocab.token_storage.data() + static_cast<size_t>(offset), static_cast<size_t>(length));
}

inline bool unescape_rwkv_token(const std::string_view escaped,
                                std::string &out) {
  using process_hex_handler_t =
      void (*)(std::string &, uint8_t &, uint8_t &, bool &, char) noexcept;
  auto process_hex_none = +[](std::string &, uint8_t &, uint8_t &, bool &, char) noexcept {};
  auto process_hex_some = +[](std::string &out_value,
                              uint8_t &hex_remaining_value,
                              uint8_t &hex_acc_value,
                              bool &consumed_value,
                              char c) noexcept {
    const uint8_t byte = static_cast<uint8_t>(c);
    const bool alpha = byte >= static_cast<uint8_t>('a');
    const uint8_t alpha_value = static_cast<uint8_t>(byte - static_cast<uint8_t>('a') + 10u);
    const uint8_t digit_value = static_cast<uint8_t>(byte - static_cast<uint8_t>('0'));
    const uint8_t nibble = select_u8(alpha, alpha_value, digit_value);
    hex_acc_value = static_cast<uint8_t>((hex_acc_value << 4u) + nibble);
    hex_remaining_value = static_cast<uint8_t>(hex_remaining_value - 1u);
    using emit_hex_handler_t = void (*)(std::string &, uint8_t &, uint8_t &) noexcept;
    auto emit_hex_none = +[](std::string &, uint8_t &, uint8_t &) noexcept {};
    auto emit_hex_some = +[](std::string &out_emit,
                             uint8_t &hex_acc_emit,
                             uint8_t &) noexcept {
      out_emit.push_back(static_cast<char>(hex_acc_emit));
      hex_acc_emit = 0;
    };
    const emit_hex_handler_t emit_hex_handlers[2] = {
        emit_hex_none,
        emit_hex_some,
    };
    emit_hex_handlers[static_cast<size_t>(hex_remaining_value == 0)](
        out_value, hex_acc_value, hex_remaining_value);
    consumed_value = true;
  };

  using process_escape_handler_t =
      void (*)(std::string &, bool &, uint8_t &, bool &, char) noexcept;
  auto process_escape_none =
      +[](std::string &, bool &, uint8_t &, bool &, char) noexcept {};
  auto process_escape_some = +[](std::string &out_value,
                                 bool &escaping_value,
                                 uint8_t &hex_remaining_value,
                                 bool &consumed_value,
                                 char c) noexcept {
    const bool esc_t = c == 't';
    const bool esc_n = c == 'n';
    const bool esc_r = c == 'r';
    const bool esc_x = c == 'x';
    char mapped = c;
    mapped = static_cast<char>(
        select_i32(esc_r, static_cast<int32_t>('\r'), static_cast<int32_t>(mapped)));
    mapped = static_cast<char>(
        select_i32(esc_n, static_cast<int32_t>('\n'), static_cast<int32_t>(mapped)));
    mapped = static_cast<char>(
        select_i32(esc_t, static_cast<int32_t>('\t'), static_cast<int32_t>(mapped)));
    using emit_char_handler_t = void (*)(std::string &, char) noexcept;
    auto emit_char_none = +[](std::string &, char) noexcept {};
    auto emit_char_some = +[](std::string &out_emit, char mapped_emit) noexcept {
      out_emit.push_back(mapped_emit);
    };
    const emit_char_handler_t emit_char_handlers[2] = {emit_char_none, emit_char_some};
    emit_char_handlers[static_cast<size_t>(!esc_x)](out_value, mapped);
    hex_remaining_value = select_u8(esc_x, static_cast<uint8_t>(2), hex_remaining_value);
    escaping_value = false;
    consumed_value = true;
  };

  using begin_escape_handler_t = void (*)(bool &, bool &) noexcept;
  auto begin_escape_none = +[](bool &, bool &) noexcept {};
  auto begin_escape_some = +[](bool &escaping_value, bool &consumed_value) noexcept {
    escaping_value = true;
    consumed_value = true;
  };

  using emit_plain_handler_t = void (*)(std::string &, char) noexcept;
  auto emit_plain_none = +[](std::string &, char) noexcept {};
  auto emit_plain_some = +[](std::string &out_value, char c) noexcept {
    out_value.push_back(c);
  };

  out.clear();
  out.reserve(escaped.size());
  bool escaping = false;
  uint8_t hex_remaining = 0;
  uint8_t hex_acc = 0;

  for (const char c : escaped) {
    bool consumed = false;
    const process_hex_handler_t process_hex_handlers[2] = {
        process_hex_none,
        process_hex_some,
    };
    process_hex_handlers[static_cast<size_t>(hex_remaining != 0)](
        out, hex_remaining, hex_acc, consumed, c);

    const process_escape_handler_t process_escape_handlers[2] = {
        process_escape_none,
        process_escape_some,
    };
    process_escape_handlers[static_cast<size_t>((!consumed) && escaping)](
        out, escaping, hex_remaining, consumed, c);

    const begin_escape_handler_t begin_escape_handlers[2] = {
        begin_escape_none,
        begin_escape_some,
    };
    begin_escape_handlers[static_cast<size_t>((!consumed) && (c == '\\'))](escaping, consumed);

    const emit_plain_handler_t emit_plain_handlers[2] = {
        emit_plain_none,
        emit_plain_some,
    };
    emit_plain_handlers[static_cast<size_t>(!consumed)](out, c);
  }
  return hex_remaining == 0;
}

inline bool rwkv_tables_ready(const emel::text::encoders::rwkv::action::context &ctx,
                              const emel::model::data::vocab &vocab) noexcept {
  return ctx.rwkv_tables_ready && ctx.rwkv_vocab == &vocab;
}

inline bool ensure_rwkv_tables(emel::text::encoders::rwkv::action::context &ctx,
                               const emel::model::data::vocab &vocab) {
  auto process_text_none = +[](emel::text::encoders::rwkv::action::context &,
                               const std::string_view,
                               int32_t,
                               std::string &,
                               bool &) {};
  auto process_text_some = +[](emel::text::encoders::rwkv::action::context &ctx_process,
                               const std::string_view text_process,
                               int32_t id_process,
                               std::string &unescaped_process,
                               bool &ok_process) {
    const bool unescaped_ok = unescape_rwkv_token(text_process, unescaped_process);
    ok_process = ok_process && unescaped_ok;
    using insert_token_handler_t =
      void (*)(emel::text::encoders::rwkv::action::context &, const std::string &, int32_t);
    auto insert_token_none = +[](emel::text::encoders::rwkv::action::context &,
                                 const std::string &,
                                 int32_t) {};
    auto insert_token_some = +[](emel::text::encoders::rwkv::action::context &ctx_insert,
                                 const std::string &unescaped_insert,
                                 int32_t id_insert) {
      ctx_insert.token_matcher.insert(
        unescaped_insert.data(), unescaped_insert.size(), id_insert);
    };
    const insert_token_handler_t insert_token_handlers[2] = {
      insert_token_none,
      insert_token_some,
    };
    const bool insert_token = unescaped_ok && !unescaped_process.empty();
    insert_token_handlers[static_cast<size_t>(insert_token)](
      ctx_process, unescaped_process, id_process);
  };

  ctx.rwkv_vocab = &vocab;
  ctx.rwkv_tables_ready = false;
  ctx.token_matcher = emel::text::encoders::detail::naive_trie{};

  std::string unescaped;
  bool ok = true;
  for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
    const bool step_active = ok;
    const std::string_view text = rwkv_token_text(vocab, static_cast<int32_t>(id));
    using process_text_handler_t = void (*)(emel::text::encoders::rwkv::action::context &,
                                            std::string_view,
                                            int32_t,
                                            std::string &,
                                            bool &);
    const process_text_handler_t process_text_handlers[2] = {
      process_text_none,
      process_text_some,
    };
    process_text_handlers[static_cast<size_t>(step_active && !text.empty())](
      ctx, text, static_cast<int32_t>(id), unescaped, ok);
  }

  ctx.rwkv_tables_ready = ok;
  return ok;
}

}  // namespace emel::text::encoders::rwkv::detail
