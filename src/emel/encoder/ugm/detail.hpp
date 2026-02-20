#pragma once

#include <cctype>
#include <cstring>
#include <string>

#include "emel/encoder/ugm/context.hpp"
#include "emel/encoder/detail.hpp"
#include "emel/encoder/events.hpp"
#include "emel/model/data.hpp"

namespace emel::encoder::ugm::detail {

using emel::encoder::detail::encode_result;
using emel::encoder::detail::k_token_null;

inline bool xcda_table(const emel::encoder::action::context &ctx,
                       const uint32_t *&table,
                       size_t &table_size,
                       const char *&replacements,
                       size_t &replacements_size) {
  if (ctx.vocab == nullptr || ctx.vocab->precompiled_charsmap_size == 0) {
    table = nullptr;
    table_size = 0;
    replacements = nullptr;
    replacements_size = 0;
    return false;
  }
  const uint8_t *data = ctx.vocab->precompiled_charsmap.data();
  const uint32_t blob_size = *reinterpret_cast<const uint32_t *>(data);
  if (blob_size + sizeof(blob_size) > ctx.vocab->precompiled_charsmap_size) {
    return false;
  }
  table = reinterpret_cast<const uint32_t *>(data + sizeof(blob_size));
  table_size = blob_size / sizeof(uint32_t);
  replacements = reinterpret_cast<const char *>(data + sizeof(blob_size) + blob_size);
  replacements_size = ctx.vocab->precompiled_charsmap_size - sizeof(blob_size) - blob_size;
  return true;
}

inline bool xcda_next(const emel::encoder::action::context &ctx,
                      const uint32_t state,
                      const uint8_t c,
                      uint32_t &next) {
  const uint32_t *table = nullptr;
  size_t table_size = 0;
  const char *replacements = nullptr;
  size_t replacements_size = 0;
  if (!xcda_table(ctx, table, table_size, replacements, replacements_size)) {
    return false;
  }
  if (state >= table_size) {
    return false;
  }
  const uint32_t entry = table[state];
  const uint32_t base = entry >> 8;
  const uint32_t next_idx = base ^ c;
  if (next_idx >= table_size) {
    return false;
  }
  const uint32_t next_entry = table[next_idx];
  const uint8_t check = static_cast<uint8_t>(next_entry & 0xFFu);
  if (check != c) {
    return false;
  }
  next = next_idx;
  return true;
}

inline bool xcda_value(const emel::encoder::action::context &ctx,
                       const uint32_t state,
                       uint32_t &value) {
  const uint32_t *table = nullptr;
  size_t table_size = 0;
  const char *replacements = nullptr;
  size_t replacements_size = 0;
  if (!xcda_table(ctx, table, table_size, replacements, replacements_size)) {
    return false;
  }
  if (state >= table_size) {
    return false;
  }
  uint32_t leaf = 0;
  if (!xcda_next(ctx, state, 0, leaf)) {
    return false;
  }
  if (leaf >= table_size) {
    return false;
  }
  const uint32_t entry = table[leaf];
  const uint32_t base = entry >> 8;
  if (base >= replacements_size) {
    return false;
  }
  value = base;
  return true;
}

inline std::string apply_precompiled_charsmap(const emel::encoder::action::context &ctx,
                                              const std::string_view text) {
  const uint32_t *table = nullptr;
  size_t table_size = 0;
  const char *replacements = nullptr;
  size_t replacements_size = 0;
  if (!xcda_table(ctx, table, table_size, replacements, replacements_size)) {
    return std::string(text);
  }

  uint32_t state = 0;
  uint32_t last_value = 0;
  size_t last_end = 0;
  bool has_value = false;

  for (size_t i = 0; i < text.size(); ++i) {
    uint32_t next = 0;
    if (!xcda_next(ctx, state, static_cast<uint8_t>(text[i]), next)) {
      break;
    }
    state = next;
    uint32_t value = 0;
    if (xcda_value(ctx, state, value)) {
      last_value = value;
      last_end = i + 1;
      has_value = true;
    }
  }

  if (!has_value || last_value >= replacements_size) {
    return std::string(text);
  }

  const char *replacement = replacements + last_value;
  const size_t replacement_len = std::strlen(replacement);
  std::string output;
  output.reserve(replacement_len + (text.size() - last_end));
  output.append(replacement, replacement_len);
  output.append(text.substr(last_end));
  return output;
}

inline bool apply_precompiled_charsmap_into(const emel::encoder::action::context &ctx,
                                            const std::string_view text,
                                            char *out,
                                            const size_t out_cap,
                                            size_t &out_len) {
  out_len = 0;
  const uint32_t *table = nullptr;
  size_t table_size = 0;
  const char *replacements = nullptr;
  size_t replacements_size = 0;
  if (!xcda_table(ctx, table, table_size, replacements, replacements_size)) {
    if (text.size() > out_cap) {
      return false;
    }
    std::memcpy(out, text.data(), text.size());
    out_len = text.size();
    return true;
  }

  uint32_t state = 0;
  uint32_t last_value = 0;
  size_t last_end = 0;
  bool has_value = false;

  for (size_t i = 0; i < text.size(); ++i) {
    uint32_t next = 0;
    if (!xcda_next(ctx, state, static_cast<uint8_t>(text[i]), next)) {
      break;
    }
    state = next;
    uint32_t value = 0;
    if (xcda_value(ctx, state, value)) {
      last_value = value;
      last_end = i + 1;
      has_value = true;
    }
  }

  if (!has_value || last_value >= replacements_size) {
    if (text.size() > out_cap) {
      return false;
    }
    std::memcpy(out, text.data(), text.size());
    out_len = text.size();
    return true;
  }

  const char *replacement = replacements + last_value;
  const size_t replacement_len = std::strlen(replacement);
  const size_t remainder_len = text.size() - last_end;
  if (replacement_len + remainder_len > out_cap) {
    return false;
  }
  std::memcpy(out, replacement, replacement_len);
  if (remainder_len > 0) {
    std::memcpy(out + replacement_len, text.data() + last_end, remainder_len);
  }
  out_len = replacement_len + remainder_len;
  return true;
}

inline std::string normalize_ugm(const emel::model::data::vocab &vocab,
                                 emel::encoder::action::context &ctx,
                                 const std::string_view text) {
  std::string normalized = apply_precompiled_charsmap(ctx, text);
  if (vocab.remove_extra_whitespaces) {
    std::string collapsed;
    collapsed.reserve(normalized.size());
    bool in_space = false;
    for (const unsigned char c : normalized) {
      if (std::isspace(c) != 0) {
        if (!in_space) {
          collapsed.push_back(' ');
          in_space = true;
        }
        continue;
      }
      in_space = false;
      collapsed.push_back(static_cast<char>(c));
    }
    if (!vocab.treat_whitespace_as_suffix) {
      while (!collapsed.empty() && collapsed.front() == ' ') {
        collapsed.erase(collapsed.begin());
      }
      while (!collapsed.empty() && collapsed.back() == ' ') {
        collapsed.pop_back();
      }
    }
    normalized.swap(collapsed);
  }

  if (vocab.add_space_prefix && !normalized.empty() && normalized.front() != ' ') {
    normalized.insert(normalized.begin(), ' ');
  }

  if (vocab.escape_whitespaces) {
    std::string escaped;
    escaped.reserve(normalized.size() * 2);
    for (const char c : normalized) {
      if (c == ' ') {
        escaped.append("\xE2\x96\x81");
      } else {
        escaped.push_back(c);
      }
    }
    normalized.swap(escaped);
  }
  return normalized;
}

inline bool normalize_ugm_into(const emel::model::data::vocab &vocab,
                               emel::encoder::action::context &ctx,
                               const std::string_view text,
                               std::string_view &out_view) {
  size_t len = 0;
  if (!apply_precompiled_charsmap_into(ctx, text,
                                       ctx.scratch.buffer.data(),
                                       ctx.scratch.buffer.size(), len)) {
    return false;
  }
  std::string_view current(ctx.scratch.buffer.data(), len);

  if (vocab.remove_extra_whitespaces) {
    size_t out_len = 0;
    bool in_space = false;
    for (size_t i = 0; i < current.size(); ++i) {
      const unsigned char c = static_cast<unsigned char>(current[i]);
      if (std::isspace(c) != 0) {
        if (!in_space) {
          ctx.scratch.buffer_alt[out_len++] = ' ';
          in_space = true;
        }
        continue;
      }
      in_space = false;
      ctx.scratch.buffer_alt[out_len++] = static_cast<char>(c);
    }
    if (!vocab.treat_whitespace_as_suffix) {
      size_t start = 0;
      while (start < out_len && ctx.scratch.buffer_alt[start] == ' ') {
        start += 1;
      }
      size_t end = out_len;
      while (end > start && ctx.scratch.buffer_alt[end - 1] == ' ') {
        end -= 1;
      }
      out_len = end - start;
      if (out_len > 0 && start > 0) {
        std::memmove(ctx.scratch.buffer_alt.data(),
                     ctx.scratch.buffer_alt.data() + start,
                     out_len);
      }
    }
    current = std::string_view(ctx.scratch.buffer_alt.data(), out_len);
  }

  if (vocab.add_space_prefix && !current.empty() && current.front() != ' ') {
    if (current.size() + 1 > ctx.scratch.buffer.size()) {
      return false;
    }
    ctx.scratch.buffer[0] = ' ';
    std::memcpy(ctx.scratch.buffer.data() + 1, current.data(), current.size());
    current = std::string_view(ctx.scratch.buffer.data(), current.size() + 1);
  }

  if (vocab.escape_whitespaces) {
    size_t out_len = 0;
    for (const char c : current) {
      if (c == ' ') {
        if (out_len + 3 > ctx.scratch.buffer_alt.size()) {
          return false;
        }
        ctx.scratch.buffer_alt[out_len++] = '\xE2';
        ctx.scratch.buffer_alt[out_len++] = '\x96';
        ctx.scratch.buffer_alt[out_len++] = '\x81';
      } else {
        if (out_len + 1 > ctx.scratch.buffer_alt.size()) {
          return false;
        }
        ctx.scratch.buffer_alt[out_len++] = c;
      }
    }
    current = std::string_view(ctx.scratch.buffer_alt.data(), out_len);
  }

  out_view = current;
  return true;
}

inline encode_result encode_ugm(const event::encode &ev,
                                emel::encoder::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::encoder::detail::ensure_tables(ctx);

  std::string_view normalized;
  if (!normalize_ugm_into(vocab, ctx, ev.text, normalized)) {
    result.error = EMEL_ERR_INVALID_ARGUMENT;
    return result;
  }

  int32_t count = 0;
  size_t pos = 0;
  while (pos < normalized.size()) {
    bool found = false;
    const size_t max_len = std::min(normalized.size() - pos,
                                    static_cast<size_t>(ctx.max_token_len));
    for (size_t len = max_len; len > 0; --len) {
      const std::string_view piece = normalized.substr(pos, len);
      const int32_t token = emel::encoder::detail::lookup_token(ctx, piece);
      if (token != k_token_null) {
        if (!emel::encoder::detail::push_token(ev, token, count)) {
          result.error = EMEL_ERR_INVALID_ARGUMENT;
          return result;
        }
        pos += len;
        found = true;
        break;
      }
    }
    if (!found) {
      int32_t unk = vocab.unk_id;
      if (unk == k_token_null) {
        unk = emel::encoder::detail::lookup_token(ctx, "<unk>");
      }
      if (unk != k_token_null) {
        if (!emel::encoder::detail::push_token(ev, unk, count)) {
          result.error = EMEL_ERR_INVALID_ARGUMENT;
          return result;
        }
      }
      pos += emel::encoder::detail::utf8_len(normalized[pos]);
    }
  }

  result.token_count = count;
  result.error = EMEL_OK;
  return result;
}

}  // namespace emel::encoder::ugm::detail
