#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "emel/emel.h"

namespace emel::jinja {

enum class token_type : uint8_t {
  eof = 0,
  text,
  numeric_literal,
  string_literal,
  identifier,
  equals,
  open_paren,
  close_paren,
  open_statement,
  close_statement,
  open_expression,
  close_expression,
  open_square_bracket,
  close_square_bracket,
  open_curly_bracket,
  close_curly_bracket,
  comma,
  dot,
  colon,
  pipe,
  call_operator,
  additive_binary_operator,
  multiplicative_binary_operator,
  comparison_binary_operator,
  unary_operator,
  comment
};

struct token {
  token_type type = token_type::eof;
  std::string value;
  size_t pos = 0;
};

struct lexer_result {
  std::vector<token> tokens;
  std::string source;
  int32_t error = EMEL_OK;
  size_t error_pos = 0;
};

struct lexer {
  lexer_result tokenize(std::string_view source) const {
    lexer_result result{};
    result.source = std::string(source);
    std::string & src = result.source;

    if (src.empty()) {
      return result;
    }

    bool ok = true;
    auto set_error = [&](int32_t code, size_t pos) {
      if (!ok) {
        return;
      }
      ok = false;
      result.error = code;
      result.error_pos = pos;
    };

    auto string_lstrip = [](std::string & s, const char * chars) {
      size_t start = s.find_first_not_of(chars);
      if (start == std::string::npos) {
        s.clear();
      } else {
        s.erase(0, start);
      }
    };

    auto string_rstrip = [](std::string & s, const char * chars) {
      size_t end = s.find_last_not_of(chars);
      if (end == std::string::npos) {
        s.clear();
      } else {
        s.erase(end + 1);
      }
    };

    auto decode_escape = [&](char ch, char & out) -> bool {
      switch (ch) {
        case 'n':
          out = '\n';
          return true;
        case 't':
          out = '\t';
          return true;
        case 'r':
          out = '\r';
          return true;
        case 'b':
          out = '\b';
          return true;
        case 'f':
          out = '\f';
          return true;
        case 'v':
          out = '\v';
          return true;
        case '\\':
          out = '\\';
          return true;
        case '\'':
          out = '\'';
          return true;
        case '"':
          out = '"';
          return true;
        default:
          return false;
      }
    };

    auto is_word = [](char ch) -> bool {
      return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
    };

    auto is_integer = [](char ch) -> bool {
      return std::isdigit(static_cast<unsigned char>(ch)) != 0;
    };

    struct mapping {
      std::string_view seq;
      token_type type;
    };

    static constexpr mapping k_mapping_table[] = {
      {"{%-", token_type::open_statement},
      {"-%}", token_type::close_statement},
      {"{{-", token_type::open_expression},
      {"-}}", token_type::close_expression},
      {"{%", token_type::open_statement},
      {"%}", token_type::close_statement},
      {"{{", token_type::open_expression},
      {"}}", token_type::close_expression},
      {"(", token_type::open_paren},
      {")", token_type::close_paren},
      {"{", token_type::open_curly_bracket},
      {"}", token_type::close_curly_bracket},
      {"[", token_type::open_square_bracket},
      {"]", token_type::close_square_bracket},
      {",", token_type::comma},
      {".", token_type::dot},
      {":", token_type::colon},
      {"|", token_type::pipe},
      {"<=", token_type::comparison_binary_operator},
      {">=", token_type::comparison_binary_operator},
      {"==", token_type::comparison_binary_operator},
      {"!=", token_type::comparison_binary_operator},
      {"<", token_type::comparison_binary_operator},
      {">", token_type::comparison_binary_operator},
      {"+", token_type::additive_binary_operator},
      {"-", token_type::additive_binary_operator},
      {"~", token_type::additive_binary_operator},
      {"*", token_type::multiplicative_binary_operator},
      {"/", token_type::multiplicative_binary_operator},
      {"%", token_type::multiplicative_binary_operator},
      {"=", token_type::equals},
    };

    // normalize \r\n or \r to \n
    for (std::string::size_type pos = 0;
         (pos = src.find("\r\n", pos)) != std::string::npos; ) {
      src.erase(pos, 1);
      ++pos;
    }
    for (std::string::size_type pos = 0;
         (pos = src.find('\r', pos)) != std::string::npos; ) {
      src.replace(pos, 1, 1, '\n');
      ++pos;
    }

    if (!src.empty() && src.back() == '\n') {
      src.pop_back();
    }

    size_t pos = 0;
    size_t start_pos = 0;
    size_t curly_bracket_depth = 0;

    auto next_pos_is = [&](std::initializer_list<char> chars, size_t n = 1) -> bool {
      size_t idx = pos + n;
      if (idx >= src.size()) {
        return false;
      }
      for (char c : chars) {
        if (src[idx] == c) {
          return true;
        }
      }
      return false;
    };

    auto consume_while = [&](const auto & predicate) -> std::string {
      std::string out;
      while (pos < src.size() && predicate(src[pos])) {
        if (src[pos] == '\\') {
          ++pos;
          if (pos >= src.size()) {
            set_error(EMEL_ERR_PARSE_FAILED, pos);
            return out;
          }
          char escaped = src[pos++];
          char decoded = '\0';
          if (!decode_escape(escaped, decoded)) {
            set_error(EMEL_ERR_PARSE_FAILED, pos);
            return out;
          }
          out += decoded;
          continue;
        }
        out += src[pos++];
      }
      return out;
    };

    auto consume_numeric = [&]() -> std::string {
      std::string num = consume_while(is_integer);
      if (!ok) {
        return num;
      }
      if (pos < src.size() && src[pos] == '.' && pos + 1 < src.size() &&
          is_integer(src[pos + 1])) {
        ++pos;
        std::string frac = consume_while(is_integer);
        num += ".";
        num += frac;
      }
      return num;
    };

    bool opt_lstrip_blocks = true;
    bool opt_trim_blocks = true;
    bool is_lstrip_block = false;
    bool is_rstrip_block = false;

    std::vector<token> tokens;

    while (pos < src.size() && ok) {
      start_pos = pos;
      token_type last_token_type = tokens.empty()
                                      ? token_type::close_statement
                                      : tokens.back().type;

      if (last_token_type == token_type::close_statement ||
          last_token_type == token_type::close_expression ||
          last_token_type == token_type::comment) {
        bool last_block_can_rm_newline = false;
        is_rstrip_block = false;
        if (pos > 3) {
          char c0 = src[pos - 3];
          char c1 = src[pos - 2];
          char c2 = src[pos - 1];
          is_rstrip_block = c0 == '-' && (c1 == '%' || c1 == '}' || c1 == '#') && c2 == '}';
          last_block_can_rm_newline = (c1 == '#' || c1 == '%' || c1 == '-') && c2 == '}';
        }

        size_t start = pos;
        size_t end = start;
        while (pos < src.size() && !(src[pos] == '{' && next_pos_is({'%', '{', '#'}))) {
          end = ++pos;
        }

        if (opt_lstrip_blocks && pos < src.size() && src[pos] == '{' &&
            next_pos_is({'%', '#', '-'})) {
          size_t current = end;
          while (current > start) {
            char c = src[current - 1];
            if (current == 1) {
              end = 0;
              break;
            }
            if (c == '\n') {
              end = current;
              break;
            }
            if (!std::isspace(static_cast<unsigned char>(c))) {
              break;
            }
            --current;
          }
        }

        std::string text = src.substr(start, end - start);

        if (opt_trim_blocks && last_block_can_rm_newline) {
          if (!text.empty() && text.front() == '\n') {
            text.erase(text.begin());
          }
        }

        if (is_rstrip_block) {
          string_lstrip(text, " \t\r\n");
        }

        is_lstrip_block =
          pos < src.size() && src[pos] == '{' && next_pos_is({'{', '%', '#'}) &&
          next_pos_is({'-'}, 2);
        if (is_lstrip_block) {
          string_rstrip(text, " \t\r\n");
        }

        if (!text.empty()) {
          tokens.push_back({token_type::text, text, start_pos});
          continue;
        }
      }

      if (pos < src.size() && src[pos] == '{' && next_pos_is({'#'})) {
        start_pos = pos;
        pos += 2;
        std::string comment;
        while (pos < src.size() &&
               !(src[pos] == '#' && next_pos_is({'}'}))) {
          if (pos + 2 >= src.size()) {
            set_error(EMEL_ERR_PARSE_FAILED, pos);
            break;
          }
          comment += src[pos++];
        }
        if (!ok) {
          break;
        }
        tokens.push_back({token_type::comment, comment, start_pos});
        pos += 2;
        continue;
      }

      if (pos < src.size() && src[pos] == '-' &&
          (last_token_type == token_type::open_expression ||
           last_token_type == token_type::open_statement)) {
        ++pos;
        if (pos >= src.size()) {
          break;
        }
      }

      consume_while([](char c) {
        return std::isspace(static_cast<unsigned char>(c)) != 0;
      });
      if (!ok) {
        break;
      }
      if (pos >= src.size()) {
        break;
      }

      char ch = src[pos];
      bool is_closing_block = ch == '-' && next_pos_is({'%', '}'});

      if (!is_closing_block && (ch == '-' || ch == '+')) {
        start_pos = pos;
      token_type last = tokens.empty() ? token_type::eof : tokens.back().type;
        if (last == token_type::text || last == token_type::eof) {
          set_error(EMEL_ERR_PARSE_FAILED, pos);
          break;
        }
        switch (last) {
          case token_type::identifier:
          case token_type::numeric_literal:
          case token_type::string_literal:
          case token_type::close_paren:
          case token_type::close_square_bracket:
            break;
          default: {
            ++pos;
            std::string num = consume_numeric();
            if (!ok) {
              break;
            }
            std::string value;
            value.reserve(num.size() + 1);
            value.push_back(ch);
            value += num;
            token_type t = num.empty() ? token_type::unary_operator : token_type::numeric_literal;
            tokens.push_back({t, std::move(value), start_pos});
            continue;
          }
        }
      }

      bool matched = false;
      for (const auto & entry : k_mapping_table) {
        start_pos = pos;
        if (entry.seq == "}}" && curly_bracket_depth > 0) {
          continue;
        }
        if (pos + entry.seq.size() <= src.size() &&
            src.compare(pos, entry.seq.size(), entry.seq) == 0) {
          tokens.push_back({entry.type, std::string(entry.seq), start_pos});
        if (entry.type == token_type::open_expression) {
          curly_bracket_depth = 0;
        } else if (entry.type == token_type::open_curly_bracket) {
          ++curly_bracket_depth;
        } else if (entry.type == token_type::close_curly_bracket) {
          if (curly_bracket_depth > 0) {
            --curly_bracket_depth;
          }
          }
          pos += entry.seq.size();
          matched = true;
          break;
        }
      }
      if (matched) {
        continue;
      }

      if (ch == '\'' || ch == '"') {
        start_pos = pos;
        ++pos;
        std::string str = consume_while([ch](char c) { return c != ch; });
        if (!ok) {
          break;
        }
      tokens.push_back({token_type::string_literal, std::move(str), start_pos});
        if (pos < src.size()) {
          ++pos;
        } else {
          set_error(EMEL_ERR_PARSE_FAILED, pos);
          break;
        }
        continue;
      }

      if (is_integer(ch)) {
        start_pos = pos;
        std::string num = consume_numeric();
        if (!ok) {
          break;
        }
      tokens.push_back({token_type::numeric_literal, std::move(num), start_pos});
        continue;
      }

      if (is_word(ch)) {
        start_pos = pos;
        std::string word = consume_while(is_word);
        if (!ok) {
          break;
        }
      tokens.push_back({token_type::identifier, std::move(word), start_pos});
        continue;
      }

      set_error(EMEL_ERR_PARSE_FAILED, pos);
      break;
    }

    if (!ok) {
      result.tokens = std::move(tokens);
      return result;
    }

    result.tokens = std::move(tokens);
    return result;
  }
};

} // namespace emel::jinja
