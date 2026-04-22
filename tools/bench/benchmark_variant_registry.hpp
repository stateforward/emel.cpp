#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace emel::bench {

inline std::string read_benchmark_manifest_file(const std::filesystem::path & path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  return std::string{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

inline std::size_t skip_benchmark_json_ws(const std::string & text, std::size_t cursor) {
  while (cursor < text.size() &&
         std::isspace(static_cast<unsigned char>(text[cursor])) != 0) {
    ++cursor;
  }
  return cursor;
}

inline bool parse_benchmark_json_string_token(const std::string & text,
                                              std::size_t cursor,
                                              std::string & out,
                                              std::size_t * end_out = nullptr) {
  cursor = skip_benchmark_json_ws(text, cursor);
  if (cursor >= text.size() || text[cursor] != '"') {
    return false;
  }

  ++cursor;
  std::string parsed = {};
  while (cursor < text.size()) {
    const char ch = text[cursor++];
    if (ch == '"') {
      out = std::move(parsed);
      if (end_out != nullptr) {
        *end_out = cursor;
      }
      return true;
    }
    if (ch != '\\') {
      parsed.push_back(ch);
      continue;
    }
    if (cursor >= text.size()) {
      return false;
    }
    const char escaped = text[cursor++];
    switch (escaped) {
      case '\\':
      case '"':
      case '/':
        parsed.push_back(escaped);
        break;
      case 'b':
        parsed.push_back('\b');
        break;
      case 'f':
        parsed.push_back('\f');
        break;
      case 'n':
        parsed.push_back('\n');
        break;
      case 'r':
        parsed.push_back('\r');
        break;
      case 't':
        parsed.push_back('\t');
        break;
      default:
        return false;
    }
  }
  return false;
}

inline bool locate_benchmark_json_value(const std::string & text,
                                        const std::string_view key,
                                        std::size_t & value_pos) {
  int object_depth = 0;
  int array_depth = 0;
  std::size_t cursor = 0u;
  while (cursor < text.size()) {
    const char ch = text[cursor];
    if (ch == '"') {
      const std::size_t token_start = cursor;
      std::string parsed_key = {};
      std::size_t token_end = 0u;
      if (!parse_benchmark_json_string_token(text, token_start, parsed_key, &token_end)) {
        return false;
      }
      const std::size_t colon_pos = skip_benchmark_json_ws(text, token_end);
      if (object_depth == 1 && array_depth == 0 && colon_pos < text.size() &&
          text[colon_pos] == ':' && parsed_key == key) {
        value_pos = skip_benchmark_json_ws(text, colon_pos + 1u);
        return value_pos < text.size();
      }
      cursor = token_end;
      continue;
    }
    if (ch == '{') {
      ++object_depth;
      ++cursor;
      continue;
    }
    if (ch == '}') {
      if (object_depth > 0) {
        --object_depth;
      }
      ++cursor;
      continue;
    }
    if (ch == '[') {
      ++array_depth;
      ++cursor;
      continue;
    }
    if (ch == ']') {
      if (array_depth > 0) {
        --array_depth;
      }
      ++cursor;
      continue;
    }
    ++cursor;
  }
  return false;
}

inline bool extract_benchmark_json_string(const std::string & text,
                                          const std::string_view key,
                                          std::string & out) {
  std::size_t value_pos = 0u;
  return locate_benchmark_json_value(text, key, value_pos) &&
      parse_benchmark_json_string_token(text, value_pos, out);
}

inline bool extract_benchmark_json_bool(const std::string & text,
                                        const std::string_view key,
                                        bool & out) {
  std::size_t value_pos = 0u;
  if (!locate_benchmark_json_value(text, key, value_pos)) {
    return false;
  }
  if (text.compare(value_pos, 4u, "true") == 0) {
    out = true;
    return true;
  }
  if (text.compare(value_pos, 5u, "false") == 0) {
    out = false;
    return true;
  }
  return false;
}

inline bool discover_benchmark_manifest_paths(const std::filesystem::path & directory,
                                              std::vector<std::filesystem::path> & out,
                                              std::string * error_out = nullptr) {
  out.clear();
  std::error_code ec = {};
  if (!std::filesystem::exists(directory, ec) || ec) {
    if (error_out != nullptr) {
      *error_out = "manifest directory missing: " + directory.string();
    }
    return false;
  }
  if (!std::filesystem::is_directory(directory, ec) || ec) {
    if (error_out != nullptr) {
      *error_out = "manifest path is not a directory: " + directory.string();
    }
    return false;
  }
  std::filesystem::recursive_directory_iterator iter(directory, ec);
  if (ec) {
    if (error_out != nullptr) {
      *error_out = "failed to read manifest directory: " + directory.string();
    }
    return false;
  }
  for (const auto & entry : iter) {
    if (ec) {
      if (error_out != nullptr) {
        *error_out = "failed to read manifest directory: " + directory.string();
      }
      return false;
    }
    if (!entry.is_regular_file(ec) || ec || entry.path().extension() != ".json") {
      ec.clear();
      continue;
    }
    const std::filesystem::path rel = std::filesystem::relative(entry.path(), directory, ec);
    if (ec) {
      if (error_out != nullptr) {
        *error_out = "failed to resolve manifest path: " + entry.path().string();
      }
      return false;
    }
    if (!rel.has_parent_path()) {
      if (error_out != nullptr) {
        *error_out = "manifest must live in an isolation subdirectory: " + entry.path().string();
      }
      return false;
    }
    out.push_back(entry.path());
  }
  std::sort(out.begin(), out.end(), [](const auto & lhs, const auto & rhs) {
    return lhs.generic_string() < rhs.generic_string();
  });
  return true;
}

inline bool validate_benchmark_manifest_ids(
    const std::vector<std::pair<std::string, std::filesystem::path>> & ids,
    std::string * error_out = nullptr) {
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (ids[i].first.empty()) {
      if (error_out != nullptr) {
        *error_out = "manifest id missing: " + ids[i].second.string();
      }
      return false;
    }
    for (std::size_t j = i + 1u; j < ids.size(); ++j) {
      if (ids[i].first != ids[j].first) {
        continue;
      }
      if (error_out != nullptr) {
        *error_out = "duplicate manifest id: " + ids[i].first;
      }
      return false;
    }
  }
  return true;
}

}  // namespace emel::bench
