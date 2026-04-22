#pragma once

#include "benchmark_variant_registry.hpp"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace emel::bench {

inline constexpr char k_generation_prompt_fixture_schema[] = "generation_prompt_fixture/v1";
inline constexpr char k_generation_workload_schema[] = "generation_workload/v1";

struct generation_prompt_fixture {
  std::string id = {};
  std::string shape = {};
  std::string text = {};
  std::string prompt_id = {};
};

struct generation_workload_manifest {
  std::string id = {};
  std::string workload_manifest_path = {};
  std::string case_name = {};
  std::string compare_group = {};
  std::string fixture_name = {};
  std::string fixture_rel = {};
  std::string fixture_slug = {};
  std::string prompt_fixture_id = {};
  std::string prompt_fixture_path = {};
  std::string prompt_shape = {};
  std::string prompt_text = {};
  std::string prompt_id = {};
  std::string formatter_mode = {};
  std::string formatter_contract = {};
  std::string sampling_id = {};
  std::string stop_id = {};
  std::string comparison_mode = {};
  std::string comparability_note = {};
  std::int64_t seed = 0;
  std::uint64_t max_output_tokens = 0u;
  bool comparable = false;
  bool current_publication = false;
};

inline std::string read_generation_manifest_file(const std::filesystem::path & path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  return std::string{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

inline std::size_t skip_generation_json_ws(const std::string & text, std::size_t cursor) {
  while (cursor < text.size() &&
         std::isspace(static_cast<unsigned char>(text[cursor])) != 0) {
    ++cursor;
  }
  return cursor;
}

inline bool parse_generation_json_string_token(const std::string & text,
                                               std::size_t cursor,
                                               std::string & out,
                                               std::size_t * end_out = nullptr) {
  cursor = skip_generation_json_ws(text, cursor);
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

inline bool locate_generation_json_value(const std::string & text,
                                         const std::string_view key,
                                         std::size_t & value_pos) {
  int object_depth = 0;
  int array_depth = 0;
  std::size_t cursor = 0u;
  while (cursor < text.size()) {
    const char ch = text[cursor];
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
    if (ch != '"') {
      ++cursor;
      continue;
    }

    const std::size_t token_start = cursor;
    ++cursor;
    bool escaped = false;
    bool closed = false;
    while (cursor < text.size()) {
      const char token_ch = text[cursor++];
      if (escaped) {
        escaped = false;
        continue;
      }
      if (token_ch == '\\') {
        escaped = true;
        continue;
      }
      if (token_ch == '"') {
        closed = true;
        break;
      }
    }
    if (!closed) {
      return false;
    }
    const std::size_t token_end = cursor;
    const std::size_t colon_pos = skip_generation_json_ws(text, token_end);
    if (object_depth == 1 && array_depth == 0 && colon_pos < text.size() &&
        text[colon_pos] == ':') {
      std::string parsed_key = {};
      if (!parse_generation_json_string_token(text, token_start, parsed_key)) {
        return false;
      }
      if (parsed_key == key) {
        value_pos = skip_generation_json_ws(text, colon_pos + 1u);
        return value_pos < text.size();
      }
    }
    cursor = token_end;
  }
  return false;
}

inline bool extract_generation_json_string(const std::string & text,
                                           const std::string_view key,
                                           std::string & out) {
  std::size_t value_pos = 0u;
  return locate_generation_json_value(text, key, value_pos) &&
      parse_generation_json_string_token(text, value_pos, out);
}

inline bool extract_generation_json_bool(const std::string & text,
                                         const std::string_view key,
                                         bool & out) {
  std::size_t value_pos = 0u;
  if (!locate_generation_json_value(text, key, value_pos)) {
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

inline bool extract_generation_json_i64(const std::string & text,
                                        const std::string_view key,
                                        std::int64_t & out) {
  std::size_t value_pos = 0u;
  if (!locate_generation_json_value(text, key, value_pos)) {
    return false;
  }
  char * end = nullptr;
  const long long parsed = std::strtoll(text.c_str() + value_pos, &end, 10);
  if (end == text.c_str() + value_pos) {
    return false;
  }
  out = static_cast<std::int64_t>(parsed);
  return true;
}

inline bool extract_generation_json_u64(const std::string & text,
                                        const std::string_view key,
                                        std::uint64_t & out) {
  std::size_t value_pos = 0u;
  if (!locate_generation_json_value(text, key, value_pos)) {
    return false;
  }
  char * end = nullptr;
  const unsigned long long parsed = std::strtoull(text.c_str() + value_pos, &end, 10);
  if (end == text.c_str() + value_pos) {
    return false;
  }
  out = static_cast<std::uint64_t>(parsed);
  return true;
}

inline bool load_generation_prompt_fixture(const std::filesystem::path & path,
                                           generation_prompt_fixture & out,
                                           std::string * error_out = nullptr) {
  const std::string text = read_generation_manifest_file(path);
  if (text.empty()) {
    if (error_out != nullptr) {
      *error_out = "prompt fixture file missing or unreadable";
    }
    return false;
  }

  std::string schema = {};
  if (!extract_generation_json_string(text, "schema", schema) ||
      schema != k_generation_prompt_fixture_schema ||
      !extract_generation_json_string(text, "id", out.id) ||
      !extract_generation_json_string(text, "shape", out.shape) ||
      !extract_generation_json_string(text, "text", out.text) ||
      !extract_generation_json_string(text, "prompt_id", out.prompt_id)) {
    if (error_out != nullptr) {
      *error_out = "invalid generation prompt fixture";
    }
    return false;
  }

  if (out.shape != "single_user_text_v1") {
    if (error_out != nullptr) {
      *error_out = "unsupported prompt fixture shape";
    }
    return false;
  }

  return true;
}

inline bool load_generation_workload_manifest(const std::filesystem::path & repo_root,
                                              const std::filesystem::path & manifest_path,
                                              generation_workload_manifest & out,
                                              std::string * error_out = nullptr) {
  const std::string text = read_generation_manifest_file(manifest_path);
  if (text.empty()) {
    if (error_out != nullptr) {
      *error_out = "generation workload manifest missing or unreadable";
    }
    return false;
  }

  std::string schema = {};
  if (!extract_generation_json_string(text, "schema", schema) ||
      schema != k_generation_workload_schema ||
      !extract_generation_json_string(text, "id", out.id) ||
      !extract_generation_json_string(text, "case_name", out.case_name) ||
      !extract_generation_json_string(text, "compare_group", out.compare_group) ||
      !extract_generation_json_string(text, "fixture_name", out.fixture_name) ||
      !extract_generation_json_string(text, "fixture_rel", out.fixture_rel) ||
      !extract_generation_json_string(text, "fixture_slug", out.fixture_slug) ||
      !extract_generation_json_string(text, "prompt_fixture_id", out.prompt_fixture_id) ||
      !extract_generation_json_string(text, "prompt_fixture_path", out.prompt_fixture_path) ||
      !extract_generation_json_string(text, "formatter_mode", out.formatter_mode) ||
      !extract_generation_json_string(text, "formatter_contract", out.formatter_contract) ||
      !extract_generation_json_string(text, "sampling_id", out.sampling_id) ||
      !extract_generation_json_string(text, "stop_id", out.stop_id) ||
      !extract_generation_json_string(text, "comparison_mode", out.comparison_mode) ||
      !extract_generation_json_string(text, "comparability_note", out.comparability_note) ||
      !extract_generation_json_i64(text, "seed", out.seed) ||
      !extract_generation_json_u64(text, "max_output_tokens", out.max_output_tokens) ||
      !extract_generation_json_bool(text, "comparable", out.comparable) ||
      !extract_generation_json_bool(text, "current_publication", out.current_publication)) {
    if (error_out != nullptr) {
      *error_out = "invalid generation workload manifest";
    }
    return false;
  }

  generation_prompt_fixture prompt = {};
  const std::filesystem::path prompt_path = repo_root / out.prompt_fixture_path;
  if (!load_generation_prompt_fixture(prompt_path, prompt, error_out)) {
    return false;
  }
  if (prompt.id != out.prompt_fixture_id) {
    if (error_out != nullptr) {
      *error_out = "prompt fixture id mismatch";
    }
    return false;
  }

  out.prompt_shape = prompt.shape;
  out.prompt_text = prompt.text;
  out.prompt_id = prompt.prompt_id;
  std::error_code ec = {};
  out.workload_manifest_path = std::filesystem::relative(manifest_path, repo_root, ec).string();
  if (ec) {
    out.workload_manifest_path = manifest_path.string();
  }
  return true;
}

inline bool load_generation_workload_manifests(const std::filesystem::path & repo_root,
                                               const std::filesystem::path & directory,
                                               std::vector<generation_workload_manifest> & out,
                                               std::string * error_out = nullptr) {
  out.clear();
  std::vector<std::filesystem::path> paths = {};
  if (!discover_benchmark_manifest_paths(directory, paths, error_out)) {
    return false;
  }

  std::vector<std::pair<std::string, std::filesystem::path>> ids = {};
  for (const auto & path : paths) {
    generation_workload_manifest manifest = {};
    if (!load_generation_workload_manifest(repo_root, path, manifest, error_out)) {
      return false;
    }
    ids.push_back({manifest.id, path});
    out.push_back(std::move(manifest));
  }
  return validate_benchmark_manifest_ids(ids, error_out);
}

inline bool load_generation_workload_manifests(const std::filesystem::path & repo_root,
                                               std::vector<generation_workload_manifest> & out,
                                               std::string * error_out = nullptr) {
  return load_generation_workload_manifests(
      repo_root, repo_root / "tools" / "bench" / "generation_variants", out, error_out);
}

}  // namespace emel::bench
