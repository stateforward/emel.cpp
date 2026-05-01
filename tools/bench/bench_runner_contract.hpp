#pragma once

#include "bench_common.hpp"

#include <charconv>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace emel::bench {

enum class runner_mode {
  emel,
  reference,
  compare,
  kernel_emel,
  kernel_reference,
  kernel_compare,
};

struct runner_request {
  runner_mode mode = runner_mode::emel;
  config cfg = {};
  std::string suite = {};
  bool generation_jsonl = false;
  bool diarization_jsonl = false;
};

struct runner_result {
  std::int32_t exit_code = 0;
  std::string error_kind = {};
  std::string error_message = {};
};

inline std::string_view runner_mode_name(const runner_mode mode) noexcept {
  switch (mode) {
    case runner_mode::emel:
      return "emel";
    case runner_mode::reference:
      return "reference";
    case runner_mode::compare:
      return "compare";
    case runner_mode::kernel_emel:
      return "kernel-emel";
    case runner_mode::kernel_reference:
      return "kernel-reference";
    case runner_mode::kernel_compare:
      return "kernel-compare";
  }
  return "emel";
}

inline bool parse_runner_mode_name(const std::string_view text, runner_mode & out) noexcept {
  if (text == "emel") {
    out = runner_mode::emel;
    return true;
  }
  if (text == "reference") {
    out = runner_mode::reference;
    return true;
  }
  if (text == "compare") {
    out = runner_mode::compare;
    return true;
  }
  if (text == "kernel-emel") {
    out = runner_mode::kernel_emel;
    return true;
  }
  if (text == "kernel-reference") {
    out = runner_mode::kernel_reference;
    return true;
  }
  if (text == "kernel-compare") {
    out = runner_mode::kernel_compare;
    return true;
  }
  return false;
}

inline bool parse_runner_mode_arg(const std::string_view arg, runner_mode & out) noexcept {
  constexpr std::string_view prefix = "--mode=";
  if (arg.size() <= prefix.size() || arg.substr(0u, prefix.size()) != prefix) {
    return false;
  }
  return parse_runner_mode_name(arg.substr(prefix.size()), out);
}

inline bool parse_runner_u64(const std::string_view text, std::uint64_t & out) noexcept {
  if (text.empty()) {
    return false;
  }
  std::uint64_t value = 0u;
  const char * begin = text.data();
  const char * end = text.data() + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  if (parsed.ec != std::errc{} || parsed.ptr != end) {
    return false;
  }
  out = value;
  return true;
}

inline bool parse_runner_i32(const std::string_view text, std::int32_t & out) noexcept {
  if (text.empty()) {
    return false;
  }
  std::int32_t value = 0;
  const char * begin = text.data();
  const char * end = text.data() + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  if (parsed.ec != std::errc{} || parsed.ptr != end) {
    return false;
  }
  out = value;
  return true;
}

inline bool parse_runner_size(const std::string_view text, std::size_t & out) noexcept {
  std::uint64_t parsed = 0u;
  if (!parse_runner_u64(text, parsed) ||
      parsed > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  out = static_cast<std::size_t>(parsed);
  return true;
}

inline bool parse_runner_bool(const std::string_view text, bool & out) noexcept {
  if (text == "1") {
    out = true;
    return true;
  }
  if (text == "0") {
    out = false;
    return true;
  }
  return false;
}

inline void append_runner_contract_line(std::string & out,
                                        const std::string_view key,
                                        const std::string_view value) {
  out.append(key);
  out.push_back('=');
  out.append(value);
  out.push_back('\n');
}

inline void append_runner_contract_line(std::string & out,
                                        const std::string_view key,
                                        const std::uint64_t value) {
  append_runner_contract_line(out, key, std::to_string(value));
}

inline void append_runner_contract_line(std::string & out,
                                        const std::string_view key,
                                        const std::int32_t value) {
  append_runner_contract_line(out, key, std::to_string(value));
}

inline std::string serialize_runner_request(const runner_request & request) {
  std::string out = {};
  append_runner_contract_line(out, "schema", "bench_runner_request/v1");
  append_runner_contract_line(out, "mode", runner_mode_name(request.mode));
  append_runner_contract_line(out, "suite", request.suite);
  append_runner_contract_line(out, "iterations", request.cfg.iterations);
  append_runner_contract_line(out, "runs", static_cast<std::uint64_t>(request.cfg.runs));
  append_runner_contract_line(out, "warmup_iterations", request.cfg.warmup_iterations);
  append_runner_contract_line(out,
                              "warmup_runs",
                              static_cast<std::uint64_t>(request.cfg.warmup_runs));
  append_runner_contract_line(out, "generation_jsonl", request.generation_jsonl ? "1" : "0");
  append_runner_contract_line(out, "diarization_jsonl", request.diarization_jsonl ? "1" : "0");
  return out;
}

inline std::string serialize_runner_result(const runner_result & result) {
  std::string out = {};
  append_runner_contract_line(out, "schema", "bench_runner_result/v1");
  append_runner_contract_line(out, "exit_code", result.exit_code);
  append_runner_contract_line(out, "error_kind", result.error_kind);
  append_runner_contract_line(out, "error_message", result.error_message);
  return out;
}

inline bool parse_runner_request_line(const std::string_view key,
                                      const std::string_view value,
                                      runner_request & out,
                                      bool & saw_schema,
                                      bool & saw_mode,
                                      bool & saw_iterations,
                                      bool & saw_runs,
                                      bool & saw_warmup_iterations,
                                      bool & saw_warmup_runs,
                                      bool & saw_generation_jsonl,
                                      bool & saw_diarization_jsonl) noexcept {
  if (key == "schema") {
    saw_schema = value == "bench_runner_request/v1";
    return saw_schema;
  }
  if (key == "mode") {
    saw_mode = parse_runner_mode_name(value, out.mode);
    return saw_mode;
  }
  if (key == "suite") {
    out.suite = std::string{value};
    return true;
  }
  if (key == "iterations") {
    saw_iterations = parse_runner_u64(value, out.cfg.iterations);
    return saw_iterations;
  }
  if (key == "runs") {
    saw_runs = parse_runner_size(value, out.cfg.runs);
    return saw_runs;
  }
  if (key == "warmup_iterations") {
    saw_warmup_iterations = parse_runner_u64(value, out.cfg.warmup_iterations);
    return saw_warmup_iterations;
  }
  if (key == "warmup_runs") {
    saw_warmup_runs = parse_runner_size(value, out.cfg.warmup_runs);
    return saw_warmup_runs;
  }
  if (key == "generation_jsonl") {
    saw_generation_jsonl = parse_runner_bool(value, out.generation_jsonl);
    return saw_generation_jsonl;
  }
  if (key == "diarization_jsonl") {
    saw_diarization_jsonl = parse_runner_bool(value, out.diarization_jsonl);
    return saw_diarization_jsonl;
  }
  return false;
}

inline bool parse_runner_request(const std::string_view text, runner_request & out) {
  bool saw_schema = false;
  bool saw_mode = false;
  bool saw_iterations = false;
  bool saw_runs = false;
  bool saw_warmup_iterations = false;
  bool saw_warmup_runs = false;
  bool saw_generation_jsonl = false;
  bool saw_diarization_jsonl = false;

  std::size_t cursor = 0u;
  while (cursor < text.size()) {
    const std::size_t end = text.find('\n', cursor);
    const std::string_view line =
      end == std::string_view::npos ? text.substr(cursor) : text.substr(cursor, end - cursor);
    cursor = end == std::string_view::npos ? text.size() : end + 1u;
    if (line.empty()) {
      continue;
    }
    const std::size_t separator = line.find('=');
    if (separator == std::string_view::npos) {
      return false;
    }
    if (!parse_runner_request_line(line.substr(0u, separator),
                                   line.substr(separator + 1u),
                                   out,
                                   saw_schema,
                                   saw_mode,
                                   saw_iterations,
                                   saw_runs,
                                   saw_warmup_iterations,
                                   saw_warmup_runs,
                                   saw_generation_jsonl,
                                   saw_diarization_jsonl)) {
      return false;
    }
  }

  return saw_schema && saw_mode && saw_iterations && saw_runs && saw_warmup_iterations &&
    saw_warmup_runs && saw_generation_jsonl && saw_diarization_jsonl;
}

inline bool parse_runner_result_line(const std::string_view key,
                                     const std::string_view value,
                                     runner_result & out,
                                     bool & saw_schema,
                                     bool & saw_exit_code) noexcept {
  if (key == "schema") {
    saw_schema = value == "bench_runner_result/v1";
    return saw_schema;
  }
  if (key == "exit_code") {
    saw_exit_code = parse_runner_i32(value, out.exit_code);
    return saw_exit_code;
  }
  if (key == "error_kind") {
    out.error_kind = std::string{value};
    return true;
  }
  if (key == "error_message") {
    out.error_message = std::string{value};
    return true;
  }
  return false;
}

inline bool parse_runner_result(const std::string_view text, runner_result & out) {
  bool saw_schema = false;
  bool saw_exit_code = false;

  std::size_t cursor = 0u;
  while (cursor < text.size()) {
    const std::size_t end = text.find('\n', cursor);
    const std::string_view line =
      end == std::string_view::npos ? text.substr(cursor) : text.substr(cursor, end - cursor);
    cursor = end == std::string_view::npos ? text.size() : end + 1u;
    if (line.empty()) {
      continue;
    }
    const std::size_t separator = line.find('=');
    if (separator == std::string_view::npos) {
      return false;
    }
    if (!parse_runner_result_line(line.substr(0u, separator),
                                  line.substr(separator + 1u),
                                  out,
                                  saw_schema,
                                  saw_exit_code)) {
      return false;
    }
  }

  return saw_schema && saw_exit_code;
}

}  // namespace emel::bench
