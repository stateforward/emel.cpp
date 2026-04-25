#pragma once

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace emel::bench {

inline constexpr char k_generation_compare_format_env[] = "EMEL_GENERATION_BENCH_FORMAT";
inline constexpr char k_generation_compare_result_dir_env[] = "EMEL_GENERATION_RESULT_DIR";
inline constexpr char k_generation_compare_schema[] = "generation_compare/v1";

struct generation_compare_record {
  std::string record_type = "result";
  std::string status = "ok";
  std::string case_name = {};
  std::string compare_group = {};
  std::string lane = {};
  std::string backend_id = {};
  std::string backend_language = {};
  std::string workload_id = {};
  std::string workload_manifest_path = {};
  std::string comparison_mode = {};
  std::string model_id = {};
  std::string fixture_id = {};
  std::string prompt_fixture_id = {};
  std::string prompt_fixture_path = {};
  std::string prompt_id = {};
  std::string formatter_mode = {};
  std::string formatter_contract = {};
  std::string sampling_id = {};
  std::string stop_id = {};
  std::int64_t seed = 0;
  std::uint64_t max_output_tokens = 0u;
  bool comparable = false;
  double ns_per_op = 0.0;
  double ns_min_per_op = 0.0;
  double ns_mean_per_op = 0.0;
  double ns_max_per_op = 0.0;
  double prepare_ns_per_op = 0.0;
  double encode_ns_per_op = 0.0;
  double publish_ns_per_op = 0.0;
  std::uint64_t output_tokens = 0u;
  std::uint64_t output_bytes = 0u;
  std::uint64_t output_checksum = 0u;
  std::uint64_t iterations = 0u;
  std::size_t runs = 0u;
  std::string output_text = {};
  std::string output_path = {};
  std::string note = {};
  std::string error_kind = {};
  std::string error_message = {};
};

inline bool generation_compare_emit_jsonl() {
  const char * format = std::getenv(k_generation_compare_format_env);
  return format != nullptr && std::string_view{format} == "jsonl";
}

inline std::string generation_compare_json_escape(std::string_view input) {
  std::string out = {};
  out.reserve(input.size() + 8u);
  for (const char ch : input) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(ch);
        break;
    }
  }
  return out;
}

inline std::string generation_compare_sanitize_case_name(std::string_view input) {
  std::string out = {};
  out.reserve(input.size());
  for (const char ch : input) {
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) {
      out.push_back(ch);
      continue;
    }
    out.push_back('_');
  }
  return out;
}

inline void maybe_dump_generation_output(generation_compare_record & record) {
  const char * output_dir = std::getenv(k_generation_compare_result_dir_env);
  if (output_dir == nullptr || output_dir[0] == '\0' || record.output_text.empty()) {
    return;
  }

  const std::filesystem::path root(output_dir);
  std::error_code ec = {};
  std::filesystem::create_directories(root, ec);
  if (ec) {
    std::fprintf(stderr,
                 "warning: failed to create generation compare output dir %s: %s\n",
                 root.string().c_str(),
                 ec.message().c_str());
    return;
  }

  const std::string backend = generation_compare_sanitize_case_name(record.backend_id);
  const std::string case_name = generation_compare_sanitize_case_name(record.case_name);
  const std::filesystem::path output_path = root / (backend + "__" + case_name + ".txt");
  std::ofstream output(output_path, std::ios::binary);
  if (!output.good()) {
    std::fprintf(stderr,
                 "warning: failed to open generation compare output file %s\n",
                 output_path.string().c_str());
    return;
  }

  output.write(record.output_text.data(),
               static_cast<std::streamsize>(record.output_text.size()));
  if (!output.good()) {
    std::fprintf(stderr,
                 "warning: failed to write generation compare output file %s\n",
                 output_path.string().c_str());
    return;
  }

  record.output_path = output_path.string();
}

inline void print_generation_compare_record_jsonl(const generation_compare_record & record) {
  std::printf(
    "{\"schema\":\"%s\",\"record_type\":\"%s\",\"status\":\"%s\",\"case_name\":\"%s\","
    "\"compare_group\":\"%s\",\"lane\":\"%s\",\"backend_id\":\"%s\","
    "\"backend_language\":\"%s\",\"workload_id\":\"%s\",\"workload_manifest_path\":\"%s\","
    "\"comparison_mode\":\"%s\",\"model_id\":\"%s\",\"fixture_id\":\"%s\","
    "\"prompt_fixture_id\":\"%s\",\"prompt_fixture_path\":\"%s\",\"prompt_id\":\"%s\","
    "\"formatter_mode\":\"%s\",\"formatter_contract\":\"%s\","
    "\"sampling_id\":\"%s\",\"stop_id\":\"%s\",\"seed\":%" PRId64 ","
    "\"comparable\":%s,"
    "\"max_output_tokens\":%" PRIu64 ",\"ns_per_op\":%.6f,"
    "\"ns_min_per_op\":%.6f,\"ns_mean_per_op\":%.6f,\"ns_max_per_op\":%.6f,"
    "\"prepare_ns_per_op\":%.6f,\"encode_ns_per_op\":%.6f,\"publish_ns_per_op\":%.6f,"
    "\"output_tokens\":%" PRIu64 ",\"output_bytes\":%" PRIu64 ","
    "\"output_checksum\":%" PRIu64 ",\"iterations\":%" PRIu64 ",\"runs\":%zu,"
    "\"output_path\":\"%s\",\"note\":\"%s\",\"error_kind\":\"%s\","
    "\"error_message\":\"%s\"}\n",
    k_generation_compare_schema,
    generation_compare_json_escape(record.record_type).c_str(),
    generation_compare_json_escape(record.status).c_str(),
    generation_compare_json_escape(record.case_name).c_str(),
    generation_compare_json_escape(record.compare_group).c_str(),
    generation_compare_json_escape(record.lane).c_str(),
    generation_compare_json_escape(record.backend_id).c_str(),
    generation_compare_json_escape(record.backend_language).c_str(),
    generation_compare_json_escape(record.workload_id).c_str(),
    generation_compare_json_escape(record.workload_manifest_path).c_str(),
    generation_compare_json_escape(record.comparison_mode).c_str(),
    generation_compare_json_escape(record.model_id).c_str(),
    generation_compare_json_escape(record.fixture_id).c_str(),
    generation_compare_json_escape(record.prompt_fixture_id).c_str(),
    generation_compare_json_escape(record.prompt_fixture_path).c_str(),
    generation_compare_json_escape(record.prompt_id).c_str(),
    generation_compare_json_escape(record.formatter_mode).c_str(),
    generation_compare_json_escape(record.formatter_contract).c_str(),
    generation_compare_json_escape(record.sampling_id).c_str(),
    generation_compare_json_escape(record.stop_id).c_str(),
    record.seed,
    record.comparable ? "true" : "false",
    record.max_output_tokens,
    record.ns_per_op,
    record.ns_min_per_op,
    record.ns_mean_per_op,
    record.ns_max_per_op,
    record.prepare_ns_per_op,
    record.encode_ns_per_op,
    record.publish_ns_per_op,
    record.output_tokens,
    record.output_bytes,
    record.output_checksum,
    record.iterations,
    record.runs,
    generation_compare_json_escape(record.output_path).c_str(),
    generation_compare_json_escape(record.note).c_str(),
    generation_compare_json_escape(record.error_kind).c_str(),
    generation_compare_json_escape(record.error_message).c_str());
}

}  // namespace emel::bench
