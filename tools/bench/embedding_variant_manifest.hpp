#pragma once

#include "benchmark_variant_registry.hpp"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace emel::bench {

inline constexpr char k_embedding_variant_schema[] = "embedding_variant/v1";

struct embedding_variant_manifest {
  std::string id = {};
  std::string case_name = {};
  std::string compare_group = {};
  std::string modality = {};
  std::string payload_id = {};
  std::string comparison_mode = {};
  std::string note = {};
  bool current_publication = false;
};

inline bool load_embedding_variant_manifest(const std::filesystem::path & path,
                                            embedding_variant_manifest & out,
                                            std::string * error_out = nullptr) {
  const std::string text = read_benchmark_manifest_file(path);
  if (text.empty()) {
    if (error_out != nullptr) {
      *error_out = "embedding variant manifest missing or unreadable";
    }
    return false;
  }

  std::string schema = {};
  if (!extract_benchmark_json_string(text, "schema", schema) ||
      schema != k_embedding_variant_schema ||
      !extract_benchmark_json_string(text, "id", out.id) ||
      !extract_benchmark_json_string(text, "case_name", out.case_name) ||
      !extract_benchmark_json_string(text, "compare_group", out.compare_group) ||
      !extract_benchmark_json_string(text, "modality", out.modality) ||
      !extract_benchmark_json_string(text, "payload_id", out.payload_id) ||
      !extract_benchmark_json_string(text, "comparison_mode", out.comparison_mode) ||
      !extract_benchmark_json_string(text, "note", out.note) ||
      !extract_benchmark_json_bool(text, "current_publication", out.current_publication)) {
    if (error_out != nullptr) {
      *error_out = "invalid embedding variant manifest";
    }
    return false;
  }
  if (out.id.empty() || out.case_name.empty() || out.compare_group.empty() ||
      out.modality.empty() || out.payload_id.empty() || out.comparison_mode.empty() ||
      out.note.empty()) {
    if (error_out != nullptr) {
      *error_out = "invalid embedding variant manifest: empty required string";
    }
    return false;
  }
  if (out.modality != "text" && out.modality != "image" && out.modality != "audio") {
    if (error_out != nullptr) {
      *error_out = "unsupported embedding variant modality";
    }
    return false;
  }
  return true;
}

inline bool load_embedding_variant_manifests(const std::filesystem::path & directory,
                                             std::vector<embedding_variant_manifest> & out,
                                             std::string * error_out = nullptr) {
  out.clear();
  std::vector<std::filesystem::path> paths = {};
  if (!discover_benchmark_manifest_paths(directory, paths, error_out)) {
    return false;
  }

  std::vector<std::pair<std::string, std::filesystem::path>> ids = {};
  for (const auto & path : paths) {
    embedding_variant_manifest manifest = {};
    if (!load_embedding_variant_manifest(path, manifest, error_out)) {
      return false;
    }
    ids.push_back({manifest.id, path});
    out.push_back(std::move(manifest));
  }
  return validate_benchmark_manifest_ids(ids, error_out);
}

}  // namespace emel::bench
