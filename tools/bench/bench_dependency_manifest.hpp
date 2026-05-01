#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace emel::bench::dependency_manifest {

inline constexpr std::string_view k_schema = "bench_dependency_manifest/v1";

enum class dependency_kind : std::uint8_t {
  source,
  config,
  fixture,
  model,
  script,
  snapshot,
};

struct dependency_record {
  std::string_view runner = {};
  dependency_kind kind = dependency_kind::source;
  std::string_view path = {};
  std::string_view reason = {};
};

struct freshness_state {
  bool missing = false;
  bool stale = false;
  bool uncertain = false;
};

std::string_view kind_name(dependency_kind kind);
std::span<const dependency_record> records();
std::span<const dependency_record> records_for(std::string_view runner);
bool requires_full_gate(freshness_state state);
freshness_state inspect(const std::filesystem::path & path, bool uncertain);
std::string render();
bool write(const std::filesystem::path & path);

}  // namespace emel::bench::dependency_manifest
