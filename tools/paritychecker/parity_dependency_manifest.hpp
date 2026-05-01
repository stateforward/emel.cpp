#pragma once

#include "parity_runner.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace emel::paritychecker::dependency_manifest {

inline constexpr std::string_view k_schema = "parity_dependency_manifest/v1";

enum class dependency_kind : uint8_t {
  source,
  config,
  fixture,
  model,
  script,
  snapshot,
};

struct dependency_record {
  parity_mode mode = parity_mode::tokenizer;
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
std::span<const dependency_record> records_for(parity_mode mode);
bool requires_full_gate(freshness_state state);
freshness_state inspect(const std::filesystem::path & path, bool uncertain);
std::string render();
bool write(const std::filesystem::path & path);

}  // namespace emel::paritychecker::dependency_manifest
