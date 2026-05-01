#include "parity_assets.hpp"

#include <cstdio>

namespace emel::paritychecker::assets {

std::filesystem::path repo_root_path() {
#ifdef PARITYCHECKER_REPO_ROOT
  return std::filesystem::path(PARITYCHECKER_REPO_ROOT);
#else
  return std::filesystem::current_path();
#endif
}

std::filesystem::path generation_baseline_directory_path() {
  return repo_root_path() / "snapshots" / "parity";
}

bool file_exists(const std::string & path) {
  std::FILE * file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return false;
  }
  std::fclose(file);
  return true;
}

bool read_file_bytes(const std::string & path, std::vector<uint8_t> & out) {
  out.clear();

  std::FILE * file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return false;
  }

  const bool seek_end_ok = std::fseek(file, 0, SEEK_END) == 0;
  const long file_size = seek_end_ok ? std::ftell(file) : -1L;
  const bool seek_start_ok = file_size >= 0 && std::fseek(file, 0, SEEK_SET) == 0;
  if (!seek_end_ok || file_size < 0 || !seek_start_ok) {
    std::fclose(file);
    return false;
  }

  out.resize(static_cast<size_t>(file_size));
  const size_t read_size = out.empty() ? 0u : std::fread(out.data(), 1u, out.size(), file);
  const bool read_ok = read_size == out.size();
  std::fclose(file);
  return read_ok;
}

std::filesystem::path normalize_path(const std::filesystem::path & path) {
  std::error_code ec;
  const std::filesystem::path absolute_path = std::filesystem::absolute(path, ec);
  if (ec) {
    return {};
  }
  return absolute_path.lexically_normal();
}

std::filesystem::path expected_generation_fixture_path(
    const maintained_generation_fixture & fixture) {
  return repo_root_path() / fixture.fixture_rel;
}

const maintained_generation_fixture * find_generation_fixture(const std::string & model_path) {
  const std::filesystem::path provided_path = normalize_path(std::filesystem::path(model_path));
  if (provided_path.empty()) {
    return nullptr;
  }

  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_maintained_generation_fixtures) {
    const std::filesystem::path expected_path =
        normalize_path(expected_generation_fixture_path(fixture));
    if (!expected_path.empty() && expected_path == provided_path) {
      return &fixture;
    }
  }
  return nullptr;
}

std::string maintained_generation_fixture_list() {
  std::string list = {};
  bool first = true;
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_maintained_generation_fixtures) {
    if (!first) {
      list += ", ";
    }
    first = false;
    list += fixture.fixture_rel;
  }
  return list;
}

}  // namespace emel::paritychecker::assets
