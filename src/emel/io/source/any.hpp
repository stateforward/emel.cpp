#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "emel/error/error.hpp"
#include "emel/io/read/errors.hpp"

namespace emel::io::source {

// Setup-time source-byte loading for maintained read/copy evidence lanes.
// This API is outside SML dispatch; callers pass the loaded immutable span into
// `io/read`, which remains the actor-owned copy boundary.
inline emel::error::type load_file_bytes(const std::string_view path,
                                         std::vector<uint8_t> &out) {
  out.clear();
  if (path.empty() || path.size() > ::emel::io::read::k_max_file_path_bytes ||
      path.find('\0') != std::string_view::npos) {
    return emel::error::cast(::emel::io::read::error::invalid_request);
  }

  std::string path_copy{path};
  std::FILE *file = std::fopen(path_copy.c_str(), "rb");
  if (file == nullptr) {
    return emel::error::cast(::emel::io::read::error::file_open_failed);
  }

  std::error_code file_size_error;
  const auto file_size = std::filesystem::file_size(
      std::filesystem::path{path_copy}, file_size_error);
  if (file_size_error) {
    std::fclose(file);
    return emel::error::cast(::emel::io::read::error::file_seek_failed);
  }

  const auto file_size_u64 = static_cast<uint64_t>(file_size);
  constexpr uint64_t size_max =
      static_cast<uint64_t>(static_cast<std::size_t>(-1));
  if (file_size_u64 > ::emel::io::read::k_max_read_bytes ||
      file_size_u64 > size_max) {
    std::fclose(file);
    return emel::error::cast(::emel::io::read::error::unsupported_resource);
  }

  out.resize(static_cast<std::size_t>(file_size_u64));
  const std::size_t read_size =
      out.empty() ? 0u : std::fread(out.data(), 1u, out.size(), file);
  std::fclose(file);
  return read_size == out.size()
             ? emel::error::cast(::emel::io::read::error::none)
             : emel::error::cast(::emel::io::read::error::file_read_failed);
}

} // namespace emel::io::source
