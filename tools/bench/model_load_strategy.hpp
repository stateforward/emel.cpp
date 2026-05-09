#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#include "emel/io/loader/events.hpp"
#include "emel/io/loader/sm.hpp"
#include "emel/model/loader/events.hpp"

namespace emel::tools {

inline constexpr const char *k_model_load_io_strategy_env =
    "EMEL_MODEL_LOAD_IO_STRATEGY";
inline constexpr const char *k_model_load_chunk_bytes_env =
    "EMEL_MODEL_LOAD_CHUNK_BYTES";
inline constexpr const char *k_model_load_constrained_ram_bytes_env =
    "EMEL_MODEL_LOAD_CONSTRAINED_RAM_BYTES";

inline uint64_t read_model_load_env_u64(const char *name,
                                        const uint64_t fallback) noexcept {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char *end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value || parsed == 0u) {
    return fallback;
  }
  return static_cast<uint64_t>(parsed);
}

inline emel::io::loader::event::strategy_kind
selected_model_load_io_strategy() noexcept {
  const char *value = std::getenv(k_model_load_io_strategy_env);
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "none") == 0) {
    return emel::io::loader::event::strategy_kind::none;
  }
  if (std::strcmp(value, "read") == 0 || std::strcmp(value, "read_copy") == 0) {
    return emel::io::loader::event::strategy_kind::read_copy;
  }
  if (std::strcmp(value, "staged") == 0 ||
      std::strcmp(value, "staged_read") == 0) {
    return emel::io::loader::event::strategy_kind::staged_read;
  }
  if (std::strcmp(value, "async") == 0 ||
      std::strcmp(value, "cooperative_async") == 0) {
    return emel::io::loader::event::strategy_kind::cooperative_async;
  }
  if (std::strcmp(value, "mapped_file") == 0 ||
      std::strcmp(value, "mmap") == 0) {
    return emel::io::loader::event::strategy_kind::mapped_file;
  }
  if (std::strcmp(value, "external_buffer") == 0) {
    return emel::io::loader::event::strategy_kind::external_buffer;
  }
  return emel::io::loader::event::strategy_kind::none;
}

inline std::string_view model_load_io_strategy_name(
    const emel::io::loader::event::strategy_kind strategy) noexcept {
  switch (strategy) {
  case emel::io::loader::event::strategy_kind::none:
    return "none";
  case emel::io::loader::event::strategy_kind::mapped_file:
    return "mapped_file";
  case emel::io::loader::event::strategy_kind::read_copy:
    return "read_copy";
  case emel::io::loader::event::strategy_kind::staged_read:
    return "staged_read";
  case emel::io::loader::event::strategy_kind::external_buffer:
    return "external_buffer";
  case emel::io::loader::event::strategy_kind::cooperative_async:
    return "cooperative_async";
  }
  return "unknown";
}

inline uint64_t selected_model_load_chunk_bytes() noexcept {
  return read_model_load_env_u64(
      k_model_load_chunk_bytes_env,
      emel::io::loader::event::k_default_staged_read_chunk_bytes);
}

inline uint64_t selected_model_load_constrained_ram_bytes() noexcept {
  return read_model_load_env_u64(k_model_load_constrained_ram_bytes_env,
                                 selected_model_load_chunk_bytes());
}

inline uint64_t selected_model_load_effective_chunk_bytes() noexcept {
  const auto chunk_bytes = selected_model_load_chunk_bytes();
  const auto constrained_ram_bytes = selected_model_load_constrained_ram_bytes();
  return chunk_bytes < constrained_ram_bytes ? chunk_bytes
                                             : constrained_ram_bytes;
}

inline void
bind_model_load_io_strategy(emel::model::loader::event::load &load,
                            emel::io::loader::sm &io_loader) noexcept {
  const auto strategy = selected_model_load_io_strategy();
  load.io_strategy = strategy;
  load.io_staged_chunk_bytes = selected_model_load_effective_chunk_bytes();
  if (strategy != emel::io::loader::event::strategy_kind::none) {
    load.io_loader = &io_loader;
  }
}

inline std::string append_model_load_io_strategy_note(
    const std::string_view note,
    const emel::io::loader::event::strategy_kind strategy) {
  std::string out{note};
  if (!out.empty()) {
    out += ' ';
  }
  out += "load_strategy=";
  out += model_load_io_strategy_name(strategy);
  return out;
}

} // namespace emel::tools
